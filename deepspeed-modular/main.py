import os
from argparse import ArgumentParser, Namespace
from typing import Tuple

import ipdb
import wandb
from accelerate import Accelerator
from accelerate.utils import DummyOptim
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast, PreTrainedModel, AutoTokenizer, AutoModelForSeq2SeqLM

from dataset import Seq2SeqDataset
from pipeline import train_one_epoch, evaluate
from util import set_seed


def parse_args():
    args = ArgumentParser(description="Seq2Seq Training")

    # Experiments
    args.add_argument("--seed", default=999, type=int, help="Random seed.")
    args.add_argument("--save_ckpt", action="store_true", help="Save best checkpoint or not.")
    args.add_argument("--save_dir", default="./models", help="Model saving directory.")
    args.add_argument("--disabled", action="store_true", help="Disable wandb.")
    args.add_argument("--model_name", default="google/pegasus-large", help="PLM name.")

    # Hyperparameters
    args.add_argument("--learning_rate", default=5e-5, type=float)
    args.add_argument("--train_batch_size", default=2, type=int)
    args.add_argument("--accumulation_steps", default=1, type=int)
    args.add_argument("--valid_batch_size", default=4, type=int)
    args.add_argument("--num_epochs", default=3, type=int)
    args.add_argument("--max_grad_norm", default=1.0, type=float)

    # Debugging
    args.add_argument("--num_samples", default=2000, type=int)

    args = args.parse_args()

    # Files
    args.train_filenames = [
        "./data/cnn_dm/train.jsonl"
    ]
    args.valid_filenames = [
        "./data/cnn_dm/validation.jsonl"
    ]

    # Accelerator
    args.accelerator = Accelerator(log_with="wandb" if not args.disabled else None,
                                   gradient_accumulation_steps=args.accumulation_steps)
    args.device = args.accelerator.device

    # # Update gradient accumulation step if deepspeed config specifies it
    # if args.accelerator.state.deepspeed_plugin is not None and \
    #         "gradient_accumulation_steps" in args.accelerator.state.deepspeed_plugin.deepspeed_config:
    #     args.accumulation_steps = args.accelerator.state.deepspeed_plugin.deepspeed_config[
    #         "gradient_accumulation_steps"
    #     ]

    return args


def init_tokenizer_and_model(args: Namespace) -> Tuple[PreTrainedTokenizerFast, PreTrainedModel]:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, legacy=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    return tokenizer, model


def main(args):
    set_seed(args.seed)

    # Initialize wandb
    args.accelerator.init_trackers(
        "Seq2Seq Training",
        config=args,
    )

    with args.accelerator.main_process_first():
        run = args.accelerator.get_tracker("wandb") if not args.disabled else None
        wandb_table = wandb.Table(columns=["Epoch", "Generation"]) if not args.disabled else None

    if args.accelerator.is_main_process and not args.disabled:
        args.ckpt_dir = os.path.join(args.save_dir, run.run.name)

        if args.save_ckpt:
            os.makedirs(args.ckpt_dir, exist_ok=True)
            args.accelerator.print(f"Model-saving directory: {args.ckpt_dir}")

    # Initialize tokenizer, model
    tokenizer, model = init_tokenizer_and_model(args)

    # Initialize optimizer
    # Creates Dummy Optimizer if `optimizer` was specified in the config file else creates AdamW Optimizer
    if args.accelerator.state.deepspeed_plugin is None or \
            "optimizer" not in args.accelerator.state.deepspeed_plugin.deepspeed_config:
        optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    else:
        optimizer = DummyOptim(model.parameters(), lr=args.learning_rate)

    # Initialize datasets
    train_dataset = Seq2SeqDataset.from_file(tokenizer, args.train_filenames, args.num_samples)
    valid_dataset = Seq2SeqDataset.from_file(tokenizer, args.valid_filenames, args.num_samples)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True,
                                  collate_fn=Seq2SeqDataset.collate_fn, num_workers=4)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.valid_batch_size, shuffle=False,
                                  collate_fn=Seq2SeqDataset.collate_fn, num_workers=4)

    # Prepare with accelerator
    model, optimizer, train_dataloader, valid_dataloader = args.accelerator.prepare(
        model, optimizer, train_dataloader, valid_dataloader
    )

    # Main routine
    best_ppl = 1000
    for epoch in range(1, args.num_epochs + 1):
        train_one_epoch(model, train_dataloader, optimizer, tokenizer, args, epoch)

        ppl, wandb_table = evaluate(model, valid_dataloader, tokenizer, args, epoch, wandb_table)

        if args.accelerator.is_main_process and ppl < best_ppl:
            best_ppl = ppl

            if args.save_ckpt:
                args.accelerator.unwrap_model(model).save_pretrained(args.ckpt_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args)

# Run: accelerate launch --config_file=./configs/bf16_accelerate.yaml main.py --save_ckpt
# Debug: accelerate launch --config_file ./configs/bf16_accelerate.yaml main.py --disabled
