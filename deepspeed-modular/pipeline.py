from argparse import Namespace

import ipdb
import torch
import wandb
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer, AdamW
from tqdm import tqdm
from transformers import GenerationConfig, PreTrainedTokenizerFast


def train_one_epoch(model: nn.Module, dataloader: DataLoader, optimizer: Optimizer, tokenizer: PreTrainedTokenizerFast,
                    args: Namespace, epoch: int):
    model.train()

    with tqdm(dataloader, desc=f"Train Ep {epoch}", total=len(dataloader),
              disable=not args.accelerator.is_local_main_process) as tq:
        for batch in tq:
            with args.accelerator.accumulate(model):
                text_encoding = batch["text_encoding"]
                summary_encoding = batch["summary_encoding"]

                labels = summary_encoding.input_ids.masked_fill(
                    summary_encoding.input_ids == tokenizer.pad_token_id, -100
                )

                output = model(**text_encoding, labels=labels)
                loss = output.loss
                args.accelerator.backward(loss)
                if args.accelerator.sync_gradients:
                    args.accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

                loss = loss.detach().item()

                args.accelerator.log({"train_loss": loss})


def evaluate(model: nn.Module, dataloader: DataLoader, tokenizer: PreTrainedTokenizerFast,
             args: Namespace, epoch: int, wandb_table: wandb.Table):
    model.eval()

    generation_batch = None
    loss_sum = 0.

    total_sample_count = 0.
    with tqdm(dataloader, desc=f"Eval", total=len(dataloader),
              disable=not args.accelerator.is_local_main_process) as tq:
        for batch in tq:
            text_encoding = batch["text_encoding"]
            summary_encoding = batch["summary_encoding"]

            labels = summary_encoding.input_ids.masked_fill(
                summary_encoding.input_ids == tokenizer.pad_token_id, -100
            )

            with torch.no_grad():
                output = model(**text_encoding, labels=labels)
                loss = output.loss

                # number of all samples in the batch, across all GPUs
                gathered_batch_sample_count = args.accelerator.gather(torch.tensor(labels.size(0)).to(args.device))
                gathered_batch_sample_count = torch.sum(gathered_batch_sample_count).item()

                # sum of all loss across all GPUs
                gathered_loss = args.accelerator.gather(loss * labels.size(0))
                gathered_loss = torch.sum(gathered_loss).item()

                total_sample_count += gathered_batch_sample_count
                loss_sum += gathered_loss

            if generation_batch is None:
                generation_batch = batch

    avg_loss = loss_sum / total_sample_count
    args.accelerator.log({"eval_avg_loss": avg_loss})

    # Generate with greedy decoding
    generation_table = None
    if args.accelerator.is_main_process:
        generation_config = GenerationConfig(
            max_new_tokens=200, num_return_sequences=1,
            do_sample=False, top_p=1, temperature=0,
            pad_token_id=tokenizer.pad_token_id
        )
        text_encoding = generation_batch["text_encoding"].to(args.accelerator.device)
        output_sequences = tokenizer.batch_decode(args.accelerator.unwrap_model(model).generate(
            **text_encoding,
            generation_config=generation_config,
        ), skip_special_tokens=False)

        if wandb_table is not None:
            for sequence in output_sequences:
                wandb_table.add_data(epoch, sequence)
            generation_table = wandb.Table(columns=wandb_table.columns, data=wandb_table.data)
            args.accelerator.log({"Generation": generation_table})

    return avg_loss, generation_table



