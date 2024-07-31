import json
from argparse import Namespace
from pathlib import Path
from typing import List

import ipdb
import jsonlines
from tqdm import tqdm

from models.openai_model import OpenAIModel


def save_to_file(sample_list: List[dict], out_filename: str | Path, save_mode: str = 'w') -> object:
    assert save_mode in ['w', 'a'], "Save mode should be either `w` or `a`."

    if len(sample_list) == 0:
        return

    sample_str_list = [json.dumps(sample) for sample in sample_list]
    with open(out_filename, save_mode) as f:
        f.write("\n".join(sample_str_list) + "\n")


def parse_args():
    args = Namespace()

    args.dataset_name = "strategy_qa/train"
    args.model_name = "gpt-3.5-turbo"

    args.batch_size = 16
    args.start_idx = 0
    args.end_idx = 10000

    args.out_filename = Path(f"./result/{args.dataset_name}-{args.model_name}-{args.start_idx}-{args.end_idx}.jsonl.tmp")
    args.in_filename = Path(f"./data/{args.dataset_name}.jsonl")

    print(f"Will be saved to: {args.out_filename}")

    return args


def extract_answer(generation: str):
    generation = generation.lower()
    if "answer: yes" in generation:
        answer = True
    elif "answer: no" in generation:
        answer = False
    else:
        answer = None

    return answer


if __name__ == "__main__":
    args = parse_args()

    # if generation file already exists, exit
    if args.out_filename.exists():
        raise FileExistsError

    # load data
    with jsonlines.open(args.in_filename) as f:
        samples = list(f)[args.start_idx:args.end_idx]

    model = OpenAIModel(model_name=args.model_name, dataset_name=args.dataset_name)

    for batch_start_idx in tqdm(range(0, len(samples), args.batch_size)):
        batch = samples[batch_start_idx:batch_start_idx + args.batch_size]
        questions: List[str] = [sample["question"] for sample in batch]

        batch_generations = model.generate_y(questions, 1)  # List[List[str]]

        for sample_idx in range(len(batch)):
            sample_generations = batch_generations[sample_idx]
            sample = batch[sample_idx]
            sample["generation"] = sample_generations[0]
            sample["inferred_answer"] = extract_answer(sample["generation"])

        # save generation to .tmp file
        save_to_file(batch, args.out_filename, save_mode='a')


