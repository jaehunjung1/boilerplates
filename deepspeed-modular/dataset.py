import re
from argparse import Namespace
from dataclasses import dataclass
from typing import Dict, List

import ipdb
import jsonlines
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerFast, AutoTokenizer


@dataclass
class Seq2SeqSample:
    text: str
    summary: str

    @staticmethod
    def from_json(json_data: Dict):
        return Seq2SeqSample(
            text=json_data["text"],
            summary=json_data["summary"],
        )


class Seq2SeqDataset(Dataset):
    tokenizer: PreTrainedTokenizerFast = None

    def __init__(self, tokenizer: PreTrainedTokenizerFast, sample_list: List[Seq2SeqSample]):
        Seq2SeqDataset.tokenizer = tokenizer
        self.sample_list = sample_list

    def __len__(self):
        return len(self.sample_list)

    def __iter__(self):
        return iter(self.sample_list)

    def __getitem__(self, idx):
        return self.sample_list[idx]

    @staticmethod
    def from_file(tokenizer: PreTrainedTokenizerFast, filename: str | List[str], num_samples: int = None):
        if isinstance(filename, list):
            sample_list = []
            for fname in filename:
                with jsonlines.open(fname) as f:
                    sample_list += [Seq2SeqSample.from_json(data) for data in f][:num_samples]
        else:
            with jsonlines.open(filename) as f:
                sample_list = [Seq2SeqSample.from_json(data) for data in f][:num_samples]

        return Seq2SeqDataset(tokenizer, sample_list)

    @staticmethod
    def collate_fn(batched_samples: List[Seq2SeqSample]) -> dict:
        batched_text = [sample.text for sample in batched_samples]
        batched_summary = [sample.summary for sample in batched_samples]

        text_encoding = Seq2SeqDataset.tokenizer(batched_text, padding=True, max_length=1024,
                                                 truncation=True, return_tensors="pt")
        summary_encoding = Seq2SeqDataset.tokenizer(batched_summary, padding=True, max_length=256,
                                                    truncation=True, return_tensors="pt")

        return dict(
            text_encoding=text_encoding,
            summary_encoding=summary_encoding,
        )


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("google/pegasus-large", legacy=False)

    train_dataset = Seq2SeqDataset.from_file(tokenizer, "./data/cnn_dm/validation.jsonl")
    train_dataloader = DataLoader(train_dataset, batch_size=5, shuffle=True,
                                  collate_fn=Seq2SeqDataset.collate_fn, num_workers=0)

    for batch in train_dataloader:
        text = tokenizer.batch_decode(batch.text_encoding.input_ids, skip_special_tokens=False)
        summary = tokenizer.batch_decode(batch.summary_encoding.input_ids, skip_special_tokens=False)
        ipdb.set_trace()
        pass
