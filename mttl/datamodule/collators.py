from dataclasses import dataclass
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PaddingStrategy
from typing import List, Union, Optional
import torch

from mttl.dataloader.data_utils import ExampleInfo


@dataclass
class DefaultCollator():
    tokenizer: AutoTokenizer
    padding: Union[bool, str, PaddingStrategy] = True
    max_input_length: Optional[int] = None
    max_output_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    model_family: str = "seq2seq"

    def __call__(self, batch: List[ExampleInfo]):
        sources = [b.input for b in batch]
        # Remove multiple spaces, which mess with tiktoken (?)
        sources = [' '.join(s.split()) for s in sources]
        labels = [b.target for b in batch]
        # Add space for auto-regressive model tokenization
        labels = [' ' + l for l in labels]
        hashes = [b.hash for b in batch]
        task_ids = [b.task_id for b in batch]
        instruction_hashes = [b.instruction_hash for b in batch]

        output_batch = {}

        if self.model_family == "gpt":
            tokenized_labels = self.tokenizer(
                labels,
                max_length=self.max_output_length,
                padding=self.padding,
                return_tensors=self.return_tensors,
                truncation=True,
            )
            tok_sources_plus_labels = self.tokenizer(
                [i + t for i, t in zip(sources, labels)],
                max_length=self.max_input_length,
                padding=self.padding,
                return_tensors=self.return_tensors,
                truncation=True,
            )
            targets_len = tokenized_labels["attention_mask"].int().sum(-1)
            mask = torch.zeros_like(tok_sources_plus_labels["attention_mask"])
            mask[(torch.arange(mask.shape[0]), mask.shape[1] - targets_len)] = 1
            mask = mask.cumsum(dim=1).bool()
            labels = tok_sources_plus_labels["input_ids"].clone()
            labels = torch.masked_fill(labels, ~mask, self.label_pad_token_id)
            output_batch["input_ids"] = tok_sources_plus_labels["input_ids"]
            output_batch["attention_mask"] = tok_sources_plus_labels["attention_mask"]
            output_batch["labels"] = labels
        else:
            tokenized_labels = self.tokenizer(
                labels,
                max_length=self.max_output_length,
                padding=self.padding,
                return_tensors=self.return_tensors,
                truncation=True,
            )
            tokenized_sources = self.tokenizer(
                sources,
                max_length=self.max_input_length,
                padding=self.padding,
                return_tensors=self.return_tensors,
                truncation=True,
                pad_to_multiple_of=self.pad_to_multiple_of,
            )
            label_mask = tokenized_labels["attention_mask"].bool()
            masked_labels = tokenized_labels["input_ids"].masked_fill(
                ~label_mask, self.label_pad_token_id
            )
            output_batch["input_ids"] = tokenized_sources["input_ids"]
            output_batch["attention_mask"] = tokenized_sources["attention_mask"]
            output_batch["labels"] = labels

        output_batch["hashes"] = hashes
        output_batch["instruction_hashes"] = instruction_hashes
        output_batch["task_ids"] = torch.LongTensor(task_ids)
        return output_batch
