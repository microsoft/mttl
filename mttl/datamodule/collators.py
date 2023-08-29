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
        inputs = [b.input for b in batch]
        targets = [b.target for b in batch]
        # Add space for auto-regressive model tokenization
        targets = [' ' + l for l in targets]
        # Remove multiple spaces, which mess with tiktoken (?)
        inputs = [' '.join(s.split()) for s in inputs]
        hashes = [b.hash for b in batch]
        task_ids = [b.task_id for b in batch]
        instruction_hashes = [b.instruction_hash for b in batch]

        output_batch = {}

        tok_targets = self.tokenizer(
            targets,
            max_length=self.max_output_length,
            padding=self.padding,
            return_tensors=self.return_tensors,
            truncation=True,
        )
        label_mask = tok_targets["attention_mask"].bool()
        labels = tok_targets["input_ids"].masked_fill(
            ~label_mask, self.label_pad_token_id
        )

        if self.model_family == "gpt":
            tok_inputs_plus_targets = self.tokenizer(
                [i + t for i, t in zip(inputs, targets)],
                max_length=self.max_input_length,
                padding=self.padding,
                return_tensors=self.return_tensors,
                truncation=True,
            )
            targets_len = tok_targets["attention_mask"].int().sum(-1)
            mask = torch.zeros_like(tok_inputs_plus_targets["attention_mask"])
            mask[(torch.arange(mask.shape[0]), mask.shape[1] - targets_len)] = 1
            mask = mask.cumsum(dim=1).bool()
            labels = tok_inputs_plus_targets["input_ids"].clone()
            labels = torch.masked_fill(labels, ~mask, self.label_pad_token_id)
            output_batch["input_ids"] = tok_inputs_plus_targets["input_ids"]
            output_batch["attention_mask"] = tok_inputs_plus_targets["attention_mask"]
            output_batch["labels"] = labels
        else:
            tok_inputs = self.tokenizer(
                inputs,
                max_length=self.max_input_length,
                padding=self.padding,
                return_tensors=self.return_tensors,
                truncation=True,
                pad_to_multiple_of=self.pad_to_multiple_of,
            )
            output_batch["input_ids"] = tok_inputs["input_ids"]
            output_batch["attention_mask"] = tok_inputs["attention_mask"]
            output_batch["labels"] = labels

        output_batch["hashes"] = hashes
        output_batch["instruction_hashes"] = instruction_hashes
        output_batch["task_ids"] = torch.LongTensor(task_ids)
        return output_batch
