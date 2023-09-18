from dataclasses import dataclass
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PaddingStrategy
from typing import List, Union, Optional
import torch

from mttl.dataloader.data_utils import ExampleInfo


@dataclass
class DefaultCollator:
    """Simple collator

    Converts a batch of examples into a batch of inputs and labels for a sequence to sequence task.
    If model_family is "gpt", then the inputs and outputs are constructed for a causal language model,
    e.g. concatenated in a single string and labels are set to be -100 for all tokens in the input.
    """

    tokenizer: AutoTokenizer
    padding: Union[bool, str, PaddingStrategy] = True
    max_input_length: Optional[int] = None
    max_output_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    model_family: str = "seq2seq"
    train_on_inputs: bool = False

    def enforce_eos(self, targets):
        # simulate the default behaviour of LLamatokenizer, when adding eos token and truncating: the last token must always be eos
        # make sure the last token is eos
        if self.tokenizer.padding_side == "left":
            targets[(torch.arange(targets.shape[0]), -1)] = self.tokenizer.eos_token_id
        else:
            # make sure last token is eos if not -100
            targets[(torch.arange(targets.shape[0]), -1)] = torch.where(
                targets[(torch.arange(targets.shape[0]), -1)]
                != self.label_pad_token_id,
                self.tokenizer.eos_token_id,
                self.label_pad_token_id,
            )
        return targets

    def prepare_inputs_for_seq2seq_family(self, sources, labels):
        output_batch = {}
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
        output_batch["labels"] = masked_labels
        return output_batch

    def prepare_inputs_for_gpt_family(self, sources, labels):
        # Add eos token
        labels = [l + " " + self.tokenizer.eos_token for l in labels]

        output_batch = {}
        if self.max_input_length > 0:
            tokenized_sources = self.tokenizer(
                sources,
                max_length=self.max_input_length,
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
                pad_to_multiple_of=self.pad_to_multiple_of,
            )
        else:
            tokenized_sources = self.tokenizer(
                sources,
                padding="longest",
                return_tensors=self.return_tensors,
            )
            tok_sources_plus_labels = self.tokenizer(
                [i + t for i, t in zip(sources, labels)],
                padding="longest",
                return_tensors=self.return_tensors,
                pad_to_multiple_of=self.pad_to_multiple_of,
            )
        targets = tok_sources_plus_labels["input_ids"].clone()
        targets = torch.masked_fill(
            targets,
            ~tok_sources_plus_labels["attention_mask"].bool(),
            self.label_pad_token_id,
        )

        if not self.train_on_inputs:
            # mask targets positions corresponding to the inputs
            input_len = tokenized_sources["attention_mask"].int().sum(-1)
            pad_tokens = tok_sources_plus_labels["attention_mask"].shape[
                1
            ] - tok_sources_plus_labels["attention_mask"].int().sum(-1)
            mask = torch.zeros(
                tok_sources_plus_labels["attention_mask"].shape[0],
                tok_sources_plus_labels["attention_mask"].shape[1] + 1,
            )
            # handle right padding here!
            if self.tokenizer.padding_side == "left":
                offset = torch.clamp(pad_tokens + input_len, max=self.max_input_length)
            else:
                offset = input_len

            mask[(torch.arange(mask.shape[0]), offset)] = 1
            mask = mask.cumsum(dim=1).bool()
            mask = mask[:, :-1]
            targets = torch.masked_fill(targets, ~mask, self.label_pad_token_id)

        if getattr(self.tokenizer, "mttl_enforces_eos", False):
            targets = self.enforce_eos(targets)

        output_batch["input_ids"] = tok_sources_plus_labels["input_ids"]
        output_batch["attention_mask"] = tok_sources_plus_labels["attention_mask"]
        output_batch["labels"] = targets
        return output_batch

    def __call__(self, batch: List[ExampleInfo]):
        sources = [b.input for b in batch]
        labels = [b.target for b in batch]
        hashes = [b.hash for b in batch]
        task_ids = [b.task_id for b in batch]
        instruction_hashes = [b.instruction_hash for b in batch]

        output_batch = (
            self.prepare_inputs_for_gpt_family(sources, labels)
            if self.model_family == "gpt"
            else self.prepare_inputs_for_seq2seq_family(sources, labels)
        )
        output_batch["hashes"] = hashes
        output_batch["instruction_hashes"] = instruction_hashes
        output_batch["task_ids"] = torch.LongTensor(task_ids)
        return output_batch
