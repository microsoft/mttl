from dataclasses import dataclass
from pytorch_lightning import LightningDataModule
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PaddingStrategy
from typing import Any, Dict, Union, Optional

import torch
from torch.utils.data import DataLoader

from mttl.utils import logger
from mttl.datamodule.utils import get_tokenizer


@dataclass
class DatasetConfig:
    dataset: str = None
    data_dir: str = None
    model: str = None
    train_batch_size: int = 4
    predict_batch_size: int = 4
    max_input_length: int = 1024
    max_output_length: int = 128
    validation_portion: float = None
    padding_side: str = "right"
    model_family: str = "gpt"
    train_on_inputs: bool = False
    finetune_task_name: str = None


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

    def add_space_and_eos(self, sources, labels):
        """Some tokenizers (e.g. gpt2) merge space with the next token. This will create problems when creating the
        mask for the targets because the input will not be a subset of the concatenation of the input + label.

        This function moves the space to the targets instead, and removes it from the sources.
        """
        for i in range(len(sources)):
            if sources[i][-1] == " ":
                # remove from sources and bring space to targets
                sources[i] = sources[i][:-1]
                labels[i] = " " + labels[i]
            elif sources[i][-1] not in [" ", "\n"] and labels[i][0] not in [" ", "\n"]:
                # adds a space to targets by default
                labels[i] = " " + labels[i]

        # adds the eos token
        labels = [l + " " + self.tokenizer.eos_token for l in labels]
        return sources, labels

    def prepare_inputs_for_seq2seq_family(self, sources, labels):
        output_batch = {}

        if self.max_input_length > 0:
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
        else:
            tokenized_labels = self.tokenizer(
                labels, padding="longest", return_tensors=self.return_tensors
            )
            tokenized_sources = self.tokenizer(
                sources,
                padding="longest",
                return_tensors=self.return_tensors,
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
        sources, labels = self.add_space_and_eos(sources, labels)

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

    def __call__(self, batch: Dict):
        sources = [b["source"] for b in batch]
        labels = [b["target"] for b in batch]
        task_ids = [b.get("task_id", 0) for b in batch]
        task_names = [b.get("task_name", None) for b in batch]

        output_batch = (
            self.prepare_inputs_for_gpt_family(sources, labels)
            if self.model_family == "gpt"
            else self.prepare_inputs_for_seq2seq_family(sources, labels)
        )

        output_batch["task_ids"] = torch.LongTensor(task_ids)
        output_batch["task_names"] = task_names
        output_batch["sources_texts"] = sources
        output_batch["labels_texts"] = labels
        return output_batch


class DefaultDataModule(LightningDataModule):
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.train_batch_size,
            shuffle=True,
            num_workers=16,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dev_dataset,
            batch_size=self.config.predict_batch_size,
            shuffle=False,
            num_workers=16,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=self.collate_fn,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.predict_batch_size,
            shuffle=False,
            num_workers=16,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=self.collate_fn,
            drop_last=False,
        )

    @property
    def collate_fn(self):
        return DefaultCollator(
            tokenizer=self.tokenizer,
            padding="longest",
            max_input_length=self.config.max_input_length,
            max_output_length=self.config.max_output_length,
            pad_to_multiple_of=8,
            return_tensors="pt",
            model_family=self.config.model_family,
            train_on_inputs=self.config.train_on_inputs,
        )

    def print_infos(self):
        from mttl.utils import logger

        if len(self.train_dataset) > 0:
            logger.info("Training steps: %s" % len(self.train_dataloader()))
        if self.dev_dataset is not None:
            logger.info("Validation steps: %s" % len(self.val_dataloader()))
        if self.test_dataset is not None:
            logger.info("Test steps: %s" % len(self.test_dataloader()))
        if self.task_names:
            logger.info("Number of tasks: %s" % len(self.task_names))

    @property
    def task_names(self):
        return self._task_names

    @property
    def task_to_id(self):
        return self._task_to_id

    def create_train_valid_split(self, dataset, validation_portion=None):
        # always use the same split for the dataset
        validation_portion = validation_portion or self.config.validation_portion

        if validation_portion is None:
            logger.warn(
                "No validation portion specified, no dev set available for this dataset."
            )
            return dataset, None

        n_tr_samples = int(len(dataset) * (1 - validation_portion))

        train_dataset, dev_dataset = torch.utils.data.random_split(
            dataset,
            [
                n_tr_samples,
                len(dataset) - n_tr_samples,
            ],
            generator=self.rng,
        )
        return train_dataset, dev_dataset

    def __init__(
        self, config: Union[DatasetConfig, Any], for_generation=False, val_mixin=None
    ):
        super().__init__()
        self.rng = torch.Generator().manual_seed(1234)
        self.config = config
        self._task_names = []
        self._task_to_id = {}
        self.val_mixin = val_mixin
        self.for_generation = for_generation
        self.tokenizer = get_tokenizer(config, for_generation=for_generation)
        self.setup_dataset()

    def setup(self, stage=None):
        pass

    def setup_dataset(self):
        pass


class AutoDataModule:
    @classmethod
    def create(cls, name, for_generation=False, val_mixin=False, **kwargs):
        from mttl.datamodule.mt_seq_to_seq_module import (
            FlanModule,
            T0FlatModule,
        )
        from mttl.datamodule.mmlu_data_module import MMLUDataModule, MMLUDataConfig
        from mttl.datamodule.platypus_module import PlatypusModule
        from mttl.datamodule.alpaca_data_module import AlpacaDataModule
        from mttl.datamodule.t0_data_module import T0PretrainDataModule
        from mttl.datamodule.ni_data_module import NiDataModule

        if name in ["sordonia/t0-10k-flat", "sordonia/t0-1.6M-flat"]:
            return T0FlatModule(
                DatasetConfig(dataset=name, **kwargs),
                for_generation=for_generation,
                val_mixin=val_mixin,
            )
        elif name in ["sordonia/flan-10k-flat", "sordonia/flan-debug-flat"]:
            return FlanModule(
                DatasetConfig(dataset=name, **kwargs),
                for_generation=for_generation,
                val_mixin=val_mixin,
            )
        elif name in ["mmlu"]:
            return MMLUDataModule(
                MMLUDataConfig(dataset=name, **kwargs),
                for_generation=for_generation,
                val_mixin=val_mixin,
            )
        elif name in ["alpaca"]:
            return AlpacaDataModule(
                DatasetConfig(dataset=name, **kwargs),
                for_generation=for_generation,
                val_mixin=val_mixin,
            )
        elif name in ["platypus"]:
            return PlatypusModule(
                DatasetConfig(dataset=name, **kwargs),
                for_generation=for_generation,
                val_mixin=val_mixin,
            )
        elif name in ["t0"]:
            return T0PretrainDataModule(kwargs.pop("config"))
        elif name in ["ni"]:
            return NiDataModule(kwargs.pop("config"), for_generation=for_generation)
        else:
            raise ValueError(f"Unknown dataset {name}")
