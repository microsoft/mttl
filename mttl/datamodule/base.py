from dataclasses import dataclass
import itertools
from pytorch_lightning import LightningDataModule
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PaddingStrategy
from typing import Any, Dict, Union, Optional

import torch
from torch.utils.data import DataLoader, Dataset

import sys
import numpy as np
from mttl.utils import logger
from mttl.datamodule.utils import get_tokenizer
from datasets import Dataset as ArrowDataset, concatenate_datasets


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
    truncation_side: str = "right"
    model_family: str = "gpt"
    train_on_inputs: bool = False
    add_eos_to_targets: bool = True
    finetune_task_name: str = None
    subsample_train: int = None
    subsample_dev: int = None
    subsample_test: int = None
    subsample_per_task: bool = False  # Changing default to False
    subsample: int = -1


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
    for_generation: bool = False
    train_on_inputs: bool = False
    task_to_id: dict = None
    add_eos_to_targets: bool = True

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
        import copy

        sources_ = copy.deepcopy(sources)
        labels_ = copy.deepcopy(labels)

        for i in range(len(sources_)):
            if self.tokenizer.mttl_merges_space and sources_[i][-1] == " ":
                # remove from sources and bring space to targets
                sources_[i] = sources_[i][:-1]
                labels_[i] = " " + labels_[i]

            if (
                sources_[i][-1] not in [" ", "\n"]
                and len(labels_[i]) > 0
                and labels_[i][0] not in [" ", "\n"]
            ):
                labels_[i] = " " + labels_[i]

        # adds the eos token
        labels_ = [
            l + ((" " + self.tokenizer.eos_token) if self.add_eos_to_targets else "")
            for l in labels_
        ]
        return sources_, labels_

    def prepare_inputs_for_seq2seq_family(self, sources, labels):
        output_batch = {}

        if self.max_input_length:
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
        output_batch = {}
        sources, labels = self.add_space_and_eos(sources, labels)

        # exit early if we are generating
        if self.for_generation:
            tokenized_sources = self.tokenizer(
                sources,
                max_length=self.max_input_length,
                padding=self.padding,
                return_tensors=self.return_tensors,
                truncation=True,
            )
            tokenized_labels = self.tokenizer(
                labels,
                max_length=self.max_output_length,
                padding=self.padding,
                return_tensors=self.return_tensors,
                truncation=True,
            )
            output_batch["input_ids"] = tokenized_sources["input_ids"]
            output_batch["attention_mask"] = tokenized_sources["attention_mask"]
            output_batch["labels"] = tokenized_labels["input_ids"]
            return output_batch

        if self.max_input_length > 0:
            if self.tokenizer.truncation_side == "left":
                tokenized_labels = self.tokenizer(
                    labels,
                    max_length=self.max_input_length,
                    padding=self.padding,
                    return_tensors=self.return_tensors,
                    truncation=True,
                )
            else:
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
            mask = torch.zeros(
                tok_sources_plus_labels["attention_mask"].shape[0],
                tok_sources_plus_labels["attention_mask"].shape[1] + 1,
            )

            # mask targets positions corresponding to the inputs
            if self.tokenizer.truncation_side == "left":
                labels_len = tokenized_labels["attention_mask"].int().sum(-1)
                pad_tokens = tok_sources_plus_labels["attention_mask"].shape[
                    1
                ] - tok_sources_plus_labels["attention_mask"].int().sum(-1)

                if self.tokenizer.padding_side == "left":
                    offset = -labels_len - 1
                else:
                    offset = torch.clamp(
                        -pad_tokens - labels_len - 1, min=-self.max_input_length, max=0
                    )
            else:
                input_len = tokenized_sources["attention_mask"].int().sum(-1)
                pad_tokens = tok_sources_plus_labels["attention_mask"].shape[
                    1
                ] - tok_sources_plus_labels["attention_mask"].int().sum(-1)

                # handle right padding here!
                if self.tokenizer.padding_side == "left":
                    offset = torch.clamp(
                        pad_tokens + input_len, max=self.max_input_length
                    )
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
        task_ids = [b.get("task_id", None) for b in batch]
        task_names = [b.get("task_name", None) for b in batch]
        task_sources = [b.get("task_source", None) for b in batch]

        output_batch = (
            self.prepare_inputs_for_gpt_family(sources, labels)
            if self.model_family == "gpt"
            else self.prepare_inputs_for_seq2seq_family(sources, labels)
        )

        has_task_sources = all(ts is not None for ts in task_sources)
        has_task_names = all(tn is not None for tn in task_names)
        has_task_ids = all(tid is not None for tid in task_ids)

        if not has_task_ids and has_task_names and self.task_to_id:
            output_batch["task_ids"] = torch.LongTensor(
                [self.task_to_id[tn] for tn in task_names]
            )
        elif has_task_ids:
            output_batch["task_ids"] = torch.LongTensor(task_ids)

        if has_task_names and not has_task_sources:
            task_sources = task_names

        output_batch["task_names"] = task_names
        output_batch["sources_texts"] = sources
        output_batch["labels_texts"] = labels
        output_batch["task_sources"] = task_sources

        # append other fields that might be available
        for key in batch[0].keys():
            if key not in output_batch:
                output_batch[key] = [b[key] for b in batch]
        return output_batch


@dataclass
class MultipleChoiceCollator(DefaultCollator):
    """A multiple choice collator useful to compute log-likelihoods of different options."""

    multisource: bool = False

    def __call__(self, batch: Dict):
        def repeat(x, num_options):
            return [x[i] for i, j in enumerate(num_options) for _ in range(j)]

        sources = [b["source"] for b in batch]
        labels = [b["target"] for b in batch]
        label_index = [b["label_index"] for b in batch]
        task_ids = [b.get("task_id", None) for b in batch]
        task_names = [b.get("task_name", None) for b in batch]
        task_sources = [b.get("task_source", None) for b in batch]

        if self.multisource:
            num_options = [len(t) for t in sources]
            labels = repeat(labels, num_options)
            sources = list(itertools.chain(*sources))
        else:
            num_options = [len(t) for t in labels]
            sources = repeat(sources, num_options)
            labels = list(itertools.chain(*labels))

        task_names = repeat(task_names, num_options)
        task_ids = repeat(task_ids, num_options)

        output_batch = (
            self.prepare_inputs_for_gpt_family(sources, labels)
            if self.model_family == "gpt"
            else self.prepare_inputs_for_seq2seq_family(sources, labels)
        )

        has_task_names = all(tn is not None for tn in task_names)
        has_task_ids = all(tid is not None for tid in task_ids)
        has_task_sources = all(ts is not None for ts in task_sources)

        if not has_task_ids and has_task_names and self.task_to_id:
            output_batch["task_ids"] = torch.LongTensor(
                [self.task_to_id[tn] for tn in task_names]
            )
        elif has_task_ids:
            output_batch["task_ids"] = torch.LongTensor(task_ids)
        if has_task_names and not has_task_sources:
            task_sources = task_names

        output_batch["sources_texts"] = sources
        output_batch["labels_texts"] = labels
        output_batch["labels_index"] = label_index
        output_batch["task_names"] = task_names
        output_batch["num_options"] = num_options
        output_batch["task_sources"] = task_sources
        return output_batch


def subsample_dst(dataset, subsample: int, rng: torch.Generator = None):
    rng = rng or torch.Generator().manual_seed(1234)
    subsample = max(len(dataset) // subsample, 1)
    if isinstance(dataset, torch.utils.data.Subset):
        idxs = dataset.indices
        idxs = idxs[:subsample]
        dataset.indices = idxs
    elif isinstance(dataset, Dataset):
        idxs = torch.randperm(len(dataset), generator=rng)[:subsample]
        dataset = torch.utils.data.Subset(dataset, idxs)
    # hugginface datasets
    elif isinstance(dataset, ArrowDataset):
        # randomly select subsample indices
        idxs = torch.randperm(len(dataset), generator=rng)[:subsample]
        dataset = dataset.select(idxs)

    return dataset


class DefaultDataModule(LightningDataModule):
    def train_dataloader(self, subsample=None):
        subsample = subsample or self.config.subsample
        train_dataset = self.train_dataset
        if subsample and subsample > 0:
            train_dataset = subsample_dst(train_dataset, subsample)

        return DataLoader(
            train_dataset,
            batch_size=self.config.train_batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            persistent_workers=False,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self, subsample=None, shuffle=False):
        subsample = subsample or self.config.subsample
        dev_dataset = self.dev_dataset
        if subsample and subsample > 0:
            dev_dataset = subsample_dst(dev_dataset, subsample)
        return DataLoader(
            dev_dataset,
            batch_size=self.config.predict_batch_size,
            shuffle=shuffle,
            num_workers=8,
            pin_memory=False,
            persistent_workers=False,
            collate_fn=self.collate_fn,
            drop_last=False,
        )

    def test_dataloader(self, subsample=None, shuffle=False):
        subsample = subsample or self.config.subsample
        test_dataset = self.test_dataset
        if subsample and subsample > 0:
            test_dataset = subsample_dst(test_dataset, subsample)
        return DataLoader(
            test_dataset,
            batch_size=self.config.predict_batch_size,
            shuffle=shuffle,
            num_workers=8,
            pin_memory=False,
            persistent_workers=False,
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
            for_generation=self.for_generation,
            train_on_inputs=self.config.train_on_inputs,
            add_eos_to_targets=self.config.add_eos_to_targets,
            task_to_id=self.task_to_id,
        )

    def print_infos(self):
        from mttl.utils import logger

        logger.info("Dataset name: %s", self.config.dataset)
        logger.info("Reader class: %s", self.__class__.__name__)
        if self.train_dataset is not None and len(self.train_dataset) > 0:
            logger.info("Training steps: %s" % len(self.train_dataloader()))
            logger.info("Training samples: %s" % len(self.train_dataset))
        if self.dev_dataset is not None:
            logger.info("Validation steps: %s" % len(self.val_dataloader()))
            logger.info("Validation samples: %s" % len(self.dev_dataset))
        if self.test_dataset is not None:
            logger.info("Test steps: %s" % len(self.test_dataloader()))
            logger.info("Test samples: %s" % len(self.test_dataset))
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

    def subsample_dataset(self, dataset, n_samples, per_task=False):
        """
        Subsamples a dataset by randomly selecting a specified number of samples.

        Args:
            train_dataset: The dataset to subsample.
            n_samples (int or float): The number of samples to subsample. If `n_samples` is less than 1, it is treated as a fraction of the total dataset size.
            per_task (bool, optional): Whether to subsample per task. Defaults to False.

        Returns:
            None

        Raises:
            AssertionError: If `per_task` is True and the dataset is not an ArrowDataset.

        """

        def get_dst_idxs_sampled(n_samples, total_size, rng):
            if n_samples < 1:
                n_samples = int(n_samples * total_size)
            idxs = torch.randperm(total_size, generator=rng)[:n_samples]
            return idxs

        total_size = len(dataset)
        # make this deterministic to always sample the same subset
        rng = torch.Generator().manual_seed(1234)
        if isinstance(dataset, ArrowDataset):
            if per_task:
                task_names = dataset.unique("task_name")
                subsampled_dataset = []
                for i, task_name in enumerate(task_names):
                    logger.info(
                        f"Subsampling task {task_name}, {i+1}/{len(task_names)}"
                    )
                    task_idxs = torch.tensor(
                        [
                            index
                            for index, value in enumerate(dataset["task_name"])
                            if value == task_name
                        ]
                    )
                    idxs = get_dst_idxs_sampled(n_samples, len(task_idxs), rng)
                    task_idxs = task_idxs[idxs]
                    task_dataset = dataset.select(task_idxs)
                    subsampled_dataset.append(task_dataset)
                    assert all([t == task_name for t in task_dataset["task_name"]])
                subsampled_dataset = concatenate_datasets(subsampled_dataset)
            else:
                idxs = get_dst_idxs_sampled(n_samples, total_size, rng)
                subsampled_dataset = dataset.select(idxs)
        else:
            assert (
                per_task is False
            ), "per_task subsampling is only supported for ArrowDataset"
            subsampled_dataset = torch.utils.data.Subset(dataset, idxs)
        return subsampled_dataset

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
        self.post_setup_dataset()

    def setup(self, stage=None):
        pass

    def setup_dataset(self):
        pass

    def post_setup_dataset(self):
        for split in ["train", "dev", "test"]:
            subsample = getattr(self.config, f"subsample_{split}", None)

            if subsample and subsample > 0:
                logger.warn(f"subsampling the {split} dataset to {subsample} samples")
                dataset = getattr(self, f"{split}_dataset")
                sub_dataset = self.subsample_dataset(
                    dataset, subsample, per_task=self.config.subsample_per_task
                )
                setattr(self, f"{split}_dataset", sub_dataset)

        self.print_infos()


class MultiChoiceDataModule(DefaultDataModule):
    @property
    def collate_fn(self):
        return MultipleChoiceCollator(
            tokenizer=self.tokenizer,
            padding="longest",
            max_input_length=self.config.max_input_length,
            max_output_length=self.config.max_output_length,
            pad_to_multiple_of=8,
            return_tensors="pt",
            model_family=self.config.model_family,
            for_generation=self.for_generation,
            train_on_inputs=self.config.train_on_inputs,
            task_to_id=self.task_to_id,
        )


class MultiChoiceSourceDataModule(DefaultDataModule):
    """
    Collates multiple sources for the same target, it's when the target is the same,
    but the source is different.
    """

    @property
    def collate_fn(self):
        return MultipleChoiceCollator(
            tokenizer=self.tokenizer,
            padding="longest",
            max_input_length=self.config.max_input_length,
            max_output_length=self.config.max_output_length,
            pad_to_multiple_of=8,
            return_tensors="pt",
            model_family=self.config.model_family,
            for_generation=self.for_generation,
            train_on_inputs=self.config.train_on_inputs,
            task_to_id=self.task_to_id,
            multisource=True,
        )


def get_datamodule(args, for_generation=False, dataset_override=None):
    from mttl.datamodule.hellaswag_data_module import (
        HellaswagDataConfig,
        HellaswagMultiChoiceDataModule,
    )

    from mttl.datamodule.openbookqa_data_module import (
        OpenbookQADataConfig,
        OpenbookQAMultiChoiceDataModule,
    )
    from mttl.datamodule.piqa_data_module import (
        PiqaDataConfig,
        PiqaMultiChoiceDataModule,
    )
    from mttl.datamodule.superglue_data_module import (
        BoolQDataModule,
        SuperGLUEDataConfig,
    )
    from mttl.datamodule.winogrande_data_module import (
        WinograndeDataConfig,
        WinograndeMultiChoiceDataModule,
    )

    from mttl.datamodule.mmlu_data_module import MMLUDataConfig, MMLUDataModule
    from mttl.datamodule.arc_data_module import ArcDataConfig, ArcMultiChoiceDataModule
    from mttl.datamodule.codex_data_module import CodexDataConfig, CodexDataModule
    from mttl.datamodule.mt_seq_to_seq_module import (
        FlanConfig,
        FlanModule,
        FlatMultiTaskConfig,
        FlatMultiTaskModule,
    )

    # refactor all the common arguments below into a dict common kwargs
    dataset = args.dataset if not dataset_override else dataset_override

    common_kwargs = {
        "model": args.model,
        "train_batch_size": args.train_batch_size,
        "predict_batch_size": args.predict_batch_size,
        "max_input_length": args.max_input_length,
        "max_output_length": args.max_output_length,
        "validation_portion": args.validation_portion,
        "model_family": args.model_family,
        "finetune_task_name": args.finetune_task_name,
        "truncation_side": args.truncation_side,
        "dataset": dataset,
        "train_on_inputs": False,
        "add_eos_to_targets": True,
        "subsample_train": args.subsample_train,
        "subsample_dev": args.subsample_dev,
        "subsample_test": args.subsample_test,
    }

    if dataset in [
        "arc-easy",
        "arc-challenge",
        "arc_easy",
        "arc_challenge",
        "openbookqa",
        "boolq",
        "piqa",
        "winogrande",
        "hellaswag",
    ]:
        dataset_to_klass_map = {
            "arc-easy": (
                ArcDataConfig(**common_kwargs, arc_type="ARC-Easy"),
                ArcMultiChoiceDataModule,
            ),
            "arc_easy": (
                ArcDataConfig(**common_kwargs, arc_type="ARC-Easy"),
                ArcMultiChoiceDataModule,
            ),
            "arc-challenge": (
                ArcDataConfig(**common_kwargs, arc_type="ARC-Challenge"),
                ArcMultiChoiceDataModule,
            ),
            "arc_challenge": (
                ArcDataConfig(**common_kwargs, arc_type="ARC-Challenge"),
                ArcMultiChoiceDataModule,
            ),
            "openbookqa": (
                OpenbookQADataConfig(**common_kwargs),
                OpenbookQAMultiChoiceDataModule,
            ),
            "boolq": (SuperGLUEDataConfig(**common_kwargs), BoolQDataModule),
            "piqa": (PiqaDataConfig(**common_kwargs), PiqaMultiChoiceDataModule),
            "winogrande": (
                WinograndeDataConfig(**common_kwargs),
                WinograndeMultiChoiceDataModule,
            ),
            "hellaswag": (
                HellaswagDataConfig(**common_kwargs),
                HellaswagMultiChoiceDataModule,
            ),
        }
        assert not for_generation
        config = dataset_to_klass_map[dataset][0]
        dm = dataset_to_klass_map[dataset][1](config)
    elif "flan" in dataset:
        config = FlanConfig(
            **common_kwargs,
            remove_phi_eval_tasks=args.remove_phi_eval_tasks,
            include_task_source=args.include_task_source,
        )
        dm = FlanModule(config, for_generation=for_generation)
    elif "flat" in dataset:
        config = FlatMultiTaskConfig(
            **common_kwargs,
            source_template=args.source_template,
            augment_few_shot=args.augment_few_shot,
        )
        dm = FlatMultiTaskModule(config, for_generation=for_generation)
    elif "mmlu" in dataset:
        config = MMLUDataConfig(
            **common_kwargs,
        )
        dm = MMLUDataModule(config, for_generation=for_generation)
    elif "codex" in dataset:
        config = CodexDataConfig(
            **common_kwargs,
        )
        dm = CodexDataModule(config, for_generation=for_generation)
    else:
        raise ValueError(f"Unknown dataset {args.dataset}")
    return dm


class AutoDataModule:
    @classmethod
    def create(cls, name, for_generation=False, val_mixin=False, **kwargs):
        from mttl.datamodule.mt_seq_to_seq_module import (
            FlanModule,
            FlanConfig,
            T0FlatModule,
            T0FlatConfig,
            FlatMultiTaskConfig,
            FlatMultiTaskModule,
        )
        from mttl.datamodule.mmlu_data_module import MMLUDataModule, MMLUDataConfig
        from mttl.datamodule.platypus_module import PlatypusModule
        from mttl.datamodule.alpaca_data_module import AlpacaDataModule
        from mttl.datamodule.t0_data_module import T0PretrainDataModule
        from mttl.datamodule.ni_data_module import NiDataModule

        if name in ["sordonia/t0-10k-flat", "sordonia/t0-1.6M-flat"]:
            return T0FlatModule(
                T0FlatConfig(dataset=name, **kwargs),
                for_generation=for_generation,
                val_mixin=val_mixin,
            )
        elif "adauni-v1-flat" in name or "platypus-flat" in name:
            return FlatMultiTaskModule(
                FlatMultiTaskConfig(dataset=name, **kwargs),
                for_generation=for_generation,
                val_mixin=val_mixin,
            )
        elif name in ["sordonia/flan-10k-flat", "sordonia/flan-debug-flat"]:
            return FlanModule(
                FlanConfig(dataset=name, **kwargs),
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
