import itertools
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import torch
import torch.nn.functional as F
from datasets import Dataset as ArrowDataset
from datasets import concatenate_datasets
from pytorch_lightning import LightningDataModule
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PaddingStrategy

from mttl.datamodule.utils import get_tokenizer
from mttl.logging import logger
from mttl.registrable import Registrable


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
    pack_sequences: bool = False  # True
    pad_to_multiple_of: int = 8
    max_seq_per_pack: int = 4


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

    def _get_nested_type(self, item):
        while isinstance(item, (list, tuple)):
            item = item[0]
        return type(item)

    def _tensor_dtype(self, item):
        dtype = self._get_nested_type(item)
        return {"int": torch.int64, "float": torch.float32, "bool": torch.bool}.get(
            dtype.__name__, None
        )

    def __call__(self, batch: Dict):
        # is our input already tokenized ?
        # trim according to the attention mask and return

        def pad_sequence_wrapper(tensor_list, batch_first, padding_value, side="right"):
            """Padding Sequence Fn that supports left padding"""
            if side == "left":
                tensor_list = [x.flip(0) for x in tensor_list]

            padded = pad_sequence(
                tensor_list, batch_first=batch_first, padding_value=padding_value
            )

            if side == "left":
                padded = padded.flip(1)

            return padded

        if "input_ids" in batch[0]:
            output_batch = defaultdict(list)
            for batch_item in batch:
                for key, value in batch_item.items():
                    dtype = self._tensor_dtype(value)

                    if dtype:
                        output_batch[key].append(torch.Tensor(value).to(dtype))
                    else:
                        output_batch[key].append(value)

            # create proper containers
            for key, value in output_batch.items():
                if isinstance(value[0], torch.Tensor):
                    pad_token = {
                        "input_ids": self.tokenizer.pad_token_id,
                        "labels": self.label_pad_token_id,
                    }.get(key, 0)
                    value = pad_sequence_wrapper(
                        value,
                        batch_first=True,
                        padding_value=pad_token,
                        side=self.tokenizer.padding_side,
                    )
                    output_batch[key] = value

            packed_seq_lens = output_batch["seq_lens"].flatten().cumsum(0)
            output_batch["packed_seq_lens"] = F.pad(packed_seq_lens, (1, 0)).to(
                torch.int32
            )

            # build the appropriate "block"-lower triangular mask for sdpa attention
            bs, seq_len = output_batch["input_ids"].shape
            packed_attn_mask = torch.zeros(bs, 1, seq_len, seq_len, dtype=torch.bool)
            for i in range(bs):
                start_idx = 0
                for seq_len in output_batch["seq_lens"][i]:
                    packed_attn_mask[
                        i,
                        :,
                        start_idx : start_idx + seq_len,
                        start_idx : start_idx + seq_len,
                    ] = True
                    start_idx += seq_len

                # For whatever reason, we need to let padding tokens attend the previous context ¯\_(ツ)_/¯
                # Otherwise SDPA has nans
                packed_attn_mask[i, :, start_idx:, :start_idx] = True

            packed_attn_mask = packed_attn_mask.tril()
            output_batch["packed_attn_mask"] = packed_attn_mask

            return dict(output_batch)

        # Otherwise process as expected
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


class DataModule(LightningDataModule, Registrable):
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
            pad_to_multiple_of=self.config.pad_to_multiple_of,
            return_tensors="pt",
            model_family=self.config.model_family,
            for_generation=self.for_generation,
            train_on_inputs=self.config.train_on_inputs,
            add_eos_to_targets=self.config.add_eos_to_targets,
            task_to_id=self.task_to_id,
        )

    def print_infos(self):
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
            logger.warning(
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

        def get_dst_idxs_sampled(n_samples, total_size):
            rng = torch.Generator().manual_seed(1234)
            if n_samples < 1:
                n_samples = int(n_samples * total_size)
            idxs = torch.randperm(total_size, generator=rng)[:n_samples]
            return idxs

        total_size = len(dataset)
        # make this deterministic to always sample the same subset
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
                    idxs = get_dst_idxs_sampled(n_samples, len(task_idxs))
                    task_idxs = task_idxs[idxs]
                    task_dataset = dataset.select(task_idxs)
                    subsampled_dataset.append(task_dataset)
                    assert all([t == task_name for t in task_dataset["task_name"]])
                subsampled_dataset = concatenate_datasets(subsampled_dataset)
            else:
                idxs = get_dst_idxs_sampled(n_samples, total_size)
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

    def tokenize_dataset(self, dataset):

        # NOTE: padding is hardcoded to `longest` already.
        # return tensors is harcoded to `pt`, but tokenizer in dataset.map overwrites this
        # TODO: write a test for this
        pad_to_multiple = self.config.pad_to_multiple_of
        self.config.pad_to_multiple_of = 1

        # remove `rng` before mapping, as it's not pickleable
        rng = self.rng
        self.rng = None

        def collate_fn_wrapper(batch):
            out = self.collate_fn([batch])
            return {k: v[0] for k, v in out.items()}

        dataset = dataset.map(collate_fn_wrapper, batched=False, num_proc=20)
        self.rng = rng
        self.collate_fn.pad_to_multiple_of = pad_to_multiple

        return dataset

    def pack_sequences(self, dataset, max_sequences=4, shuffle=True):
        """
        Combine sequences together in larger chunks closer to `max_input_length`
        """
        # first, let's shuffle the dataset
        if shuffle:
            dataset = dataset.shuffle(seed=42)

        # TODO: first partition dataset according to `task_name`, and
        # pack each task individually to ensure that we don't mix tasks

        # Very basic code that will iterate over sequences one by one,
        # and merge together until the max_input_length is reached
        # This is not optimal, but it's a start
        max_length = self.config.max_input_length

        def group(examples):

            def new_container():
                # for when starting a new packed batch
                return {k: [] for k in list(examples.keys()) + ["seq_lens"]}

            grouped_samples = new_container()

            def append_to_running_seq(container, example):
                for k, v in example.items():
                    if isinstance(v, int) or isinstance(v, str):
                        container[k] += [v]
                    elif isinstance(v, list):
                        container[k] += v
                    else:
                        raise ValueError(f"Unknown type {type(v)}")

                # TODO: THis is SOMEHOW WRONG. CHECK.
                container["seq_lens"] += [len(example["input_ids"])]

            def add_finished_sequence(container, example):
                for k, v in example.items():
                    container[k].append(v)

            def trim_ex(ex):
                for key in ex.keys():
                    value = ex[key]
                    if isinstance(value, list):
                        ex[key] = value[:max_length]

            def dict_get_item(ex, i):
                return {k: v[i] for k, v in ex.items()}

            num_examples = len(examples["input_ids"])
            packed = new_container()
            current_lens = []
            for i in range(num_examples):
                ex = dict_get_item(examples, i)
                ex_len = len(ex["input_ids"])
                # can pack
                if (
                    sum(current_lens) + ex_len <= max_length
                    and len(current_lens) < max_sequences
                ):
                    append_to_running_seq(packed, ex)
                    current_lens += [ex_len]
                else:
                    if len(current_lens) > 0:
                        add_finished_sequence(grouped_samples, packed)
                    packed = new_container()
                    current_lens = []
                    trim_ex(ex)
                    append_to_running_seq(packed, ex)
                    current_lens += [ex_len]

            if len(current_lens) > 0:
                add_finished_sequence(grouped_samples, packed)

            return grouped_samples

        dataset = dataset.map(
            group,
            num_proc=20,
            batched=True,
            batch_size=10_000,
            remove_columns=list(dataset.features),
        )
        return dataset

    def post_setup_dataset(self):
        for split in ["train", "dev", "test"]:

            subsample = getattr(self.config, f"subsample_{split}", None)
            if subsample and subsample > 0:
                dataset = getattr(self, f"{split}_dataset")
                logger.warning(
                    f"subsampling the {split} dataset to {subsample} samples"
                )
                sub_dataset = self.subsample_dataset(
                    dataset, subsample, per_task=self.config.subsample_per_task
                )

                setattr(self, f"{split}_dataset", sub_dataset)

            if self.config.pack_sequences and split == "train":
                dataset = getattr(self, f"{split}_dataset")
                logger.info(f"Packing sequences for {split} dataset")
                dataset = self.tokenize_dataset(dataset)
                dataset = self.pack_sequences(
                    dataset, max_sequences=self.config.max_seq_per_pack
                )
                setattr(self, f"{split}_dataset", dataset)

        self.print_infos()


class MultiChoiceDataModule(DataModule):
    @property
    def collate_fn(self):
        return MultipleChoiceCollator(
            tokenizer=self.tokenizer,
            padding="longest",
            max_input_length=self.config.max_input_length,
            max_output_length=self.config.max_output_length,
            pad_to_multiple_of=self.config.pad_to_multiple_of,
            return_tensors="pt",
            model_family=self.config.model_family,
            for_generation=self.for_generation,
            train_on_inputs=self.config.train_on_inputs,
            task_to_id=self.task_to_id,
            add_eos_to_targets=self.config.add_eos_to_targets,
        )


class MultiChoiceSourceDataModule(DataModule):
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
            add_eos_to_targets=self.config.add_eos_to_targets,
        )


def get_datamodule(args, for_generation=False, dataset_override=None):
    from mttl.datamodule.arc_data_module import ArcDataConfig, ArcMultiChoiceDataModule
    from mttl.datamodule.codex_data_module import CodexDataConfig, CodexDataModule
    from mttl.datamodule.hellaswag_data_module import (
        HellaswagDataConfig,
        HellaswagMultiChoiceDataModule,
    )
    from mttl.datamodule.mmlu_data_module import MMLUDataConfig, MMLUDataModule
    from mttl.datamodule.mt_seq_to_seq_module import (
        FlanConfig,
        FlanModule,
        FlatMultiTaskConfig,
        FlatMultiTaskModule,
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
        "subsample_per_task": args.subsample_per_task,
        "pad_to_multiple_of": args.pad_to_multiple_of,
        "padding_side": args.padding_side,
        "max_seq_per_pack": args.max_seq_per_pack,
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
            pack_sequences=args.pack_sequences,
        )
        dm = FlanModule(config, for_generation=for_generation)
    elif "flat" in dataset:
        config = FlatMultiTaskConfig(
            **common_kwargs,
            source_template=args.source_template,
            augment_few_shot=args.augment_few_shot,
            pack_sequences=args.pack_sequences,
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
