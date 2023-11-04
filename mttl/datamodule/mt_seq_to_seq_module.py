from typing import List

import torch
from mttl.utils import logger
from datasets import load_dataset
from mttl.datamodule.base import DefaultDataModule, DatasetConfig, DefaultCollator
from mttl.datamodule.utils import maybe_filter_hf_dataset_by_task
from dataclasses import dataclass


@dataclass
class DataCollatorMTSeq2Seq(DefaultCollator):
    task_field: str = "task_name"
    task_to_id: dict = None

    def __call__(self, batch: List):
        sources = [b["source"] for b in batch]
        labels = [b["target"] for b in batch]
        task_names = [b[self.task_field] for b in batch]

        output_batch = (
            self.prepare_inputs_for_gpt_family(sources, labels)
            if self.model_family == "gpt"
            else self.prepare_inputs_for_seq2seq_family(sources, labels)
        )

        output_batch["task_names"] = task_names
        if self.task_to_id:
            output_batch["task_ids"] = torch.tensor(
                [self.task_to_id[t] for t in task_names]
            ).long()
        output_batch["source_texts"] = sources
        output_batch["label_texts"] = labels
        return output_batch


@dataclass
class FlanConfig(DatasetConfig):
    pass


class FlanModule(DefaultDataModule):
    TASK_FIELD = "task_name"

    @property
    def collate_fn(self):
        return DataCollatorMTSeq2Seq(
            tokenizer=self.tokenizer,
            model_family=self.config.model_family,
            max_output_length=self.config.max_output_length,
            max_input_length=self.config.max_input_length,
            train_on_inputs=self.config.train_on_inputs,
            task_to_id=self._task_to_id,
            task_field=self.TASK_FIELD,
        )

    def setup_dataset(self):
        dataset = load_dataset(self.config.dataset)

        (
            self._task_names,
            self._task_to_id,
            train_dataset,
            _,
            _,
        ) = maybe_filter_hf_dataset_by_task(
            dataset, "task_name", self.config.finetune_task_name
        )

        self.train_dataset, self.dev_dataset = self.create_train_valid_split(
            train_dataset
        )
        self.test_dataset = self.dev_dataset
        self.print_infos()


@dataclass
class T0FlatConfig(DatasetConfig):
    pass


class T0FlatModule(FlanModule):
    pass
