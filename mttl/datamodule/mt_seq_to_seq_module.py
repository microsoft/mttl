from typing import List

import torch
from mttl.utils import logger
from datasets import load_dataset
from mttl.datamodule.base import DefaultDataModule, DatasetConfig, DefaultCollator
from mttl.datamodule.utils import maybe_filter_hf_dataset_by_task
from dataclasses import dataclass


@dataclass
class FlanConfig(DatasetConfig):
    pass


class FlanModule(DefaultDataModule):
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
