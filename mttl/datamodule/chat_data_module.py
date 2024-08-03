import os
from functools import partial

import numpy

from mttl.datamodule.base import DatasetConfig, DefaultDataModule
from mttl.datamodule.mt_seq_to_seq_module import (
    FlatMultiTaskConfig,
    apply_source_template,
)
from mttl.datamodule.utils import maybe_filter_hf_dataset_by_task
from mttl.logging import logger
from mttl.models.library.expert_library import DatasetLibrary


class ChatDataConfig(FlatMultiTaskConfig):
    seed: str = 42


class ChatDataModule(DefaultDataModule):
    def setup_dataset(self):
        n_proc = int(os.environ.get("MTTL_NUM_PROC_DATASETS", 16))
        dataset = DatasetLibrary.pull_dataset(self.config.dataset)

        (
            self._task_names,
            self._task_to_id,
            train_dataset,
            _,
            _,
        ) = maybe_filter_hf_dataset_by_task(
            dataset,
            "cluster_id",
            self.config.finetune_task_name,
            n_proc=n_proc,
        )

        num_examples = len(train_dataset)
        num_train = int(0.9 * num_examples)
        num_dev = int(0.1 * num_examples)

        train_dataset = train_dataset.shuffle(seed=self.config.seed)
        train_dataset = train_dataset.rename_column("task_name", "original_task_name")
        train_dataset = train_dataset.rename_column("cluster_id", "task_name")

        self.train_dataset = train_dataset.select(range(num_train))
        self.dev_dataset = train_dataset.select(range(num_train, num_train + num_dev))
        self.test_dataset = self.dev_dataset
