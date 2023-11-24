from functools import partial
from typing import List
import os
from datasets import load_dataset
from mttl.datamodule.base import DefaultDataModule, DatasetConfig
from mttl.datamodule.utils import maybe_filter_hf_dataset_by_task
from dataclasses import dataclass


@dataclass
class AdaUniConfig(DatasetConfig):
    pass


class AdaUniModule(DefaultDataModule):
    def setup_dataset(self):
        dataset = load_dataset(self.config.dataset)
        n_proc = int(os.environ.get("MTTL_NUM_PROC_DATASETS", 16))
        if "split" not in dataset.column_names["train"]:
            raise ValueError(
                "Dataset must have a 'split' column, try removing the dataset manually from the cache."
            )
        (
            self._task_names,
            self._task_to_id,
            train_dataset,
            _,
            _,
        ) = maybe_filter_hf_dataset_by_task(
            dataset, "task_name", self.config.finetune_task_name, n_proc=n_proc
        )

        if "split" in dataset.column_names["train"]:
            self.train_dataset = train_dataset.filter(
                lambda x: x["split"] == "train",
                num_proc=n_proc,
                desc="Creating train set",
            )
            self.dev_dataset = train_dataset.filter(
                lambda x: x["split"] == "validation",
                num_proc=n_proc,
                desc="Creating valid set",
            )
            self.test_dataset = train_dataset.filter(
                lambda x: x["split"] == "test",
                num_proc=n_proc,
                desc="Creating test set",
            )
        else:
            self.train_dataset, self.dev_dataset = self.create_train_valid_split(
                train_dataset
            )
            self.test_dataset = self.dev_dataset
        self.print_infos()
