import torch
from mttl.datamodule.mt_seq_to_seq_module import FlanModule, FlanConfig
from dataclasses import dataclass
from mttl.datamodule.base import DefaultDataModule, DatasetConfig
from datasets import load_dataset
from mttl.datamodule.utils import maybe_filter_hf_dataset_by_task
import os


@dataclass
class ClassificationConfig(FlanConfig):
    pass


@dataclass
class DataCollatorForClassification:
    def __call__(self, batch):
        # change the tasks_names to tasks_ids
        inputs = [b["source"] for b in batch]
        targets = [b["target"] for b in batch]
        task_name = [b["task_name"] for b in batch]

        return {
            "input": inputs,
            "target": targets,
            "task_name": task_name,
        }


@dataclass
class ClassificationFlatMutiTaskConfig(DatasetConfig):
    pass


class ClassificationDataModule(FlanModule):
    @property
    def collate_fn(self):
        # change the tasks_names to tasks_ids
        return DataCollatorForClassification()


class ClassificationDataModuleFlatMultiTask(DefaultDataModule):
    def setup_dataset(self):
        n_proc = int(os.environ.get("ADAUNI_NUM_PROC_DATASETS", 16))
        dataset = load_dataset(self.config.dataset)

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
                lambda x: x["split"] in ["validation", "valid"],
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
        if len(self.test_dataset) == 0:
            self.test_dataset = self.dev_dataset
        self.print_infos()

    @property
    def collate_fn(self):
        # change the tasks_names to tasks_ids
        return DataCollatorForClassification()
