from typing import List

from datasets import load_dataset
from mttl.datamodule.base import DefaultDataModule, DatasetConfig
from mttl.datamodule.utils import maybe_filter_hf_dataset_by_task
from dataclasses import dataclass


@dataclass
class FlanConfig(DatasetConfig):
    include_template_type: str = "zs_noopt"
    include_task_source: str = "P3,Flan2021"


class FlanModule(DefaultDataModule):
    def setup_dataset(self):
        dataset = load_dataset(self.config.dataset)

        if self.config.include_template_type != "*":
            dataset = dataset.filter(
                lambda x: x["template_type"]
                in self.config.include_template_type.split(","),
            )

        if self.config.include_task_source != "*":
            dataset = dataset.filter(
                lambda x: x["task_source"]
                in self.config.include_task_source.split(","),
            )

        (
            self._task_names,
            self._task_to_id,
            train_dataset,
            _,
            _,
        ) = maybe_filter_hf_dataset_by_task(
            dataset, "task_name", self.config.finetune_task_name
        )

        if "split" in dataset.column_names:
            self.train_dataset = train_dataset.filter(lambda x: x["split"] == "train")
            self.dev_dataset = train_dataset.filter(
                lambda x: x["split"] == "validation"
            )
            self.test_dataset = train_dataset.filter(lambda x: x["split"] == "test")
        else:
            self.train_dataset, self.dev_dataset = self.create_train_valid_split(
                train_dataset
            )
            self.test_dataset = self.dev_dataset
        self.print_infos()


@dataclass
class T0FlatConfig(DatasetConfig):
    use_templates_as_tasks: bool = False


class T0FlatModule(FlanModule):
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

        if self.config.use_templates_as_tasks:

            def concat_templates_and_task(example):
                example["task_name"] = (
                    example["task_name"]
                    + "/"
                    + example["template_type"].strip().replace(" ", "_")
                )
                return example

            train_dataset = train_dataset.map(concat_templates_and_task, num_proc=16)

            self._task_names = sorted(list(set(train_dataset["task_name"])))
            self._task_to_id = {
                task_name: i for i, task_name in enumerate(self._task_names)
            }

        self.train_dataset, self.dev_dataset = self.create_train_valid_split(
            train_dataset
        )
        self.test_dataset = self.dev_dataset
        self.print_infos()
