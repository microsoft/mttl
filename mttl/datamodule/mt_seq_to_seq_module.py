from functools import partial
from typing import List

from datasets import load_dataset
from mttl.datamodule.base import DefaultDataModule, DatasetConfig
from mttl.datamodule.utils import maybe_filter_hf_dataset_by_task
from dataclasses import dataclass


@dataclass
class FlanConfig(DatasetConfig):
    include_template_type: str = "zs_noopt"
    include_task_source: str = "P3,Flan2021"


def filter_template_type(include_template_type, example):
    return example["template_type"] in include_template_type


def filter_task_source(include_task_source, example):
    return example["task_source"] in include_task_source


class FlanModule(DefaultDataModule):
    def setup_dataset(self):
        dataset = load_dataset(self.config.dataset)

        if self.config.include_template_type != "*":
            dataset = dataset.filter(
                partial(
                    set(self.config.include_template_type.split(",")),
                    filter_template_type,
                ),
                num_proc=16,
                desc="Filtering template types",
            )

        if self.config.include_task_source != "*":
            dataset = dataset.filter(
                partial(
                    filter_task_source, set(self.config.include_task_source.split(","))
                ),
                num_proc=16,
                desc="Filtering task sources",
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

        if "split" in dataset.column_names["train"]:
            self.train_dataset = train_dataset.filter(
                lambda x: x["split"] == "train",
                num_proc=16,
                desc="Creating train set",
            )
            self.dev_dataset = train_dataset.filter(
                lambda x: x["split"] == "validation",
                num_proc=16,
                desc="Creating valid set",
            )
            self.test_dataset = train_dataset.filter(
                lambda x: x["split"] == "test",
                num_proc=16,
                desc="Creating test set",
            )
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
