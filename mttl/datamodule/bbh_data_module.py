from functools import partial
from datasets import load_dataset, get_dataset_config_names, concatenate_datasets
from mttl.datamodule.base import DefaultDataModule, DatasetConfig
from dataclasses import dataclass
import os


@dataclass
class BBHConfig(DatasetConfig):
    apply_source_template: str = "Instruct: {}\nResponse:"


class BBHDataModule(DefaultDataModule):
    def setup_dataset(self):
        n_proc = int(os.environ.get("MTTL_NUM_PROC_DATASETS", 16))
        self._task_names = get_dataset_config_names("maveriq/bigbenchhard")
        self._task_to_id = {
            task_name: i for i, task_name in enumerate(self._task_names)
        }

        datasets = []
        task_names = (
            sorted(self.config.finetune_task_name.split(","))
            if isinstance(self.config.finetune_task_name, str)
            else self._task_names
        )

        for task_name in self._task_names:
            if task_name not in task_names:
                continue

            task_dataset = load_dataset("maveriq/bigbenchhard", task_name)["train"]
            task_dataset = task_dataset.map(
                lambda x: {
                    "source": x["input"],
                    "task_name": task_name,
                    "task_source": "bbh",
                },
                remove_columns=["input"],
                num_proc=n_proc,
            )
            datasets.append(task_dataset)

        dataset = concatenate_datasets(datasets)

        if self.config.apply_source_template:

            def apply_source_template(source_template, example):
                example["source"] = source_template.format(example["source"])
                return example

            dataset = dataset.map(
                partial(apply_source_template, self.config.apply_source_template),
                num_proc=n_proc,
            )

        self.train_dataset = self.dev_dataset = dataset
        self.test_dataset = self.dev_dataset
        self.print_infos()
