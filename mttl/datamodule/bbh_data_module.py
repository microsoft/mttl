from datasets import get_dataset_config_names, concatenate_datasets
from mttl.datamodule.base import DefaultDataModule, DatasetConfig
from dataclasses import dataclass
import os

from mttl.datamodule.mt_seq_to_seq_module import augment_few_shot_task
from mttl.models.modifiers.expert_containers.expert_library import DatasetLibrary


@dataclass
class BBHConfig(DatasetConfig):
    augment_few_shot: int = -1
    source_template: str = "Solve the following problem: {}\nAnswer:"


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

            task_dataset = DatasetLibrary.pull_dataset(
                "maveriq/bigbenchhard", task_name, split="train"
            )
            task_dataset = task_dataset.map(
                lambda x: {
                    "source": x["input"],
                    "task_name": task_name,
                    "task_source": "bbh",
                    "split": "test",
                },
                remove_columns=["input"],
                num_proc=n_proc,
            )

            if self.config.source_template is not None:
                from mttl.datamodule.mt_seq_to_seq_module import apply_source_template

                task_dataset = apply_source_template(
                    task_dataset, self.config.source_template
                )

            if self.config.augment_few_shot > 0:
                few_shots = task_dataset.select(range(self.config.augment_few_shot))

                task_dataset = augment_few_shot_task(
                    task_dataset,
                    few_shots=few_shots,
                    tokenizer=self.tokenizer,
                    max_input_length=self.config.max_input_length,
                )

            datasets.append(task_dataset)

        dataset = concatenate_datasets(datasets)
        self.train_dataset = self.dev_dataset = dataset
        self.test_dataset = self.dev_dataset
