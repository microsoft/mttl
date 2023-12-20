from functools import partial
from datasets import load_dataset
from mttl.datamodule.base import DefaultDataModule, DatasetConfig
from dataclasses import dataclass
import os

from mttl.datamodule.utils import maybe_filter_hf_dataset_by_task


@dataclass
class HumanEvalConfig(DatasetConfig):
    apply_source_template: str = None


class HumanEvalDataModule(DefaultDataModule):
    def setup_dataset(self):
        n_proc = int(os.environ.get("MTTL_NUM_PROC_DATASETS", 16))
        dataset = load_dataset("openai_humaneval", name="openai_humaneval")

        # convert task_id to task_name and labels
        def map_example(example):
            example["task_name"] = example["task_id"]
            example["task_source"] = "humaneval"
            example["target"] = (
                example["test"] + "\n" + f"check({example['entry_point']})"
            )
            example["source"] = example["prompt"]
            return example

        dataset = dataset.map(
            map_example,
            num_proc=n_proc,
            desc="Mapping task_id to task_name.",
            remove_columns=["prompt", "test", "entry_point", "task_id"],
        )

        (
            self._task_names,
            self._task_to_id,
            _,
            _,
            test_dataset,
        ) = maybe_filter_hf_dataset_by_task(
            dataset, "task_name", self.config.finetune_task_name, n_proc=n_proc
        )

        self.train_dataset = self.dev_dataset = self.test_dataset = test_dataset
        self.print_infos()
