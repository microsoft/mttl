from functools import partial
from datasets import load_dataset
from mttl.datamodule.base import DefaultDataModule, DatasetConfig
from dataclasses import dataclass
import os

from mttl.datamodule.utils import maybe_filter_hf_dataset_by_task


@dataclass
class MBPPDataConfig(DatasetConfig):
    apply_source_template: str = None


class MBPPDataModule(DefaultDataModule):
    def setup_dataset(self):
        n_proc = int(os.environ.get("MTTL_NUM_PROC_DATASETS", 16))
        dataset = load_dataset("mbpp", name="sanitized")

        # convert task_id to task_name and labels
        def map_example(example):
            source = example["prompt"]
            code_header = example["code"].split(":")[0] + ":"
            source_template = '{}\n\t"""\n\t{}\n\t{}\n\t"""\n\t'

            example["task_source"] = "mbpp"
            example["task_name"] = "mbpp"
            example["source"] = source_template.format(
                code_header, source, "\n\t".join(example["test_list"])
            )
            example["target"] = "\n".join(example["test_list"])
            return example

        dataset = dataset.map(
            map_example,
            num_proc=n_proc,
            remove_columns=["prompt", "task_id"],
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

        self.train_dataset = None
        self.dev_dataset = self.test_dataset = test_dataset
