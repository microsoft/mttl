from datasets import load_dataset
from mttl.datamodule.base import (
    MultiChoiceDataModule,
    DatasetConfig,
)
from dataclasses import dataclass
import os, re


@dataclass
class HellaswagDataConfig(DatasetConfig):
    pass


def _pre_process_text(text: str) -> str:
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


class HellaswagMultiChoiceDataModule(MultiChoiceDataModule):
    def setup_dataset(self):
        n_proc = int(os.environ.get("MTTL_NUM_PROC_DATASETS", 16))
        dataset = load_dataset("hellaswag", name="default")

        # convert task_id to task_name and labels
        def map_example(example):
            prompt = "{}: {}"

            activity_label = _pre_process_text(example["activity_label"])
            context = _pre_process_text(
                example["ctx_a"] + " " + example["ctx_b"].capitalize()
            )
            targets = [_pre_process_text(ending) for ending in example["endings"]]

            example["task_name"] = "hellaswag"
            example["source"] = prompt.format(activity_label, context)
            example["target"] = targets
            example["label_index"] = int(example["label"])
            return example

        dataset = dataset.map(
            map_example,
            num_proc=n_proc,
        )

        self._task_to_id = {}
        self._task_names = []

        self.train_dataset = dataset["train"]
        self.dev_dataset = dataset["validation"]
        self.test_dataset = dataset["validation"]
