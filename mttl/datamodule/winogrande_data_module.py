from datasets import load_dataset
from mttl.datamodule.base import (
    MultiChoiceSourceDataModule,
    DatasetConfig,
)
from dataclasses import dataclass
import os


def doc_to_text(doc):
    answer_to_num = {"1": 0, "2": 1}
    return answer_to_num[doc["answer"]]


def doc_to_target(doc):
    idx = doc["sentence"].index("_") + 1
    return doc["sentence"][idx:].strip()


def doc_to_choice(doc):
    idx = doc["sentence"].index("_")
    options = [doc["option1"], doc["option2"]]
    return [doc["sentence"][:idx] + opt for opt in options]


@dataclass
class WinograndeDataConfig(DatasetConfig):
    pass


class WinograndeMultiChoiceDataModule(MultiChoiceSourceDataModule):
    def setup_dataset(self):
        n_proc = int(os.environ.get("MTTL_NUM_PROC_DATASETS", 16))
        dataset = load_dataset("winogrande", name="winogrande_xl")["validation"]

        # convert task_id to task_name and labels
        def map_example(example):
            sources = doc_to_choice(example)
            target = doc_to_target(example)
            example_label = doc_to_text(example)

            example["source"] = sources
            example["target"] = target
            example["label_index"] = int(example_label)
            example["task_name"] = "winogrande"
            return example

        dataset = dataset.map(
            map_example,
            num_proc=n_proc,
        )

        self._task_to_id = {}
        self._task_names = []
        self.train_dataset = None
        self.dev_dataset = dataset
        self.test_dataset = dataset
