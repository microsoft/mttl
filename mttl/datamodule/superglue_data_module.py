import os
from dataclasses import dataclass
from datasets import load_dataset
from mttl.datamodule.base import DatasetConfig, MultiChoiceDataModule


@dataclass
class SuperGLUEDataConfig(DatasetConfig):
    pass


class SuperGLUEMultiChoiceDataModule(MultiChoiceDataModule):
    TASK_NAME = None
    DATASET_SPLIT = "validation"

    @classmethod
    def map_example(example):
        pass

    def setup_dataset(self):
        n_proc = int(os.environ.get("MTTL_NUM_PROC_DATASETS", 16))

        dataset = load_dataset("super_glue", name=self.TASK_NAME)

        # convert task_id to task_name and labels
        dataset = dataset.map(
            self.map_example,
            num_proc=n_proc,
        )

        self._task_to_id = {}
        self._task_names = []

        self.train_dataset = dataset["train"]
        self.dev_dataset = dataset[self.DATASET_SPLIT]
        self.test_dataset = dataset[self.DATASET_SPLIT]


class BoolQDataModule(SuperGLUEMultiChoiceDataModule):
    TASK_NAME = "boolq"
    DATASET_SPLIT = "validation"

    @classmethod
    def map_example(cls, example):
        prompt = "{}\nQuestion: {}?\nAnswer:"
        targets = ["no", "yes"]

        example["source"] = prompt.format(example["passage"], example["question"])
        example["target"] = targets
        example["label_index"] = example["label"]
        example["task_name"] = "boolq"
        return example
