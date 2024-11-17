import os
from dataclasses import dataclass

from mttl.datamodule.base import DataModule, DatasetConfig, MultiChoiceDataModule
from mttl.models.library.dataset_library import DatasetLibrary


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

        dataset = DatasetLibrary.pull_dataset("super_glue", name=self.TASK_NAME)

        # convert task_id to task_name and labels
        self._task_to_id = {}
        self._task_names = []

        self.train_dataset = dataset["train"].map(
            self.map_example,
            num_proc=n_proc,
        )
        self.dev_dataset = self.test_dataset = dataset[self.DATASET_SPLIT].map(
            self.map_example,
            num_proc=n_proc,
        )


@DataModule.register("boolq", config_cls=SuperGLUEDataConfig)
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
