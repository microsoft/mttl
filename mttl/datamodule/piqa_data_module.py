import os
from dataclasses import dataclass

from mttl.datamodule.base import DatasetConfig, MultiChoiceDataModule
from mttl.models.library.expert_library import DatasetLibrary


@dataclass
class PiqaDataConfig(DatasetConfig):
    pass


class PiqaMultiChoiceDataModule(MultiChoiceDataModule):
    def setup_dataset(self):
        n_proc = int(os.environ.get("MTTL_NUM_PROC_DATASETS", 16))
        dataset = DatasetLibrary.pull_dataset("piqa")

        # convert task_id to task_name and labels
        def map_example(example):
            prompt = "Question: {}\nAnswer:"
            targets = [example["sol1"], example["sol2"]]

            example["source"] = prompt.format(example["goal"])
            example["target"] = targets
            example["label_index"] = example["label"]
            example["task_name"] = "piqa"
            return example

        dataset = dataset.map(
            map_example,
            num_proc=n_proc,
        )

        self._task_to_id = {}
        self._task_names = []
        self.train_dataset = dataset["train"]
        self.dev_dataset = self.test_dataset = dataset["validation"]
