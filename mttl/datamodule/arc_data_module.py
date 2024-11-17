import os
from dataclasses import dataclass
from functools import partial

from mttl.datamodule.base import DataModule, DatasetConfig, MultiChoiceDataModule
from mttl.models.library.dataset_library import DatasetLibrary


@dataclass
class ArcDataConfig(DatasetConfig):
    arc_type: str = "ARC-Easy"


@DataModule.register("arc", config_cls=ArcDataConfig)
class ArcMultiChoiceDataModule(MultiChoiceDataModule):
    def setup_dataset(self):
        n_proc = int(os.environ.get("MTTL_NUM_PROC_DATASETS", 16))
        dataset = DatasetLibrary.pull_dataset("ai2_arc", name=self.config.arc_type)

        # convert task_id to task_name and labels
        def map_example(arc_type, example):
            prompt = "Question: {}\nAnswer:"
            targets = [choice for choice in example["choices"]["text"]]

            # Prevents `label` from having wrong values due to dataset values
            # mixed between strings and integers
            answer_key_map = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}
            answer_key = answer_key_map.get(example["answerKey"], example["answerKey"])

            example["task_name"] = arc_type
            example["source"] = prompt.format(example["question"])
            example["target"] = targets
            example["label_index"] = ["A", "B", "C", "D", "E"].index(answer_key)
            return example

        dataset = dataset.map(
            partial(map_example, self.config.arc_type),
            num_proc=n_proc,
        )

        self._task_to_id = {}
        self._task_names = []

        self.train_dataset = dataset["train"]
        self.dev_dataset = dataset["validation"]
        self.test_dataset = dataset["test"]
