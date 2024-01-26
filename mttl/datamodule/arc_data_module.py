from functools import partial
from datasets import load_dataset
from mttl.datamodule.base import (
    MultiChoiceDataModule,
    DatasetConfig,
    MultipleChoiceCollator,
)
from dataclasses import dataclass
import os


@dataclass
class ArcDataConfig(DatasetConfig):
    arc_type: str = "ARC-Easy"


class ArcMultiChoiceDataModule(MultiChoiceDataModule):
    def setup_dataset(self):
        n_proc = int(os.environ.get("MTTL_NUM_PROC_DATASETS", 16))
        dataset = load_dataset("ai2_arc", name=self.config.arc_type)

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
        self.dev_dataset = dataset["test"]
        self.test_dataset = dataset["test"]
