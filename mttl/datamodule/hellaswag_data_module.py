import os
import re
from dataclasses import dataclass

from mttl.datamodule.base import DataModule, DatasetConfig, MultiChoiceDataModule
from mttl.models.library.expert_library import DatasetLibrary


@dataclass
class HellaswagDataConfig(DatasetConfig):
    pass


def _pre_process_text(text: str) -> str:
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


@DataModule.register("hellaswag", config_cls=HellaswagDataConfig)
class HellaswagMultiChoiceDataModule(MultiChoiceDataModule):
    def setup_dataset(self):
        n_proc = int(os.environ.get("MTTL_NUM_PROC_DATASETS", 16))
        dataset = DatasetLibrary.pull_dataset("hellaswag", name="default")

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

        self._task_to_id = {}
        self._task_names = []

        self.train_dataset = dataset["train"].map(
            map_example,
            num_proc=n_proc,
        )
        self.dev_dataset = self.test_dataset = dataset["validation"].map(
            map_example,
            num_proc=n_proc,
        )
