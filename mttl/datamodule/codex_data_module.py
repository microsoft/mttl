from functools import partial
import os
import numpy
from mttl.datamodule.base import DefaultDataModule, DatasetConfig
from mttl.models.modifiers.expert_containers.expert_library import DatasetLibrary


class CodexDataConfig(DatasetConfig):
    pass


class CodexDataModule(DefaultDataModule):
    def setup_dataset(self):
        dataset = DatasetLibrary.pull_dataset("jinaai/code_exercises", split="train")
        n_proc = int(os.environ.get("MTTL_NUM_PROC_DATASETS", 16))

        def process(rng, ex):
            return {
                "source": ex["problem"],
                "target": ex["solution"],
                "task_name": "code-exercises",
                "task_source": "code-exercises",
                "split": "train" if rng.random() < 0.9 else "validation",
            }

        dataset = dataset.map(
            partial(process, numpy.random.RandomState(42)),
            num_proc=n_proc,
            desc="Creating split",
            remove_columns=dataset.column_names,
        )

        self.train_dataset = dataset.filter(
            lambda x: x["split"] == "train",
            num_proc=n_proc,
            desc="Creating train set",
        )
        self.dev_dataset = dataset.filter(
            lambda x: x["split"] in ["validation", "valid"],
            num_proc=n_proc,
            desc="Creating valid set",
        )
        self.test_dataset = self.dev_dataset
