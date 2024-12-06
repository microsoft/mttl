import os
from dataclasses import dataclass

from mttl.datamodule.base import DataModule, DatasetConfig
from mttl.models.library.dataset_library import DatasetLibrary


@dataclass
class GsmDataConfig(DatasetConfig):
    pass


def instruct_templete(example):
    example["source"] = example["input"]
    example["target"] = str(example["answer"])
    return example


@DataModule.register("gsm", config_cls=GsmDataConfig)
class GsmDataModule(DataModule):
    def setup_dataset(self):
        n_proc = int(os.environ.get("MTTL_NUM_PROC_DATASETS", 16))
        dataset = DatasetLibrary.pull_dataset("reasoning-machines/gsm-hard")
        dataset = dataset.rename_column("target", "answer")
        dataset = dataset.map(instruct_templete, num_proc=n_proc)
        self.train_dataset = dataset["train"]
        self.dev_dataset = self.test_dataset = dataset["train"]


if __name__ == "__main__":
    config = GsmDataConfig(model="microsoft/Phi-3-mini-4k-instruct")

    datamodule = GsmDataModule(config)
    train_dataloader = datamodule.train_dataloader()
    val_dataloder = datamodule.val_dataloader()
    for batch in val_dataloder:
        print(batch)
        breakpoint()
