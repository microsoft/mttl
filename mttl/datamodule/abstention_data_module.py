from mttl.datamodule.base import DataModule, DatasetConfig
from dataclasses import dataclass
import os
from mttl.models.library.dataset_library import DatasetLibrary


@dataclass
class AbstentionDataConfig(DatasetConfig):
    predict_output_dir: str = "./"


DataModule.register("abstention", config_cls=AbstentionDataConfig)


class AbstentionDataModule(DataModule):
    def setup_dataset(self):
        n_proc = int(os.environ.get("MTTL_NUM_PROC_DATASETS", 4))
        dataset = DatasetLibrary.pull_dataset(self.config.dataset)
        dataset = dataset.rename_column("question", "source")
        dataset = dataset.rename_column("answer", "target")
        self.train_dataset = dataset["train"]
        self.dev_dataset = self.test_dataset = dataset["train"]


if __name__ == "__main__":
    config = AbstentionDataConfig(
        model="microsoft/Phi-3-mini-4k-instruct",
        dataset="zhan1993/coconot_original_train_routing",
    )

    datamodule = AbstentionDataModule(config, for_generation=True)

    train_dataloader = datamodule.train_dataloader()
    val_dataloder = datamodule.val_dataloader()
    for batch in val_dataloder:
        print(batch)
        breakpoint()
