from dataclasses import dataclass

from mttl.datamodule.base import DataModule, DatasetConfig, DefaultCollator
from mttl.models.library.dataset_library import DatasetLibrary


@dataclass
class MathQADataConfig(DatasetConfig):
    pass


@dataclass
class MathQADataModule(DataModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup_dataset(self):
        train_dataset = DatasetLibrary.pull_dataset_with_retry("meta-math/MetaMathQA")[
            "train"
        ]

        train_dataset = train_dataset.rename_column("query", "source")
        train_dataset = train_dataset.rename_column("response", "target")

        # filter out the rows where the source is empty
        train_dataset = train_dataset.filter(lambda x: x["source"] != "")

        self.train_dataset, self.dev_dataset= self.create_train_valid_split(train_dataset, validation_portion=0.01)
        self.test_dataset = self.dev_dataset

        self.print_infos()

    @property
    def collate_fn(self):
        return DefaultCollator(
            tokenizer=self.tokenizer,
            padding="longest",
            max_input_length=self.config.max_input_length,
            max_output_length=self.config.max_output_length,
            return_tensors="pt",
            model_family=self.config.model_family,
            for_generation=self.for_generation,
        )


if __name__ == "__main__":
    config = MathQADataConfig(model="microsoft/Phi-3-mini-4k-instruct")

    datamodule = MathQADataModule(config)
    train_dataloader = datamodule.train_dataloader()
    val_dataloder = datamodule.val_dataloader()
    for batch in val_dataloder:
        print(batch)
        breakpoint()
