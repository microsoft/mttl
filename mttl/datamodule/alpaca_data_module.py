from mttl.dataloader.alpaca_dataset_readers import (
    AlpacaCodeDataset,
    AlpacaDataset,
    MathQaAlpacaCodeDataset,
    MathQallamaDataset,
)
from mttl.datamodule.base import DataModule, DatasetConfig


@DataModule.register("alpaca", config_cls=DatasetConfig)
class AlpacaDataModule(DataModule):
    @property
    def all_instructions(self):
        return self.dataset.read_all_instructions()

    def __init__(self, config, for_generation=False, val_mixin=None):
        super().__init__(config, for_generation, val_mixin)

    def setup_dataset(self):
        dataset = AlpacaDataset()

        self.train_dataset, self.dev_dataset = self.create_train_valid_split(dataset)
        self.test_dataset = self.dev_dataset


@DataModule.register("alpaca_code", config_cls=DatasetConfig)
class AlpacaCodeDataModule(DataModule):
    @property
    def all_instructions(self):
        return self.dataset.read_all_instructions()

    def __init__(self, config, for_generation=False, val_mixin=None):
        super().__init__(config, for_generation, val_mixin)

    def setup_dataset(self):
        dataset = AlpacaCodeDataset()
        self.train_dataset, self.dev_dataset = self.create_train_valid_split(dataset, validation_portion=0.01)
        self.test_dataset = self.dev_dataset


@DataModule.register("mathqa_alpaca_code", config_cls=DatasetConfig)
class MathQaAlpacaCodeDataModule(AlpacaDataModule):
    def setup_dataset(self):
        dataset = MathQaAlpacaCodeDataset()
        self.train_dataset, self.dev_dataset = self.create_train_valid_split(dataset, validation_portion=0.01)
        self.test_dataset = self.dev_dataset

@DataModule.register("mathqa_llama", config_cls=DatasetConfig)
class MathQallamaDataModule(AlpacaDataModule):
    def setup_dataset(self):
        dataset = MathQallamaDataset()
        self.train_dataset, self.dev_dataset = self.create_train_valid_split(dataset, validation_portion=0.01)
        self.test_dataset = self.dev_dataset


if __name__ == "__main__":
    # alpaca_code_module = AlpacaCodeDataModule(
    #     DatasetConfig(model="meta-llama/Llama-2-7b-hf")
    # )
    # alpaca_code_module.setup_dataset()
    # val_dataloder = alpaca_code_module.val_dataloader()
    # print(alpaca_data_module.train_dataset)

    alpaca_code_data_module = MathQallamaDataModule(
        DatasetConfig(model="yahma/llama-7b-hf")
    )
    alpaca_code_data_module.setup_dataset()
    val_dataloder = alpaca_code_data_module.val_dataloader()
    for batch in val_dataloder:
        print(batch)
        breakpoint()
