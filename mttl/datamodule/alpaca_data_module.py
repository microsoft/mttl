from mttl.dataloader.alpaca_dataset_readers import AlpacaDataset
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


class AlpacaPretrainDataModule(AlpacaDataModule):
    pass


class AlpacaFinetuneDataModule(AlpacaDataModule):
    pass
