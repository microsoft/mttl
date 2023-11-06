from mttl.datamodule.base import DefaultDataModule
from mttl.dataloader.platypus_dataset_reader import PlatypusDataset


class PlatypusModule(DefaultDataModule):
    def __init__(self, config, for_generation=False, val_mixin=None):
        super().__init__(config, for_generation, val_mixin)

    def setup_dataset(self):
        dataset = PlatypusDataset()

        self.train_dataset, self.dev_dataset = self.create_train_valid_split(dataset)
        self.test_dataset = self.dev_dataset
        self.print_infos()
