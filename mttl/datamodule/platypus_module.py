
from mttl.dataloader.platypus_dataset_reader import InversePlatypusDataset, PlatypusDataset, PlatypusQADataset
from mttl.datamodule.collators import DefaultDataModule, DatasetConfig
from dataclasses import dataclass


@dataclass
class PlatypusConfig(DatasetConfig):
    train_on_reverse: bool = False


class PlatypusModule(DefaultDataModule):
    def setup_dataset(self):
        if getattr(self.config, 'train_on_reverse', False):
            dataset = InversePlatypusDataset(self.config.data_dir)
        else:
            dataset = PlatypusDataset(self.config.data_dir)

        self.train_dataset, self.dev_dataset = self.create_train_valid_split(dataset)
        self.print_infos()


class PlatypusQAModule(PlatypusModule):
    def setup_dataset(self):
        if getattr(self.config, 'train_on_reverse', False):
            raise NotImplementedError("train_on_reverse is not implemented for QA.")
        else:
            train_dataset = PlatypusQADataset(
                self.config.data_dir,
                dataset_name=self.config.dataset,
                filter_by_subject=self.config.finetune_task_name
            )

        self.train_dataset, self.dev_dataset = self.create_train_valid_split(train_dataset)
        self.test_dataset = self.dev_dataset
        self.print_infos()
