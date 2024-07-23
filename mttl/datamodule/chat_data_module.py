from mttl.datamodule.base import DatasetConfig, DefaultDataModule
from mttl.models.library.expert_library import DatasetLibrary


class ChatDataConfig(DatasetConfig):
    pass


class ChatDataModule(DefaultDataModule):

    def setup_dataset(self):
        dataset = DatasetLibrary.pull_dataset(self.config.dataset, split="train")
        # TODO: continue implementation

