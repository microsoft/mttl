from dataclasses import dataclass

from mttl.dataloader.oasst1_readers import InverseOasst1Dataset, Oasst1Dataset
from mttl.datamodule.base import DataModule, DatasetConfig


@dataclass
class OA1Config(DatasetConfig):
    train_on_reverse: bool = False


@DataModule.register("oasst1", config_cls=OA1Config)
class OA1Module(DataModule):
    def setup_dataset(self):
        if getattr(self.config, "train_on_reverse", False):
            dataset = InverseOasst1Dataset(self.config.data_dir)
        else:
            dataset = Oasst1Dataset(self.config.data_dir)

        self.train_dataset, self.dev_dataset = self.create_train_valid_split(dataset)
