from typing import Any
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from mttl.datamodule.base import DefaultCollator
from mttl.dataloader.flan_10k_dataset_reader import Flan10kDataset
from mttl.datamodule.base import DatasetConfig
from mttl.datamodule.utils import get_tokenizer
from dataclasses import dataclass


@dataclass
class Flan10kConfig(DatasetConfig):
    pass


class Flan10kModule(LightningDataModule):
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.train_batch_size,
            shuffle=True,
            num_workers=16,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dev_dataset,
            batch_size=self.config.predict_batch_size,
            shuffle=False,
            num_workers=16,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.config.predict_batch_size,
            shuffle=False,
            num_workers=16,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=self.collate_fn,
        )

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.tokenizer = get_tokenizer(config)

        self.collate_fn = DefaultCollator(
            tokenizer=self.tokenizer,
            padding="longest",
            max_input_length=config.max_input_length,
            max_output_length=config.max_output_length,
            pad_to_multiple_of=8,
            return_tensors="pt",
            model_family=self.config.model_family,
        )
        self.task_to_id = {"flan_10k": 0}

    def get_dataset(self, idxs=None):
        return Flan10kDataset(
            self.config.category,
            idxs=idxs,
        )

    def setup(self, stage=None):
        dataset = self.get_dataset()
        breakpoint
        # always use the same split for the dataset
        rng = torch.Generator().manual_seed(1234)
        n_tr_samples = int(len(dataset) * (1 - self.config.validation_portion))

        self.train_dataset, self.dev_dataset = torch.utils.data.random_split(
            dataset,
            [
                n_tr_samples,
                len(dataset) - n_tr_samples,
            ],
            generator=rng,
        )
        self.test_set = self.dev_dataset
        print("Training steps:", len(self.train_dataloader()))
        print("Validation steps:", len(self.val_dataloader()))
