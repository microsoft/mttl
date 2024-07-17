from dataclasses import dataclass

import torch

from mttl.dataloader.platypus_dataset_reader import (
    InversePlatypusDataset,
    PlatypusDataset,
    PlatypusQADataset,
)
from mttl.datamodule.base import DatasetConfig, DefaultDataModule


@dataclass
class PlatypusConfig(DatasetConfig):
    train_on_reverse: bool = False


class PlatypusModule(DefaultDataModule):
    def setup_dataset(self):
        if getattr(self.config, "train_on_reverse", False):
            dataset = InversePlatypusDataset()
        else:
            dataset = PlatypusDataset()

        self.train_dataset, self.dev_dataset = self.create_train_valid_split(dataset)
        self.test_dataset = self.dev_dataset


class PlatypusQAModule(PlatypusModule):
    def setup_dataset(self):
        if getattr(self.config, "train_on_reverse", False):
            raise NotImplementedError("train_on_reverse is not implemented for QA.")
        else:
            train_dataset = PlatypusQADataset(
                dataset_name=self.config.dataset,
                filter_by_subject=self.config.finetune_task_name,
                val_mixin=self.val_mixin,
            )

        self.train_dataset, self.dev_dataset = self.create_train_valid_split(
            train_dataset
        )
        self.test_dataset = self.dev_dataset

    def create_train_valid_split(self, dataset, validation_portion=None):
        """
        Create train dev split while making sure that val_mixin examples are only in the dev set.
        """
        if not hasattr(dataset, "val_mixin") or dataset.val_mixin is None:
            return super().create_train_valid_split(dataset, validation_portion)

        validation_portion = validation_portion or self.config.validation_portion

        n_tr_samples = int(
            (len(dataset) - len(dataset.val_mixin)) * (1 - validation_portion)
        )
        idxs = torch.randperm(len(dataset) - len(dataset.val_mixin), generator=self.rng)
        idxs_train = idxs[:n_tr_samples]
        idxs_valid = idxs[n_tr_samples:]
        idxs_valid = torch.cat([idxs_valid, dataset.mixin_idxs])
        # overlap
        overlap = set(idxs_valid.tolist()).intersection(set(idxs_train.tolist()))
        assert len(overlap) == 0
        return torch.utils.data.Subset(
            dataset, idxs_train.tolist()
        ), torch.utils.data.Subset(dataset, idxs_valid.tolist())
