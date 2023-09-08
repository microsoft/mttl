import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from mttl.dataloader.alpaca_dataset_readers import AlpacaDataset
from mttl.datamodule.utils import get_tokenizer
from mttl.datamodule.collators import DefaultCollator


class AlpacaDataModule(LightningDataModule):
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
            batch_size=self.config.train_batch_size,
            shuffle=False,
            num_workers=16,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.config.train_batch_size,
            shuffle=False,
            num_workers=16,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=self.collate_fn,
        )

    @property
    def all_instructions(self):
        return self.dataset.read_all_instructions()

    def __init__(
        self,
        config,
        fast=None,
    ):
        super().__init__()
        self.fast = fast
        self.config = config
        self.tokenizer = get_tokenizer(config)
        self.collate_fn = DefaultCollator(
            tokenizer=self.tokenizer,
            padding="longest",
            max_input_length=config.max_input_length,
            max_output_length=config.max_output_length,
            pad_to_multiple_of=8,
            return_tensors="pt",
            model_family=config.model_family,
        )
        self.task_to_id = {"alpaca_full": 0}

    def get_dataset(self, idxs=None):
        return AlpacaDataset(
            self.config.data_dir,
            idxs,
        )

    def setup(self, stage=None):
        dataset = self.get_dataset()

        # always use the same split for the dataset
        rng = torch.Generator().manual_seed(1234)

        n_tr_samples = int(
            len(dataset) * (1 - self.config.validation_portion)
        )  # len(dataset) -

        self.train_dataset, self.dev_dataset = torch.utils.data.random_split(
            dataset,
            [
                n_tr_samples,
                len(dataset) - n_tr_samples,
            ],
            generator=rng,
        )
        self.test_set = self.dev_dataset
        if self.fast:
            import numpy as np
            self.train_dataset = AlpacaDataset(
                                    self.config.data_dir,
                                    idxs=np.random.choice(len(self.train_dataset), 10),
                                )

        print("Training steps:", len(self.train_dataloader()))
        print("Validation steps:", len(self.val_dataloader()))


class AlpacaPretrainDataModule(AlpacaDataModule):
    pass


class AlpacaFinetuneDataModule(AlpacaDataModule):
    pass
