import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from mttl.datamodule.collators import DefaultCollator
from mttl.dataloader.flan_100k_dataset_reader import Flan100KDataset
from transformers import AutoTokenizer

from mttl.datamodule.utils import get_tokenizer


class Flan100kModule(LightningDataModule):
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

    @property
    def all_instructions(self):
        return self.dataset.read_all_instructions()

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
            model_family=config.model_family,
        )
        self.task_to_id = {"flan_100k": 0}

    def get_dataset(self, idxs=None, loss_for_keywords=True):
        return Flan100KDataset(
            self.config.data_dir,
            idxs,
            loss_for_keywords=loss_for_keywords,
        )

    def setup(self, stage=None):
        dataset = self.get_dataset()

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
