import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from mttl.dataloader.alpaca_dataset_readers import AlpacaDataset
from transformers import AutoTokenizer
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
    ):
        super().__init__()

        self.config = config

        self.tokenizer = AutoTokenizer.from_pretrained(config.model, add_eos_token=False)
        if not self.tokenizer.pad_token_id:
            self.tokenizer.pad_token_id = self.pad_token_id = 0

        if self.config.padding_side == "left":
            self.tokenizer.padding_side = (
                "left"  # Allow batched inference, used by tloen also in training
            )

        self.collate_fn = DefaultCollator(
            tokenizer=self.tokenizer,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
        self.task2id = {"alpaca_full": 0}

    def get_dataset(self, idxs=None, loss_for_keywords=True):
        return AlpacaDataset(
            self.tokenizer,
            self.config.max_input_length,
            self.config.max_output_length,
            self.config.train_dir,
            self.config.train_on_inputs,
            self.config.dst_dir,
            idxs,
            loss_for_keywords=loss_for_keywords,
            subset=100 if self.config.fast_debug_run else None,
        )

    def setup(self, stage=None):
        idxs_cluster = []
        dataset = self.get_dataset()

        # always use the same split for the dataset
        rng = torch.Generator().manual_seed(1234)
        if len(idxs_cluster) > 0:
            dataset.dataset = dataset.dataset.select(idxs_cluster)
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

        print("Training steps:", len(self.train_dataloader()))
        print("Validation steps:", len(self.val_dataloader()))


class AlpacaPretrainDataModule(AlpacaDataModule):
    pass


class AlpacaFinetuneDataModule(AlpacaDataModule):
    pass
