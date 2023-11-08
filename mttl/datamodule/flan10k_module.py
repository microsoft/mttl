import torch
from torch.utils.data import DataLoader

from mttl.datamodule.base import DefaultCollator
from mttl.dataloader.flan_10k_dataset_reader import Flan10kDataset
from mttl.datamodule.base import DatasetConfig, DefaultDataModule
from dataclasses import dataclass
import numpy as np


@dataclass
class Flan10kConfig(DatasetConfig):
    pass


@dataclass
class DataCollatorForFlan10k(DefaultCollator):
    def __call__(self, batch, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        sources = [b["source"] for b in batch]
        labels = [b["target"] for b in batch]
        task_ids = [b.get("task_id", 0) for b in batch]
        # give a random experts
        experts_ids = [b.get("expert_id", np.random.randint(0, 245 - 1)) for b in batch]
        task_names = [b.get("task_name", None) for b in batch]

        output_batch = (
            self.prepare_inputs_for_gpt_family(sources, labels)
            if self.model_family == "gpt"
            else self.prepare_inputs_for_seq2seq_family(sources, labels)
        )

        output_batch["task_ids"] = torch.LongTensor(task_ids)
        output_batch["task_names"] = task_names
        output_batch["sources_texts"] = sources
        output_batch["labels_texts"] = labels
        output_batch["expert_ids"] = torch.LongTensor(experts_ids)
        return output_batch


class Flan10kModule(DefaultDataModule):
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
            collate_fn=self.test_collate_fn,
        )

    def __init__(self, config, for_generation=False, val_mixin=None):
        super().__init__(config, for_generation=for_generation, val_mixin=val_mixin)

        self.config = config

    @property
    def test_collate_fn(self):
        return DataCollatorForFlan10k(
            tokenizer=self.tokenizer,
            padding="longest",
            max_input_length=self.config.max_input_length,
            max_output_length=self.config.max_output_length,
            pad_to_multiple_of=8,
            return_tensors="pt",
            model_family=self.config.model_family,
        )

    @property
    def collate_fn(self):
        return DefaultCollator(
            tokenizer=self.tokenizer,
            padding="longest",
            max_input_length=self.config.max_input_length,
            max_output_length=self.config.max_output_length,
            pad_to_multiple_of=8,
            return_tensors="pt",
            model_family=self.config.model_family,
        )

    def get_dataset(self, idxs=None):
        return Flan10kDataset(
            self.config.category,
            idxs=idxs,
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
