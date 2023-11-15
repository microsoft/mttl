import torch
import pytorch_lightning as pl
from mttl.datamodule.mt_seq_to_seq_module import FlanModule, FlanConfig
import pandas as pd
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ClassificationDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, subsample=1000):
        super().__init__()
        self.batch_size = batch_size
        self.subsample = subsample
        df = pd.read_csv("tasks_names.csv")
        self.tasks_names_to_ids = dict(zip(df["task_name"], df["task_ids"]))

    def setup(self, stage: str) -> None:
        data_module = FlanModule(
            FlanConfig(
                dataset="sordonia/flan-10k-flat", model="EleutherAI/gpt-neo-125m"
            )
        )

        self.train_dataset = data_module.train_dataset
        self.test_dataset = data_module.test_dataset

    def collate_fn(self, batch):
        # change the tasks_names to tasks_ids
        inputs = [b["source"] for b in batch]
        targets = [b["target"] for b in batch]

        tasks_ids = [self.tasks_names_to_ids[b["task_name"]] for b in batch]
        return {
            "input": inputs,
            "target": targets,
            "label": torch.tensor(tasks_ids).to(device),
        }

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
        )
