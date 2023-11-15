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


class RetrievalDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.sample_input = []
        self.acc = []
        self.expert = []
        self.expert_index = {"abstract_algebra": 0, "global_facts": 1}

    def setup(self, stage=None):
        # Load data from file
        with open(self.data_path, "r") as f:
            for line in f:
                json_unit = json.loads(line)
                self.sample_input.append(json_unit["input"])
                self.acc.append(json_unit["accuracy"])
                self.expert.append(json_unit["expertname"])

    def __len__(self):
        return len(self.sample_input)

    def __getitem__(self, index):
        sample = self.sample_input[index]
        acc = torch.tensor(self.acc[index]).to(device)
        if acc == 100:
            acc = 1
        else:
            acc = 0
        expert_name = self.expert[index]
        label = torch.tensor(self.expert_index[expert_name]).to(device)

        return sample, expert_name, acc, label
