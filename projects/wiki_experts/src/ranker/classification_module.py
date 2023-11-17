import torch
from mttl.datamodule.mt_seq_to_seq_module import FlanModule, FlanConfig
from projects.wiki_experts.src.config import tasks_names_to_ids, ids_to_tasks_names
from dataclasses import dataclass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class ClassificationConfig(FlanConfig):
    pass


@dataclass
class DataCollatorForClassification:
    def __call__(self, batch):
        # change the tasks_names to tasks_ids
        inputs = [b["source"] for b in batch]
        targets = [b["target"] for b in batch]

        tasks_ids = [tasks_names_to_ids[b["task_name"]] for b in batch]
        return {
            "input": inputs,
            "target": targets,
            "label": torch.tensor(tasks_ids),
        }


class ClassificationDataModule(FlanModule):
    @property
    def collate_fn(self):
        # change the tasks_names to tasks_ids
        return DataCollatorForClassification()
