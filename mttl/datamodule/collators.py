from dataclasses import dataclass
from transformers import DataCollatorForSeq2Seq
from typing import List
import torch

from mttl.dataloader.data_utils import ExampleInfo


@dataclass
class DefaultCollator(DataCollatorForSeq2Seq):
    def __call__(self, batch: List[ExampleInfo]):
        input_ids = [b.input_ids for b in batch]
        target_ids = [b.target_ids for b in batch]
        hashes = [b.hash for b in batch]
        task_ids = [b.task_id for b in batch]
        instruction_hashes = [b.instruction_hash for b in batch]
        task_ids = torch.LongTensor(task_ids)
        collated_features = super().__call__(
            [
                {"input_ids": i, "labels": t}
                for i, t in zip(input_ids, target_ids)
            ]
        )
        output_batch = {
            "input_ids": collated_features["input_ids"],
            "labels": collated_features["labels"],
            "task_ids": task_ids,
            "hashes": hashes,
            "instruction_hashes": instruction_hashes,
        }
        return output_batch