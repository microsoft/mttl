import numpy as np
import torch
from dataclasses import dataclass


@dataclass
class ExampleInfo:
    input_ids: torch.Tensor
    target_ids: torch.Tensor
    task_id: int
    hash: str
    example_id: int
    input_text: str = None
    instruction_hash: str = None


@dataclass
class MultiChoiceExampleInfo:
    input_ids: torch.Tensor
    target_ids: torch.Tensor
    answer_choices_ids: torch.Tensor
    label: torch.Tensor
    idx: torch.Tensor
    task_id: int
    hash: str
    example_id: int
    instruction_hash: str = None
