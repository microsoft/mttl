import numpy as np
import torch
from dataclasses import dataclass


@dataclass
class ExampleInfo:
    input: str
    target: str
    task_id: int
    hash: str
    example_id: int
    input_text: str = None
    instruction_hash: str = None


@dataclass
class MultiChoiceExampleInfo:
    input: str
    target: str
    answer_choices: str
    label: torch.Tensor
    idx: torch.Tensor
    task_id: int
    hash: str
    example_id: int
    input_text: str = None
    instruction_hash: str = None
