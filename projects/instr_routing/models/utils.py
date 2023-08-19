from pytorch_lightning import LightningModule
import torch
from dataclasses import dataclass
from typing import List
from collections import defaultdict
from dataclasses import dataclass, field

@dataclass    
class RoutingInfo:   
    task_ids: torch.Tensor
    hashes: List[str]
    instruction_hashes: List[str] = None
    example_ids: List[int] = None
    pad_token_mask: torch.Tensor = None
    inst_token_mask: torch.Tensor = None
    labels: torch.Tensor = None
    # empty list
    aux_loss: List[torch.Tensor] = field(default_factory=list)
    # dict
    metrics:defaultdict(list) = field(default_factory= lambda: defaultdict(list))
    save_oracle_routings: bool = False
    oracle_routings: List[torch.Tensor] = field(default_factory=list)
    routings: List[torch.Tensor] = None

    @classmethod 
    def from_batch(cls, batch: dict):
        ri = RoutingInfo(
            task_ids=batch["task_ids"],   
            hashes=batch.get("hashes", None),
            example_ids=batch.get("example_ids", None),
            instruction_hashes=batch.get("instruction_hashes", None),
            pad_token_mask = batch.get("pad_token_mask", None),
            labels=batch.get("labels", None),
        )
        if "distances" in batch:
            # used for evaluation of soft clustering tuned models
            setattr(ri, "distances", batch["distances"])
        return ri

    def repeat_interleave(self, repeats):
        # print("Repeating routing info", repeats, "times")
        # useful for beam search
        self.task_ids = self.task_ids.repeat_interleave(repeats)
        if self.hashes:
            self.hashes = [h for h in self.hashes for _ in range(repeats)]
        if self.instruction_hashes:
            self.instruction_hashes = [
                h for h in self.instruction_hashes for _ in range(repeats)
            ]
        self.example_ids = (
            self.example_ids.repeat_interleave(repeats)
            if self.example_ids is not None
            else None
        )