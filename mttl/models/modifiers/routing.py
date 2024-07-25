from dataclasses import dataclass, field
from typing import Dict, List

import torch


@dataclass
class RoutingInfo:
    task_ids: torch.Tensor = None
    task_names: List[str] = None
    task_sources: List[str] = None
    example_ids: List[int] = None
    labels: torch.Tensor = None
    input_ids: torch.Tensor = None
    sources_texts: List[str] = None
    attention_mask: torch.Tensor = None
    task_weights: torch.nn.ParameterDict = None
    aux_losses: Dict = field(default_factory=dict)

    @classmethod
    def from_batch(cls, batch: dict, **kwargs):
        task_ids = batch.get("task_ids").long() if "task_ids" in batch else None
        task_names = batch.get("task_names", None)
        task_weights = batch.get("task_weights", None)
        task_sources = batch.get("task_sources", None)

        ri = cls(
            task_ids=task_ids,
            task_names=task_names,
            task_weights=task_weights,
            task_sources=task_sources,
            input_ids=batch.get("input_ids", None),
            example_ids=batch.get("example_ids", None),
            sources_texts=batch.get("sources_texts", None),
            labels=batch.get("labels", None),
            attention_mask=batch.get("attention_mask", None),
            **kwargs,
        )
        return ri

    def _repeat(self, inputs, n):
        if inputs is not None:
            if isinstance(inputs, torch.Tensor):
                return inputs.repeat_interleave(n)
            else:
                return [item for item in inputs for _ in range(n)]
        return inputs

    def repeat_interleave(self, repeats):
        # useful for beam search
        self.task_ids = self._repeat(self.task_ids, repeats)
        self.task_names = self._repeat(self.task_names, repeats)
        self.task_sources = self._repeat(self.task_sources, repeats)
        self.example_ids = self._repeat(self.example_ids, repeats)
        self.task_weights = self._repeat(self.task_weights, repeats)
