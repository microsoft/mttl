from dataclasses import dataclass, field, fields
from typing import Dict, List

import torch


@dataclass
class RoutingInfo:
    input_ids: torch.Tensor = None
    labels: torch.Tensor = None
    attention_mask: torch.Tensor = None
    task_ids: torch.Tensor = None
    task_names: List[str] = None
    task_sources: List[str] = None
    example_ids: List[int] = None
    sources_texts: List[str] = None
    labels_texts: List[str] = None
    task_weights: torch.nn.ParameterDict = None
    aux_losses: Dict = field(default_factory=dict)
    packed_seq_lens: List[int] = None
    seq_lens: List[int] = None
    packed_attn_mask: torch.Tensor = None

    @classmethod
    def pop_elements(cls, batch, keep=None):
        """We don't want to pass these elements to the model."""
        keep = keep or []
        return [
            batch.pop(k.name)
            for k in fields(cls)
            if k.name in batch and k.name not in keep
        ]

    @classmethod
    def prepare_for_forward(cls, batch):
        cls.pop_elements(batch, keep=["input_ids", "attention_mask", "labels"])

    @classmethod
    def prepare_for_generate(cls, batch):
        cls.pop_elements(batch, keep=["input_ids", "attention_mask"])

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
            packed_seq_lens=batch.get("packed_seq_lens", None),
            seq_lens=batch.get("seq_lens", None),
            packed_attn_mask=batch.get("packed_attn_mask", None),
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
