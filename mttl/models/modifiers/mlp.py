import math
from dataclasses import dataclass
from typing import List

import bitsandbytes as bnb
import numpy as np
import torch
from torch import nn
from transformers.activations import ACT2FN

from mttl.models.modifiers import register_modifier
from mttl.models.modifiers.base import Modifier, ModifierConfig


@dataclass
class MLPConfig(ModifierConfig):
    n_embd: int = 2560


@Modifier.register("mlp", config_cls=MLPConfig)
class MLPModifier(Modifier):
    def __init__(
        self,
        config: MLPConfig,
        layer: nn.Module,
    ):
        super().__init__()

        self.config = config
        self.layer = layer

        device = next(layer.parameters()).device
        dtype = next(layer.parameters()).dtype

        self.mod_fc1 = nn.Linear(config.n_embd, config.n_embd).to(
            device=device, dtype=dtype
        )
        self.mod_fc2 = nn.Linear(config.n_embd, config.n_embd).to(
            device=device, dtype=dtype
        )
        self.act = ACT2FN["gelu_new"]

    def _modifier_forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        hidden_states = self.mod_fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.mod_fc2(hidden_states)
        return hidden_states

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        return self.layer(hidden_states) + self._modifier_forward(hidden_states)

    @classmethod
    def parallel_linear_forward(
        cls, input: torch.Tensor, mlps: List["MLPModifier"]
    ) -> torch.Tensor:
        if len(set([mlp.layer for mlp in mlps])) > 1:
            raise ValueError("Cannot parallelize adapters applied to different layers.")

        mlp_to_index = {}
        for i, mlp in enumerate(mlps):
            mlp_to_index.setdefault(mlp, []).append(i)

        if input.ndim == 3:
            raise ValueError("Cannot parallelize MLP modules applied to 3D inputs.")

        output = torch.zeros(
            input.shape,
            dtype=input.dtype,
            device=input.device,
        )
        for mlp, indices in mlp_to_index.items():
            hidden_states = input[indices]
            hidden_states = mlp._modifier_forward(hidden_states)
            output.index_add_(0, indices, hidden_states.to(input.dtype))
        return mlps[0].layer(input) + output
