import math
from dataclasses import dataclass
from typing import Dict

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from mttl.models.containers.selectors.base import (
    Selector,
    SelectorConfig,
    forward_with_cache,
)
from mttl.models.containers.selectors.selector_output import (
    MultiheadBatchSequenceExpertsAndWeightsSelectorOutput,
)
from mttl.models.library.expert import ExpertInfo


@dataclass
class PKSelectorConfig(SelectorConfig):
    num_heads: int = 8
    emb_dim: int = 128
    top_k: int = -1
    non_competitive_gates: bool = False
    moe_num_experts: int = (
        -1
    )  # for num experts to be set it must have the same name as the num of experts variable in MoEExpertConfig
    pk_use_batchnorm: bool = False


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def init_(t, dim=None):
    dim = default(dim, t.shape[-1])
    std = 1.0 / math.sqrt(dim)
    return nn.init.normal_(t, mean=0, std=std)


@Selector.register("moe_pk_router", PKSelectorConfig)
class PKSSelector(Selector):
    def __init__(self, config, **kwargs) -> None:
        super().__init__(config)

        if "in_d" in kwargs:
            self.input_dim = kwargs["in_d"]
        elif "layer" in kwargs:
            self.input_dim = kwargs["layer"].in_features
        else:
            raise ValueError(
                "PKSSelector requires in_d or layer (assumed to be MLP block ith fc1) to be passed."
            )

        self.top_k = config.top_k
        self.num_experts = config.moe_num_experts
        self.num_heads = config.num_heads
        self.emb_dim = config.emb_dim

        self.q = nn.Linear(
            self.input_dim, self.num_heads * 2 * self.emb_dim, bias=False
        )

        self.gate_activation = (
            nn.Softmax(dim=-1) if not self.config.non_competitive_gates else nn.ReLU()
        )

        assert self.num_experts > 0
        assert math.sqrt(self.num_experts).is_integer(), "N must be a perfect square"
        self.N = int(math.sqrt(self.num_experts))

        self.norm = (
            nn.BatchNorm1d(self.emb_dim) if config.pk_use_batchnorm else nn.Identity()
        )
        self._init_keys()

    def _init_keys(self, device=None):
        assert self.num_experts > 0
        assert math.sqrt(self.num_experts).is_integer(), "N must be a perfect square"
        self.keys = nn.Parameter(
            torch.randn((self.num_heads, self.N, 2, self.emb_dim)).to(device)
        )  # H, (2 x emb_dim), N
        init_(self.keys)

    @forward_with_cache
    def forward(
        self, input, **kwargs
    ) -> MultiheadBatchSequenceExpertsAndWeightsSelectorOutput:
        input = input.to(dtype=self.q.weight.dtype)
        assert math.sqrt(self.num_experts).is_integer(), "N must be a perfect square"
        b, s, d = input.shape

        # get query
        q_x = self.q(input)  # B x s x (num_heads * 2 * emb_dim)
        q_x = rearrange(
            q_x, "b s (h p d) -> (h p b) d s", d=self.emb_dim, h=self.num_heads, p=2
        )
        q_x = self.norm(q_x)
        q_x = rearrange(
            q_x, "(h p b) d s -> b s h p d", d=self.emb_dim, h=self.num_heads, p=2
        )

        sim_scores = torch.einsum(
            "bshpd,hnpd->bshpn", q_x, self.keys
        )  # B x s x H x 2 x N
        scores, indices = sim_scores.topk(k=self.top_k, dim=-1)

        # all selected experts
        (i_1, i_2), (i_1_idx, i_2_idx) = map(
            lambda t: t.chunk(2, dim=3), (scores, indices)
        )
        i_2 = i_2.swapaxes(-1, -2)
        i_2_idx = i_2_idx.swapaxes(-1, -2)

        # map top-k on sub-keys to indices on cartesian product
        idx_full = (i_1_idx * self.top_k + i_2_idx).view(b, s, self.num_heads, -1)
        # get full cartesian scores of k^2 pairs
        scores_full = (i_1 + i_2).view(b, s, self.num_heads, -1)

        # redo top-k on full cartesian scores
        logits, idx = torch.topk(scores_full, self.top_k, dim=-1)
        selected_experts = idx_full.gather(-1, idx)

        routing_weights = self.gate_activation(logits)

        return MultiheadBatchSequenceExpertsAndWeightsSelectorOutput(
            experts=selected_experts, weights=routing_weights
        )

    def on_add_expert(
        self, expert_name: str, expert_info: ExpertInfo = None, is_default=False
    ):
        # No need to do anything here, as it relied on moe_num_experts to be passed at instantiation.
        pass
