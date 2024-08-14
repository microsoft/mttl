from dataclasses import dataclass
from typing import Dict

import math
import torch
from torch import nn
from torch.nn import functional as F

from mttl.models.containers.selectors.base import (
    Selector,
    SelectorConfig,
    forward_with_cache,
)
from mttl.models.containers.selectors.selector_output import (
    BatchSequenceExpertsAndWeightsSelectorOutput,
    SelectorOutput,
)
from mttl.models.library.expert import ExpertInfo


@dataclass
class PKSelectorConfig(SelectorConfig):
    emb_dim: int = 128
    top_k: int = -1
    moe_num_experts: int = (
        -1
    )  # for num experts to be set it must have the same name as the num of experts variable in MoEExpertConfig


@Selector.register("moe_pk_router", PKSelectorConfig)
class PKSSelector(Selector):
    def __init__(self, config, **kwargs) -> None:
        super().__init__(config)

        if "layer" not in kwargs:
            raise ValueError(
                "MOERKHSSelector requires a layer to be passed in kwargs to infer the input dimension."
            )
        assert self.config.emb_dim % 2 == 0, "emb_dim must be even"

        self.top_k = config.top_k
        self.input_dim = kwargs["layer"].weight.data.shape[-1]

        self.q = nn.Linear(self.input_dim, self.config.emb_dim, bias=False)
        self._N = config.moe_num_experts
        assert self._N > 0
        assert math.sqrt(self._N).is_integer(), "N must be a perfect square"
        self._init_keys()

    def _init_keys(self, device=None):
        assert self._N > 0
        assert math.sqrt(self._N).is_integer(), "N must be a perfect square"
        self.keys = nn.ParameterList(
            [
                nn.Parameter(
                    torch.randn(
                        (int(math.sqrt(self._N)), int(self.config.emb_dim / 2)),
                        device=device,
                    )
                ),
                nn.Parameter(
                    torch.randn(
                        (int(math.sqrt(self._N)), int(self.config.emb_dim / 2)),
                        device=device,
                    )
                ),
            ]
        )

    @staticmethod
    def _calc_full_idx(i_1_idx, i_2_idx, top_k):
        """
        Map the top-k indices from both sqrt(N) halves to the index int he coresponding cartesian product
        """
        b, s = i_1_idx.shape[:2]
        i_1_idx = i_1_idx.unsqueeze(-1)
        i_2_idx = i_2_idx.unsqueeze(-2)
        idx_full = i_1_idx * top_k + i_2_idx
        return idx_full.view(b, s, -1)

    @staticmethod
    def _get_full_cartesian_scores(i_1, i_2):
        """
        Compute the full cartesian sum of i_1 and i_2
        """
        b, s = i_1.shape[:2]
        i_1 = i_1.unsqueeze(-1)
        i_2 = i_2.unsqueeze(-2)
        return (i_1 + i_2).view(b, s, -1)

    @forward_with_cache
    def forward(self, input, **kwargs) -> BatchSequenceExpertsAndWeightsSelectorOutput:
        input = input.to(dtype=self.q.weight.dtype)
        assert math.sqrt(self._N).is_integer(), "N must be a perfect square"

        # get query
        q_x = self.q(input)  # B x d
        q_1, q_2 = torch.chunk(q_x, 2, dim=-1)  # B x d/2

        # get sub-keys
        i_1 = torch.matmul(q_1, self.keys[0].T)  # B x sqrt(N)
        i_2 = torch.matmul(q_2, self.keys[1].T)  # B x sqrt(N)

        # top-k on sub-keys
        i_1, i_1_idx = torch.topk(i_1, self.top_k, dim=-1)
        i_2, i_2_idx = torch.topk(i_2, self.top_k, dim=-1)

        # map top-k on sub-keys to indices on cartesian product
        idx_full = self._calc_full_idx(i_1_idx, i_2_idx, self.top_k)
        # get full cartesian scores of k^2 pairs
        scores_full = self._get_full_cartesian_scores(i_1, i_2)

        # redo top-k on full cartesian scores
        logits, idx = torch.topk(scores_full, self.top_k, dim=-1)
        selected_experts = idx_full.gather(-1, idx)

        routing_weights = F.softmax(logits, dim=-1, dtype=torch.float)

        return BatchSequenceExpertsAndWeightsSelectorOutput(
            experts=selected_experts, weights=routing_weights
        )

    def on_add_expert(
        self, expert_name: str, expert_info: ExpertInfo = None, is_default=False
    ):
        # No need to do anything here, as it relied on moe_num_experts to be passed at instantiation.
        pass
