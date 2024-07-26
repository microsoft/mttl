from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F

from mttl.models.containers.selectors.base import (
    BatchSequenceExpertsAndWeightsSelectorOutput,
    Selector,
    SelectorConfig,
    SelectorOutput,
    forward_with_cache,
)
from mttl.models.library.expert import ExpertInfo


@dataclass
class MOERKHSSelectorConfig(SelectorConfig):
    rkhs_dim: int = 512
    emb_dim: int = 128
    top_k: int = -1


@Selector.register("moe_rkhs_router", MOERKHSSelectorConfig)
class MOERKHSSelector(Selector):
    def __init__(self, config, **kwargs) -> None:
        super().__init__(config)

        if "layer" not in kwargs:
            raise ValueError(
                "MOERKHSSelector requires a layer to be passed in kwargs to infer the input dimension."
            )

        self.top_k = config.top_k
        self.input_dim = kwargs["layer"].weight.data.shape[-1]
        self.rkhs_dim = config.rkhs_dim
        self.emb_dim = config.emb_dim

        device = kwargs["layer"].weight.device

        self.rkhs_exp = nn.Linear(self.emb_dim, self.rkhs_dim, device=device)
        self.rkhs_hid = nn.Linear(self.input_dim, self.rkhs_dim, device=device)
        self.rkhs_embeddings = nn.Parameter(
            torch.empty((0, self.emb_dim), device=device)
        )

    def _get_weights(self, input):
        input_view = input.view(-1, input.shape[-1])
        return self.rkhs_hid(input_view).reshape(input.shape[0], input.shape[1], -1)

    @forward_with_cache
    def forward(self, input, **kwargs) -> BatchSequenceExpertsAndWeightsSelectorOutput:
        # do routing business on fp32
        input = input.to(dtype=self.rkhs_exp.weight.dtype)

        rkhs_enc = self._get_weights(input)
        rkhs_emb = self.rkhs_exp(self.rkhs_embeddings)

        router_logits = torch.matmul(rkhs_enc, rkhs_emb.T)
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)

        if self.top_k > 0:
            routing_weights, selected_experts = torch.topk(
                routing_weights, self.top_k, dim=-1
            )
            # we cast back to the input dtype
            routing_weights = routing_weights.to(input.dtype)
        else:
            # soft routing
            selected_experts = SelectorOutput.ALL_EXPERTS

        g = getattr(self.info_container, "routing_gates", [])
        g.append(router_logits)
        self.info_container.routing_gates = g

        return BatchSequenceExpertsAndWeightsSelectorOutput(
            experts=selected_experts, weights=routing_weights
        )

    def get_merging_weights(self, **selector_kwargs) -> Dict:
        raise ValueError(
            f"Not supported for {self.__class__} since routing depends on input."
        )

    def get_routing_weights(self):
        raise ValueError("Not supported for MOESelector.")

    def on_add_expert(
        self, expert_name: str, expert_info: ExpertInfo = None, is_default=False
    ):
        # just initialize the expert embeddings
        self.rkhs_embeddings.data = torch.cat(
            [
                self.rkhs_embeddings.data,
                torch.zeros(
                    1, self.emb_dim, device=self.rkhs_embeddings.device
                ).uniform_(-0.02, 0.02),
            ],
            dim=0,
        )
