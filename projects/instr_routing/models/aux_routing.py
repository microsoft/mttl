import torch
import copy
import torch.nn as nn
from enum import Enum

import torch.nn.functional as F
from mttl.models.modify_model import patch_layers, register_modifier
from mttl.models.poly import PolyLoRALinear
from mttl.models.routing import (
    RouterWrapper,
    RoutingSelector,
    register_selector,
)


@register_selector("aux_var_router")
class VariationalRouter(RoutingSelector):
    def __init__(self, config, in_d):
        super().__init__()

        self.config = config
        self.in_d = in_d
        self.n_splits = config.n_splits
        assert self.n_splits == 1
        self.input_ln = nn.LayerNorm(in_d)

        self.prior_router = nn.Linear(in_d, config.n_skills * self.n_splits)
        self.prior_router_ln = nn.LayerNorm(in_d)
        self.prior_router_ln.weight = nn.Parameter(torch.ones(in_d))

        self.post_router = nn.Linear(in_d, config.n_skills * self.n_splits)
        self.post_router.bias.data.fill_(0)
        self.post_router_ln = nn.LayerNorm(in_d)
        self.post_router_ln.weight = nn.Parameter(torch.ones(self.in_d))

    @property
    def W_norm(self):
        W = self.ff.weight
        norm = torch.norm(W, p=1, keepdim=True)
        return norm.item()

    def route(self, router: nn.Linear, layer_norm: nn.LayerNorm, x):
        x = self.input_ln(x)
        weights = layer_norm(router.weight)
        return F.linear(x, weights, router.bias)

    def apply_mask_and_average(self, x, padding_mask):
        x_rout = x * padding_mask.unsqueeze(-1).to(x.device)
        lengths = x_rout.ne(0).sum(dim=1)
        x_rout = (x_rout.sum(dim=1) / lengths)
        return x_rout

    def forward(self, routing_infos, input: torch.Tensor):
        bs, seq, _ = input.shape
        padding_mask = routing_infos.pad_token_mask
        inst_padding_mask = routing_infos.inst_token_mask

        prior_input = self.apply_mask_and_average(input, inst_padding_mask)
        post_input = self.apply_mask_and_average(input, padding_mask)

        prior_routes = self.route(self.prior_router, self.prior_router_ln, prior_input)
        post_routes = self.route(self.post_router, self.post_router_ln, post_input)

        routing_probs = F.softmax(post_routes / 0.25, dim=-1)
        auxiliary_loss = F.kl_div(
            F.log_softmax(prior_routes / 0.25, dim=-1),
            F.log_softmax(post_routes / 0.25, dim=-1),
            log_target=True,
        ).mean(0)
        return routing_probs.unsqueeze(1), auxiliary_loss


class AuxRoutingLoRALinear(PolyLoRALinear):
    def __init__(self, config, task_id_ptr, linear_layer, selector=None, **kwargs):
        super().__init__(config, task_id_ptr, linear_layer, selector, **kwargs)

        # store losses and metrics
        self.losses = []
        self.metrics = {}
        self.training_steps = 0
        self.reset_parameters()

    def forward(self, input):
        if self.training:
            self.training_steps += 1

        task_id = self.routing_infos.task_ids
        repeat = input.size(0) // task_id.size(0)

        # this repeat follows the patten in `model.predict()` line 152
        if repeat:
            self.routing_infos.repeat_interleave(repeat)

        if self.selector is not None:
            mixing_weights = self.selector(self.routing_infos, input=input)

            if isinstance(mixing_weights, tuple):
                mixing_weights, kl = mixing_weights
                self.losses.append(kl)

            mixing_weights.to(input.device)
        else:
            bs = input.size(0)
            mixing_weights = torch.ones(
                bs, self.n_splits, self.n_skills, device=input.device, dtype=input.dtype
            )
        self.metrics["routing"] = mixing_weights.detach().cpu().float()

        bs, n_splits, n_skills = mixing_weights.size()
        A = torch.einsum("bqs,qsdr->bqdr", (mixing_weights, self.lora_a))
        B = torch.einsum("bqs,qsrd->bqrd", (mixing_weights, self.lora_b))
        A = A.reshape(bs, self.in_features, self.rank)
        B = B.transpose(1, 2).reshape(bs, self.rank, self.out_features)
        adapter_out = input.bmm(A).bmm(B) * self.scaling

        warmup = min(self.training_steps / 10_000, 1)
        if self.use_warmup:
            adapter_out = adapter_out * warmup
        return self.linear_layer(input) + adapter_out


@register_modifier("aux_lora")
def modify_with_variational_router(transformer, config):
    return patch_layers(transformer, config, AuxRoutingLoRALinear, RouterWrapper)
