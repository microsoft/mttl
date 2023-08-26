import torch
import copy
import torch.nn as nn
from enum import Enum

import torch.nn.functional as F
from mttl.models.modify_model import patch_layers, register_modifier
from mttl.models.poly import PolyLoRALinear
from mttl.models.routing import (
    RouterWrapper,
    RoutingAdapter,
    RoutingSelector,
    get_selector,
    register_selector,
)


@register_selector("vsmear")
class VariationalRouter(RoutingSelector):
    def __init__(self, config, in_d):
        super().__init__()

        self.config = config
        self.in_d = in_d
        self.n_splits = config.n_splits
        assert self.n_splits == 1

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

    def route(self, router: nn.Linear, layer_norm: nn.LayerNorm, x, ln=False):
        if ln:
            weights = layer_norm(router.weight)
        else:
            weights = router.weight
        return F.linear(x, weights, router.bias)

    def apply_mask_and_average(self, x, padding_mask):
        x_rout = x * padding_mask.unsqueeze(-1).to(x.device)
        lengths = x_rout.ne(0).sum(dim=1)
        x_rout = x_rout.sum(dim=1) / lengths
        return x_rout

    def forward(self, routing_infos, input: torch.Tensor):
        bs, seq, _ = input.shape
        padding_mask = routing_infos.pad_token_mask
        inst_padding_mask = routing_infos.inst_token_mask

        prior_input = self.apply_mask_and_average(input, inst_padding_mask)
        prior_routes = self.route(self.prior_router, self.prior_router_ln, prior_input)

        if self.training:
            # during training :-)
            post_input = self.apply_mask_and_average(input, padding_mask)
            post_routes = self.route(self.post_router, self.post_router_ln, post_input)
            routing_probs = F.softmax(post_routes, dim=-1)

            # compute auxiliary loss (KL divergence)
            auxiliary_loss = routing_probs * F.log_softmax(
                post_routes, -1
            ) - routing_probs * F.log_softmax(prior_routes, dim=-1)

            auxiliary_loss = auxiliary_loss.sum(dim=-1).mean()
        else:
            # during eval :-(
            routing_probs = F.softmax(prior_routes, dim=-1)
            auxiliary_loss = routing_probs.sum().detach() * 0.0
        return routing_probs.unsqueeze(1), auxiliary_loss


class AuxRoutingLoRALinear(RoutingAdapter):
    def __init__(self, config, task_id_ptr, linear_layer, selector=None, **kwargs):
        super().__init__()

        self.n_splits = config.n_splits
        self.n_tasks = config.n_tasks
        self.n_skills = config.n_skills
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.use_warmup = config.lora_warmup
        self.rank = config.lora_rank
        self.lora_alpha = config.lora_alpha
        self.scaling = self.lora_alpha / self.rank
        self.linear_layer = linear_layer
        self.weight = linear_layer.weight
        self.bias = linear_layer.bias
        self.task_id_ptr = task_id_ptr
        self.training_steps = 0.0

        if selector is None:
            self.selector = get_selector(config, in_d=self.in_features)
        else:
            self.selector = selector

        self.lora_a = nn.Parameter(
            torch.empty(
                self.n_skills,
                linear_layer.in_features * self.rank,
            )
        )
        self.lora_b = nn.Parameter(
            torch.empty(
                self.n_skills,
                self.rank * linear_layer.out_features,
            )
        )

        # store losses and metrics
        self.losses = []
        self.metrics = {}
        self.training_steps = 0
        self.reset_parameters()

    def reset_parameters(self):
        import math

        gain = nn.init.calculate_gain(nonlinearity="leaky_relu", param=math.sqrt(5))
        std = gain / math.sqrt(self.in_features)

        with torch.no_grad():
            self.lora_a.uniform_(-std, std)

        # ensure that initially, adding the adapter does not change the output
        if self.use_warmup:
            with torch.no_grad():
                self.lora_b.uniform_(-std, std)
        else:
            torch.nn.init.zeros_(self.lora_b)

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
        else:
            bs = input.size(0)
            mixing_weights = torch.ones(
                bs, self.n_splits, self.n_skills, device=input.device, dtype=input.dtype
            )

        self.metrics["routing"] = mixing_weights.detach().cpu().float()

        mixing_weights = mixing_weights.squeeze()
        mixed_a = torch.matmul(mixing_weights, self.lora_a).view(-1, self.in_features, self.rank)
        mixed_b = torch.matmul(mixing_weights, self.lora_b).view(-1, self.rank, self.out_features)
        adapter_out = input.bmm(mixed_a).bmm(mixed_b) * self.scaling

        warmup = min(self.training_steps / 10_000, 1)
        if self.use_warmup:
            adapter_out = adapter_out * warmup

        return self.linear_layer(input) + adapter_out


@register_modifier("vsmear")
def modify_with_vsmear(transformer, config):
    config.router_selector = config.router_selector or "vsmear"

    if config.adapter_type in ["lora", None]:
        return patch_layers(transformer, config, AuxRoutingLoRALinear, RouterWrapper)
    else:
        raise NotImplementedError(f"Adapter type {config.adapter_type} not implemented for vsmear modifier.")
