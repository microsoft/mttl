import torch
import numpy as np
import copy
import torch.nn as nn
from enum import Enum

import torch.nn.functional as F
from mttl.models.adapters import SkilledLoRA
from mttl.models.modifiers import modify_with_routing, register_modifier
from mttl.models.modifiers.routing import (
    RouterWrapper,
    RoutingMixin,
    RoutingSelector,
    get_selector,
    register_selector,
)
from mttl.models.modifiers.poly import PolytroponSelector
from torch.distributions.categorical import Categorical
from torch.distributions.kl import kl_divergence

@register_selector("smear")
class SMEARRouter(RoutingSelector):
    def __init__(self, config, in_d):
        super().__init__()

        self.config = config
        self.in_d = in_d
        self.n_splits = config.n_splits
        self.temperature = config.router_temperature
        assert in_d % self.n_splits == 0

        self.prior_router_weight = nn.Parameter(
            torch.empty(self.n_splits, in_d // self.n_splits, config.n_skills)
        )
        self.prior_router_bias = nn.Parameter(
            torch.empty(self.n_splits, config.n_skills)
        )
        bound = 1. / np.sqrt(in_d // self.n_splits)
        self.prior_router_weight.data.uniform_(-bound, bound)
        self.prior_router_bias.data.uniform_(-bound, bound)

        self.prior_router_ln = None
        self.activation_cache = None
        self.cache_size = 32

    @property
    def W_norm(self):
        W = self.ff.weight
        norm = torch.norm(W, p=1, keepdim=True)
        return norm.item()

    def route(self, router_w: nn.Parameter, router_b: nn.Parameter, layer_norm: nn.LayerNorm, x, ln=False):
        x = x.reshape(x.size(0), self.n_splits, -1)
        sim_scores = torch.einsum('bsd,sdk->bsk', x, router_w) + router_b.unsqueeze(0) 
        return sim_scores / self.temperature

    def apply_mask_and_average(self, x, padding_mask):
        x_rout = x * padding_mask.unsqueeze(-1).to(x.device)
        lengths = padding_mask.ne(0).sum(1, keepdim=True)
        x_rout = x_rout.sum(dim=1) / lengths
        return x_rout

    def forward(self, routing_infos, input: torch.Tensor):
        if routing_infos.encoder_output is not None:
            input = routing_infos.encoder_output
        inst_padding_mask = routing_infos.inst_token_mask
        prior_input = self.apply_mask_and_average(input, inst_padding_mask)
        prior_routes = self.route(self.prior_router_weight, self.prior_router_bias, self.prior_router_ln, prior_input)
        routing_probs = F.softmax(prior_routes, dim=-1) # bs, n_splits, n_skills

        if self.activation_cache is None:
            self.activation_cache = routing_probs.detach()
        else:
            self.activation_cache = torch.cat((routing_probs.detach(), self.activation_cache), dim=0)

        self.activation_cache = self.activation_cache[:self.cache_size]

        auxiliary_loss = routing_probs.sum().detach() * 0.0
        return routing_probs, auxiliary_loss

@register_selector("vsmear")
class VSMEARRouter(SMEARRouter):
    # Polytropon as the posterior
    # Smear as the prior
    def __init__(self, config, in_d):
        super().__init__(config, in_d)
        self.post_poly = PolytroponSelector(config)

    def forward(self, routing_infos, input: torch.Tensor):
        repeat = 1
        if routing_infos.encoder_output is not None:
            repeat = input.size(0) // routing_infos.encoder_output.size(0)
            input = routing_infos.encoder_output

        inst_padding_mask = routing_infos.inst_token_mask
        prior_input = self.apply_mask_and_average(input, inst_padding_mask)
        prior_logits = self.route(self.prior_router_weight, self.prior_router_bias, self.prior_router_ln, prior_input)
        prior_probs = F.softmax(prior_logits, -1)
        if repeat > 1:
            prior_logits = prior_logits.repeat_interleave(repeat, dim=0)
        
        prior = Categorical(logits=prior_logits)
        if self.training:
            post_probs = self.post_poly(routing_infos)
            posterior = Categorical(probs=post_probs)
            auxiliary_loss = kl_divergence(posterior, prior)
            routing_probs = post_probs
        else:
            routing_probs = prior_probs
            auxiliary_loss = prior_logits.sum().detach() * 0.0
        
        if self.activation_cache is None:
            self.activation_cache = prior_probs.detach()
        else:
            self.activation_cache = torch.cat((prior_probs.detach(), self.activation_cache), dim=0)
        self.activation_cache = self.activation_cache[:self.cache_size]


        return routing_probs, auxiliary_loss

class AuxRoutingLoRALinear(SkilledLoRA, RoutingMixin):
    def __init__(self, config, task_id_ptr, layer, selector=None, **kwargs):
        RoutingMixin.__init__(self, task_id_ptr)
        SkilledLoRA.__init__(self, config, layer, **kwargs)

        if selector is None:
            self.selector = get_selector(config, in_d=self.in_features)
        else:
            self.selector = selector

        # store losses and metrics
        self.losses = []
        self.metrics = {}

    def forward(self, input):
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
        return SkilledLoRA.forward(self, input, mixing_weights)


@register_modifier("smear")
def modify_with_smear(transformer, config):
    config.router_selector = config.router_selector or "smear"
    config.adapter_type = config.adapter_type or "lora"

    if config.adapter_type in ["lora"]:
        return modify_with_routing(
            transformer, config, AuxRoutingLoRALinear, RouterWrapper
        )
    else:
        raise NotImplementedError(
            f"Adapter type {config.adapter_type} not implemented for vsmear modifier."
        )


@register_modifier("vsmear")
def modify_with_vsmear(transformer, config):
    config.router_selector = "vsmear"

    return modify_with_smear(transformer, config)