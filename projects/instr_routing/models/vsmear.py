import math
import torch
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
from projects.instr_routing.models.clm import AugmentedRoutingInfo


class Metrics:
    def __init__(self) -> None:
        self.metrics = {}

    def add(self, key, value):
        self.metrics[key] = value.detach().cpu()

    def __setitem__(self, key, value):
        self.add(key, value)

    def clear(self):
        self.metrics.clear()

    def items(self):
        return self.metrics.items()


@register_selector("smear")
class SMEARRouter(RoutingSelector):
    def __init__(self, config, in_d):
        super().__init__()

        self.config = config
        self.in_d = in_d
        self.n_skills = config.n_skills
        self.n_splits = config.n_splits
        self.temperature = config.router_temperature
        self.normalize_weights = config.router_normalize_weights
        assert self.n_splits == 1

        # during generation we want to cache the input encoding
        # because in successive generation steps, the input is going to be only the encoding for the last token
        self._router_input_cache = None
        self.prior_router = nn.Linear(in_d, config.n_skills * self.n_splits, bias=False)
        self.prior_router_ln = nn.LayerNorm(in_d)
        self.prior_router_ln.weight = nn.Parameter(torch.ones(in_d))

    @property
    def W_norm(self):
        W = self.ff.weight
        norm = torch.norm(W, p=1, keepdim=True)
        return norm.item()

    def route(self, router: nn.Linear, layer_norm: nn.LayerNorm, x):
        x = self.prior_router_ln(x)

        if self.normalize_weights:
            weights = router.weight / torch.norm(router.weight, p=2, keepdim=True)
            return F.linear(x, weights, None) / self.temperature
        else:
            return F.linear(x, router.weight, None) / self.temperature

    def apply_mask_and_average(self, x, padding_mask):
        x_rout = x * padding_mask.unsqueeze(-1).to(x.device)
        lengths = padding_mask.ne(0).sum(dim=1, keepdim=True) + 1e-5
        x_rout = x_rout.sum(dim=1) / lengths
        return x_rout

    def _get_router_inputs(self, input: torch.Tensor, routing_infos: AugmentedRoutingInfo):
        """When generating, the successive forward calls only receive the last token (bs, 1, d).

        Therefore, at the first forward call (context encoding), we need to cache the input encodings
        so that we can use it to compute the prior routing probabilities.
        """
        if routing_infos.generation_mode:
            if self._router_input_cache is None:
                router_input = self.apply_mask_and_average(input, routing_infos.inst_token_mask)
                self._router_input_cache = router_input
            else:
                router_input = self._router_input_cache
        else:
            self._router_input_cache = None
            router_input = self.apply_mask_and_average(input, routing_infos.inst_token_mask)
        return router_input

    def forward(self, routing_infos: AugmentedRoutingInfo, input: torch.Tensor):
        prior_input = self._get_router_inputs(input, routing_infos)
        prior_routes = self.route(self.prior_router, self.prior_router_ln, prior_input)
        routing_probs = F.softmax(prior_routes, dim=-1)
        return routing_probs.unsqueeze(1)


@register_selector("vsmear")
class VSMEARRouter(SMEARRouter):
    def __init__(self, config, in_d):
        super().__init__(config, in_d)

        self.metrics = Metrics()

    def forward(self, routing_infos: AugmentedRoutingInfo, input: torch.Tensor):
        self.metrics.clear()

        prior_input = self._get_router_inputs(input, routing_infos)
        prior_routes = self.route(self.prior_router, self.prior_router_ln, prior_input)

        if self.training:
            # during training :-)
            assert routing_infos.generation_mode is False, "We are not expecting to be in generation mode during training."

            padding_mask = routing_infos.pad_token_mask
            post_input = self.apply_mask_and_average(input, padding_mask)
            post_routes = self.route(self.prior_router, self.prior_router_ln, post_input)
            post_probs = routing_probs = F.softmax(post_routes, dim=-1)
            prior_probs = F.softmax(prior_routes, dim=-1)

            # compute auxiliary loss (KL divergence), KL = - H(posterior) + Xent(posterior, prior)
            h_post = -(post_probs * F.log_softmax(post_routes, -1)).sum(1).mean()
            h_pri = -(prior_probs * F.log_softmax(prior_routes, -1)).sum(1).mean()
            x_ent = -(post_probs * F.log_softmax(prior_routes, -1)).sum(1).mean()

            self.routings = post_probs.detach().cpu()
            self.metrics["h_post"] = h_post / math.log(self.n_skills)
            self.metrics["h_pri"] = h_pri / math.log(self.n_skills)
            self.metrics["x_ent"] = x_ent
            self.auxiliary_loss = x_ent
        else:
            # during eval :-(
            prior_probs = routing_probs = F.softmax(prior_routes, dim=-1)
            h_pri = -(prior_probs * F.log_softmax(prior_routes, -1)).sum(1).mean()

            self.routings = prior_probs.detach().cpu()
            self.metrics["h_pri"] = h_pri / math.log(self.n_skills)
            self.auxiliary_loss = h_pri.sum() * 0.0
        return routing_probs.unsqueeze(1)


class AuxRoutingLoRALinear(SkilledLoRA, RoutingMixin):
    def __init__(self, config, task_id_ptr, layer, selector=None, **kwargs):
        RoutingMixin.__init__(self, task_id_ptr)
        SkilledLoRA.__init__(self, config, layer, **kwargs)

        if selector is None:
            self.selector = get_selector(config, in_d=self.in_features)
        else:
            self.selector = selector

    def forward(self, input):
        task_id = self.routing_infos.task_ids
        repeat = input.size(0) // task_id.size(0)

        # this repeat follows the patten in `model.predict()` line 152
        if repeat:
            self.routing_infos.repeat_interleave(repeat)

        if self.selector is not None:
            mixing_weights = self.selector(self.routing_infos, input=input)
        else:
            bs = input.size(0)
            mixing_weights = torch.ones(
                bs, self.n_splits, self.n_skills, device=input.device, dtype=input.dtype
            )
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
