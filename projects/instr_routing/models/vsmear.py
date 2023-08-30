import torch
import copy
import torch.nn as nn
from enum import Enum

import torch.nn.functional as F
from mttl.models.adapters import SkilledLoRA, LoRA
from mttl.models.modifiers import modify_with_routing, register_modifier
from mttl.models.modifiers.routing import (
    RouterWrapper,
    RoutingMixin,
    RoutingSelector,
    get_selector,
    register_selector,
)

                     
class SkilledLoRA_MergeLoraAfterOP(SkilledLoRA):
    def __init__(
        self,
        config,
        layer,
    ):
        super().__init__(config, layer)

    def forward_linear_(self, input, weights):
        if self.training:     
            self.training_steps += 1

        bs, _, _ = weights.size()         
        adapter_out = torch.einsum("bsd,qkdr->bsqkr", (input, self.lora_a)) # bs x n_splits x n_skills x rank")       
        adapter_out = torch.einsum("bsqkr,qkrd->bsqkd", (adapter_out, self.lora_b)) # bs x seq x n_splits x n_skills x D        
        adapter_out = torch.einsum("bsqkd,bqk->bsd", (adapter_out, weights)) # bs x seq x n_splits x D
        adapter_out *= self.scaling
        warmup = min(self.training_steps / 10_000, 1)
        if self.use_warmup:
            adapter_out = adapter_out * warmup

        return self.layer(input) + adapter_out

@register_selector("smear")
class SMEARRouter(RoutingSelector):
    def __init__(self, config, in_d):
        super().__init__()

        self.config = config
        self.in_d = in_d
        self.n_splits = config.n_splits
        self.temperature = config.router_temperature
        assert self.n_splits == 1

        self.prior_router = nn.Linear(in_d, config.n_skills * self.n_splits)
        self.prior_router_ln = nn.LayerNorm(in_d)
        self.prior_router_ln.weight = nn.Parameter(torch.ones(in_d))

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
        return F.linear(x, weights, router.bias) / self.temperature

    def apply_mask_and_average(self, x, padding_mask):
        x_rout = x * padding_mask.unsqueeze(-1).to(x.device)
        lengths = padding_mask.ne(0).sum(dim=1, keepdim=True) + 1e-5
        x_rout = x_rout.sum(dim=1) / lengths
        return x_rout

    def forward(self, routing_infos, input: torch.Tensor):
        inst_padding_mask = routing_infos.inst_token_mask
        prior_input = self.apply_mask_and_average(input, inst_padding_mask)
        prior_routes = self.route(self.prior_router, self.prior_router_ln, prior_input)
        routing_probs = F.softmax(prior_routes, dim=-1)
        auxiliary_loss = routing_probs.sum().detach() * 0.0
        return routing_probs.unsqueeze(1), auxiliary_loss


@register_selector("vsmear")
class VSMEARRouter(SMEARRouter):
    def __init__(self, config, in_d):
        super().__init__(config, in_d)

        self.post_router = nn.Linear(in_d, config.n_skills * self.n_splits)
        self.post_router.bias.data.fill_(0)
        self.post_router_ln = nn.LayerNorm(in_d)
        self.post_router_ln.weight = nn.Parameter(torch.ones(self.in_d))

    def forward(self, routing_infos, input: torch.Tensor):
        padding_mask = routing_infos.pad_token_mask
        inst_padding_mask = routing_infos.inst_token_mask

        prior_input = self.apply_mask_and_average(input, inst_padding_mask)
        prior_routes = self.route(self.prior_router, self.prior_router_ln, prior_input)

        if self.training:
            # during training :-)
            post_input = self.apply_mask_and_average(input, padding_mask)
            post_routes = self.route(self.post_router, self.post_router_ln, post_input)
            routing_probs = F.softmax(post_routes, dim=-1)

            # compute auxiliary loss (KL divergence), KL = - H(posterior) + Xent(posterior, prior)
            auxiliary_loss = routing_probs * F.log_softmax(
                post_routes, -1
            ) - routing_probs * F.log_softmax(prior_routes, dim=-1)
            auxiliary_loss = auxiliary_loss.sum(dim=-1).mean()
        else:
            # during eval :-(
            routing_probs = F.softmax(prior_routes, dim=-1)
            auxiliary_loss = routing_probs.sum().detach() * 0.0
        return routing_probs.unsqueeze(1), auxiliary_loss


class AuxRoutingLoRALinear(SkilledLoRA, RoutingMixin):
    def __init__(self, config, task_id_ptr, layer, selector=None, **kwargs):
        RoutingMixin.__init__(self, task_id_ptr)
        SkilledLoRA.__init__(self, config, layer, **kwargs)

        if selector is None:
            self.selector = get_selector(config, in_d=self.in_features)
        else:
            self.selector = selector

        self._losses = []
        self._metrics = {}

    @property
    def losses(self):
        return self._losses

    @property
    def metrics(self):
        return self._metrics

    def clear(self):
        self._losses.clear()
        self._metrics.clear()

    def forward(self, input):
        # Need to clear losses and metrics before forward pass!
        self.clear()

        task_id = self.routing_infos.task_ids
        repeat = input.size(0) // task_id.size(0)

        # this repeat follows the patten in `model.predict()` line 152
        if repeat:
            self.routing_infos.repeat_interleave(repeat)

        if self.selector is not None:
            mixing_weights = self.selector(self.routing_infos, input=input)

            if isinstance(mixing_weights, tuple):
                mixing_weights, kl = mixing_weights
                self._losses.append(kl)
        else:
            bs = input.size(0)
            mixing_weights = torch.ones(
                bs, self.n_splits, self.n_skills, device=input.device, dtype=input.dtype
            )

        self._metrics["routing"] = mixing_weights.detach().cpu().float()
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


  
@register_selector("vsmear_wreg")          
class VSMEARRouterExperimental(VSMEARRouter):
    def __init__(self, config, in_d):
        super().__init__(config, in_d)

    def forward(self, routing_infos, input: torch.Tensor):
        padding_mask = routing_infos.pad_token_mask # 1 for everythin that is not a pad token, i.e. instuction, input, output
        inst_padding_mask = routing_infos.inst_token_mask # 1 for everything that is instruction

        prior_input = self.apply_mask_and_average(input, inst_padding_mask)
        prior_routes = self.route(self.prior_router, self.prior_router_ln, prior_input)

        if self.training:
            # during training :-)     
            post_input = self.apply_mask_and_average(input, padding_mask)
            post_routes = self.route(self.post_router, self.post_router_ln, post_input)
             
            routing_probs = F.softmax(prior_routes, dim=-1) # output and teacher
            auxiliary_loss = 1 - F.cosine_similarity(post_routes, prior_routes, dim=-1).mean()
            
        else:
            # during eval :-(
            routing_probs = F.softmax(prior_routes, dim=-1)
            auxiliary_loss = routing_probs.sum().detach() * 0.0
        return routing_probs.unsqueeze(1), auxiliary_loss


class AuxRoutingLoRALinear_wreg(SkilledLoRA_MergeLoraAfterOP, RoutingMixin):
    def __init__(self, config, task_id_ptr, layer, selector=None, **kwargs):
        RoutingMixin.__init__(self, task_id_ptr)
        SkilledLoRA_MergeLoraAfterOP.__init__(self, config, layer, **kwargs)

        self.selector = VSMEARRouterExperimental(config, self.in_features)

        self._losses = []
        self._metrics = {}

    @property
    def losses(self):
        return self._losses

    @property
    def metrics(self):
        return self._metrics

    def clear(self):
        self._losses.clear()
        self._metrics.clear()
        
    def forward(self, input):
        self.clear()    
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
        return super(SkilledLoRA_MergeLoraAfterOP, self).forward(input, mixing_weights)

  
@register_modifier("vsmear_wreg")            
def modify_with_vsmear_reg(transformer, config):
    config.router_selector = "vsmear_wreg"
    config.adapter_type = "lora"

    if config.adapter_type in ["lora"]:
        return modify_with_routing(     
            transformer, config, AuxRoutingLoRALinear_wreg, RouterWrapper
        )
    else:
        raise NotImplementedError(
            f"Adapter type {config.adapter_type} not implemented for vsmear modifier."
        )
