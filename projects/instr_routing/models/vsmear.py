import math
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
from projects.instr_routing.models.clm import AugmentedRoutingInfo

                     
class SkilledLoRA_MergeLoraAfterOP(SkilledLoRA):
    def __init__(
        self,
        config,
        layer,
    ):
        super().__init__(config, layer)
        self.merge_after_op = config.merge_after_op

    def forward_linear_(self, input, weights):
        if not self.merge_after_op:
            return super().forward_linear_(input, weights)
        
        if self.training:     
            self.training_steps += 1

        bs, _, _ = weights.size()         
        adapter_out = torch.einsum("bsd,qkdr->bsqkr", (input, self.lora_a)) # bs x n_splits x n_skills x rank")       
        adapter_out = torch.einsum("bsqkr,qkrd->bsqkd", (adapter_out, self.lora_b)) # bs x seq x n_splits x n_skills x D        
        adapter_out = torch.einsum("bsqkd,bqk->bsd", (adapter_out, weights)) # bs x seq x n_splits x D
        adapter_out *= self.scaling # (adapter_out[0,:,:]*weights[0].unsqueeze(0).unsqueeze(-1)).sum(-2)[0][0] like adapter_out[0][0]
        warmup = min(self.training_steps / 10_000, 1)
        if self.use_warmup:
            adapter_out = adapter_out * warmup

        return self.layer(input) + adapter_out

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

        self.prior_router = nn.Linear(in_d, config.n_skills * self.n_splits, bias=True)
        self.prior_router_ln = nn.LayerNorm(in_d)
        self.prior_router_ln.weight = nn.Parameter(torch.ones(in_d))

        self.router_center_momentum = self.config.router_center_momentum
        self.register_buffer("center", torch.zeros(1, self.n_skills))
        
        # normal innit
        self.prior_router.weight.data.normal_(mean=0.0, std=0.02)
        self.prior_router.bias.data.fill_(0)

        self.metrics = Metrics()

    @torch.no_grad()
    def apply_center_update(self, routes):
        if self.training:
            self.center = (
                self.center * (1 - self.router_center_momentum)
                + torch.mean(routes, dim=0, keepdim=True) * self.router_center_momentum
            )

    def route_maybe_center(
        self, input, router, router_ln, temperature=1.0, center=False
    ):
        """Computes routes for the teacher / posterior."""
        input = router_ln(input)
        if self.normalize_weights:
            weights = router.weight / torch.norm(
                router.weight, p=2, dim=-1, keepdim=True
            )
            routes = F.linear(input, weights, None)
        else:
            routes = F.linear(input, router.weight, router.bias)
        if center:
            self.apply_center_update(routes)
        # teacher centering and sharpening
        return (routes - self.center) / temperature

    def apply_mask_and_average(self, x, padding_mask):
        x_rout = x * padding_mask.unsqueeze(-1).to(x.device)
        lengths = padding_mask.ne(0).sum(dim=1, keepdim=True) + 1e-5
        x_rout = x_rout.sum(dim=1) / lengths
        return x_rout

    def _get_router_inputs(
        self, input: torch.Tensor, routing_infos: AugmentedRoutingInfo
    ):
        """When generating, the successive forward calls only receive the last token (bs, 1, d).

        Therefore, at the first forward call (context encoding), we need to cache the input encodings
        so that we can use it to compute the prior routing probabilities.
        """
        router_input = routing_infos.inputs_cache_for_generation.get(self)
        if router_input is None:
            router_input = self.apply_mask_and_average(
                input, routing_infos.inst_token_mask # only instruction is marked with 1
            )
            # if in generation mode, cache the input encoding for the next forward calls
            if routing_infos.generation_mode:
                routing_infos.inputs_cache_for_generation[self] = router_input
        return router_input

    def forward(self, routing_infos: AugmentedRoutingInfo, input: torch.Tensor):
        self.metrics.clear()

        prior_input = self._get_router_inputs(input, routing_infos)
        prior_routes = self.route_maybe_center(
            prior_input,
            self.prior_router,
            self.prior_router_ln,
            temperature=self.temperature,
            center=self.router_center_momentum > 0.0,
        )

        routing_probs = F.softmax(prior_routes, dim=-1)
        h_pri = -(routing_probs * F.log_softmax(prior_routes, -1)).sum(1).mean()
        self.routings = routing_probs.detach().cpu()
        self.metrics["h_pri"] = h_pri / math.log(self.n_skills)
        return routing_probs.unsqueeze(1)


@register_selector("vsmear")
class VSMEARRouter(SMEARRouter):
    def __init__(self, config, in_d):
        super().__init__(config, in_d)

        self.router_shared_weights = config.router_shared_weights

        if self.router_shared_weights:
            self.post_router = self.prior_router
            self.post_router_ln = self.prior_router_ln
        else:
            self.post_router = nn.Linear(
                in_d, config.n_skills * self.n_splits, bias=True
            )
            self.post_router_ln = nn.LayerNorm(in_d)
            self.post_router_ln.weight = nn.Parameter(torch.ones(in_d))
            
        
            self.post_router.weight.data.normal_(mean=0.0, std=0.02)
            self.post_router.bias.data.fill_(0)

        self.router_teacher_ent_factor = self.config.router_teacher_ent_factor
        self.router_teacher_temperature = self.config.router_teacher_temperature
        self.router_center_momentum = config.router_center_momentum

    def get_posterior_input(self, input, routing_infos):
        padding_mask = routing_infos.pad_token_mask
        post_input = self.apply_mask_and_average(input, padding_mask)
        return post_input

    def forward(self, routing_infos: AugmentedRoutingInfo, input: torch.Tensor):
        self.metrics.clear()

        prior_input = self._get_router_inputs(input, routing_infos)

        # do not center the student, center only the teacher now
        prior_routes = self.route_maybe_center(
            prior_input,
            self.prior_router,
            self.prior_router_ln,
            temperature=self.temperature,
            center=False,
        )

        if self.training:
            # during training :-)
            assert (
                routing_infos.generation_mode is False
            ), "We are not expecting to be in generation mode during training."

            post_input = self.get_posterior_input(input, routing_infos)
            post_routes = self.route_maybe_center(
                post_input,
                self.post_router,
                self.post_router_ln,
                temperature=self.router_teacher_temperature,
                center=self.router_center_momentum > 0.0,
            )
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
            self.auxiliary_loss = -self.router_teacher_ent_factor * h_post + x_ent
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


  
@register_selector("vsmear_xr4")   
class VSMEARRouterExperimental(VSMEARRouter):
    def __init__(self, config, in_d):
        super().__init__(config, in_d)
        self.prior_router_ln = nn.Identity()
        self.post_router_ln = nn.Identity()
        self.xrouter_x4_target = config.xrouter_x4_target
        self.xrouter_x4target_detach = config.xrouter_x4target_detach

    def forward(self, routing_infos, input: torch.Tensor):
        self.metrics.clear()     
        prior_input = self._get_router_inputs(input, routing_infos)

        # do not center the student, center only the teacher now
        prior_routes = self.route_maybe_center(
            prior_input,     
            self.prior_router,
            self.prior_router_ln,
            temperature=self.temperature,
            center=False,
        )        
        # padding_mask = routing_infos.pad_token_mask # 1 for everythin that is not a pad token, i.e. instuction, input, output
        # inst_padding_mask = routing_infos.inst_token_mask # 1 for everything that is instruction
        if self.training:     
            # during training :-)
            assert (
                routing_infos.generation_mode is False
            ), "We are not expecting to be in generation mode during training."

            post_input = self.get_posterior_input(input, routing_infos)
            post_routes = self.route_maybe_center(
                post_input,
                self.post_router,
                self.post_router_ln,
                temperature=self.router_teacher_temperature,
                center=self.router_center_momentum > 0.0,
            )          
            post_probs = F.softmax(post_routes, dim=-1)  
            prior_probs= F.softmax(prior_routes, dim=-1) # output and teacher
            
                
            if self.xrouter_x4_target=="posterior": 
                target = post_routes * self.router_teacher_temperature
                student_logit = prior_routes * self.temperature
                routing_probs = post_probs
            elif self.xrouter_x4_target=="prior":
                target = prior_routes * self.temperature
                student_logit = post_routes * self.router_teacher_temperature
                routing_probs = prior_probs
            
            
            self.auxiliary_loss = 1 - F.cosine_similarity(student_logit, 
                                                          target.detach() if self.xrouter_x4target_detach else target, dim=-1).mean()
            
            
            h_post = -(post_probs * F.log_softmax(post_routes, -1)).sum(1).mean()
            h_pri = -(prior_probs * F.log_softmax(prior_routes, -1)).sum(1).mean()
            x_ent = -(post_probs * F.log_softmax(prior_routes, -1)).sum(1).mean()
            self.routings = routing_probs.detach().cpu()
            self.metrics["h_post"] = h_post / math.log(self.n_skills)
            self.metrics["h_pri"] = h_pri / math.log(self.n_skills)
            self.metrics["x_ent"] = x_ent
            
        else:
            # during eval :-(
            prior_probs = routing_probs = F.softmax(prior_routes, dim=-1)
            h_pri = -(prior_probs * F.log_softmax(prior_routes, -1)).sum(1).mean()

            self.routings = prior_probs.detach().cpu()
            self.metrics["h_pri"] = h_pri / math.log(self.n_skills)
            self.auxiliary_loss = h_pri.sum() * 0.0
        return routing_probs.unsqueeze(1)

@register_selector("smear_oracle")   
class VSMEARRouterOracle(VSMEARRouter):
    def __init__(self, config, in_d):
        super().__init__(config, in_d)
        self.prior_router_ln = nn.Identity()
        self.post_router_ln = nn.Identity()

    def forward(self, routing_infos, input: torch.Tensor):
        self.metrics.clear()     
        prior_input = self._get_router_inputs(input, routing_infos)

        # do not center the student, center only the teacher now
        prior_routes = self.route_maybe_center(
            prior_input,     
            self.prior_router,
            self.prior_router_ln,
            temperature=self.temperature,
            center=False,
        )        
        # padding_mask = routing_infos.pad_token_mask # 1 for everythin that is not a pad token, i.e. instuction, input, output
        # inst_padding_mask = routing_infos.inst_token_mask # 1 for everything that is instruction
        # if self.training:     
            # during training and eval route with posterior
        # assert (
        #     routing_infos.generation_mode is False
        # ), "We are not expecting to be in generation mode during training."

        post_input = self.get_posterior_input(input, routing_infos)
        post_routes = self.route_maybe_center(
            post_input,
            self.post_router,
            self.post_router_ln,
            temperature=self.router_teacher_temperature,
            center=self.router_center_momentum > 0.0,
        )          
        routing_probs = post_probs = F.softmax(post_routes, dim=-1)  
        prior_probs=F.softmax(prior_routes, dim=-1) # route with posterior
        
        h_post = -(post_probs * F.log_softmax(post_routes, -1)).sum(1).mean()
        h_pri = -(prior_probs * F.log_softmax(prior_routes, -1)).sum(1).mean()
        x_ent = -(post_probs * F.log_softmax(prior_routes, -1)).sum(1).mean()
        self.routings = routing_probs.detach().cpu()
        self.metrics["h_post"] = h_post / math.log(self.n_skills)
        self.metrics["h_pri"] = h_pri / math.log(self.n_skills)
        self.metrics["x_ent"] = x_ent
        self.auxiliary_loss  = h_pri.sum() * 0.0
            
        # else:
        #     # during eval also route with posterior
        #     prior_probs = routing_probs = F.softmax(prior_routes, dim=-1)
        #     h_pri = -(prior_probs * F.log_softmax(prior_routes, -1)).sum(1).mean()

        #     self.routings = prior_probs.detach().cpu()
        #     self.metrics["h_pri"] = h_pri / math.log(self.n_skills)
        #     self.auxiliary_loss = h_pri.sum() * 0.0
        return routing_probs.unsqueeze(1)



class AuxRoutingLoRALinear_MergeAfterOP(SkilledLoRA_MergeLoraAfterOP, RoutingMixin):
    def __init__(self, config, task_id_ptr, layer, selector=None, **kwargs):
        RoutingMixin.__init__(self, task_id_ptr)
        SkilledLoRA_MergeLoraAfterOP.__init__(self, config, layer, **kwargs)

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
        return super(SkilledLoRA_MergeLoraAfterOP, self).forward(input, mixing_weights)

  
@register_modifier("vsmear_xr4")            
def modify_with_vsmear_reg(transformer, config):
    config.router_selector = "vsmear_xr4"
    config.adapter_type = "lora"

    if config.adapter_type in ["lora"]:
        return modify_with_routing(     
            transformer, config, AuxRoutingLoRALinear_MergeAfterOP, RouterWrapper
        )
    else:
        raise NotImplementedError(
            f"Adapter type {config.adapter_type} not implemented for vsmear modifier."
        )

  
@register_selector("vsmear_xr1")           
class VSMEARRouterExperimentalXR1(SMEARRouter):
    def __init__(self, config, in_d):
        super().__init__(config, in_d)

    def forward(self, routing_infos, input: torch.Tensor):
        return super().forward(routing_infos, input)

# same as smear, but uses merging after the ouyter product
@register_modifier("vsmear_xr1")    
def modify_with_vsmear_reg(transformer, config):
    config.router_selector = "vsmear_xr1"
    config.adapter_type = "lora"

    if config.adapter_type in ["lora"]:
        return modify_with_routing(     
            transformer, config, AuxRoutingLoRALinear_MergeAfterOP, RouterWrapper
        )
    else:
        raise NotImplementedError(
            f"Adapter type {config.adapter_type} not implemented for vsmear modifier."
        )
     
# same as smear, but uses merging after the ouyter product
@register_modifier("smear_oracle")    
def modify_with_vsmear_reg(transformer, config):
    config.router_selector = "smear_oracle"
    config.adapter_type = "lora"

    if config.adapter_type in ["lora"]:
        return modify_with_routing(     
            transformer, config, AuxRoutingLoRALinear_MergeAfterOP, RouterWrapper
        )
    else:
        raise NotImplementedError(
            f"Adapter type {config.adapter_type} not implemented for vsmear modifier."
        )
