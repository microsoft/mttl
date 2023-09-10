from typing import Union
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
from mttl.global_vars import EPS

from mttl.models.adapters import SkilledLoRA
from mttl.models.modifiers.modify_model import register_modifier
from mttl.models.modifiers.routing import (
    RouterWrapper,
    RoutingMixin,
    RoutingSelector,
    register_selector,
    modify_with_routing,
)


class SkillWrapper(RouterWrapper):
    @classmethod
    def switch_selector_to_average(cls, object, selector_to_replace=None, **kwargs):
        super(SkillWrapper, cls).switch_selector_to_average(
            object, PolytroponSelector, **kwargs
        )

    @classmethod
    def resize_module_logits(cls, object, n_tasks):
        """Resizes the vector routing, in case of fine-tuning."""
        for name, selector in object.get_selectors().items():
            print("Resizing module_logits of selector", name, "with", n_tasks, "tasks.")
            selector.resize_module_logits(n_tasks)


@register_selector("poly")
class PolytroponSelector(RoutingSelector):
    def __init__(self, config, **kwargs):
        super().__init__(config)

        self.n_splits = config.n_splits
        self.n_skills = config.n_skills
        self.dropout = config.module_logits_dropout
        self.use_l2_norm = config.module_logits_l2_norm
        self.use_relaxed_bernoulli = config.module_logits_relaxed_bernoulli
        self.use_straight_through = config.module_logits_straight_through
        self.poly_average_correction = config.poly_average_correction
        self.poly_use_shared_skill = config.poly_use_shared_skill

        if self.use_relaxed_bernoulli and self.use_straight_through:
            raise ValueError("Cannot use both relaxed and straight through.")

        self.module_logits = nn.Parameter(
            torch.empty((config.n_tasks, config.n_splits * config.n_skills)).uniform_(
                -1e-3, 1e-3
            )
        )

    def resize_module_logits(self, n_tasks):
        self.module_logits.data = torch.empty(
            (n_tasks, self.n_splits * self.n_skills)
        ).uniform_(-1e-3, 1e-3)

    def forward(self, routing_infos, **kwargs):
        if torch.max(routing_infos.task_ids).item() >= self.module_logits.shape[0]:
            raise ValueError(
                "Poly selector encountered a larger number of tasks than provided at init. {} vs {}".format(
                    torch.max(routing_infos.task_ids).item(),
                    self.module_logits.shape[0],
                )
            )

        module_logits = self.module_logits[routing_infos.task_ids]
        module_logits = module_logits.view(-1, self.n_splits, self.n_skills)

        if self.use_l2_norm:
            module_weights = F.normalize(module_logits, p=2, dim=-1)
        else:
            if self.training and self.use_relaxed_bernoulli:
                module_logits = RelaxedBernoulli(
                    temperature=1.0, logits=module_logits
                ).rsample()
            elif self.use_straight_through:
                module_logits = torch.sigmoid(module_logits)
                module_logits_disc = torch.round(module_logits)
                # straight through estimator
                module_logits = (
                    module_logits + (module_logits_disc - module_logits).detach()
                )
            else:
                module_logits = torch.sigmoid(module_logits)

            if self.dropout > 0.0:
                module_logits = nn.Dropout(self.dropout)(module_logits)

            if self.poly_use_shared_skill:
                # last skill is always active whatever the task that has been selected
                module_logits = torch.cat(
                    (module_logits[:, :, :-1], module_logits[:, :, -1:] * 0.0 + 1.0),
                    dim=-1,
                )

            if self.poly_average_correction:
                module_weights = module_logits * (
                    np.sqrt(self.n_splits) / np.sqrt(self.n_skills)
                )
            else:
                module_weights = module_logits / (
                    module_logits.sum(dim=-1, keepdim=True) + EPS
                )
        return module_weights


class PolyLoRA(SkilledLoRA, RoutingMixin):
    def __init__(self, config, task_id_ptr, layer, selector):
        RoutingMixin.__init__(self, task_id_ptr)
        SkilledLoRA.__init__(self, config, layer)
        self.selector = selector

    def forward(self, input):
        task_id = self.routing_infos.task_ids
        repeat = input.size(0) // task_id.size(0)

        # this repeat follows the patten in `model.predict()` line 152
        if repeat:
            self.routing_infos.repeat_interleave(repeat)

        mixing_weights = self.selector(self.routing_infos).to(dtype=input.dtype)
        # add n_splits dimension
        if mixing_weights.ndim == 2:
            mixing_weights = mixing_weights.unsqueeze(1)
        return SkilledLoRA.forward(self, input, mixing_weights)


@register_modifier("poly")
def modify_with_poly_ia3(transformer, config):
    config.router_selector = config.router_selector or "poly"
    config.adapter_type = config.adapter_type or "lora"

    if config.adapter_type == "lora":
        return modify_with_routing(transformer, config, PolyLoRA, SkillWrapper)
    else:
        raise NotImplementedError(f"Poly modifier not implemented for adapter {config.adapter_type}.")


@register_modifier("skilled")
def modify_with_poly_ia3(transformer, config):
    # setting router_selector to private
    config.router_selector = "private"
    config.adapter_type = config.adapter_type or "lora"
    # setting n_skills to n_tasks in case of skilled modifier
    config.n_skills = config.n_tasks

    if config.adapter_type == "lora":
        return modify_with_routing(transformer, config, PolyLoRA, SkillWrapper)
    else:
        raise NotImplementedError(f"Skilled modifier not implemented for adapter {config.adapter_type}.")
