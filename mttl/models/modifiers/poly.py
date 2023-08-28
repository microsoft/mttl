import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
from mttl.global_vars import EPS

from mttl.models.adapters import SkilledLoRA
from mttl.models.modifiers.modify_model import register_modifier
from mttl.models.modifiers.routing import (
    RoutingAdapter,
    RouterWrapper,
    RoutingSelector,
    get_selector,
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

    @classmethod
    def remove_skills(cls, object, skill_ids_to_keep):
        print("Removing skills, keeping", skill_ids_to_keep)
        for name, adapter in object.get_adapters().items():
            if isinstance(adapter, PolytroponAdapter):
                if adapter.lora_a.shape[1] > 1:
                    adapter.lora_a = nn.Parameter(
                        adapter.lora_a[:, skill_ids_to_keep, :, :]
                    )
                if adapter.lora_b.shape[1] > 1:
                    adapter.lora_b = nn.Parameter(
                        adapter.lora_b[:, skill_ids_to_keep, :, :]
                    )
                adapter.n_skills = len(skill_ids_to_keep)
                adapter.selector.n_skills = len(skill_ids_to_keep)


class PolytroponAdapter(RoutingAdapter):
    pass


@register_selector("poly")
class PolytroponSelector(RoutingSelector):
    def __init__(self, config, **kwargs):
        super().__init__()

        self.config = config
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


class PolyLoRALinear(PolytroponAdapter):
    def __init__(self, config, task_id_ptr, linear_layer, selector=None):
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
            self.selector = get_selector(config)
        else:
            self.selector = selector

        self.lora_a = nn.Parameter(
            torch.empty(
                self.n_splits,
                self.n_skills,
                linear_layer.in_features // self.n_splits,
                self.rank,
                device=self.weight.device,
            )
        )
        self.lora_b = nn.Parameter(
            torch.empty(
                self.n_splits,
                self.n_skills,
                self.rank,
                linear_layer.out_features // self.n_splits,
                device=self.weight.device,
            )
        )
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

        mixing_weights = self.selector(self.routing_infos).to(dtype=input.dtype)

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


class PolyIA3Linear(PolytroponAdapter):
    def __init__(self, config, task_id_ptr, linear_layer, selector=None):
        super().__init__()

        self.n_splits = config.n_splits
        self.n_tasks = config.n_tasks
        self.n_skills = config.n_skills
        self.linear_layer = linear_layer
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.weight = linear_layer.weight
        self.bias = linear_layer.bias
        self.task_id_ptr = task_id_ptr

        assert self.out_features % config.n_splits == 0

        data = torch.ones(
            self.n_skills, self.n_splits, self.out_features // self.n_splits
        )
        self.lora_a = nn.Parameter(
            data,
            dtype=self.weight.dtype
            if self.weight.dtype != torch.int8
            else torch.float32,
        )

        if selector is None:
            self.selector = get_selector(config)
        else:
            self.selector = selector

    def forward(self, input):
        task_id = self.routing_infos.task_ids

        repeat = input.size(0) // task_id.size(0)

        # this repeat follows the patten in `model.predict()` line 152
        if repeat:
            self.routing_infos.repeat_interleave(repeat)

        # bs, n_splits, n_skills
        mixing_weights = self.selector(self.routing_infos)

        # n_skills, n_splits, D // n_splits
        weight = self.lora_a

        A = torch.einsum("bqs,sqd->bqd", (mixing_weights, weight))
        A = A.reshape(input.size(0), 1, -1)
        return self.linear_layer(input) * A

    def extra_repr(self):
        return "n_skills={}, in_features={}, out_features={}, bias={}".format(
            self.n_skills, self.in_features, self.out_features, self.bias is not None
        )


@register_modifier("poly")
def modify_with_poly_ia3(transformer, config):
    config.router_selector = config.router_selector or "poly"

    if config.adapter_type == "ia3":
        return modify_with_routing(transformer, config, PolyIA3Linear, SkillWrapper)
    else:
        return modify_with_routing(transformer, config, PolyLoRALinear, SkillWrapper)
