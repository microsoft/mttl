from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
from mttl.global_vars import EPS
from mttl.utils import logger

from mttl.models.modifiers.base import ModifierConfig, ModifyMixin
from mttl.models.modifiers.lora import SkilledLoRA, SkilledLoRAConfig
from mttl.models.modifiers.modify_model import register_modifier
from mttl.models.modifiers.routing import (
    modify_with_routing,
    RouterWrapper,
    RoutingMixin,
    RoutingSelector,
    register_selector,
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


@dataclass
class PolytroponConfig(ModifierConfig):
    n_tasks: int = None
    n_skills: int = 1
    n_splits: int = 1
    router_selector: str = "poly"
    module_logits_dropout: float = 0.0
    module_logits_l2_norm: bool = False
    module_logits_relaxed_bernoulli: bool = False
    module_logits_straight_through: bool = False
    poly_average_correction: bool = False
    poly_use_shared_skill: bool = False
    router_granularity: str = "finegrained"
    model_family: str = "gpt"


@register_selector("poly")
class PolytroponSelector(RoutingSelector):
    seen_samples_per_task = None

    def __init__(self, config, **kwargs):
        super().__init__(config)

        self.n_tasks = config.n_tasks
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

        if PolytroponSelector.seen_samples_per_task is None:
            PolytroponSelector.seen_samples_per_task = torch.zeros(
                config.n_tasks, dtype=torch.long, device="cpu"
            )

        self.use_avg_last = False

    def resize_module_logits(self, n_tasks):
        self.module_logits.data = torch.empty(
            (n_tasks, self.n_splits * self.n_skills)
        ).uniform_(-1e-3, 1e-3)

    def forward(self, routing_infos, **kwargs):
        task_ids = routing_infos.task_ids

        if task_ids is None:
            if not self.use_avg_last:
                logger.warning("task_ids is None, using AverageSelector instead")

            self.use_avg_last = True
            bs = routing_infos.input_ids.size(0)
            module_weights = torch.ones(
                (bs, self.n_splits, self.n_skills),
                device=routing_infos.input_ids.device,
            )
            module_weights = (
                module_weights / module_weights.sum(dim=-1, keepdim=True) + EPS
            )
            return module_weights

        self.use_avg_last = False
        if self.training and not hasattr(routing_infos, "logged_task_ids"):
            PolytroponSelector.seen_samples_per_task += torch.bincount(
                task_ids, minlength=self.n_tasks
            ).cpu()
            routing_infos.logged_task_ids = True

        if torch.max(task_ids).item() >= self.module_logits.shape[0]:
            raise ValueError(
                "Poly selector encountered a larger number of tasks than provided at init. {} vs {}".format(
                    torch.max(task_ids).item(),
                    self.module_logits.shape[0],
                )
            )

        module_logits = self.module_logits[task_ids]
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


@dataclass
class PerTokenPolytroponConfig(PolytroponConfig):
    skip_unseen_tokens: bool = True  # during evaluation, if token has not been seen (and no mapping has been learned yet) skip it


@register_selector("per_token_poly")
class PerTokenPolytroponSelector(RoutingSelector):
    seen_samples_per_token = None

    def __init__(self, config, **kwargs):
        super().__init__(config)

        assert config.model_family == "gpt", "only decoder models supported for now."

        self.n_splits = config.n_splits
        self.n_skills = config.n_skills
        self.vocab_size = config.vocab_size
        self.dropout = config.module_logits_dropout
        self.skip_unseen_tokens = config.skip_unseen_tokens

        self.module_logits = nn.Parameter(
            torch.empty(
                (config.vocab_size, config.n_splits * config.n_skills)
            ).uniform_(-1e-3, 1e-3)
        )

        if PerTokenPolytroponSelector.seen_samples_per_token is None:
            PerTokenPolytroponSelector.seen_samples_per_token = torch.zeros(
                config.vocab_size, dtype=torch.long, device="cpu"
            )

    def resize_module_logits(self, n_tasks):
        logger.warning(
            "Resizing module logits is a no-op for PerTokenPolytroponSelector"
        )

    def forward(self, routing_infos, **kwargs):
        # Note : encoder - decoder models not currently supported.
        input_ids = routing_infos.input_ids

        if self.training and not hasattr(routing_infos, "logged_task_ids"):
            PerTokenPolytroponSelector.seen_samples_per_token += torch.bincount(
                input_ids.view(-1), minlength=self.vocab_size
            ).cpu()
            routing_infos.logged_task_ids = True
        if input_ids.max().item() >= self.module_logits.shape[0]:
            raise ValueError(
                "Poly selector encountered a larger number of tasks than provided at init. {} vs {}".format(
                    input_ids.max().item(),
                    self.module_logits.shape[0],
                )
            )

        module_logits = self.module_logits[input_ids]
        module_logits = module_logits.view(
            *input_ids.shape, self.n_splits, self.n_skills
        )
        module_logits = torch.sigmoid(module_logits)
        module_weights = module_logits / (module_logits.sum(dim=-1, keepdim=True) + EPS)

        if not self.training and self.config.skip_unseen_tokens:
            is_seen = (
                PerTokenPolytroponSelector.seen_samples_per_token[input_ids.cpu()] > 0
            )
            module_weights[~is_seen] = 0.0

        return module_weights


@dataclass
class PolyLoRAConfig(SkilledLoRAConfig, PolytroponConfig):
    pass


@register_modifier("poly", config_cls=PolyLoRAConfig)
class PolyLoRA(SkilledLoRA, RoutingMixin):
    def __init__(self, config, task_id_ptr, layer, selector):
        SkilledLoRA.__init__(self, config, layer)
        RoutingMixin.__init__(self, task_id_ptr)
        self.selector = selector

    def forward(self, input):
        task_id = self.routing_infos.task_ids

        if task_id is not None:
            repeat = input.size(0) // task_id.size(0)

            # this repeat follows the patten in `model.predict()` line 152
            if repeat:
                self.routing_infos.repeat_interleave(repeat)

        mixing_weights = self.selector(self.routing_infos).to(dtype=input.dtype)
        # add n_splits dimension
        if mixing_weights.ndim == 2:
            mixing_weights = mixing_weights.unsqueeze(1)
        return SkilledLoRA.forward(self, input, mixing_weights)

    @classmethod
    def modify_transformer(cls, transformer, config, optional_wrapper=None):
        if config.router_selector is None:
            config.router_selector = "poly"

        return modify_with_routing(cls, transformer, config, SkillWrapper)


@dataclass
class PerTokenPolyLoRAConfig(SkilledLoRAConfig, PerTokenPolytroponConfig):
    pass


@register_modifier("per_token_poly", config_cls=PerTokenPolyLoRAConfig)
class PerTokenPolyLoRA(PolyLoRA):
    def forward(self, input):
        input_ids = self.routing_infos.input_ids
        repeat = input.size(0) / input_ids.size(0)

        # this repeat follows the patten in `model.predict()` line 152
        if repeat != 1.0:
            self.routing_infos.input_ids = (
                self.routing_infos.input_ids.repeat_interleave(repeat, dim=0)
            )

        # Phi-2 generation specific
        if input.size(1) != input_ids.size(1):
            assert input.size(1) == 1
            self.routing_infos.input_ids = self.routing_infos.input_ids[:, [-1]]

        mixing_weights = self.selector(self.routing_infos).to(dtype=input.dtype)
        # add n_splits dimension
        if mixing_weights.ndim == 3:
            mixing_weights = mixing_weights.unsqueeze(1)

        return SkilledLoRA.forward(self, input, mixing_weights)

    @classmethod
    def modify_transformer(cls, transformer, config, optional_wrapper=None):
        config.router_selector = "per_token_poly"
        if not hasattr(config, "vocab_size"):
            config.vocab_size = transformer.get_input_embeddings().num_embeddings

        return modify_with_routing(cls, transformer, config, SkillWrapper)


@dataclass
class SkilledPolyConfig(SkilledLoRAConfig, PolytroponConfig):
    router_selector: str = "private"


@register_modifier("skilled", config_cls=SkilledPolyConfig)
class SkilledPolyLoRA(PolyLoRA):
    pass
