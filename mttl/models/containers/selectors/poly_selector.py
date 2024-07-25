from dataclasses import dataclass
from typing import Dict, List, Union

import torch
from torch import nn

from mttl.logging import logger
from mttl.models.containers.selectors.base_selectors import (
    EPS,
    BatchExpertsSplitsAndWeightsSelectorOutput,
    ExpertsAndWeightsSelectorOutput,
    ExpertsSplitsAndWeightsSelectorOutput,
    Selector,
    SelectorConfig,
    SelectorOutput,
    forward_with_cache,
    register_multi_expert_selector,
)


@dataclass
class PolySelectorConfig(SelectorConfig):
    n_splits: int = 1
    task_names: List[str] = None


@register_multi_expert_selector("poly_router", PolySelectorConfig)
class PolySelector(Selector):
    """
    Implements routing at a per-layer or per-model level
    """

    avg_selector_warned: bool = False

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.n_tasks = len(self.config.task_names) if self.config.task_names else 0

        # We add an extra task for the default (average) expert if not found
        self.module_logits = nn.Parameter(
            torch.empty(self.n_tasks + 1, self.config.n_splits).uniform_(-1e-3, 1e-3)
        )

        if self.n_tasks == 0:
            logger.warning(
                "No task names found in the config. Using a single task for PolySelector."
            )

    def _convert_task_names_to_ids(self, task_names: List[str]) -> torch.LongTensor:
        """Converts task names to task ids (indices in the module_logits routing tensor)."""
        return torch.LongTensor(
            [
                (
                    self.config.task_names.index(t)
                    if t in self.config.task_names
                    else self.n_tasks
                )
                for t in task_names
            ],
        ).to(self.module_logits.device)

    def _get_weights(self, task_names: List[str] = None) -> torch.Tensor:
        """Gets the routing weights for the corresponding task names.

        If `task_names` is None, read task names from the routing infos structure.
        """
        # Poly used for finetuning a single task
        if self.n_tasks == 0:
            task_ids = [0]
        else:
            # if task names was specified, then we use that
            if task_names is not None:
                task_ids = self._convert_task_names_to_ids(task_names)
            else:
                routing_info: RoutingInfo = self.routing_infos

                if hasattr(routing_info, "task_ids_from_name"):
                    task_ids = routing_info.task_ids_from_name
                else:
                    task_ids = self._convert_task_names_to_ids(routing_info.task_names)
                    # cache the computation for future use
                    self.routing_infos.task_ids_from_name = task_ids

            if task_ids.max() < self.n_tasks:
                if PolySelector.avg_selector_warned:
                    logger.warning(
                        f"Task ids were found. Reverting to default task-based routing"
                    )

                PolySelector.avg_selector_warned = False
            else:
                if not PolySelector.avg_selector_warned:
                    not_found_tasks = set(
                        [
                            t
                            for t in self.routing_infos.task_names
                            if t not in self.config.task_names
                        ]
                    )
                    logger.warning(
                        f"Tasks {not_found_tasks} not in taining tasks. Defaulting to average selector."
                    )
                    PolySelector.avg_selector_warned = True

                assert not self.training, "Unknown tasks during training"

        module_logits = torch.sigmoid(self.module_logits[task_ids])
        module_logits = module_logits.view(
            module_logits.size(0), self.config.n_splits, self.n_experts
        )
        module_weights = module_logits / (module_logits.sum(dim=-1, keepdim=True) + EPS)
        return module_weights

    @forward_with_cache
    def forward(self, input, **kwargs) -> Union[
        BatchExpertsSplitsAndWeightsSelectorOutput,
        ExpertsSplitsAndWeightsSelectorOutput,
    ]:
        """Returns the experts and weights for the task names used in the current batch.

        If there is only one task, we return a ExpertsAndWeightsSelectorOutput. This is to show that the weights are shared across the batch,
        and therefore can allow speedups in the forward pass.
        """
        weights = self._get_weights()

        if self.n_tasks == 0:
            return ExpertsSplitsAndWeightsSelectorOutput(
                SelectorOutput.ALL_EXPERTS, weights.squeeze(0)
            )

        return BatchExpertsSplitsAndWeightsSelectorOutput(
            SelectorOutput.ALL_EXPERTS, weights
        )

    def get_merging_weights(self, **selector_kwargs) -> Dict:
        return self.get_routing_weights(**selector_kwargs)

    def get_routing_weights(self, task_name, **selector_kwargs) -> Dict:
        assert task_name in self.config.task_names, f"Task {task_name} not found."
        weights = self._get_weights(task_names=[task_name])
        return {k: v.detach().item() for k, v in zip(self.expert_names, weights[0][0])}

    def on_add_expert(self, expert_name: str, **kwargs):
        self.module_logits.data = torch.empty(
            self.n_tasks + 1, self.config.n_splits * (self.n_experts + 1)
        ).uniform_(-1e-3, 1e-3)

        # Last expert is exactly uniform
        self.module_logits.data[-1] = 0.0


@dataclass
class PolySelectorDirectConfig(PolySelectorConfig):
    pass


@register_multi_expert_selector("poly_router_dir", PolySelectorDirectConfig)
class PolySelectorDirect(PolySelector):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.module_logits_dict = nn.ParameterDict()
        self.training_config = kwargs["training_config"]
        self.init_gap = [-1e-3, 1e-3]

        self.device = kwargs["layer"].weight.device

    def _get_weights(self):
        weights = torch.cat(
            [self.module_logits_dict[k] for k in self.module_logits_dict.keys()]
        )
        return weights

    def get_merging_weights(self, **selector_kwargs) -> Dict:
        return self.get_routing_weights(**selector_kwargs)

    def get_routing_weights(self):
        return {k: v.detach().item() for k, v in self.module_logits_dict.items()}

    def on_add_expert(self, expert_name: str, **kwargs):
        """
        Assume:
        expert_task_name -- task name expert is pecialized at
        self.config.finetune_task_name -- name of the task the model is currently trained on

        If we encounter a module for the current task, we init it with one hot, otherwise with uniform.
        """
        main_m = 1

        expert_task_name = kwargs["expert_info"].expert_task_name
        if expert_name not in self.module_logits_dict:
            if self.training_config.finetune_task_name == expert_task_name:
                self.init_gap = [
                    0,
                    0,
                ]  # encountered module for current task, init one hot
                self.module_logits_dict[expert_name] = torch.nn.Parameter(
                    torch.ones(1).to(self.device)
                )
                self.init_logits_uniform()
                self.module_logits_dict[expert_name].data *= main_m

            else:
                self.module_logits_dict[expert_name] = torch.nn.Parameter(
                    torch.empty(1).uniform_(*self.init_gap).to(self.device)
                )

    def load_state_dict(self, state_dict, strict=True):
        self._initialized = True
        return super().load_state_dict(state_dict, strict=strict)

    def init_logits_uniform(self):
        if sum([p for p in self.module_logits_dict.values()]) == 0:
            for name, param in self.module_logits_dict.items():
                self.module_logits_dict[name].data = (
                    torch.empty(1).uniform_(-1e-3, 1e-3).to(self.device)
                )
        self._initialized = True

    @forward_with_cache
    def forward(self, *args, **kwargs):
        weights = self._get_weights()
        experts = list(self.module_logits_dict.keys())
        return ExpertsAndWeightsSelectorOutput(experts, weights)


@dataclass
class PolySelectorDirectConfigUniform(PolySelectorConfig):
    pass


@register_multi_expert_selector("uniform", PolySelectorDirectConfigUniform)
class PolyUniform(PolySelectorDirect):
    """
    Currently only used for uniform merging of experts.
    """

    def on_add_expert(self, expert_name: str, **kwargs):
        if expert_name not in self.module_logits_dict:
            self.module_logits_dict[expert_name] = torch.nn.Parameter(
                torch.ones(1).to(self.device)
            )
            for name in self.module_logits_dict.keys():
                self.module_logits_dict[name].data = torch.ones(1).to(self.device)
                self.module_logits_dict[name].data /= len(self.module_logits_dict)
