from dataclasses import dataclass, field

import torch

from mttl.logging import warn_once
from mttl.models.containers.selectors.base import (
    Selector,
    SelectorConfig,
    forward_with_cache,
)
from mttl.models.containers.selectors.selector_output import (
    BatchExpertsAndWeightsSelectorOutput,
    BatchExpertsSelectorOutput,
)


@dataclass
class KnowledgeExtractorSelectorConfig(SelectorConfig):
    # name of the KE expert
    ke_expert_name: str = "KE"
    # Optionally support the case when a KM is missing
    allow_missing_kms: bool = False
    # only need a single selector
    router_granularity: str = "coarsegrained"
    # merge after
    lora_merge_after: bool = True
    # learn w
    learn_w: bool = False


@Selector.register("ke_selector", KnowledgeExtractorSelectorConfig)
class KnowledgeExtractorSelector(Selector):
    """Offloads experts to CPUs given that it is likely to go OOM."""

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        if self.config.learn_w:
            self.register_parameter(
                "KEe_w",
                torch.nn.Parameter(
                    torch.tensor([0.5], device=self.device), requires_grad=True
                ),
            )
            self.register_parameter(
                "KEm_w",
                torch.nn.Parameter(
                    torch.tensor([0.5], device=self.device), requires_grad=True
                ),
            )

    @forward_with_cache
    def forward(self, input, **kwargs) -> BatchExpertsSelectorOutput:
        task_names = self.routing_infos.task_names

        # Make sure there's only one KE expert
        assert (
            sum([self.config.ke_expert_name == e_name for e_name in self.expert_names])
            == 1
        )

        # Make sure we are not missing any KMs, except if we allow it
        missing_km_idx = [
            i for i in range(len(task_names)) if task_names[i] not in self.expert_names
        ]
        assert len(missing_km_idx) == 0 or self.config.allow_missing_kms, breakpoint()

        # given a `batch_size` list of task names, we build a `batch_size` list of [ke_expert_name, expert_i_name]
        if self.config.learn_w:
            weights = torch.cat(
                [
                    self.KEe_w.unsqueeze(0).repeat(len(task_names), 1),
                    self.KEm_w.unsqueeze(0).repeat(len(task_names), 1),
                ],
                dim=1,
            )
        else:
            weights = input.new_ones(size=(len(task_names), 2)).fill_(1.0)
        expert_names = [
            (
                [self.config.ke_expert_name, task_names[i]]
                if i not in missing_km_idx
                else [self.config.ke_expert_name, self.config.ke_expert_name]
            )
            for i in range(len(task_names))
        ]
        return BatchExpertsAndWeightsSelectorOutput(expert_names, weights)
