from dataclasses import dataclass, field

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
    # name of the field matching the expert names
    field_name: str = "task_names"
    # name of the KE expert
    ke_expert_name: str = "KE"
    # Optionally support the case when a KM is missing
    allow_missing_kms: bool = False
    # only need a single selecter
    router_granularity = "coarsegrained"


@Selector.register("ke_selector", KnowledgeExtractorSelectorConfig)
class KnowledgeExtractorSelector(Selector):
    @forward_with_cache
    def forward(self, input, **kwargs) -> BatchExpertsSelectorOutput:

        assert self.routing_infos and hasattr(
            self.routing_infos, self.config.field_name
        )
        task_names = getattr(self.routing_infos, self.config.field_name)

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
        weights = input.new_ones(size=(len(task_names), 2)).fill_(0.5)
        expert_names = [
            (
                [self.config.ke_expert_name, task_names[i]]
                if i not in missing_km_idx
                else [self.config.ke_expert_name, self.config.ke_expert_name]
            )
            for i in range(len(task_names))
        ]

        return BatchExpertsAndWeightsSelectorOutput(expert_names, weights)
