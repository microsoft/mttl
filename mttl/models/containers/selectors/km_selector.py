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
    field_name: str = "task_names"
    ke_expert_name: str = "KE"
    # argument that allows heterogenous batches, with some having a KM and some not
    allow_missing_kms: bool = False


@Selector.register("ke_selector", KnowledgeExtractorSelectorConfig)
class KnowledgeExtractorSelector(Selector):
    MODE = None
    """
    Selector which supports both 
        1) activation of the desired KM + the shared KE
        2) activation of only the shared KE
    """

    @forward_with_cache
    def forward(self, input, **kwargs) -> BatchExpertsSelectorOutput:

        assert self.routing_infos and hasattr(
            self.routing_infos, self.config.field_name
        )
        task_names = getattr(self.routing_infos, self.config.field_name)
        ke_expert_name = [
            task for task in self.expert_names if self.config.ke_expert_name == task
        ]

        if len(ke_expert_name) != 1:
            raise ValueError(
                f"Expected exactly one KE expert with name {self.config.ke_expert_name}, found {ke_expert_name}"
            )

        ke_expert_name = ke_expert_name[0]

        if all(task in self.expert_names for task in task_names):
            mode = "KM"
            expert_names = [[ke_expert_name, task_name] for task_name in task_names]
            warn_once(f"Knowledge Extractor Mode : {mode}")
        elif any(task in self.expert_names for task in task_names):
            if not self.config.allow_missing_kms:
                raise ValueError(
                    "Some, *but not all*, tasks are not in the expert names!"
                )
            mode = "mixed"
            # To keep a consistent (batchable structure), we will add twice the KE expert
            # whenever no KM exists.
            expert_names, missing_km_idx = [], []
            for i, task_name in enumerate(task_names):
                if task_name in self.expert_names:
                    expert_names.append([ke_expert_name, task_name])
                else:
                    expert_names.append([ke_expert_name, ke_expert_name])
                    missing_km_idx.append(i)
        else:
            mode = "KE"  # no knowledge modules
            expert_names = [[ke_expert_name]] * len(task_names)
            warn_once(f"Knowledge Extractor Mode : {mode}")

        if KnowledgeExtractorSelector.MODE is None:
            KnowledgeExtractorSelector.MODE = mode
        elif KnowledgeExtractorSelector.MODE != mode:
            msg = f"Different modes for KE selector: {KnowledgeExtractorSelector.MODE} != {mode}"
            if not self.config.allow_missing_kms:
                raise ValueError(msg)
            warn_once(msg)

        # build weight vector
        weights = input.new(size=(len(expert_names), len(expert_names[0]))).fill_(1.0)
        if mode == "mixed":
            weights[missing_km_idx] = 0.5
        return BatchExpertsAndWeightsSelectorOutput(expert_names, weights)
