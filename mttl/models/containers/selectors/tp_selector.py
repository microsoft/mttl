from dataclasses import dataclass
from typing import Dict

import torch

from mttl.logging import logger
from mttl.models.containers.selectors.base import (
    BatchExpertsAndWeightsSelectorOutput,
    Selector,
    SelectorConfig,
    forward_with_cache,
)
from mttl.models.library.expert import ExpertInfo
from mttl.models.ranker.adapter_ranker import AdapterRankerHelper
from mttl.models.ranker.classifier_ranker import ClusterPredictor


@dataclass
class TaskPredictorSelectorConfig(SelectorConfig):
    ranker_path: str = None
    ranker_model: str = None
    ranker_top_k: int = 1
    library_id: str = None


@Selector.register("task_predictor_selector", TaskPredictorSelectorConfig)
class TaskPredictorSelector(Selector):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.ranker_top_k = self.config.ranker_top_k

        # get the routing model
        self.expert_ranker = AdapterRankerHelper.get_ranker_instance(
            ranker_model=self.config.ranker_model,
            ranker_path=self.config.ranker_path,
        )

        if isinstance(self.expert_ranker, ClusterPredictor):
            self.expert_ranker.init_clusters(self.config.library_id)

    @forward_with_cache
    def forward(self, input, **kwargs) -> BatchExpertsAndWeightsSelectorOutput:
        # get the sources_texts from routing_infos
        routing_infos = self.routing_infos

        if hasattr(routing_infos, "sources_texts"):
            sources_texts = routing_infos.sources_texts
            self.expert_ranker.set_available_tasks(self.expert_names)

            experts, weights = self.expert_ranker.predict_task(
                sources_texts, n=self.ranker_top_k
            )
            logger.debug(f"Predicted tasks: {experts} with weights {weights}")

            weights = torch.tensor(weights, device=input.device)
            return BatchExpertsAndWeightsSelectorOutput(experts, weights)
        else:
            raise ValueError(
                "Routing infos does not contain sources_texts, cannot predict tasks."
            )

    def get_merging_weights(self, **selector_kwargs) -> Dict:
        raise ValueError(
            f"Not supported for {self.__class__} since routing depends on input."
        )

    def on_add_expert(
        self, expert_name: str, expert_info: ExpertInfo = None, is_default=False
    ):
        pass
