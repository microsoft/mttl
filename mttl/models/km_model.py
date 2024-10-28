from dataclasses import dataclass
from typing import List

from mttl.logging import logger
from mttl.models.containers.selectors.base import AutoSelectorConfig
from mttl.models.containers.selectors.km_selector import (
    KnowledgeExtractorSelectorConfig,
)
from mttl.models.expert_model import BaseExpertModel, MoEModelConfig, MultiExpertMixin
from mttl.models.library.expert_library import ExpertLibrary
from mttl.models.modifiers.base import AutoModifierConfig


@dataclass
class KMMoEModelConfig(MoEModelConfig):
    ke_expert_name: str = "KE"
    library_id: str = None
    expert_selection: List[str] = None
    # if selector_config is not None, then we use it to select experts
    selector_config: AutoSelectorConfig = None
    # if modifier_config is not None, then we create moe_num_experts with this modifier
    modifier_config: AutoModifierConfig = None


@BaseExpertModel.register("moe_km", config_cls=KMMoEModelConfig)
class KMMoEModel(BaseExpertModel, MultiExpertMixin):
    """MoeModel that can accomodate a Knowledge Extractor"""

    def __init__(self, config, **kwargs):

        # If no selectors have been provided, we default to the KnowledgeExtractorSelector
        if config.selector_config is None:
            logger.info(
                "No selector_config provided, defaulting to KnowledgeExtractorSelector"
            )
            config.selector_config = KnowledgeExtractorSelectorConfig(
                ke_expert_name=config.ke_expert_name, router_granularity="coarsegrained"
            )
        elif not isinstance(config.selector_config, KnowledgeExtractorSelectorConfig):
            raise ValueError(
                f"Expected `selector_config` to be of type `KnowledgeExtractorSelectorConfig`, found {type(config.selector_config)}"
            )

        super().__init__(config, **kwargs)

        # Now, we may want to try and test multiple knowledge extractors on the same library.
        # To do so, we need to be able to not load previously trained ones
        expert_library = ExpertLibrary.get_expert_library(
            repo_id=config.library_id, selection=config.expert_selection
        )
        self.add_experts_from_library(expert_library)

        # make sure existing expert are not trainable
        for param in self.parameters():
            param.requires_grad = False

        # also need to add an additional expert for the KE
        # we will use the `ExpertConfig` of the first expert
        an_expert = self.get_expert_instance(self.experts_names[0])

        self.ke_expert_name = self.config.ke_expert_name
        self.add_empty_expert(
            self.ke_expert_name, expert_config=an_expert.expert_config
        )
        logger.info("Added KE expert: %s", self.ke_expert_name)
