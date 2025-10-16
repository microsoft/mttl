from copy import deepcopy
from dataclasses import dataclass
from typing import List, Union

import torch

from mttl.logging import logger
from mttl.models.containers.lora_containers import LoRAExpertContainer
from mttl.models.containers.selectors.base import (
    AutoSelectorConfig,
    DefaultExpertSelectorConfig,
)
from mttl.models.containers.selectors.poly_selector import PolySelectorConfig
from mttl.models.expert_context import InfoContainer
from mttl.models.expert_model import (
    BaseExpertModel,
    BaseExpertModelConfig,
    ExpertModel,
    ExpertModelConfig,
    MoEModelConfig,
    MultiExpertMixin,
    MultiExpertModel,
    MultiExpertModelConfig,
)
from mttl.models.library.expert_library import ExpertLibrary
from mttl.models.modifiers.base import AutoModifierConfig
from projects.kms.utils.km_selector import KnowledgeExtractorSelectorConfig
from projects.kms.utils.simple_utils import cpu_offload


@dataclass
class KMMoEModelConfig(BaseExpertModelConfig):
    library_id: str = None
    expert_selection: List[str] = None
    # if selector_config is not None, then we use it to select experts
    selector_config: AutoSelectorConfig = None
    # if modifier_config is not None, then we create moe_num_experts with this modifier
    modifier_config: AutoModifierConfig = None
    # if cpu_offload is True, then we offload the computation to the CPU
    eval_cpu_offload: bool = False


@BaseExpertModel.register("moe_km", config_cls=KMMoEModelConfig)
class KMMoEModel(BaseExpertModel, MultiExpertMixin):

    @InfoContainer.create_context
    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        **kwargs,
    ):
        should_cpu_offload = not self.training and self.config.eval_cpu_offload
        with cpu_offload(
            self,
            InfoContainer.get().routing_infos.task_names,
            enable=should_cpu_offload,
        ):
            outputs = self.model.forward(
                input_ids, attention_mask=attention_mask, labels=labels, **kwargs
            )
        return outputs

    @InfoContainer.create_context
    def generate(
        self,
        input_ids,
        attention_mask=None,
        **kwargs,
    ):
        should_cpu_offload = not self.training and self.config.eval_cpu_offload
        with cpu_offload(
            self,
            InfoContainer.get().routing_infos.task_names,
            enable=should_cpu_offload,
        ):
            generations = self.model.generate(
                inputs=input_ids,
                attention_mask=attention_mask,
                **kwargs,
            )
        return generations


@dataclass
class KEMoEModelConfig(KMMoEModelConfig):
    # expert name
    ke_expert_name: str = "KE"
    # expert path
    ke_expert_path: str = None
    # if cpu_offload is True, then we offload the computation to the CPU
    eval_cpu_offload: bool = False


# Copying the setup of MoEModel
@BaseExpertModel.register("moe_ke", config_cls=KEMoEModelConfig)
class KEMoEModel(KMMoEModel):
    """MoeModel that can accomodate a Knowledge Extractor"""

    def __init__(self, config, **kwargs):
        # If no selectors have been provided, we default to the KnowledgeExtractorSelector
        if config.selector_config is None:
            logger.info(
                "No selector_config provided, defaulting to KnowledgeExtractorSelector"
            )
            config.selector_config = KnowledgeExtractorSelectorConfig(
                ke_expert_name=config.ke_expert_name,
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

        self.ke_expert_name = self.config.ke_expert_name

        if self.config.ke_expert_path:
            # pretrained expert path!
            ke_model = ExpertModel.from_pretrained(
                self.config.ke_expert_path, device_map="cpu"
            )
            ke_expert = ke_model.as_expert()
            self.add_expert_instance(ke_expert, self.config.ke_expert_name)
        else:
            # also need to add an additional expert for the KE
            # we will use the `ExpertConfig` of the first expert
            an_expert = self.get_expert_instance(self.experts_names[0])
            self.add_empty_expert(
                self.ke_expert_name, expert_config=an_expert.expert_config
            )
