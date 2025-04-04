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
class KEMoEModelConfig(MoEModelConfig):
    # expert name
    ke_expert_name: str = "KE"
    # expert path
    ke_expert_path: str = None
    library_id: str = None
    expert_selection: List[str] = None
    # if selector_config is not None, then we use it to select experts
    selector_config: AutoSelectorConfig = None
    # if modifier_config is not None, then we create moe_num_experts with this modifier
    modifier_config: AutoModifierConfig = None
    # if cpu_offload is True, then we offload the computation to the CPU
    eval_cpu_offload: bool = False


@BaseExpertModel.register("moe_ke", config_cls=KEMoEModelConfig)
class KEMoEModel(BaseExpertModel, MultiExpertMixin):
    """MoeModel that can accomodate a Knowledge Extractor"""

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


@dataclass
class EMAExpertModelConfig(ExpertModelConfig, MoEModelConfig):
    ema_coef: float = 0.999
    modifier_config: AutoModifierConfig = None
    default_expert: str = "KM"
    downscale_factor: int = 16


@BaseExpertModel.register("ema_km", config_cls=EMAExpertModelConfig)
class EMAExpertModel(BaseExpertModel, MultiExpertMixin):
    """MoeModel that can accomodate a Knowledge Extractor"""

    def __init__(self, config, **kwargs):

        config.selector_config = DefaultExpertSelectorConfig()
        super().__init__(config, **kwargs)

        ema_modif_config = deepcopy(config.modifier_config)
        ema_modif_config.lora_alpha /= self.config.downscale_factor

        self.add_empty_expert("EMA", expert_config=ema_modif_config)
        # Make sure existing experts are not trainable
        for param in self.parameters():
            param.requires_grad = False

        self.add_empty_expert("KM", expert_config=config.modifier_config)

        # Set EMA weights to match the KM weights
        self.ema_update(ema_coef=0.0)

    @torch.no_grad()
    def ema_update(self, ema_coef=None):
        if ema_coef is None:
            ema_coef = self.config.ema_coef

        expert = self.get_expert_instance("KM")
        ema_expert = self.get_expert_instance(f"EMA")

        state_dict_keys = expert.expert_weights.keys()
        for key in state_dict_keys:
            p, ema_p = expert.expert_weights[key], ema_expert.expert_weights[key]
            ema_p.data.mul_(ema_coef).add_(p.data, alpha=1 - ema_coef)


@dataclass
class KMMoEModelConfig(MultiExpertModelConfig):
    cpu_offload: bool = False


# For training models in a multitask setup
@BaseExpertModel.register("moe_km", config_cls=KMMoEModelConfig)
class KMMoEModel(MultiExpertModel):
    def forward(self, *args, **kwargs):
        if not self.training and self.config.cpu_offload:
            with cpu_offload(self, kwargs["task_names"], enable=True):
                return super().forward(*args, **kwargs)
        else:
            return super().forward(*args, **kwargs)
