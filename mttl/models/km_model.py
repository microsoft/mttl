from dataclasses import dataclass

from mttl.logging import logger
from mttl.models.expert_model import BaseExpertModel, MoEModelConfig, MultiExpertMixin
from mttl.models.library.expert_library import ExpertLibrary


@dataclass
class KMMoEModelConfig(MoEModelConfig):
    # TODO: figure out how to control from the command line
    ke_expert_name: str = "KE"
    ke_experts_prefix: str = "KE"
    dummy: int = 1


@BaseExpertModel.register("moe_km", config_cls=KMMoEModelConfig)
class KMMoEModel(BaseExpertModel, MultiExpertMixin):
    """MoeModel that can accomodate a Knowledge Extractor"""

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        self.modifier_config = config.modifier_config
        expert_library = ExpertLibrary.get_expert_library(self.config.library_id)

        # Now, we may want to try and test multiple knowledge extractors on the same library.
        # To do so, we need to be able to not load previously trained ones
        expert_names = expert_library.keys()
        ke_experts = list(
            filter(lambda x: x.startswith(self.config.ke_experts_prefix), expert_names)
        )
        for expert in sorted(list(expert_library.keys())):
            if expert not in ke_experts:
                self.add_expert_instance(expert_library[expert], expert_name=expert)

        assert len(self.experts_names) > 0, "No experts found in the library."

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