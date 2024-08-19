from dataclasses import dataclass

from mttl.models.expert_configuration import BaseExpertModelConfig
from mttl.models.expert_modeling_base import BaseExpertModel
from mttl.models.library.expert import Expert, ExpertInfo
from mttl.models.modifiers.base import AutoModifierConfig, ModifierConfig
from mttl.models.modifiers.modify_model import modify_transformer
from mttl.models.utils import model_loader_helper


@dataclass
class SingleExpertModelConfig(BaseExpertModelConfig):
    task_name: str = None
    expert_name: str = None
    modifier_config: AutoModifierConfig = None


@BaseExpertModel.register("single_expert_model", config_cls=SingleExpertModelConfig)
class SingleExpertModel(BaseExpertModel):
    def __init__(
        self,
        config: SingleExpertModelConfig,
        **loading_kwargs,
    ):
        super().__init__(config, **loading_kwargs)

        if config.modifier_config is not None:
            self.model = modify_transformer(self.model, config.modifier_config)

        self.expert_name = config.expert_name
        self.task_name = config.task_name

    # write a repr function
    def __repr__(self):
        return f"{self.__class__.__name__}(expert_name={self.expert_name}, task_name={self.task_name}, config={self.config})"

    def as_expert(self, training_config=None):
        state_dict = self.state_dict()
        self._delete_non_trainable_params(state_dict)

        # to use as an expert, we need to remove a `model.` prefix
        state_dict = {k[len("model.") :]: v for k, v in state_dict.items()}

        # inject expert info in the expert checkpoint
        expert_info = ExpertInfo(
            expert_name=self.expert_name,
            expert_task_name=self.task_name,
            expert_config=self.modifier_config,
            training_config=training_config,
        )
        return Expert(
            expert_info=expert_info,
            expert_weights=state_dict,
        )
