from mttl.models.containers.selectors.moe_selector import MOERKHSSelectorConfig
from mttl.models.expert_modeling_multi import MultiExpertModel, MultiExpertModelConfig
from mttl.models.expert_modeling_single import (
    SingleExpertModel,
    SingleExpertModelConfig,
)
from mttl.models.modifiers.lora import LoRAConfig


def test_save_load(tmp_path):
    model = SingleExpertModel(
        SingleExpertModelConfig(
            "EleutherAI/gpt-neo-125m",
            modifier_config=LoRAConfig(modify_layers=".*k_proj.*"),
        )
    )
    model.save_pretrained(tmp_path)
    new_model = SingleExpertModel.from_pretrained(tmp_path)
    assert model.config == new_model.config

    model = MultiExpertModel(
        MultiExpertModelConfig(
            "EleutherAI/gpt-neo-125m", selector_config=MOERKHSSelectorConfig()
        )
    )
    model.save_pretrained(tmp_path)
    new_model = MultiExpertModel.from_pretrained(tmp_path)
    assert model.config == new_model.config
