import pytest

from mttl.arguments import ExpertConfig
from mttl.models.containers.selectors.base import UniformSelectorConfig
from mttl.models.expert_model import MultiExpertModel, MultiExpertModelConfig
from mttl.models.modifiers.lora import LoRAConfig, SkilledLoRA


@pytest.mark.parametrize("lora_merge_after", [True, False])
def test_lora_merge_after(
    lora_merge_after, dummy_batch, tmp_path, create_dummy_expert, mocker
):
    model = MultiExpertModel(
        MultiExpertModelConfig(base_model="EleutherAI/gpt-neo-125m"), device_map="cpu"
    )
    config = LoRAConfig(
        modify_layers="k_proj|v_proj|q_proj|o_proj",
        lora_rank=4,
        lora_init_b_random=True,
    )
    model.add_empty_expert(
        "expert_1",
        config,
    )
    model.add_empty_expert(
        "expert_2",
        config,
    )
    spy = mocker.spy(SkilledLoRA, "parallel_linear_weighted_forward")

    model.set_selector("lora", UniformSelectorConfig(lora_merge_after=lora_merge_after))
    output = model(**dummy_batch)

    assert spy.call_args[1]["merge_after"] == lora_merge_after
