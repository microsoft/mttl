from mttl.models.expert_model import MultiExpertModel
from mttl.models.modifiers.lora import LoRAConfig
from mttl.models.modifiers.expert_containers.selectors import (
    PolySelector,
    PolySelectorConfig,
    TaskNameSelectorConfig,
    TaskNameSelector,
)


def test_expert_model():
    model = MultiExpertModel("EleutherAI/gpt-neo-125m", device_map="cpu")
    model.add_empty_expert("a", LoRAConfig(modify_layers=".*out_proj.*"))
    model.add_empty_expert("b", LoRAConfig(modify_layers=".*out_proj.*"))
    assert len(model.selectors) == 0

    # plug a poly selector
    model.set_selector("lora", PolySelectorConfig(task_names=["t1", "t2", "t3"]))

    assert len(model.selectors["lora"]) == 12
    assert isinstance(next(iter(model.selectors["lora"].values())), PolySelector)

    # switch selector for lora to task name
    model.set_selector("lora", TaskNameSelectorConfig())

    assert len(model.selectors["lora"]) == 12
    assert isinstance(next(iter(model.selectors["lora"].values())), TaskNameSelector)
