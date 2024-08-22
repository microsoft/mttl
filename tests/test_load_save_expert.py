import pytest

from mttl.models.base_model import BaseExpertModel
from mttl.models.containers.selectors.poly_selector import (
    PolySelector,
    PolySelectorConfig,
)
from mttl.models.expert_model import MultiExpertModel, MultiExpertModelConfig
from mttl.models.modifiers.lora import LoRAConfig


def test_load_expert_from_checkpoint(tmp_path):
    from mttl.models.expert_model import ExpertModel, ExpertModelConfig
    from mttl.models.library.expert_library import LocalExpertLibrary

    model = ExpertModel(
        ExpertModelConfig(
            "EleutherAI/gpt-neo-125m",
            expert_name="a",
            modifier_config=LoRAConfig(modify_layers="k_proj"),
        )
    )
    model.save_pretrained(tmp_path)

    library = LocalExpertLibrary(tmp_path)
    library.add_expert_from_ckpt(tmp_path)

    assert len(library) == 1
    assert library["a"] is not None


def test_load_model_inited_from_model(tmp_path, tiny_llama):
    model = MultiExpertModel.from_model(
        MultiExpertModelConfig(selector_config=PolySelectorConfig()),
        model=tiny_llama,
    )
    model.add_empty_expert("b", LoRAConfig(modify_layers="k_proj"))
    model.save_pretrained(tmp_path)

    with pytest.raises(ValueError):
        reloaded = MultiExpertModel.from_pretrained(tmp_path)

    reloaded = MultiExpertModel.from_pretrained(tmp_path, model_object=tiny_llama)
    assert isinstance(reloaded.selector_config, PolySelectorConfig)
    assert len(reloaded.experts_names) == 1
    assert "b" in reloaded.experts_names
