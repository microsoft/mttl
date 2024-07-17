import numpy as np
import pytest
from pytorch_lightning import seed_everything
from transformers import AutoModelForCausalLM

from mttl.models.containers import get_modules_to_modify_trie
from mttl.models.containers.selectors import (
    ArrowSelector,
    PolySelector,
    PolySelectorConfig,
    TaskNameSelector,
    TaskNameSelectorConfig,
)
from mttl.models.expert_model import Expert, MultiExpertModel
from mttl.models.modifiers.lora import LoRAConfig


def test_expert_model():
    seed_everything(0)
    model = MultiExpertModel(model="EleutherAI/gpt-neo-125m", device_map="cpu")
    model.add_empty_expert("a", LoRAConfig(modify_layers=".*out_proj.*"))
    model.add_empty_expert("b", LoRAConfig(modify_layers=".*out_proj.*"))
    assert len(model.selectors) == 0

    # plug a poly selector
    model.set_selector("lora", PolySelectorConfig(task_names=["t1", "t2", "t3"]))
    assert len(model.selectors["lora"]) == 12
    assert isinstance(next(iter(model.selectors["lora"].values())), PolySelector)

    expert_a: Expert = model.get_expert_instance("a")
    assert len(expert_a.expert_weights) == 24
    assert expert_a.expert_config.modify_layers == ".*out_proj.*"
    expert_merged = model.get_merged_expert(task_name="t1")
    assert len(expert_merged.expert_weights) == 24
    assert np.allclose(
        sum([p.sum().item() for p in expert_merged.expert_weights.values()]),
        -0.407,
        atol=0.1,
    )

    # switch selector for lora to task name
    model.set_selector("lora", TaskNameSelectorConfig())

    # this should raise an error
    with pytest.raises(NotImplementedError):
        model.get_merged_expert()

    assert len(model.selectors["lora"]) == 12
    assert isinstance(next(iter(model.selectors["lora"].values())), TaskNameSelector)


def test_from_pretrained():
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdirname:
        # create a dummy library
        model = MultiExpertModel(model="EleutherAI/gpt-neo-125m", device_map="cpu")
        model.add_empty_expert("a", LoRAConfig(modify_layers=".*out_proj.*"))
        model.add_empty_expert("b", LoRAConfig(modify_layers=".*out_proj.*"))
        library = model.save_to_library(f"local://{tmpdirname}")

        # from pretrained library
        model = MultiExpertModel.from_pretrained_library(library)
        assert len(model.experts_names) == 2
        # the order might be different due to multi-threading in adding experts in parallel
        assert "a" in model.experts_names
        assert "b" in model.experts_names


def test_from_pretrained_with_arrow():
    import tempfile

    from mttl.models.containers.selectors import ArrowSelectorConfig
    from mttl.models.library.library_transforms import ArrowConfig, ArrowTransform

    with tempfile.TemporaryDirectory() as tmpdirname:
        # create a dummy library
        model = MultiExpertModel(model="EleutherAI/gpt-neo-125m", device_map="cpu")
        model.add_empty_expert(
            "a", LoRAConfig(modify_layers=".*out_proj.*", lora_init_b_random=True)
        )
        model.add_empty_expert(
            "b", LoRAConfig(modify_layers=".*out_proj.*", lora_init_b_random=True)
        )
        library = model.save_to_library(f"local://{tmpdirname}")

        # store arrow experts
        protos = ArrowTransform(ArrowConfig()).transform(library, persist=True)

        # from pretrained library
        selector_config = ArrowSelectorConfig(moe_top_k=4)
        model = MultiExpertModel.from_pretrained_library(
            library, selector_configs={"lora": selector_config}
        )
        assert len(model.experts_names) == 2
        # the order might be different due to multi-threading in adding experts in parallel
        assert "a" in model.experts_names
        assert "b" in model.experts_names
        assert model.selectors["lora"][0].config == selector_config
        assert isinstance(model.selectors["lora"][0], ArrowSelector)
        # loaded two experts
        assert model.selectors["lora"][0].prototypes.shape[0] == 2
        name1 = model.selectors["lora"][0].expert_names[0]
        name2 = model.selectors["lora"][0].expert_names[1]
        ln = model.selectors["lora"][0].layer_name.replace(".selector", "")
        assert np.allclose(
            model.selectors["lora"][0].prototypes[0].sum().item(),
            protos[name1][ln].sum().item(),
        )
        assert np.allclose(
            model.selectors["lora"][0].prototypes[1].sum().item(),
            protos[name2][ln].sum().item(),
        )


def test_get_modules_to_modify_trie():
    model_name = "EleutherAI/gpt-neo-125m"
    transformer = AutoModelForCausalLM.from_pretrained(model_name)
    multi_expert_model = MultiExpertModel(model=model_name, device_map="cpu")
    transformer_modules = dict(get_modules_to_modify_trie(transformer))
    clean_multi_expert_modules = dict(
        get_modules_to_modify_trie(multi_expert_model.model)
    )
    assert clean_multi_expert_modules.keys() == transformer_modules.keys()

    # add an expert
    multi_expert_model.add_empty_expert("a", LoRAConfig(modify_layers=".*out_proj.*"))
    one_expert_modules = dict(get_modules_to_modify_trie(multi_expert_model.model))
    one_expert_all_modules = dict(multi_expert_model.model.named_modules())
    assert len(one_expert_all_modules.keys()) == 248
    assert one_expert_modules.keys() == transformer_modules.keys()
    assert len(one_expert_all_modules) > len(transformer_modules)

    # add another expert
    multi_expert_model.add_empty_expert("b", LoRAConfig(modify_layers=".*out_proj.*"))
    two_expert_modules = dict(get_modules_to_modify_trie(multi_expert_model.model))
    two_expert_all_modules = dict(multi_expert_model.model.named_modules())
    assert two_expert_modules.keys() == transformer_modules.keys()
    assert len(two_expert_all_modules) > len(one_expert_all_modules)


if __name__ == "__main__":
    pytest.main([__file__])
