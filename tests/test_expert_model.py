import numpy as np
import pytest
from pytorch_lightning import seed_everything
from transformers import AutoModelForCausalLM

from mttl.models.containers.lora_containers import LoRAExpertContainer
from mttl.models.containers.selectors.arrow_selector import (
    ArrowSelector,
    ArrowSelectorConfig,
)
from mttl.models.containers.selectors.base import (
    TaskNameSelector,
    TaskNameSelectorConfig,
)
from mttl.models.containers.selectors.moe_selector import MOERKHSSelectorConfig
from mttl.models.containers.selectors.poly_selector import (
    PolySelector,
    PolySelectorConfig,
)
from mttl.models.containers.utils import get_modifiable_modules
from mttl.models.expert_model import (
    ExpertModel,
    ExpertModelConfig,
    MultiExpertModel,
    MultiExpertModelConfig,
)
from mttl.models.library.expert import Expert
from mttl.models.library.library_transforms import ArrowTransform, ArrowTransformConfig
from mttl.models.modifiers.lora import LoRAConfig, SkilledLoRAConfig


def test_expert_model(monkeypatch):
    seed_everything(0)

    model = MultiExpertModel(MultiExpertModelConfig("EleutherAI/gpt-neo-125m"))
    model.add_empty_expert("a", LoRAConfig(modify_layers=".*out_proj.*"))
    assert model.experts_containers[0].default_expert_name is None

    model.add_empty_expert(
        "b", LoRAConfig(modify_layers=".*out_proj.*"), is_default=True
    )
    assert len(model.selectors["lora"]) == 0
    assert model.experts_containers[0].default_expert_name == "b"

    # plug a poly selector
    model.set_selector("lora", PolySelectorConfig(task_names=["t1", "t2", "t3"]))
    selector = model.selectors["lora"][0]
    assert len(model.selectors["lora"]) == 12
    assert isinstance(selector, PolySelector)

    expert_a: Expert = model.get_expert_instance("a")
    assert len(expert_a.expert_weights) == 24
    assert expert_a.expert_config.modify_layers == ".*out_proj.*"

    # switch selector for lora to task name
    model.set_selector("lora", TaskNameSelectorConfig())

    assert len(model.selectors["lora"]) == 12
    assert isinstance(model.selectors["lora"][0], TaskNameSelector)


def test_disable_enable_adapters(monkeypatch, mocker, dummy_batch):
    from mttl.models.expert_model import disable_adapters

    seed_everything(0)

    model = MultiExpertModel(MultiExpertModelConfig("EleutherAI/gpt-neo-125m"))
    model.add_empty_expert(
        "a", LoRAConfig(modify_layers=".*out_proj.*"), is_default=True
    )
    assert model.experts_containers[0].default_expert_name == "a"
    container: LoRAExpertContainer = model.experts_containers[0]
    mock = mocker.spy(container, "container_forward")

    assert container._enabled

    with disable_adapters(model):
        assert not container._enabled
        model(**dummy_batch)
        assert mock.call_count == 0

    assert container._enabled
    model(**dummy_batch)
    assert mock.call_count == 1

    model = ExpertModel(
        ExpertModelConfig(
            "EleutherAI/gpt-neo-125m",
            modifier_config=LoRAConfig(modify_layers=".*out_proj.*"),
        )
    )

    assert model.modifiers[0]._enabled
    mock = mocker.spy(model.modifiers[0], "dropout_layer")

    with disable_adapters(model):
        assert not model.modifiers[0]._enabled
        model(**dummy_batch)
        assert mock.call_count == 0

    assert container._enabled
    model(**dummy_batch)
    assert mock.call_count == 1


def test_expert_model_skilled(monkeypatch):
    seed_everything(0)

    model = MultiExpertModel(MultiExpertModelConfig("EleutherAI/gpt-neo-125m"))
    model.add_empty_expert("a", SkilledLoRAConfig(modify_layers=".*out_proj.*"))
    assert model.experts_containers[0].default_expert_name is None

    model.add_empty_expert(
        "b", SkilledLoRAConfig(modify_layers=".*out_proj.*"), is_default=True
    )
    assert len(model.selectors["lora"]) == 0
    assert model.experts_containers[0].default_expert_name == "b"

    # plug a poly selector
    model.set_selector(
        "skilled_lora", PolySelectorConfig(task_names=["t1", "t2", "t3"])
    )

    assert len(model.selectors["skilled_lora"]) == 12
    selector = model.selectors["skilled_lora"][0]
    assert isinstance(selector, PolySelector)

    expert_a: Expert = model.get_expert_instance("a")
    assert len(expert_a.expert_weights) == 24
    assert expert_a.expert_config.modify_layers == ".*out_proj.*"

    # switch selector for skilled lora to task name
    model.set_selector("skilled_lora", TaskNameSelectorConfig())

    assert len(model.selectors["skilled_lora"]) == 12
    assert isinstance(model.selectors["skilled_lora"][0], TaskNameSelector)


def test_from_pretrained(tmp_path):
    from mttl.models.expert_model import MultiExpertModel, MultiExpertModelConfig

    # create a dummy library
    model = MultiExpertModel(MultiExpertModelConfig("EleutherAI/gpt-neo-125m"))
    model.add_empty_expert("a", LoRAConfig(modify_layers=".*out_proj.*"))
    model.add_empty_expert("b", LoRAConfig(modify_layers=".*out_proj.*"))
    library = model.save_to_library(f"local://{tmp_path}")

    # from pretrained library
    model = MultiExpertModel.from_pretrained_library(library)
    assert len(model.experts_names) == 2
    # the order might be different due to multi-threading in adding experts in parallel
    assert "a" in model.experts_names
    assert "b" in model.experts_names


def test_from_pretrained_multi_selector(tmp_path):
    from mttl.models.containers.selectors.base import UniformSelectorConfig
    from mttl.models.expert_model import MultiExpertModel, MultiExpertModelConfig

    # create a dummy library
    model = MultiExpertModel(MultiExpertModelConfig("EleutherAI/gpt-neo-125m"))
    model.add_empty_expert("a", LoRAConfig(modify_layers=".*out_proj.*"))
    model.add_empty_expert("b", LoRAConfig(modify_layers=".*out_proj.*"))
    model.set_selector("lora", UniformSelectorConfig())
    model.save_pretrained(tmp_path)

    # from pretrained library
    model = MultiExpertModel.from_pretrained(tmp_path)
    assert "a" in model.experts_names
    assert "b" in model.experts_names
    assert len(model.experts_names) == 2
    assert model.selector_config.get("lora").__class__ == UniformSelectorConfig


def test_from_pretrained_with_arrow_save_and_reload(tmp_path):
    # create a dummy library
    model = MultiExpertModel(MultiExpertModelConfig("EleutherAI/gpt-neo-125m"))
    model.add_empty_expert(
        "a", LoRAConfig(modify_layers=".*out_proj", lora_init_b_random=True)
    )
    model.add_empty_expert(
        "b", LoRAConfig(modify_layers=".*out_proj", lora_init_b_random=True)
    )
    library = model.save_to_library(f"local://{tmp_path}")

    # store arrow experts
    protos = ArrowTransform(ArrowTransformConfig()).transform(library, persist=True)

    # from pretrained library
    selector_config = ArrowSelectorConfig(top_k=4)
    model = MultiExpertModel.from_pretrained_library(
        library, selector_config=selector_config
    )
    assert len(model.experts_names) == 2
    # the order might be different due to multi-threading in adding experts in parallel
    assert "a" in model.experts_names
    assert "b" in model.experts_names

    selector = model.selectors["lora"][0]
    assert selector.config == selector_config
    assert isinstance(selector, ArrowSelector)
    model.save_pretrained(tmp_path)

    # reload from the checkpoint
    model = MultiExpertModel.from_pretrained(tmp_path)
    selector = model.selectors["lora"][0]
    assert selector.prototypes.shape[0] == 2
    name1 = selector.expert_names[0]
    name2 = selector.expert_names[1]
    ln = selector.layer_name.replace(".selector", "")

    assert np.allclose(
        selector.prototypes[0].sum().item(),
        protos[name1][ln].sum().item(),
    )

    assert np.allclose(
        selector.prototypes[1].sum().item(),
        protos[name2][ln].sum().item(),
    )


def test_from_pretrained_with_arrow(tmp_path):
    # create a dummy library
    model = MultiExpertModel(MultiExpertModelConfig("EleutherAI/gpt-neo-125m"))
    model.add_empty_expert(
        "a", LoRAConfig(modify_layers=".*out_proj", lora_init_b_random=True)
    )
    model.add_empty_expert(
        "b", LoRAConfig(modify_layers=".*out_proj", lora_init_b_random=True)
    )
    library = model.save_to_library(f"local://{tmp_path}")

    # store arrow experts
    protos = ArrowTransform(ArrowTransformConfig()).transform(library, persist=True)

    # from pretrained library
    selector_config = ArrowSelectorConfig(top_k=4)
    model = MultiExpertModel.from_pretrained_library(
        library, selector_config=selector_config
    )
    assert len(model.experts_names) == 2
    # the order might be different due to multi-threading in adding experts in parallel
    assert "a" in model.experts_names
    assert "b" in model.experts_names

    selector = model.selectors["lora"][0]
    assert selector.config == selector_config
    assert isinstance(selector, ArrowSelector)
    # loaded two experts
    assert selector.prototypes.shape[0] == 2
    name1 = selector.expert_names[0]
    name2 = selector.expert_names[1]
    ln = selector.layer_name.replace(".selector", "")

    assert np.allclose(
        selector.prototypes[0].sum().item(),
        protos[name1][ln].sum().item(),
    )

    assert np.allclose(
        selector.prototypes[1].sum().item(),
        protos[name2][ln].sum().item(),
    )


def test_get_modifiable_modules(monkeypatch):
    model_name = "EleutherAI/gpt-neo-125m"
    transformer = AutoModelForCausalLM.from_pretrained(model_name)

    multi_expert_model = MultiExpertModel(MultiExpertModelConfig(model_name))
    transformer_modules = dict(get_modifiable_modules(transformer))
    clean_multi_expert_modules = dict(get_modifiable_modules(multi_expert_model.model))
    assert clean_multi_expert_modules.keys() == transformer_modules.keys()

    # add an expert
    multi_expert_model.add_empty_expert("a", LoRAConfig(modify_layers=".*out_proj"))
    one_expert_modules = dict(get_modifiable_modules(multi_expert_model.model))
    one_expert_all_modules = dict(multi_expert_model.model.named_modules())
    one_expert_all_params = dict(multi_expert_model.model.named_parameters())
    assert len(one_expert_all_modules.keys()) == 248
    assert one_expert_modules.keys() == transformer_modules.keys()
    assert len(one_expert_all_modules) > len(transformer_modules)

    # add another expert
    multi_expert_model.add_empty_expert("b", LoRAConfig(modify_layers=".*out_proj"))
    two_expert_modules = dict(get_modifiable_modules(multi_expert_model.model))
    two_expert_all_modules = dict(multi_expert_model.model.named_modules())
    two_expert_all_params = dict(multi_expert_model.model.named_parameters())
    assert two_expert_modules.keys() == transformer_modules.keys()
    assert len(two_expert_all_modules) == len(one_expert_all_modules)
    assert len(two_expert_all_params) > len(one_expert_all_params)


def test_get_modifiable_modules_skilled(monkeypatch):
    model_name = "EleutherAI/gpt-neo-125m"

    transformer = AutoModelForCausalLM.from_pretrained(model_name)
    multi_expert_model = MultiExpertModel(MultiExpertModelConfig(model_name))
    transformer_modules = dict(get_modifiable_modules(transformer))
    clean_multi_expert_modules = dict(get_modifiable_modules(multi_expert_model.model))
    assert clean_multi_expert_modules.keys() == transformer_modules.keys()

    # add an expert
    multi_expert_model.add_empty_expert(
        "a", SkilledLoRAConfig(modify_layers=".*out_proj")
    )
    one_expert_modules = dict(get_modifiable_modules(multi_expert_model.model))
    one_expert_all_modules = dict(multi_expert_model.model.named_modules())

    assert len(one_expert_all_modules.keys()) == 260
    assert one_expert_modules.keys() == transformer_modules.keys()
    assert len(one_expert_all_modules) > len(transformer_modules)

    # add another expert
    multi_expert_model.add_empty_expert(
        "b", SkilledLoRAConfig(modify_layers=".*out_proj")
    )
    two_expert_modules = dict(get_modifiable_modules(multi_expert_model.model))
    two_expert_all_modules = dict(multi_expert_model.model.named_modules())
    assert two_expert_modules.keys() == transformer_modules.keys()
    assert len(two_expert_all_modules) == len(one_expert_all_modules)


def test_save_load(tmp_path):
    model = ExpertModel(
        ExpertModelConfig(
            "EleutherAI/gpt-neo-125m",
            modifier_config=LoRAConfig(modify_layers=".*k_proj.*"),
        )
    )
    model.save_pretrained(tmp_path)
    new_model = ExpertModel.from_pretrained(tmp_path)
    assert model.config == new_model.config

    model = MultiExpertModel(
        MultiExpertModelConfig(
            "EleutherAI/gpt-neo-125m", selector_config=MOERKHSSelectorConfig()
        )
    )
    model.save_pretrained(tmp_path)
    new_model = MultiExpertModel.from_pretrained(tmp_path)
    assert model.config == new_model.config


if __name__ == "__main__":
    pytest.main([__file__])
