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
from mttl.models.expert_config import ExpertConfig
from mttl.models.expert_model import Expert, ExpertModel, MultiExpertModel
from mttl.models.modifiers.lora import LoRAConfig


def test_expert_model():
    seed_everything(0)

    model = MultiExpertModel("EleutherAI/gpt-neo-125m", device_map="cpu")
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


def test_from_pretrained(tmp_path):
    # create a dummy library
    model = MultiExpertModel("EleutherAI/gpt-neo-125m", device_map="cpu")
    model.add_empty_expert("a", LoRAConfig(modify_layers=".*out_proj.*"))
    model.add_empty_expert("b", LoRAConfig(modify_layers=".*out_proj.*"))
    library = model.save_to_library(f"local://{tmp_path}")

    # from pretrained library
    model = MultiExpertModel.from_pretrained_library(library)
    assert len(model.experts_names) == 2
    # the order might be different due to multi-threading in adding experts in parallel
    assert "a" in model.experts_names
    assert "b" in model.experts_names


def test_from_pretrained_with_arrow(tmp_path):
    from mttl.models.containers.selectors import ArrowSelectorConfig
    from mttl.models.library.library_transforms import ArrowConfig, ArrowTransform

    # create a dummy library
    model = MultiExpertModel("EleutherAI/gpt-neo-125m", device_map="cpu")
    model.add_empty_expert(
        "a", LoRAConfig(modify_layers=".*out_proj.*", lora_init_b_random=True)
    )
    model.add_empty_expert(
        "b", LoRAConfig(modify_layers=".*out_proj.*", lora_init_b_random=True)
    )
    library = model.save_to_library(f"local://{tmp_path}")

    # store arrow experts
    protos = ArrowTransform(ArrowConfig()).transform(library, persist=True)

    # from pretrained library
    selector_config = ArrowSelectorConfig(moe_top_k=4)
    model = MultiExpertModel.from_pretrained_library(
        library, selector_config={"lora": selector_config}
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


def test_get_modules_to_modify_trie():
    model_name = "EleutherAI/gpt-neo-125m"
    transformer = AutoModelForCausalLM.from_pretrained(model_name)

    multi_expert_model = MultiExpertModel(model_name, device_map="cpu")
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


def test_expert_model_generate(tmp_path, create_dummy_expert, flan_data_module):
    config = ExpertConfig()
    config.model = "EleutherAI/gpt-neo-125m"
    config.device_map = "cpu"

    model = MultiExpertModel(
        config.model,
        device_map=config.device_map,
    )

    config = ExpertConfig(
        kwargs={
            "model_modifier": "lora",
            "modify_layers": "k_proj|v_proj|q_proj",
            "modify_modules": ".*",
            "trainable_param_names": ".*lora_[ab].*",
            "output_dir": tmp_path,
            "model": "EleutherAI/gpt-neo-125m",
        }
    )

    model.add_empty_expert(
        "e1",
        expert_config=config,
        is_default=True,
    )

    batch = next(iter(flan_data_module.val_dataloader()))

    input_shift = batch["input_ids"].shape[1]
    generation = model.generate(batch, max_new_tokens=3)[:, input_shift:]
    assert generation.cpu().numpy().tolist() == [[198, 198, 464]]

    batch["attention_mask"][:1] = 0
    generation = model.generate(batch, max_new_tokens=3)[:, input_shift:]
    assert generation.cpu().numpy().tolist() == [[355, 257, 1255]]


if __name__ == "__main__":
    pytest.main([__file__])
