from typing import Union

import numpy as np
import pytest
import torch
from pytorch_lightning import seed_everything

from mttl.arguments import MoEExpertConfig, MultiExpertConfig
from mttl.models.containers.lora_containers import (
    LoRAExpertContainer,
    SkilledLoRAExpertContainer,
)
from mttl.models.containers.selectors.moe_selector import MOERKHSSelector
from mttl.models.containers.selectors.per_token_selector import PerTokenSelector
from mttl.models.containers.selectors.poly_selector import (
    PolySelector,
    PolySelectorDirect,
)
from mttl.models.containers.selectors.selector_output import (
    BatchSequenceExpertsAndWeightsSelectorOutput,
    SelectorOutput,
)
from mttl.models.expert_model import MoEModel, MoEModelConfig
from mttl.models.library.expert import Expert
from mttl.models.lightning.expert_module import MoEModule, MultiExpertModule
from mttl.models.modifiers.lora import LoRA, LoRAConfig, SkilledLoRAConfig


@pytest.fixture
def bigger_dummy_batch():
    torch.manual_seed(0)
    bs = 5
    max_seq_len = 10
    batch = {
        "input_ids": torch.randint(10, 400, (bs, max_seq_len)),
        "labels": torch.randint(10, 400, (bs, max_seq_len)),
    }
    seq_len = torch.randint(0, max_seq_len, (bs,))
    attn_mask = torch.zeros(bs, max_seq_len, dtype=torch.int32)
    attn_mask[torch.arange(bs), seq_len] = 1
    attn_mask = 1 - attn_mask.cumsum(dim=-1)
    batch["attention_mask"] = attn_mask
    batch["task_names"] = ["dummy_task"] * bs
    batch["task_sources"] = batch["task_names"]
    return batch


bs, max_seq_len = 10, 5


def create_dummy_expert(config: MultiExpertConfig, exp_name) -> Expert:
    from mttl.models.expert_model import MultiExpertModel, MultiExpertModelConfig

    model = MultiExpertModel(
        MultiExpertModelConfig(
            base_model=config.model,
        ),
        device_map="cpu",
    )
    expert = model.add_empty_expert(exp_name, config.modifier_config)
    return expert


def create_expert_model_from_args(config: Union[MultiExpertConfig, MoEExpertConfig]):
    from mttl.models.expert_model import MultiExpertModel, MultiExpertModelConfig

    if type(config) is MultiExpertConfig:
        model = MultiExpertModel(
            MultiExpertModelConfig(
                base_model=config.model,
                selector_config=config.selector_config,
            ),
            device_map="cpu",
        )
        return model
    elif type(config) is MoEExpertConfig:
        model = MoEModel(
            MoEModelConfig(
                base_model=config.model,
                moe_num_experts=config.moe_num_experts,
                library_id=config.library_id,
                modifier_config=config.modifier_config,
                selector_config=config.selector_config,
            ),
            device_map="cpu",
        )
        return model


def test_add_expert_with_action_merge(tmp_multi_exp_config, monkeypatch):
    seed_everything(0)
    config: MultiExpertConfig = tmp_multi_exp_config

    config.router_selector = "poly_router"
    exp1_dest = create_dummy_expert(config, "exp1")
    exp2_dest = create_dummy_expert(config, "exp2")
    module_dict = {"mod1": exp1_dest, "mod2": exp2_dest}

    module = create_expert_model_from_args(config)
    module.add_experts_from_dict(module_dict, action="merge")

    assert isinstance(
        module.model.transformer.h[0].attn.attention.k_proj, LoRAExpertContainer
    )

    # expert container should be empty
    assert len(module.model.transformer.h[0].attn.attention.k_proj) == 0

    batch = {
        "input_ids": torch.randint(10, 400, (bs, max_seq_len)),
        "labels": torch.randint(10, 400, (bs, max_seq_len)),
    }
    seq_len = torch.randint(0, max_seq_len, (bs,))
    attn_mask = torch.zeros(bs, max_seq_len, dtype=torch.int32)
    attn_mask[torch.arange(bs), seq_len] = 1
    attn_mask = 1 - attn_mask.cumsum(dim=-1)
    batch["attention_mask"] = attn_mask

    # Test Base Llama model
    output = module(**batch)
    assert np.allclose(output.loss.item(), 11.31, atol=0.1)


def nonzero_B_init(model):
    gen = torch.Generator()
    gen.manual_seed(0)

    for mod in model.modules():
        if type(mod) == LoRAExpertContainer:
            for name, param in mod.lora_a.items():
                param.data = torch.rand(param.shape, generator=gen) * 0.5
            for name, param in mod.lora_b.items():
                param.data = torch.rand(param.shape, generator=gen) * 0.5
        elif type(mod) == SkilledLoRAExpertContainer:
            mod.experts.lora_a.data = (
                torch.rand(mod.experts.lora_a.shape, generator=gen) * 0.5
            )
            mod.experts.lora_b.data = (
                torch.rand(mod.experts.lora_b.shape, generator=gen) * 0.5
            )


@pytest.mark.parametrize("is_skilled", [0, 1])
def test_expert_selector_with_poly_task_routing(
    tmp_multi_exp_config, monkeypatch, is_skilled
):
    seed_everything(0)

    config: MultiExpertConfig = tmp_multi_exp_config
    config.router_selector = "poly_router"
    config.model_modifier = "skilled_lora" if is_skilled else "lora"

    # Tasks need to be specified to the selector to perform routing
    config.task_names = ["task_1", "task_2", "task_3"]

    exp1 = create_dummy_expert(config, "task_1")
    exp2 = create_dummy_expert(config, "task_2")
    module_dict = {"mod1": exp1, "mod2": exp2}

    module = create_expert_model_from_args(config)
    module.add_experts_from_dict(module_dict, action="route")

    assert isinstance(
        module.model.transformer.h[0].attn.attention.k_proj, LoRAExpertContainer
    )

    batch = {
        "input_ids": torch.randint(10, 400, (bs, max_seq_len)),
        "labels": torch.randint(10, 400, (bs, max_seq_len)),
    }
    seq_len = torch.randint(0, max_seq_len, (bs,))
    attn_mask = torch.zeros(bs, max_seq_len, dtype=torch.int32)
    attn_mask[torch.arange(bs), seq_len] = 1
    attn_mask = 1 - attn_mask.cumsum(dim=-1)
    batch["attention_mask"] = attn_mask
    batch["task_names"] = ["task_1", "task_2"] * 5

    # BASE MODEL FWD BASS (because all Bs are == 0, so functially same as backbone)
    output = module(**batch)
    assert np.allclose(output.loss.item(), 13.625 if is_skilled else 12.93, atol=0.1)

    # Now let's change the adapter params, and also the function parameterized by the model
    nonzero_B_init(module)
    output = module(**batch)
    assert np.allclose(output.loss.item(), 19.0 if is_skilled else 20.5, atol=0.1)

    """ Multi-Head Routing Test """
    # NOTE: We need to add SkilledLoRAs instead of standard LoRAs
    # We follow the procedure as in MoETrainer constructor :
    # init SkilledLoras n_skills=1, for `n_experts` amount of times
    if is_skilled:
        config.n_splits = 4
        config.n_skills = 1
        config.model_modifier = "skilled_lora"

        # Need to recreate experts so that they have the right amt of splits
        exp1 = create_dummy_expert(config, "task_1")
        exp2 = create_dummy_expert(config, "task_2")

        module_dict = {"mod1": exp1, "mod2": exp2}
        module = create_expert_model_from_args(config)
        module.add_experts_from_dict(module_dict, action="route")

        nonzero_B_init(module)
        output = module(**batch)

        # Because routing is initialized to uniform, should give same result
        assert np.allclose(output.loss.item(), 19.0, atol=0.1)

        # Now let's change the routing, to make sure the output also changes
        for mod in module.modules():
            if isinstance(mod, PolySelector):
                mod.module_logits.data.uniform_(-10, 10)
                mod.module_logits.data[:, -1] = 999

        output = module(**batch)
        assert np.allclose(output.loss.item(), 20.375, atol=0.1)

        # Finally, Test invalid tasks
        batch["task_names"][-1] = "task_10"
        with pytest.raises(AssertionError):
            output = module(**batch)


def test_expert_selector_with_task_name_routing(tmp_multi_exp_config):
    seed_everything(0)
    config: MultiExpertConfig = tmp_multi_exp_config

    config.router_selector = "task_selector"
    exp1 = create_dummy_expert(config, "exp1")
    exp2 = create_dummy_expert(config, "exp2")
    module_dict = {"mod1": exp1, "mod2": exp2, "mod3": exp1}

    module = create_expert_model_from_args(config)
    module.add_experts_from_dict(module_dict, action="route")
    module.set_default_expert("mod3")

    assert isinstance(
        module.model.transformer.h[0].attn.attention.k_proj, LoRAExpertContainer
    )

    generator = torch.Generator()
    generator.manual_seed(0)
    batch = {
        "input_ids": torch.randint(10, 400, (bs, max_seq_len), generator=generator),
        "labels": torch.randint(10, 400, (bs, max_seq_len), generator=generator),
    }
    seq_len = torch.randint(0, max_seq_len, (bs,), generator=generator)
    attn_mask = torch.zeros(bs, max_seq_len, dtype=torch.int32)
    attn_mask[torch.arange(bs), seq_len] = 1
    attn_mask = 1 - attn_mask.cumsum(dim=-1)
    batch["attention_mask"] = attn_mask
    batch["task_names"] = ["mod1", "mod2"] * 4
    batch["task_names"] += ["some_unknown_task_name"] * 2
    batch["task_sources"] = batch["task_names"]

    # Test Base Llama model
    output = module(**batch)
    assert np.allclose(output.loss.item(), 12.3125, atol=0.1)


def test_expert_selector_with_poly_routing(tmp_multi_exp_config):
    seed_everything(0)
    config: MultiExpertConfig = tmp_multi_exp_config

    config.router_selector = "poly_router_dir"
    exp1_dest = create_dummy_expert(config, "exp1")
    exp2_dest = create_dummy_expert(config, "exp2")
    module_dict = {"mod1": exp1_dest, "mod2": exp2_dest}

    module = create_expert_model_from_args(config)
    module.add_experts_from_dict(module_dict, action="route")
    assert module.selectors["lora"][0].init_gap == [-1e-3, 1e-3]

    assert isinstance(
        module.model.transformer.h[0].attn.attention.k_proj, LoRAExpertContainer
    )

    generator = torch.Generator()
    generator.manual_seed(0)
    batch = {
        "input_ids": torch.randint(10, 400, (bs, max_seq_len), generator=generator),
        "labels": torch.randint(10, 400, (bs, max_seq_len), generator=generator),
    }
    seq_len = torch.randint(0, max_seq_len, (bs,), generator=generator)
    attn_mask = torch.zeros(bs, max_seq_len, dtype=torch.int32)
    attn_mask[torch.arange(bs), seq_len] = 1
    attn_mask = 1 - attn_mask.cumsum(dim=-1)
    batch["attention_mask"] = attn_mask
    batch["task_names"] = ["mod1", "mod2"] * 4
    batch["task_names"] += ["some_unknown_task_name"] * 2
    batch["task_sources"] = batch["task_names"]

    # Test Base Llama model
    output = module(**batch)
    assert np.allclose(output.loss.item(), 12.3125, atol=0.1)

    # check the get_router_weights function
    weights = {}
    for _, selector_dict in module.selector_cache.items():
        for selector in selector_dict.values():
            weights[selector.layer_name] = selector.get_routing_weights()
    assert len(weights) == 1
    assert (
        "mod1" in weights["transformer.h.0.attn.attention.k_proj.selector"]
        and "mod2" in weights["transformer.h.0.attn.attention.k_proj.selector"]
    )
    assert "shared" in module.selector_cache.get("lora")

    assert isinstance(
        module.model.transformer.h[0].attn.attention.k_proj.selector,
        PolySelectorDirect,
    )
    assert (
        module.model.transformer.h[0].attn.attention.q_proj.selector.selector_instance
        == module.model.transformer.h[0].attn.attention.k_proj.selector
    )

    # change router_granularity to finegrained
    config.router_granularity = "finegrained"
    config.finetune_task_name = "mod1"

    module = create_expert_model_from_args(config)
    module.add_experts_from_dict(module_dict)
    selector = module.selectors["lora"][0]
    assert selector.init_gap == [0, 0]
    assert selector.module_logits_dict["mod1"].item() == 1.0
    assert selector.module_logits_dict["mod2"].item() == 0.0

    output = module(**batch)
    assert np.allclose(output.loss.item(), 12.3125, atol=0.1)

    weights = {}
    for _, selector_dict in module.selector_cache.items():
        for selector in selector_dict.values():
            weights[selector.layer_name] = selector.get_routing_weights()
    assert len(weights) > 1


def test_expert_selector_with_moe_routing_soft(mocker, tmp_moe_exp_config, dummy_batch):
    seed_everything(0)

    config: MoEExpertConfig = tmp_moe_exp_config

    config.router_selector = "moe_rkhs_router"
    config.router_granularity = "finegrained"

    module = create_expert_model_from_args(config)

    container = module.model.transformer.h[0].attn.attention.k_proj
    assert isinstance(container, LoRAExpertContainer)
    assert isinstance(container.selector, MOERKHSSelector)
    assert container.selector.top_k == -1

    # Test Base Llama model
    spy = mocker.spy(container.selector, "forward")
    output = module(**dummy_batch)
    assert np.allclose(output.loss.item(), 18, atol=0.1)
    assert container.selector.total_calls_per_forward == 1

    assert spy.call_count == 1
    assert isinstance(spy.spy_return, BatchSequenceExpertsAndWeightsSelectorOutput)
    assert spy.spy_return.experts is SelectorOutput.ALL_EXPERTS
    assert spy.spy_return.weights.shape == (2, 3, 8)


def test_expert_selector_with_moe_routing_soft_granularity(
    mocker, tmp_moe_exp_config, dummy_batch
):
    seed_everything(0)
    config: MoEExpertConfig = tmp_moe_exp_config
    config.router_selector = "moe_rkhs_router"
    config.router_granularity = "coarsegrained"
    config.moe_emb_dim = 10
    config.moe_rkhs_dim = 10

    module = create_expert_model_from_args(config)

    container = module.model.transformer.h[0].attn.attention.k_proj
    assert isinstance(container, LoRAExpertContainer)
    assert isinstance(container.selector, MOERKHSSelector)
    assert len(container.selector.views) == 71
    assert container.selector.top_k == -1
    # Test Base Llama model
    output = module(**dummy_batch)
    assert np.allclose(output.loss.item(), 18.1, atol=0.1)
    assert container.selector.total_calls_per_forward == 72

    config: MoEExpertConfig = tmp_moe_exp_config
    config.router_granularity = "mixer"

    # mixer not found
    with pytest.raises(ValueError):
        module = create_expert_model_from_args(config)


def test_expert_selector_with_moe_routing_soft(
    mocker, tmp_moe_exp_config, dummy_batch, monkeypatch
):
    seed_everything(0)
    config: MoEExpertConfig = tmp_moe_exp_config
    config.router_selector = "moe_rkhs_router"
    config.router_granularity = "finegrained"
    config.top_k = -1
    config.emb_dim = 10
    config.rkhs_dim = 10

    module = create_expert_model_from_args(config)

    container = module.model.transformer.h[0].attn.attention.k_proj
    assert isinstance(container, SkilledLoRAExpertContainer)
    assert isinstance(container.selector, MOERKHSSelector)
    assert container.selector.top_k == -1

    # Test Base Llama model
    spy = mocker.spy(container.selector, "forward")
    output = module(**dummy_batch)
    assert np.allclose(output.loss.item(), 18, atol=0.1)
    assert container.selector.total_calls_per_forward == 1

    assert spy.call_count == 1
    assert isinstance(spy.spy_return, BatchSequenceExpertsAndWeightsSelectorOutput)
    assert spy.spy_return.experts is SelectorOutput.ALL_EXPERTS
    assert spy.spy_return.weights.shape == (2, 3, 8)


def test_expert_selector_with_moe_routing_hard(mocker, tmp_moe_exp_config, dummy_batch):
    seed_everything(0)
    config: MoEExpertConfig = tmp_moe_exp_config
    config.router_selector = "moe_rkhs_router"
    config.router_granularity = "finegrained"
    config.top_k = 2
    config.emb_dim = 10
    config.rkhs_dim = 10

    module = create_expert_model_from_args(config)

    container = module.model.transformer.h[0].attn.attention.k_proj
    assert isinstance(container, LoRAExpertContainer)
    assert isinstance(container.selector, MOERKHSSelector)
    assert container.selector.top_k == 2

    # Test Base Llama model
    spy = mocker.spy(container.selector, "forward")
    output = module(**dummy_batch)
    assert np.allclose(output.loss.item(), 18, atol=0.1)
    assert container.selector.total_calls_per_forward == 1

    assert spy.call_count == 1
    assert isinstance(spy.spy_return, BatchSequenceExpertsAndWeightsSelectorOutput)
    assert spy.spy_return.experts.shape == (2, 3, 2)
    assert spy.spy_return.weights.shape == (2, 3, 2)


def test_expert_selector_with_moe_clown_routing_soft(
    mocker, tmp_moe_exp_config, bigger_dummy_batch, monkeypatch
):
    seed_everything(0)
    config: MoEExpertConfig = tmp_moe_exp_config
    config.router_selector = "arrow_router"
    config.router_granularity = "finegrained"
    config.router_temp = 0.1

    module = create_expert_model_from_args(config)

    container = module.model.transformer.h[0].attn.attention.k_proj
    assert isinstance(container, SkilledLoRAExpertContainer)
    assert isinstance(container.selector, PerTokenSelector)
    assert container.selector.config.top_k == -1
    assert container.selector.config.router_temp == 0.1

    # Test Base Llama model
    spy = mocker.spy(container.selector, "forward")

    # Initialize the prototypes
    def init_proto(fill_value=0.0, random=False):
        for sub_mod in module.modules():
            if isinstance(sub_mod, PerTokenSelector):
                protos = torch.full(
                    (len(sub_mod.expert_names), sub_mod.input_dim), fill_value
                )
                if random:
                    protos = torch.rand_like(protos)
                sub_mod.overwrite_prototypes(protos)

    init_proto(1.0)
    output = module(**bigger_dummy_batch)
    entropy_uniform = container.selector.metric_logger.meters["ent_uniform"].avg
    actual_entropy = container.selector.metric_logger.meters["ent_routing"].avg
    assert np.allclose(entropy_uniform, actual_entropy, atol=0.1)

    init_proto(random=True)
    output = module(**bigger_dummy_batch)
    entropy_uniform = container.selector.metric_logger.meters["ent_uniform"].avg
    actual_entropy = container.selector.metric_logger.meters["ent_routing"].avg
    assert actual_entropy < entropy_uniform


def test_expert_selector_with_task_predictor_selection(tmp_multi_exp_config):
    seed_everything(0)
    config: MultiExpertConfig = tmp_multi_exp_config

    config.device_map = "cpu"
    config.router_selector = "task_predictor_selector"
    config.ranker_model = "classifier"
    config.ranker_path = "zhan1993/classifier_ranker_debug"
    exp1_dest = create_dummy_expert(config, "exp1")
    exp2_dest = create_dummy_expert(config, "exp2")
    module_dict = {"niv2_sentence_compression": exp1_dest, "niv2_misc": exp2_dest}

    module = create_expert_model_from_args(config)
    module.add_experts_from_dict(module_dict, action="route")

    bs = 2
    batch = {
        "input_ids": torch.randint(bs, 400, (bs, max_seq_len)),
        "labels": torch.randint(bs, 400, (bs, max_seq_len)),
        "sources_texts": ["task predictor", "hello world"],
    }

    seq_len = torch.randint(0, max_seq_len, (bs,))
    attn_mask = torch.zeros(bs, max_seq_len, dtype=torch.int32)
    attn_mask[torch.arange(bs), seq_len] = 1
    attn_mask = 1 - attn_mask.cumsum(dim=-1)
    batch["attention_mask"] = attn_mask

    output = module(**batch)
