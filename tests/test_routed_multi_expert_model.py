import os
import torch
import pytest
import numpy as np
from mttl.config import Config
from pytorch_lightning import seed_everything
from projects.wiki_experts.src.expert_trainer import ExpertTrainer
from projects.wiki_experts.src.config import ExpertConfig
from projects.wiki_experts.src.expert_model import (
    MoETrainer,
    RoutedMultiExpertModel,
)

from mttl.models.modifiers.expert_containers.expert import Expert, load_expert
from mttl.models.modifiers.expert_containers import (
    LoRAExpertContainer,
    CoalescedLoRAExpertContainer,
)
from mttl.models.modifiers.expert_containers.selectors import (
    BatchSequenceModulesAndWeightsSelectorOutput,
    PolySelectorDirect,
    MOERKHSSelector,
    SelectorView,
)
from mttl.models.modifiers.lora import LoRA


@pytest.fixture
def tmp_exp_config(tmp_path):
    class SimpleConfig(ExpertConfig):
        def _set_defaults(self):
            super()._set_defaults()
            self.library_id = None
            self.model_modifier = "lora"
            self.modify_layers = "c_fc|c_proj|k_proj|v_proj|q_proj|out_proj"
            self.modify_modules = ".*"
            self.trainable_param_names = ".*lora_[ab].*"
            self.output_dir = tmp_path
            self.router_selector = "poly_router_dir"
            self.router_granularity = "coarsegrained"
            self.model = "EleutherAI/gpt-neo-125m"
            self.n_tasks = 1

    return SimpleConfig()


@pytest.fixture
def dummy_batch():
    torch.manual_seed(0)
    bs = 2
    max_seq_len = 3
    batch = {
        "input_ids": torch.randint(10, 400, (bs, max_seq_len)),
        "labels": torch.randint(10, 400, (bs, max_seq_len)),
    }
    seq_len = torch.randint(0, max_seq_len, (bs,))
    attn_mask = torch.zeros(bs, max_seq_len, dtype=torch.int32)
    attn_mask[torch.arange(bs), seq_len] = 1
    attn_mask = 1 - attn_mask.cumsum(dim=-1)
    batch["attention_mask"] = attn_mask
    return batch


class TestMultiExpertModel:
    def create_dummy_expert(self, config: ExpertConfig, exp_name) -> Expert:
        exp_trainer = ExpertTrainer(
            tokenizer=None,
            **vars(config),
        )
        dir = f"{config.output_dir}/{exp_name}"
        os.makedirs(dir, exist_ok=True)
        checkpoint = exp_trainer.save_pretrained(dir)
        expert = load_expert(checkpoint, exp_name)
        expert.expert_info.expert_name = exp_name
        return expert

    def test_add_expert_with_action_merge(self, tmp_exp_config):
        seed_everything(0)
        config: ExpertConfig = tmp_exp_config

        config.router_selector = "poly_router"
        exp1_dest = self.create_dummy_expert(config, "exp1")
        exp2_dest = self.create_dummy_expert(config, "exp2")
        module_dict = {"mod1": exp1_dest, "mod2": exp2_dest}

        module = RoutedMultiExpertModel(
            tokenizer=None,
            expert_info={},
            **vars(config),
        )
        module.load_from_module_dict(module_dict, action="merge")
        bs, max_seq_len = 10, 100

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
        output = module(batch)
        assert np.allclose(output.item(), 9.7, atol=0.1)

    def test_expert_selector_with_task_name_routing(self, tmp_exp_config):
        seed_everything(0)
        config: Config = tmp_exp_config

        config.router_selector = "task_selector"
        exp1 = self.create_dummy_expert(config, "exp1")
        exp2 = self.create_dummy_expert(config, "exp2")
        module_dict = {"mod1": exp1, "mod2": exp2, "default": exp1}

        module = RoutedMultiExpertModel(
            # model_object=make_tiny_llama(),
            tokenizer=None,
            **vars(config),
        )
        assert module.hparams.model_modifier == None
        module.load_from_module_dict(module_dict, action="route")
        bs, max_seq_len = 10, 100

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
        batch["task_names"] = ["mod1", "mod2"] * 4
        batch["task_names"] += ["some_unknown_task_name"] * 2

        # Test Base Llama model
        output = module(batch)
        assert np.allclose(output.item(), 11.04, atol=0.1)

    def test_expert_selector_with_poly_routing(self, tmp_exp_config):
        seed_everything(0)
        config: ExpertConfig = tmp_exp_config

        config.router_selector = "poly_router_dir"
        exp1_dest = self.create_dummy_expert(config, "exp1")
        exp2_dest = self.create_dummy_expert(config, "exp2")
        module_dict = {"mod1": exp1_dest, "mod2": exp2_dest}

        module = RoutedMultiExpertModel(
            tokenizer=None,
            expert_info={},
            **vars(config),
        )
        module.load_from_module_dict(module_dict, action="route")
        bs, max_seq_len = 10, 100

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

        # Test Base Llama model
        output = module(batch)
        assert np.allclose(output.item(), 10.15, atol=0.1)

        # check the get_router_weights function
        routing_weights = module.get_router_weights()
        assert (
            "mod1" in routing_weights["shared.selector"]
            and "mod2" in routing_weights["shared.selector"]
        )

        assert isinstance(
            module.model.transformer.h[0].attn.attention.k_proj.selector,
            PolySelectorDirect,
        )
        assert (
            module.model.transformer.h[
                0
            ].attn.attention.q_proj.selector.selector_instance
            == module.model.transformer.h[0].attn.attention.k_proj.selector
        )

        # change router_granularity to finegrained
        config.router_granularity = "finegrained"
        module = RoutedMultiExpertModel(
            tokenizer=None,
            expert_info={},
            **vars(config),
        )
        module.load_from_module_dict(module_dict)
        output = module(batch)
        routing_weights = module.get_router_weights()
        assert np.allclose(output.item(), 10.15, atol=0.1)

        expert = module.to_expert()
        assert isinstance(expert, Expert)
        module.replace_container_with_expert("mod1")
        assert isinstance(module.model.transformer.h[0].attn.attention.k_proj, LoRA)

    def test_add_expert_with_action_merge(self, tmp_exp_config):
        seed_everything(0)
        config: ExpertConfig = tmp_exp_config

        config.router_selector = "poly_router"
        exp1_dest = self.create_dummy_expert(config, "exp1")
        exp2_dest = self.create_dummy_expert(config, "exp2")
        module_dict = {"mod1": exp1_dest, "mod2": exp2_dest}

        module = RoutedMultiExpertModel(
            tokenizer=None,
            expert_info={},
            **vars(config),
        )
        module.load_from_module_dict(module_dict, action="merge")
        bs, max_seq_len = 10, 100

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
        output = module(batch)
        assert np.allclose(output.item(), 10.15, atol=0.1)

    def test_expert_selector_with_moe_routing_soft(
        self, mocker, tmp_exp_config, dummy_batch
    ):
        seed_everything(0)
        config: ExpertConfig = tmp_exp_config
        config.router_selector = "moe_rkhs_router"
        config.router_granularity = "finegrained"
        config.moe_emb_dim = 10
        config.moe_rkhs_dim = 10

        module = MoETrainer(
            tokenizer=None,
            expert_info={},
            **vars(config),
        )

        container = module.model.transformer.h[0].attn.attention.k_proj
        assert isinstance(container, LoRAExpertContainer)
        assert isinstance(container.selector, MOERKHSSelector)
        assert container.selector.top_k == -1

        # Test Base Llama model
        spy = mocker.spy(container.selector, "forward")
        output = module(dummy_batch)
        assert np.allclose(output.item(), 18, atol=0.1)
        assert container.selector.total_calls_per_forward == 1

        assert spy.call_count == 1
        assert isinstance(spy.spy_return, BatchSequenceModulesAndWeightsSelectorOutput)
        assert spy.spy_return.modules == None
        assert spy.spy_return.weights.shape == (2, 3, 8)

    def test_expert_selector_with_moe_routing_soft_granularity(
        self, mocker, tmp_exp_config, dummy_batch
    ):
        seed_everything(0)
        config: ExpertConfig = tmp_exp_config
        config.router_selector = "moe_rkhs_router"
        config.router_granularity = "coarsegrained"
        config.moe_emb_dim = 10
        config.moe_rkhs_dim = 10

        module = MoETrainer(
            tokenizer=None,
            expert_info={},
            **vars(config),
        )

        container = module.model.transformer.h[0].attn.attention.k_proj
        assert isinstance(container, LoRAExpertContainer)
        assert isinstance(container.selector, MOERKHSSelector)
        assert len(container.selector.views) == 71
        assert container.selector.top_k == -1
        # Test Base Llama model
        output = module(dummy_batch)
        assert np.allclose(output.item(), 18, atol=0.1)
        assert container.selector.total_calls_per_forward == 72

        config: ExpertConfig = tmp_exp_config
        config.router_granularity = "mixer"
        # mixer not found
        with pytest.raises(ValueError):
            module = MoETrainer(
                tokenizer=None,
                expert_info={},
                **vars(config),
            )

    def test_expert_selector_with_moe_routing_soft_coalesced(
        self, mocker, tmp_exp_config, dummy_batch, monkeypatch
    ):
        monkeypatch.setenv("COALESCED_LORA_CONTAINER", "1")

        seed_everything(0)
        config: ExpertConfig = tmp_exp_config
        config.router_selector = "moe_rkhs_router"
        config.router_granularity = "finegrained"
        config.moe_to_k = -1
        config.moe_emb_dim = 10
        config.moe_rkhs_dim = 10

        module = MoETrainer(
            tokenizer=None,
            expert_info={},
            **vars(config),
        )

        container = module.model.transformer.h[0].attn.attention.k_proj
        assert isinstance(container, CoalescedLoRAExpertContainer)
        assert isinstance(container.selector, MOERKHSSelector)
        assert container.selector.top_k == -1

        # Test Base Llama model
        spy = mocker.spy(container.selector, "forward")
        output = module(dummy_batch)
        assert np.allclose(output.item(), 18, atol=0.1)
        assert container.selector.total_calls_per_forward == 1

        assert spy.call_count == 1
        assert isinstance(spy.spy_return, BatchSequenceModulesAndWeightsSelectorOutput)
        assert spy.spy_return.modules == None
        assert spy.spy_return.weights.shape == (2, 3, 8)

    def test_expert_selector_with_moe_routing_hard(
        self, mocker, tmp_exp_config, dummy_batch
    ):
        seed_everything(0)
        config: ExpertConfig = tmp_exp_config
        config.router_selector = "moe_rkhs_router"
        config.router_granularity = "finegrained"
        config.moe_top_k = 2
        config.moe_emb_dim = 10
        config.moe_rkhs_dim = 10

        module = MoETrainer(
            tokenizer=None,
            expert_info={},
            **vars(config),
        )

        container = module.model.transformer.h[0].attn.attention.k_proj
        assert isinstance(container, LoRAExpertContainer)
        assert isinstance(container.selector, MOERKHSSelector)
        assert container.selector.top_k == 2

        # Test Base Llama model
        spy = mocker.spy(container.selector, "forward")
        output = module(dummy_batch)
        assert np.allclose(output.item(), 18, atol=0.1)
        assert container.selector.total_calls_per_forward == 1

        assert spy.call_count == 1
        assert isinstance(spy.spy_return, BatchSequenceModulesAndWeightsSelectorOutput)
        assert spy.spy_return.modules.shape == (2, 3, 2)
        assert spy.spy_return.weights.shape == (2, 3, 2)
