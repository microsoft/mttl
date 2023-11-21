import os
import torch
import pytest
import numpy as np
from mttl.config import Config
from pytorch_lightning import seed_everything
from projects.wiki_experts.src.expert_trainer import ExpertTrainer
from projects.wiki_experts.src.config import ExpertConfig
from projects.wiki_experts.src.expert_model import (
    RoutedMultiExpertModel,
)
from mttl.models.modifiers.expert_containers import LoRAExpertContainer
from mttl.models.modifiers.lora import LoRA
from conftest import make_tiny_llama


@pytest.fixture
def tmp_exp_config(tmp_path):
    class SimpleConfig(ExpertConfig):
        def _set_defaults(self):
            super()._set_defaults()
            self.model_modifier = "lora"
            self.modify_layers = "gate_proj|down_proj|up_proj"
            self.modify_modules = ".*mlp.*"
            self.trainable_param_names = ".*lora_[ab].*"
            self.output_dir = tmp_path
            self.router_selector = "poly_router"
            self.router_granularity = "coarsegrained"

    return SimpleConfig()


class TestRoutedMultiExpertModel:
    def creat_dummy_expert(self, config: ExpertConfig, exp_name):
        # create random Lora
        exp_trainer = ExpertTrainer(
            model_object=make_tiny_llama(),
            tokenizer=None,
            expert_info={},
            **vars(config),
        )
        dir = str(config.output_dir / exp_name)
        os.makedirs(dir, exist_ok=True)
        checkpoint = exp_trainer.save_pretrained(dir)
        return checkpoint

    def test_expert_selector_with_task_name_routing(self, tmp_exp_config):
        seed_everything(0)
        config: Config = tmp_exp_config

        config.router_selector = "task_selector"
        exp1_dest = self.creat_dummy_expert(config, "exp1")
        exp2_dest = self.creat_dummy_expert(config, "exp2")
        module_dict = {"mod1": exp1_dest, "mod2": exp2_dest, "default": exp1_dest}

        module = RoutedMultiExpertModel(
            model_object=make_tiny_llama(),
            tokenizer=None,
            expert_info={},
            **vars(config),
        )
        assert module.hparams.model_modifier == None
        module.load_from_module_dict(module_dict, action="route")
        bs, max_seq_len = 10, 100

        assert isinstance(
            module.model.model.layers[0].mlp.down_proj, LoRAExpertContainer
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
        assert np.allclose(output.item(), 6.0915, atol=0.1)

    def test_expert_selector_with_poly_routing(self, tmp_exp_config):
        seed_everything(0)
        config: ExpertConfig = tmp_exp_config

        config.router_selector = "poly_router"
        exp1_dest = self.creat_dummy_expert(config, "exp1")
        exp2_dest = self.creat_dummy_expert(config, "exp2")
        module_dict = {"mod1": exp1_dest, "mod2": exp2_dest}

        module = RoutedMultiExpertModel(
            model_object=make_tiny_llama(),
            tokenizer=None,
            expert_info={},
            **vars(config),
        )
        module.load_from_module_dict(module_dict, action="route")
        bs, max_seq_len = 10, 100

        assert isinstance(
            module.model.model.layers[0].mlp.down_proj, LoRAExpertContainer
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
        assert np.allclose(output.item(), 6.1158, atol=0.1)

        # check the get_router_weights function
        routing_weights = module.get_router_weights()
        assert (
            "mod1" in routing_weights["shared.selector"]
            and "mod2" in routing_weights["shared.selector"]
        )

        # change router_granularity to finegrained
        config.router_granularity = "finegrained"
        module = RoutedMultiExpertModel(
            model_object=make_tiny_llama(),
            tokenizer=None,
            expert_info={},
            **vars(config),
        )
        module.load_from_module_dict(module_dict)
        output = module(batch)
        routing_weights = module.get_router_weights()
        assert np.allclose(output.item(), 6.07, atol=0.1)
        assert (
            "mod1" in routing_weights["model_layers_0_mlp_up_proj.selector"]
            and "mod2" in routing_weights["model_layers_0_mlp_up_proj.selector"]
        )
        module.merge_experts_together()
        assert isinstance(module.model.model.layers[0].mlp.down_proj, LoRA)

    def test_add_expert_with_action_merge(self, tmp_exp_config):
        seed_everything(0)
        config: ExpertConfig = tmp_exp_config

        config.router_selector = "poly_router"
        exp1_dest = self.creat_dummy_expert(config, "exp1")
        exp2_dest = self.creat_dummy_expert(config, "exp2")
        module_dict = {"mod1": exp1_dest, "mod2": exp2_dest}

        module = RoutedMultiExpertModel(
            model_object=make_tiny_llama(),
            tokenizer=None,
            expert_info={},
            **vars(config),
        )
        module.load_from_module_dict(module_dict, action="merge")
        bs, max_seq_len = 10, 100

        assert isinstance(
            module.model.model.layers[0].mlp.down_proj, LoRAExpertContainer
        )
        # expert container should be empty
        assert len(module.model.model.layers[0].mlp.down_proj) == 0

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
        assert np.allclose(output.item(), 6.09, atol=0.1)
