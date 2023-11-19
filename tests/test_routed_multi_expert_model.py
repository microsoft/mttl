import torch
import numpy as np
from mttl.config import Config
from tempfile import TemporaryDirectory
from pytorch_lightning import seed_everything
from mttl.models.modifiers.routing import RoutingInfo
from projects.wiki_experts.src.evolution.nevergrad_opt import NGRoutingOptimizer
from projects.wiki_experts.src.expert_trainer import ExpertTrainer
from projects.wiki_experts.src.config import ExpertConfig
from projects.wiki_experts.src.expert_model import (
    MultiExpertModel,
    RoutedMultiExpertModel,
)
from mttl.models.modifiers.expert_containers import LoRAExpertContainer
from mttl.models.modifiers.lora import LoRA


def create_a_tiny_llama():
    from transformers.models.llama.configuration_llama import LlamaConfig

    small_config = LlamaConfig(
        vocab_size=400,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=5,
        num_attention_heads=8,
        max_position_embeddings=512,
    )
    from transformers.models.llama.modeling_llama import LlamaForCausalLM

    model_object = LlamaForCausalLM(small_config)
    return model_object


class TestRoutedMultiExpertModel:
    def __init__(self):
        td = TemporaryDirectory()
        self.exp_config: Config = ExpertConfig(
            kwargs={
                "model_modifier": "lora",
                "modify_layers": "gate_proj|down_proj|up_proj",
                "modify_modules": ".*mlp.*",
                "trainable_param_names": ".*lora_[ab].*",
                "output_dir": td.name,
                "router_selector": "poly_router",
                "router_granularity": "coarsegrained",
            }
        )
        # keep references so that temp dirs dont get deleted
        self._tds = [td]

    def create_dummy_expert(self):
        tiny_llama = create_a_tiny_llama()
        # create random Lora
        exp_trainer = ExpertTrainer(
            model_object=tiny_llama,
            tokenizer=None,
            expert_info={},
            **vars(self.exp_config),
        )
        td = TemporaryDirectory()
        self._tds.append(td)
        checkpoint = exp_trainer.save_pretrained(td.name)
        return checkpoint

    def test_expert_selector_with_task_name_routing(self):
        seed_everything(0)
        config: Config = self.exp_config

        config.router_selector = "task_selector"
        exp1_dest = self.create_dummy_expert()
        exp2_dest = self.create_dummy_expert()
        module_dict = {"mod1": exp1_dest, "mod2": exp2_dest, "default": exp1_dest}

        module = RoutedMultiExpertModel(
            model_object=create_a_tiny_llama(),
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
        batch["task_names"] = ["mod1", "mod2"] * 4
        batch["task_names"] += ["some_unknown_task_name"] * 2

        # Test Base Llama model
        output = module(batch)
        assert np.allclose(output.item(), 6.0915, atol=0.1)

    def test_expert_selector_with_poly_routing(self):
        seed_everything(0)
        config: Config = self.exp_config

        config.router_selector = "poly_router"
        exp1_dest = self.create_dummy_expert()
        exp2_dest = self.create_dummy_expert()
        module_dict = {"mod1": exp1_dest, "mod2": exp2_dest}

        module = RoutedMultiExpertModel(
            model_object=create_a_tiny_llama(),
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
            model_object=create_a_tiny_llama(),
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

    def test_add_expert_with_action_merge(self):
        seed_everything(0)
        config: Config = self.exp_config

        config.router_selector = "poly_router"
        exp1_dest = self.create_dummy_expert()
        exp2_dest = self.create_dummy_expert()
        module_dict = {"mod1": exp1_dest, "mod2": exp2_dest}

        module = RoutedMultiExpertModel(
            model_object=create_a_tiny_llama(),
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


def test_expert_selector_with_task_name_routing():
    test = TestRoutedMultiExpertModel()
    test.test_expert_selector_with_task_name_routing()


def test_expert_selector_with_poly_routing():
    test = TestRoutedMultiExpertModel()
    test.test_expert_selector_with_poly_routing()


def test_add_expert_with_action_merge():
    test = TestRoutedMultiExpertModel()
    test.test_add_expert_with_action_merge()


# if __name__ == "__main__":
#     test = TestRoutedMultiExpertModel()
#     test.test_expert_selector_with_task_name_routing()
#     # test.test_expert_selector_with_poly_routing()
#     # test.test_add_expert_with_action_merge()
