import torch
import pytest
import numpy as np
from mttl.config import Config
from pytorch_lightning import seed_everything
from mttl.models.expert_config import ExpertConfig

from mttl.models.modifiers.base import ModifierConfig
from mttl.models.modifiers.expert_containers.expert import Expert, load_expert
from mttl.models.modifiers.expert_containers import (
    LoRAExpertContainer,
    CoalescedLoRAExpertContainer,
)
from mttl.models.modifiers.expert_containers.selectors import (
    BatchSequenceModulesAndWeightsSelectorOutput,
    PolySelectorDirect,
    MOERKHSSelector,
    PerTokenSelector,
    PolySelector,
)
from mttl.models.expert_model import ExpertModel, MoEModel, MultiExpertModel
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


class TestMultiExpertModel:
    def create_dummy_expert(self, config: ExpertConfig, exp_name) -> Expert:
        model = MultiExpertModel(model=config.model, device_map="cpu")
        expert = model.add_empty_expert(
            exp_name, ModifierConfig.from_training_config(config)
        )
        return expert

    def test_add_expert_with_action_merge(self, tmp_exp_config):
        seed_everything(0)
        config: ExpertConfig = tmp_exp_config

        config.router_selector = "poly_router"
        exp1_dest = self.create_dummy_expert(config, "exp1")
        exp2_dest = self.create_dummy_expert(config, "exp2")
        module_dict = {"mod1": exp1_dest, "mod2": exp2_dest}

        module = MultiExpertModel(**vars(config))
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

    def nonzero_B_init(self, model):
        gen = torch.Generator()
        gen.manual_seed(0)
        for mod in model.modules():
            if isinstance(mod, LoRA):
                # Also re-initing lora_a so that we have the exact same values
                # for both the Poly and MHR test
                mod.lora_a.data = torch.rand(mod.lora_a.shape, generator=gen) * 0.5
                mod.lora_b.data = torch.rand(mod.lora_b.shape, generator=gen) * 0.5

    def test_expert_selector_with_poly_task_routing(
        self, tmp_exp_config
    ):  # this fails, why?
        seed_everything(0)
        config: Config = tmp_exp_config
        config.router_selector = "poly_router"

        # Tasks need to be specified to the selector to perform routing
        config.task_names = ["task_1", "task_2", "task_3"]

        exp1 = self.create_dummy_expert(config, "task_1")
        exp2 = self.create_dummy_expert(config, "task_2")
        module_dict = {"mod1": exp1, "mod2": exp2}

        module = MultiExpertModel(
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
        batch["task_names"] = ["task_1", "task_2"] * 5

        # BASE MODEL FWD BASS (because all Bs are == 0, so functially same as backbone)
        output = module(batch)
        assert np.allclose(output.item(), 10.20, atol=0.1)

        # Now let's change the adapter params, and also the function parameterized by the model
        self.nonzero_B_init(module)
        output = module(batch)
        assert np.allclose(output.item(), 14.69, atol=0.1)

        """ Multi-Head Routing Test """
        # NOTE: We need to add SkilledLoRAs instead of standard LoRAs
        # We follow the procedure as in MoETrainer constructor :
        # init SkilledLoras n_skills=1, for `n_experts` amount of times
        config.n_splits = 4
        config.n_skills = 1
        config.model_modifier = "skilled_lora"

        # Need to recreate experts so that they have the right amt of splits
        exp1 = self.create_dummy_expert(config, "task_1")
        exp2 = self.create_dummy_expert(config, "task_2")

        module_dict = {"mod1": exp1, "mod2": exp2}
        module = MultiExpertModel(
            **vars(config),
        )
        assert module.hparams.model_modifier == None
        module.load_from_module_dict(module_dict, action="route")
        self.nonzero_B_init(module)

        output = module(batch)

        # Because routing is initialized to uniform, should give same result
        assert np.allclose(output.item(), 15.27, atol=0.1)

        # Now let's change the routing, to make sure the output also changes
        for mod in module.modules():
            if isinstance(mod, PolySelector):
                mod.module_logits.data.uniform_(-10, 10)
                mod.module_logits.data[:, -1] = 999

        output = module(batch)
        assert np.allclose(output.item(), 15.21, atol=0.1)

        # Finally, Test invalid tasks
        batch["task_names"][-1] = "task_10"
        with pytest.raises(AssertionError):
            output = module(batch)

    def test_expert_selector_with_task_name_routing(self, tmp_exp_config):
        seed_everything(0)
        config: Config = tmp_exp_config

        def create_dummy_expert(config, exp_name):
            expert_model = MultiExpertModel(
                tokenizer=None,
                **vars(config),
            )
            expert = expert_model.add_empty_expert(
                exp_name, ModifierConfig.from_training_config(config)
            )
            return expert

        config.router_selector = "task_selector"
        exp1 = create_dummy_expert(config, "exp1")
        exp2 = create_dummy_expert(config, "exp2")
        module_dict = {"mod1": exp1, "mod2": exp2, "default": exp1}

        module = MultiExpertModel(**vars(config))
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
        batch["task_sources"] = batch["task_names"]

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

        module = MultiExpertModel(**vars(config))
        module.load_from_module_dict(module_dict, action="route")

        assert isinstance(
            module.model.transformer.h[0].attn.attention.k_proj, LoRAExpertContainer
        )

        bs, max_seq_len = 10, 100
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
        weights = {}
        for _, selector_dict in module.selectors.items():
            for _, selector in selector_dict.items():
                weights[selector.layer_name] = selector.get_routing_weights()
        assert (
            "mod1" in weights["shared.selector"]
            and "mod2" in weights["shared.selector"]
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
        module = MultiExpertModel(
            **vars(config),
        )
        module.load_from_module_dict(module_dict)
        output = module(batch)
        assert np.allclose(output.item(), 10.15, atol=0.1)

        expert = module.get_merged_expert()
        assert isinstance(expert, Expert)

    def test_expert_selector_with_moe_routing_soft(
        self, mocker, tmp_exp_config, dummy_batch
    ):
        seed_everything(0)
        config: ExpertConfig = tmp_exp_config
        config.router_selector = "moe_rkhs_router"
        config.router_granularity = "finegrained"
        config.moe_emb_dim = 10
        config.moe_rkhs_dim = 10

        module = MoEModel(**vars(config))

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

        module = MoEModel(**vars(config))

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
            module = MoEModel(
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

        module = MoEModel(**vars(config))

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

        module = MoEModel(**vars(config))

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

    def test_expert_selector_with_moe_clown_routing_soft_coalesced(
        self, mocker, tmp_exp_config, bigger_dummy_batch, monkeypatch
    ):
        monkeypatch.setenv("COALESCED_LORA_CONTAINER", "1")

        seed_everything(0)
        config: ExpertConfig = tmp_exp_config
        config.router_selector = "arrow_router"
        config.router_granularity = "finegrained"
        config.router_temp = 0.1

        module = MoEModel(**vars(config))

        container = module.model.transformer.h[0].attn.attention.k_proj
        assert isinstance(container, CoalescedLoRAExpertContainer)
        assert isinstance(container.selector, PerTokenSelector)
        assert container.selector.config.moe_top_k == -1
        assert container.selector.config.router_temp == 0.1

        # Test Base Llama model
        spy = mocker.spy(container.selector, "forward")

        # Prototypes not initialized
        with pytest.raises(ValueError):
            output = module(bigger_dummy_batch)

        # Initialize the prototypes
        def init_proto(fill_value=0.0, random=False):
            for sub_mod in module.modules():
                if isinstance(sub_mod, PerTokenSelector):
                    print("overwriting")
                    protos = torch.full(
                        (len(sub_mod.expert_names), sub_mod.input_dim), fill_value
                    )
                    if random:
                        protos = torch.rand_like(protos)
                    sub_mod.overwrite_prototypes(protos)

        init_proto(1.0)
        output = module(bigger_dummy_batch)
        entropy_uniform = container.selector.metric_logger.meters["ent_uniform"].avg
        actual_entropy = container.selector.metric_logger.meters["ent_routing"].avg
        assert np.allclose(entropy_uniform, actual_entropy, atol=0.1)

        init_proto(random=True)
        output = module(bigger_dummy_batch)
        entropy_uniform = container.selector.metric_logger.meters["ent_uniform"].avg
        actual_entropy = container.selector.metric_logger.meters["ent_routing"].avg
        assert actual_entropy < entropy_uniform

    def test_expert_selector_with_task_predictor_selection(self, tmp_exp_config):
        seed_everything(0)
        config: Config = tmp_exp_config

        config.router_selector = "task_predictor_selector"
        config.ranker_model = "classifier"
        config.ranker_path = "zhan1993/classifier_ranker_debug"
        exp1_dest = self.create_dummy_expert(config, "exp1")
        exp2_dest = self.create_dummy_expert(config, "exp2")
        module_dict = {"niv2_sentence_compression": exp1_dest, "niv2_misc": exp2_dest}

        module = MultiExpertModel(**vars(config), device_map="cpu")
        module.load_from_module_dict(module_dict, action="route")

        bs, max_seq_len = 2, 100
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

        output = module(batch)


if __name__ == "__main__":
    pytest.main([__file__])
