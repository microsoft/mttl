import pytest
import numpy as np
from pathlib import Path
from mttl.config import Config
from unittest.mock import MagicMock
from tempfile import TemporaryDirectory
from projects.wiki_experts.src.evolution.nevergrad_opt import NGRoutingOptimizer
from projects.wiki_experts.src.expert_trainer import ExpertTrainer
from projects.wiki_experts.src.config import ExpertConfig
from projects.wiki_experts.src.expert_model import MultiExpertModel


class TestNGRoutingOptimizer:
    def test_optimize(self):
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

        td = TemporaryDirectory()
        tmp_path = td.name
        config = ExpertConfig(
            kwargs={
                "model_modifier": "lora",
                "modify_layers": "gate_proj|down_proj|up_proj",
                "modify_modules": ".*mlp.*",
                "trainable_param_names": ".*lora_[ab].*",
                "output_dir": tmp_path,
            }
        )
        # create random Lora
        exp_trainer = ExpertTrainer(
            model_object=model_object, tokenizer=None, expert_info={}, **vars(config)
        )
        checkpoint = exp_trainer.save_pretrained(tmp_path)

        get_loss = lambda *args, **kwargs: 0.0

        # create a mock modules_2_dest dictionary
        modules_2_dest = {"module1": checkpoint}

        model_object = LlamaForCausalLM(small_config)
        config.model_modifier = None
        model = MultiExpertModel(
            model_object=model_object, tokenizer=None, expert_info={}, **vars(config)
        )

        # create an NGRoutingOptimizer instance
        optimizer = NGRoutingOptimizer(
            model=model,
            modules_2_dest=modules_2_dest,
            get_loss=get_loss,
            budget=1,
            task_name="new_task",
            regularizer_factor=0.0,
            action="route",
        )

        del td
        # call the optimize method
        result = optimizer.optimize()

        # assert that the result is a tuple with two elements
        assert isinstance(result, tuple)
        assert len(result) == 2

        # assert that the first element of the result is a list
        print(result[0].__class__)
        assert isinstance(result[0], np.ndarray)

        # assert that the second element of the result is a string
        assert isinstance(result[1], str)
