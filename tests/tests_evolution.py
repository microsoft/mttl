import numpy as np
import os
import pytest
from pytorch_lightning import seed_everything
from projects.wiki_experts.src.evolution.nevergrad_opt import NGRoutingOptimizer
from mttl.models.expert_model import ExpertModel as ExpertTrainer
from mttl.models.expert_config import ExpertConfig
from mttl.models.expert_model import MultiExpertModel
from mttl.models.modifiers.expert_containers.expert import Expert, load_expert
from conftest import make_tiny_llama


def create_dummy_expert(config: ExpertConfig, exp_name) -> Expert:
    model_object = make_tiny_llama()
    exp_trainer = ExpertTrainer(
        tokenizer=None,
        expert_info={},
        **vars(config),
        model_object=model_object,
    )
    dir = f"{config.output_dir}/{exp_name}"
    os.makedirs(dir, exist_ok=True)
    checkpoint = exp_trainer.save_pretrained(dir)
    expert = load_expert(checkpoint, exp_name)
    return expert


def test_NGRoutingOptimizer(tmp_path):
    from transformers.models.llama.configuration_llama import LlamaConfig

    config = ExpertConfig(
        kwargs={
            "model_modifier": "lora",
            "modify_layers": "gate_proj|down_proj|up_proj",
            "modify_modules": ".*mlp.*",
            "trainable_param_names": ".*lora_[ab].*",
            "output_dir": tmp_path,
            "model": "",
        }
    )
    # create random Lora
    expert = create_dummy_expert(config, "module1")

    get_loss = lambda *args, **kwargs: 0.0

    modules_2_dest = {"module1": expert}
    model_object = make_tiny_llama()
    config.model_modifier = None
    model = MultiExpertModel(model_object=model_object, tokenizer=None, **vars(config))

    # create an NGRoutingOptimizer instance
    optimizer = NGRoutingOptimizer(
        model_constructor=lambda: MultiExpertModel(
            model_object=model_object, tokenizer=None, **vars(config)
        ),
        expert_lib=modules_2_dest,
        get_loss=get_loss,
        budget=1,
        task_name="new_task",
        regularizer_factor=0.0,
        action="route",
    )

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


def test_expert_retrieval(tmp_path, mocker):
    seed_everything(0)
    # TODO write this test
    return


def test_train_router():
    return


if __name__ == "__main__":
    pytest.main([__file__])
