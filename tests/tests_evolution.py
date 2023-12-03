import numpy as np
import os
import pytest
from pytorch_lightning import seed_everything
from projects.wiki_experts.src.evolution.nevergrad_opt import NGRoutingOptimizer
from projects.wiki_experts.src.expert_trainer import ExpertTrainer
from projects.wiki_experts.src.config import ExpertConfig
from projects.wiki_experts.src.expert_model import MultiExpertModel
from mttl.models.modifiers.expert_containers.module_graph import Expert, load_expert
from conftest import make_tiny_llama
from mttl.models.modifiers.expert_containers.expert_library import HFExpertLibrary
from projects.wiki_experts.src.evolution.sequential_evolution import (
    retrieve_experts_for_task,
)
from projects.wiki_experts.src.evolution.evaluators import prepare_evaluator


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
        model=model,
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
    return
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

    exp_1: Expert = create_dummy_expert(config, "exp1")
    exp_2: Expert = create_dummy_expert(config, "exp2")
    exp_3: Expert = create_dummy_expert(config, "exp3")

    # save some scores in Expert Infos
    exp_2.save_score(
        task_name="ai2_arc_ARC_Challenge_1_0_0", metric_name="loss_test", score=100
    )
    exp_2.save_score(
        task_name="ai2_arc_ARC_Challenge_1_0_0", metric_name="rougeL_test", score=100
    )
    # create local expert library, with keys corresponding to task names
    module_dict = {
        "ai2_arc_ARC_Challenge_1_0_0": exp_1,
        "best_performing_task": exp_2,
        "default": exp_3,
    }

    expert_lib = HFExpertLibrary("sordonia/test-library")

    task = "abstract_algebra"

    module = MultiExpertModel(
        tokenizer=None,
        expert_info={},
        **vars(config),
    )

    def evaluate(*args, **kwargs):
        return {"all": {"mean": -1}, f"{task}": {"mean": -1}}

    evaluator_constructor = prepare_evaluator(config, config.dataset, tasks=task)
    evaluator = evaluator_constructor(split="test", subsample=100)

    evaluator.evaluate = mocker.MagicMock(side_effect=evaluate)

    # random retrieval
    random_library = retrieve_experts_for_task(
        1, "random", module, expert_lib, task, evaluator=None
    )
    # keep task's module in the library + 1 random expert
    assert len(random_library) == 2
    assert task in random_library
    # lora_sim retrieval
    lora_sim_library = retrieve_experts_for_task(
        1, "lora_sim", module, expert_lib, task, evaluator=None
    )
    assert len(lora_sim_library) == 2
    assert task in lora_sim_library
    # loss retrieval
    perf_library = retrieve_experts_for_task(
        1, "loss", module, expert_lib, task, evaluator=evaluator
    )
    assert len(perf_library) == 2
    # select task's module + "some_task_2" module as it has the largest score
    assert "ai2_arc_ARC_Challenge_1_0_0" in perf_library
    assert "best_performing_task" in perf_library
    # rougeL retrieval
    rouge_library = retrieve_experts_for_task(
        1, "rougeL", module, expert_lib, task, evaluator=evaluator
    )
    assert len(rouge_library) == 2
    # select task's module + "some_task_2" module as it has the largest score
    assert "ai2_arc_ARC_Challenge_1_0_0" in rouge_library
    assert "best_performing_task" in perf_library

    # current task's module is not in the lilbrary:
    task = "some_unknown_task"
    random_library = retrieve_experts_for_task(1, "random", module, expert_lib, task)
    assert len(random_library) == 1
    lora_sim_library = retrieve_experts_for_task(
        1, "lora_sim", module, expert_lib, task
    )
    assert len(lora_sim_library) == 3

    # remock evaluators
    task = "squad_v1_1_3_0_0"

    def evaluate(*args, **kwargs):
        return {"all": {"mean": -1}, f"{task}": {"mean": -1}}

    perf_library = retrieve_experts_for_task(
        1, "loss", module, expert_lib, task, evaluator=evaluator
    )
    assert len(perf_library) == 1

    perf_library = retrieve_experts_for_task(
        1, "rougeL", module, expert_lib, task, evaluator=evaluator
    )
    assert len(perf_library) == 1


def test_train_router():
    return


if __name__ == "__main__":
    pytest.main([__file__])
