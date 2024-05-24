# unit test for adapter_ranker
import pytest
from mttl.datamodule.mt_seq_to_seq_module import FlanModule, FlanConfig
from mttl.models.expert_model import (
    MultiExpertModel,
)
from mttl.models.modifiers.lora import LoRAConfig
from mttl.models.modifiers.expert_containers.selectors import TaskPredictorSelector
from mttl.models.ranker.classifier_ranker import SentenceTransformerClassifier
from mttl.models.ranker.clip_ranker import CLIPRanker
from mttl.models.expert_config import ExpertConfig


def test_clip_routing(tiny_flan_id):
    config = ExpertConfig()
    config.ranker_model = "clip"
    config.ranker_path = "zhan1993/clip_ranker_debug"
    config.model = "EleutherAI/gpt-neo-125m"
    config.router_selector = "task_predictor_selector"
    config.router_granularity = "coarsegrained"
    config.device_map = "cpu"

    finetune_task_name = "cot_gsm8k"
    data_module = FlanModule(
        FlanConfig(
            dataset=tiny_flan_id,
            model="EleutherAI/gpt-neo-125m",
            finetune_task_name=finetune_task_name,
            predict_batch_size=1,
            include_template_type="*",
        ),
        for_generation=True,
    )
    module = MultiExpertModel(**vars(config), tokenizer=data_module.tokenizer)
    module.add_empty_expert("a", LoRAConfig(modify_layers=".*out_proj.*"))
    module.add_empty_expert("b", LoRAConfig(modify_layers=".*out_proj.*"))
    batch = next(iter(data_module.val_dataloader()))
    selector = module.model.transformer.h[0].attn.attention.out_proj.selector
    assert isinstance(selector, TaskPredictorSelector)

    prediction_experts = selector.expert_ranker.predict_task(batch["sources_texts"])
    assert len(prediction_experts) == 2
    assert isinstance(selector.expert_ranker, CLIPRanker)
    assert prediction_experts[0][0][0] == "quarel_do_not_use"


def test_classifier_routing(tiny_flan_id):
    config = ExpertConfig()
    config.model = "EleutherAI/gpt-neo-125m"
    config.ranker_model = "classifier"
    config.ranker_path = "zhan1993/classifier_ranker_debug"
    finetune_task_name = "cot_gsm8k"
    config.router_selector = "task_predictor_selector"
    config.router_granularity = "coarsegrained"

    data_module = FlanModule(
        FlanConfig(
            dataset=tiny_flan_id,
            model="EleutherAI/gpt-neo-125m",
            finetune_task_name=finetune_task_name,
            predict_batch_size=1,
            include_template_type="*",
        ),
        for_generation=True,
    )

    module = MultiExpertModel(**vars(config), tokenizer=data_module.tokenizer)

    module.add_empty_expert("a", LoRAConfig(modify_layers=".*out_proj.*"))
    module.add_empty_expert("b", LoRAConfig(modify_layers=".*out_proj.*"))
    batch = next(iter(data_module.val_dataloader()))
    selector = module.model.transformer.h[0].attn.attention.out_proj.selector
    assert isinstance(selector, TaskPredictorSelector)
    prediction_experts = selector.expert_ranker.predict_task(batch["sources_texts"])
    assert len(prediction_experts) == 2  # experts and names
    assert isinstance(selector.expert_ranker, SentenceTransformerClassifier)
    assert prediction_experts[0][0][0] == "cot_gsm8k"


def test_expert_model_generate(tmp_path, create_dummy_expert, flan_data_module):
    config = ExpertConfig()
    config.model = "EleutherAI/gpt-neo-125m"
    config.device_map = "cpu"
    module = MultiExpertModel(**vars(config), tokenizer=flan_data_module.tokenizer)
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
    # create random Lora
    expert1 = create_dummy_expert(config, "module1")
    module.add_expert_instance(
        expert1,
        expert_name="expert1",
        action="route",
        is_default=True,
    )

    batch = next(iter(flan_data_module.val_dataloader()))

    input_shift = batch["input_ids"].shape[1]
    generation = module.generate(batch, max_new_tokens=3)[:, input_shift:]
    assert generation.cpu().numpy().tolist() == [[198, 198, 464]]

    batch["attention_mask"][:1] = 0
    generation = module.generate(batch, max_new_tokens=3)[:, input_shift:]
    assert generation.cpu().numpy().tolist() == [[355, 257, 1255]]


if __name__ == "__main__":
    pytest.main([__file__])
