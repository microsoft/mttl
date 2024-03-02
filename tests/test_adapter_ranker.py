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


def test_clip_routing():
    config = ExpertConfig()
    config.ranker_model = "clip"
    config.ranker_path = "zhan1993/clip_ranker_debug"
    config.model = "EleutherAI/gpt-neo-125m"
    config.router_selector = "task_predictor_selector"
    config.router_granularity = "coarsegrained"

    finetune_task_name = "adversarial_qa_dbert_answer_the_following_q"
    data_module = FlanModule(
        FlanConfig(
            dataset="sordonia/flan-debug-flat",
            model="EleutherAI/gpt-neo-125m",
            finetune_task_name=finetune_task_name,
            predict_batch_size=1,
            include_template_type="*",
        ),
        for_generation=True,
    )
    module = MultiExpertModel(
        **vars(config), device_map="cpu", tokenizer=data_module.tokenizer
    )
    module.add_empty_expert("a", LoRAConfig(modify_layers=".*out_proj.*"))
    module.add_empty_expert("b", LoRAConfig(modify_layers=".*out_proj.*"))
    batch = next(iter(data_module.val_dataloader()))
    selector = module.model.transformer.h[0].attn.attention.out_proj.selector
    assert isinstance(selector, TaskPredictorSelector)

    prediction_experts = selector.expert_ranker.predict_task(batch["sources_texts"])
    assert len(prediction_experts) == 2
    assert isinstance(selector.expert_ranker, CLIPRanker)
    assert (
        prediction_experts[0][0][0]
        == "race_high_Write_a_multi_choice_question_for_the_following_article"
    )


def test_classifier_routing():
    config = ExpertConfig()
    config.model = "EleutherAI/gpt-neo-125m"
    config.ranker_model = "classifier"
    config.ranker_path = "zhan1993/classifier_ranker_debug"
    finetune_task_name = "adversarial_qa_dbert_answer_the_following_q"
    config.router_selector = "task_predictor_selector"
    config.router_granularity = "coarsegrained"

    data_module = FlanModule(
        FlanConfig(
            dataset="sordonia/flan-debug-flat",
            model="EleutherAI/gpt-neo-125m",
            finetune_task_name=finetune_task_name,
            predict_batch_size=1,
            include_template_type="*",
        ),
        for_generation=True,
    )

    module = MultiExpertModel(
        **vars(config), device_map="cpu", tokenizer=data_module.tokenizer
    )

    module.add_empty_expert("a", LoRAConfig(modify_layers=".*out_proj.*"))
    module.add_empty_expert("b", LoRAConfig(modify_layers=".*out_proj.*"))
    batch = next(iter(data_module.val_dataloader()))
    selector = module.model.transformer.h[0].attn.attention.out_proj.selector
    assert isinstance(selector, TaskPredictorSelector)
    prediction_experts = selector.expert_ranker.predict_task(batch["sources_texts"])
    assert len(prediction_experts) == 2  # experts and names
    assert isinstance(selector.expert_ranker, SentenceTransformerClassifier)
    assert prediction_experts[0][0][0] == "stream_qed_ii"


def test_expert_model_generate():
    config = ExpertConfig()
    config.model = "EleutherAI/gpt-neo-125m"
    finetune_task_name = "adversarial_qa_dbert_answer_the_following_q"
    data_module = FlanModule(
        FlanConfig(
            dataset="sordonia/flan-debug-flat",
            model="EleutherAI/gpt-neo-125m",
            finetune_task_name=finetune_task_name,
            predict_batch_size=1,
            include_template_type="*",
        ),
        for_generation=True,
    )

    module = MultiExpertModel(
        **vars(config), device_map="cpu", tokenizer=data_module.tokenizer
    )
    module.load_expert(
        expert_path="zhan1993/gpt-neo_adversarial_qa_dbert_answer_the_following_q",
        expert_name="adversarial_qa_dbert_answer_the_following_q",
        action="route",
        is_default=True,
    )

    batch = next(iter(data_module.val_dataloader()))

    generation = module.generate(batch)[:, batch["input_ids"].shape[1] :]
    assert generation[0][:4].cpu().numpy().tolist() == [220]

    batch["attention_mask"][:1] = 0
    generation = module.generate(batch)[:, batch["input_ids"].shape[1] :]
    assert generation[0][:4].cpu().numpy().tolist() != [220]


if __name__ == "__main__":
    pytest.main([__file__])
