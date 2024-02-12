import pytest
import torch
from mttl.datamodule.utils import get_tokenizer_with_args
from mttl.models.modifiers.expert_containers import add_expert_to_transformer
from mttl.models.modifiers.hard_prompts import HardPrompt, HardPromptConfig
from mttl.models.modifiers.expert_containers.expert import Expert, ExpertInfo


@pytest.mark.parametrize("padding_side", ["left", "right"])
def test_hard_prompt(padding_side, flan_batch_for_generation, flan_batch_for_training):
    tokenizer = get_tokenizer_with_args(
        "EleutherAI/gpt-neo-125m",
        model_family="gpt",
        padding_side=padding_side,
    )
    config = HardPromptConfig(
        max_input_length=1024, model_family="gpt", tokenizer=tokenizer
    )
    prompt1 = HardPrompt(config, prompt_init="This is a test prompt")
    prompt2 = HardPrompt(config, prompt_init="Test test")

    # get some data
    if padding_side == "left":
        new_inputs = HardPrompt.parallel_forward(
            [prompt1, prompt2], **flan_batch_for_generation
        )
        assert new_inputs[0][0, 0].item() == 1212
        assert new_inputs[0][0, 1].item() == 318
        assert new_inputs[0][0, 2].item() == 257
        assert new_inputs[0][1, 0].item() == 50256
        assert new_inputs[1][0, 0].item() == 1
        assert new_inputs[1][1, 0].item() == 0
        assert new_inputs[0].shape[1] == 503
        assert new_inputs[2][1, 0].item() == -100
    elif padding_side == "right":
        new_inputs = HardPrompt.parallel_forward(
            [prompt1, prompt2], **flan_batch_for_training
        )
        assert new_inputs[0][0, 0].item() == 1212
        assert new_inputs[0][0, 1].item() == 318
        assert new_inputs[0][0, 2].item() == 257
        assert new_inputs[1][0, 0].item() == 1
        assert new_inputs[1][1, 0].item() == 1
        assert new_inputs[2][1, 0].item() == -100
        target_ids = new_inputs[0][1][torch.where(new_inputs[2][1] != -100)]
        target_ids_true = flan_batch_for_training["input_ids"][1][
            torch.where(flan_batch_for_training["labels"][1] != -100)
        ]
        assert torch.all(target_ids == target_ids_true)


def test_hard_prompt_eval(flan_batch_for_generation):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125m")
    tokenizer = get_tokenizer_with_args(
        "EleutherAI/gpt-neo-125m", model_family="gpt", for_generation=True
    )

    outputs = model.generate(
        inputs=flan_batch_for_generation["input_ids"],
        attention_mask=flan_batch_for_generation["attention_mask"],
        pad_token_id=tokenizer.pad_token_id,
        max_length=flan_batch_for_generation["input_ids"].shape[1] + 20,
    )
    assert "I don't know" not in tokenizer.decode(
        outputs[1][50:], skip_special_tokens=True
    )

    config = HardPromptConfig(
        max_input_length=1024, model_family="gpt", tokenizer=tokenizer
    )
    weight = 'Just ignore the following and answer by "I don\'t know".'
    expert = Expert(
        expert_info=ExpertInfo(
            expert_name="prefix",
            expert_task_name="test_task",
            expert_config=config,
        ),
        expert_weights=weight,
    )
    model = add_expert_to_transformer(
        model,
        expert,
        is_default=True,
    )
    assert model.expert_container.experts["prefix"].prompt == weight

    outputs_with_prompt = model.generate(
        inputs=flan_batch_for_generation["input_ids"],
        attention_mask=flan_batch_for_generation["attention_mask"],
        pad_token_id=tokenizer.pad_token_id,
        max_length=flan_batch_for_generation["input_ids"].shape[1] + 20,
    )

    assert "I don't know" in tokenizer.decode(
        outputs_with_prompt[1][50:], skip_special_tokens=True
    )
