import pytest
from mttl.datamodule.utils import get_tokenizer_with_args
from mttl.models.modifiers.hard_prompts import HardPrompt, HardPromptConfig


@pytest.mark.parametrize("padding_side", ["left", "right"])
def test_hard_prompt(padding_side, flan_batch_for_generation, flan_batch_for_training):
    config = HardPromptConfig(max_input_length=1024)
    tokenizer = get_tokenizer_with_args(
        "EleutherAI/gpt-neo-125m",
        model_family="gpt",
        padding_side=padding_side,
    )
    prompt1 = HardPrompt(
        config, prompt_init="This is a test prompt", tokenizer=tokenizer
    )
    prompt2 = HardPrompt(config, prompt_init="Test test", tokenizer=tokenizer)

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
    elif padding_side == "right":
        new_inputs = HardPrompt.parallel_forward(
            [prompt1, prompt2], **flan_batch_for_training
        )
        assert new_inputs[0][0, 0].item() == 1212
        assert new_inputs[0][0, 1].item() == 318
        assert new_inputs[0][0, 2].item() == 257
        assert new_inputs[0][0, 0].item() == 1212
        assert new_inputs[0][0, 1].item() == 318
        assert new_inputs[0][0, 2].item() == 257
        assert new_inputs[1][0, 0].item() == 1
        assert new_inputs[1][1, 0].item() == 1
