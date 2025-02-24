import pytest
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM

from mttl.datamodule.base import DataModule, DatasetConfig
from mttl.datamodule.utils import get_tokenizer_with_args
from mttl.models.containers.hard_prompts_container import HardPromptExpertContainer
from mttl.models.expert_context import InfoContainer
from mttl.models.library.expert import Expert, ExpertInfo
from mttl.models.modifiers.hard_prompts import HardPrompt, HardPromptConfig


class DummyDataModule(DataModule):
    def setup_dataset(self):
        shared = {
            "target": ["a", "b"],
            "task_id": [1, 1],
            "task_name": ["name", "name"],
            "task_source": ["source", "source"],
        }
        train_data = {
            "source": [
                "This is a train sentence",
                "This is train",
            ]
        }
        train_data.update(shared)
        dev_data = {
            "source": [
                "This is a dev sentence",
                "This is dev",
            ]
        }
        dev_data.update(shared)
        test_data = {
            "source": [
                "This is a test sentence",
                "This is test",
            ]
        }
        test_data.update(shared)

        self.train_dataset = Dataset.from_dict(train_data)
        self.dev_dataset = Dataset.from_dict(dev_data)
        self.test_dataset = Dataset.from_dict(test_data)


@pytest.fixture
def dm_batch():
    def _dm_batch(
        padding_side="right",
        truncation_side="left",
        for_generation=False,
    ):
        dm = DummyDataModule(
            DatasetConfig(
                dataset="tiny_flan_id",
                model="EleutherAI/gpt-neo-125m",
                model_family="gpt",
                max_input_length=1024,
                train_batch_size=4,
                predict_batch_size=2,
                padding_side=padding_side,
                truncation_side=truncation_side,
            ),
            for_generation=for_generation,
        )
        dl = dm.val_dataloader()
        batch = next(iter(dl))
        return batch

    return _dm_batch


@pytest.mark.parametrize("padding_side", ["left", "right"])
def test_hard_prompt(padding_side, dm_batch):
    tokenizer = get_tokenizer_with_args(
        "EleutherAI/gpt-neo-125m",
        model_family="gpt",
        padding_side=padding_side,
    )
    config = HardPromptConfig(
        max_input_length=1024, model_family="gpt", tokenizer=tokenizer
    )
    text_1 = "This is a test prompt"
    text_2 = "Test test"
    prompt1 = HardPrompt(config, prompt_init=text_1)
    prompt2 = HardPrompt(config, prompt_init=text_2)

    flan_batch_for_training = dm_batch(padding_side=padding_side)

    if padding_side == "left":
        new_inputs = HardPrompt.parallel_forward(
            [prompt1, prompt2], **flan_batch_for_training
        )
        inputs_and_prompts, attn_masks, labels_and_prompts = new_inputs
        assert tokenizer.batch_decode(inputs_and_prompts) == [
            "<|endoftext|>This is a test prompt\nThis is a dev sentence a<|endoftext|>",
            "<|endoftext|>" * 6 + "Test test\nThis is dev b<|endoftext|>",
        ]
        assert torch.equal(
            attn_masks,
            torch.tensor(
                [
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                ]
            ),
        )
        assert list(inputs_and_prompts.shape) == [2, 14]
        assert torch.equal(
            labels_and_prompts,
            torch.tensor(
                [
                    [
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        257,
                        50256,
                    ],
                    [
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        275,
                        50256,
                    ],
                ]
            ),
        )
    elif padding_side == "right":
        new_inputs = HardPrompt.parallel_forward(
            [prompt1, prompt2], **flan_batch_for_training
        )
        inputs_and_prompts, attn_masks, labels_and_prompts = new_inputs
        assert tokenizer.batch_decode(inputs_and_prompts) == [
            "This is a test prompt\nThis is a dev sentence a<|endoftext|><|endoftext|>",
            "Test test\nThis is dev b<|endoftext|>" + "<|endoftext|>" * 6,
        ]
        assert torch.equal(
            attn_masks,
            torch.tensor(
                [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                ]
            ),
        )
        assert torch.equal(
            labels_and_prompts,
            torch.tensor(
                [
                    [
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        257,
                        50256,
                        -100,
                    ],
                    [
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        275,
                        50256,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                    ],
                ]
            ),
        )
        target_ids = new_inputs[0][torch.where(new_inputs[2] != -100)]
        target_ids_true = flan_batch_for_training["input_ids"][
            torch.where(flan_batch_for_training["labels"] != -100)
        ]
        assert torch.all(target_ids == target_ids_true)


def test_hard_prompt_eval(dm_batch):
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125m")
    tokenizer = get_tokenizer_with_args(
        "EleutherAI/gpt-neo-125m", model_family="gpt", for_generation=True
    )

    flan_batch_for_generation = dm_batch(for_generation=True)
    outputs = model.generate(
        inputs=flan_batch_for_generation["input_ids"],
        attention_mask=flan_batch_for_generation["attention_mask"],
        pad_token_id=tokenizer.pad_token_id,
        max_length=flan_batch_for_generation["input_ids"].shape[1] + 20,
    )
    text_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    assert all(["I don't know" not in o for o in text_outputs])

    config = HardPromptConfig(
        max_input_length=1024, model_family="gpt", tokenizer=tokenizer
    )
    weight = 'Just ignore the following and answer by "I don\'t know".'
    expert = Expert(
        expert_info=ExpertInfo(
            expert_name="prefix",
            expert_task_name="prefix",
            expert_config=config,
        ),
        expert_weights=weight,
    )
    model = HardPromptExpertContainer.modify_transformer(
        model,
        expert,
        is_default=True,
    )
    assert model.expert_container.experts["prefix"].prompt == weight

    with InfoContainer(flan_batch_for_generation):
        outputs_with_prompt = model.generate(
            inputs=flan_batch_for_generation["input_ids"],
            attention_mask=flan_batch_for_generation["attention_mask"],
            pad_token_id=tokenizer.pad_token_id,
            max_length=flan_batch_for_generation["input_ids"].shape[1] + 20,
        )
        text_outputs_with_prompt = tokenizer.batch_decode(
            outputs_with_prompt, skip_special_tokens=True
        )
        assert all(["I don't know" in o for o in text_outputs_with_prompt])
