import pytest
import numpy as np
from mttl.datamodule.base import AutoDataModule
from mttl.datamodule.mt_seq_to_seq_module import FlanModule, FlanConfig
from mttl.datamodule.mmlu_data_module import MMLUDataModule, MMLUDataConfig
from mttl.datamodule.alpaca_data_module import AlpacaDataModule


@pytest.mark.parametrize("task_name", [None, "huggingface_xsum"])
def test_flan(task_name):
    flan = FlanModule(
        FlanConfig(
            "sordonia/flan-debug-flat",
            model="t5-small",
            model_family="seq2seq",
            train_batch_size=4,
            predict_batch_size=4,
            finetune_task_name=task_name,
        )
    )
    if task_name is None:
        assert len(flan.train_dataset) == 559
        assert len(flan.task_names) == 235
        assert len(flan.test_dataset) == 63
    else:
        assert len(flan.train_dataset) == 2
        assert len(flan.task_names) == 1
        assert len(flan.test_dataset) == 1

    batch = next(iter(flan.train_dataloader()))
    assert "input_ids" in batch
    assert "labels" in batch
    assert "attention_mask" in batch
    assert "sources_texts" in batch
    assert "labels_texts" in batch
    assert "task_names" in batch


def test_platypus():
    platy = AutoDataModule.create(
        "platypus",
        model="yahma/llama-7b-hf",
        max_input_length=4096,
        model_family="gpt",
        for_generation=False,
        validation_portion=0.05,
    )
    batch = next(iter(platy.val_dataloader()))
    assert "input_ids" in batch
    assert "labels" in batch
    assert "attention_mask" in batch
    assert "sources_texts" in batch
    assert "labels_texts" in batch
    # there is no space added to the labels
    assert batch["labels_texts"][0][0] != ""
    input_ids = platy.tokenizer(
        batch["sources_texts"][0] + " " + batch["labels_texts"][0]
    ).input_ids
    assert np.allclose(
        batch["input_ids"][0][: len(input_ids)].numpy().tolist(), input_ids
    )


def test_alpaca():
    alpaca = AutoDataModule.create(
        "alpaca",
        model="t5-small",
        model_family="seq2seq",
        for_generation=False,
        validation_portion=0.05,
    )
    batch = next(iter(alpaca.train_dataloader()))
    assert "input_ids" in batch
    assert "labels" in batch
    assert "attention_mask" in batch
    assert "sources_texts" in batch
    assert "labels_texts" in batch
    sources_texts = batch["sources_texts"]
    labels_texts = batch["labels_texts"]
    assert sources_texts[0][-1] == ":"
    # there is no space added to the labels
    assert labels_texts[0][0] != " "

    alpaca = AutoDataModule.create(
        "alpaca",
        model="EleutherAI/gpt-neo-125m",
        model_family="gpt",
        for_generation=False,
        validation_portion=0.05,
    )
    batch = next(iter(alpaca.val_dataloader()))

    sources_texts = batch["sources_texts"]
    labels_texts = batch["labels_texts"]

    input_ids = alpaca.tokenizer(sources_texts[0] + " " + labels_texts[0]).input_ids
    assert np.allclose(
        batch["input_ids"][0][: len(input_ids)].numpy().tolist(), input_ids
    )
    assert batch["labels"][0][0] == -100  # train on inputs == False
    assert sources_texts[0][-1] == ":"
    # there is no space added to the labels
    assert labels_texts[0][0] != ""


def test_alpaca_for_gen():
    alpaca = AutoDataModule.create(
        "alpaca",
        model="EleutherAI/gpt-neo-125m",
        model_family="gpt",
        predict_batch_size=1,
        for_generation=True,
        validation_portion=0.05,
    )
    batch = next(iter(alpaca.val_dataloader()))

    sources_texts = batch["sources_texts"]

    input_ids = alpaca.tokenizer(sources_texts[0]).input_ids
    assert np.allclose(
        batch["input_ids"][0][: len(input_ids)].numpy().tolist(), input_ids
    )


def test_auto_module():
    flan = AutoDataModule.create(
        "sordonia/flan-debug-flat",
        model="t5-small",
        model_family="seq2seq",
        train_batch_size=4,
        predict_batch_size=4,
    )
    assert len(flan.train_dataset) == 559
    assert len(flan.task_names) == 235

    flan = AutoDataModule.create(
        "sordonia/flan-debug-flat",
        model="t5-small",
        model_family="seq2seq",
        train_batch_size=4,
        predict_batch_size=4,
        include_template_type="*",
        include_task_source="*",
    )
    assert len(flan.train_dataset) == 14560
    assert len(flan.task_names) == 1820


@pytest.mark.parametrize("task_name", [None, "high_school_government_and_politics"])
def test_mmlu(task_name):
    mmlu = MMLUDataModule(
        MMLUDataConfig(
            "mmlu",
            model="t5-small",
            model_family="seq2seq",
            train_batch_size=4,
            predict_batch_size=4,
            finetune_task_name=task_name,
        )
    )
    if task_name is None:
        # include also task names from auxiliary train
        assert len(mmlu.task_names) == 65
        print(mmlu.task_names)
    if task_name is not None:
        assert len(mmlu.task_names) == 1
        assert mmlu.task_names[0] == task_name


def test_mmlu_spaces_and_merges(task_name=None):
    """this tests whether spaces are added correctly after sources or labels.

    if the tokenizer merges space w the next token (e.g. gpt2), then we strip the space from the sources
    and add it to the labels. this logic is needed if we want to ensure that the model
    creates the correct labels and input_ids.
    """

    mmlu = MMLUDataModule(
        MMLUDataConfig(
            "mmlu",
            model="yahma/llama-7b-hf",
            model_family="gpt",
            max_input_length=4096,
            train_batch_size=4,
            predict_batch_size=4,
            finetune_task_name=task_name,
        )
    )

    batch = next(iter(mmlu.test_dataloader()))

    sources_text = batch["sources_texts"]
    labels_text = batch["labels_texts"]
    input_ids = mmlu.tokenizer(sources_text[0] + " " + labels_text[0]).input_ids
    assert np.allclose(
        batch["input_ids"][0][: len(input_ids)].numpy().tolist(), input_ids
    )
    # Answer:
    assert sources_text[0][-1] == ":"
    assert labels_text[0][0] != " "

    mmlu = MMLUDataModule(
        MMLUDataConfig(
            "mmlu",
            model="EleutherAI/gpt-neo-125m",
            model_family="gpt",
            max_input_length=4096,
            train_batch_size=4,
            predict_batch_size=4,
            finetune_task_name=task_name,
        )
    )

    assert mmlu.tokenizer.mttl_merges_space
    batch = next(iter(mmlu.test_dataloader()))
    assert sources_text[0][-1] == ":"
    assert labels_text[0][0] != " "
    input_ids = mmlu.tokenizer(sources_text[0] + " " + labels_text[0]).input_ids
    assert np.allclose(
        batch["input_ids"][0][: len(input_ids)].numpy().tolist(), input_ids
    )
