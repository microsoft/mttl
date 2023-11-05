import pytest
from mttl.datamodule.base import AutoDataModule
from mttl.datamodule.mt_seq_to_seq_module import FlanModule, FlanConfig
from mttl.datamodule.mmlu_data_module import MMLUDataModule, MMLUDataConfig


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
        assert len(flan.train_dataset) == 2_460
        assert len(flan.task_names) == 246
    else:
        assert len(flan.train_dataset) == 10
        assert len(flan.task_names) == 1

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
        model="t5-small",
        model_family="seq2seq",
        for_generation=False,
        validation_portion=0.05,
    )
    batch = next(iter(platy.train_dataloader()))
    assert "input_ids" in batch
    assert "labels" in batch
    assert "attention_mask" in batch
    assert "sources_texts" in batch
    assert "labels_texts" in batch


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


def test_auto_module():
    flan = AutoDataModule.create(
        "sordonia/flan-debug-flat",
        model="t5-small",
        model_family="seq2seq",
        train_batch_size=4,
        predict_batch_size=4,
    )
    assert len(flan.train_dataset) == 2_460
    assert len(flan.task_names) == 246


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
    # there must be a space
    assert sources_text[0][-1] == " "
    # there must *not* be a space
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

    sources_text = batch["sources_texts"]
    labels_text = batch["labels_texts"]
    # there must be a space
    assert sources_text[0][-1] != ""
    # there *must* be a space
    assert labels_text[0][0] == " "
