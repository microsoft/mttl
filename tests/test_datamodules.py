import pytest
import numpy as np
from mttl.datamodule.alpaca_data_module import AlpacaDataModule
from mttl.datamodule.base import DatasetConfig
from mttl.datamodule.mt_seq_to_seq_module import (
    FlanModule,
    FlanConfig,
    FlatMultiTaskModule,
    FlatMultiTaskConfig,
)
from mttl.datamodule.mmlu_data_module import MMLUDataModule, MMLUDataConfig
from mttl.datamodule.mbpp_datamodule import MBPPDataConfig, MBPPDataModule
from mttl.datamodule.platypus_module import PlatypusModule


@pytest.mark.parametrize(
    "task_name, n_tasks, train, test",
    [
        (None, 18, 765, 73),
        ("stream_aqua", 1, 46, 5),
    ],
)
def test_flan(tiny_flan_id, task_name, n_tasks, train, test):
    flan = FlanModule(
        FlanConfig(
            tiny_flan_id,
            model="t5-small",
            model_family="seq2seq",
            train_batch_size=4,
            predict_batch_size=4,
            finetune_task_name=task_name,
            include_template_type="zs_opt",
            include_task_source="CoT",
        )
    )
    assert len(flan.task_names) == n_tasks
    assert len(flan.train_dataset) == train
    assert len(flan.test_dataset) == test

    batch = next(iter(flan.train_dataloader()))
    assert "input_ids" in batch
    assert "labels" in batch
    assert "attention_mask" in batch
    assert "sources_texts" in batch
    assert "labels_texts" in batch
    assert "task_names" in batch


def test_platypus():
    platy = PlatypusModule(
        DatasetConfig(
            dataset="platypus",
            model="yahma/llama-7b-hf",
            max_input_length=4096,
            model_family="gpt",
            validation_portion=0.05,
        ),
        for_generation=False,
        val_mixin=False,
    )
    batch = next(iter(platy.val_dataloader()))
    assert "input_ids" in batch
    assert "labels" in batch
    assert "attention_mask" in batch
    assert "sources_texts" in batch
    assert "labels_texts" in batch
    # there is no space added to the labels
    assert batch["sources_texts"][0][-1] == "\n"
    assert batch["labels_texts"][0][0] != ""
    input_ids = platy.tokenizer(
        batch["sources_texts"][0] + batch["labels_texts"][0]
    ).input_ids
    assert np.allclose(
        batch["input_ids"][0][: len(input_ids)].numpy().tolist(), input_ids
    )


def test_alpaca():
    alpaca = AlpacaDataModule(
        DatasetConfig(
            dataset="alpaca",
            model="t5-small",
            model_family="seq2seq",
            validation_portion=0.05,
        ),
        for_generation=False,
        val_mixin=False,
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

    alpaca = AlpacaDataModule(
        DatasetConfig(
            dataset="alpaca",
            model="EleutherAI/gpt-neo-125m",
            model_family="gpt",
            validation_portion=0.05,
        ),
        for_generation=False,
        val_mixin=False,
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
    alpaca = AlpacaDataModule(
        DatasetConfig(
            dataset="alpaca",
            model="EleutherAI/gpt-neo-125m",
            model_family="gpt",
            predict_batch_size=1,
            validation_portion=0.05,
        ),
        for_generation=True,
        val_mixin=False,
    )
    batch = next(iter(alpaca.val_dataloader()))

    sources_texts = batch["sources_texts"]

    input_ids = alpaca.tokenizer(sources_texts[0]).input_ids
    assert np.allclose(
        batch["input_ids"][0][: len(input_ids)].numpy().tolist(), input_ids
    )


def test_truncation_side(tiny_flan_id):
    flan = FlanModule(
        FlanConfig(
            dataset=tiny_flan_id,
            model="EleutherAI/gpt-neo-125m",
            model_family="gpt",
            max_input_length=16,
            train_batch_size=4,
            predict_batch_size=100,
            truncation_side="left",
            include_template_type="zs_opt",
            include_task_source="CoT",
        ),
        for_generation=False,
        val_mixin=False,
    )
    dl = flan.val_dataloader()
    batch = next(iter(dl))

    batch["labels"][batch["labels"] == -100] = flan.tokenizer.pad_token_id
    decoded = flan.tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)
    decoded = [d.strip() for d in decoded]

    for i, (true, dec) in enumerate(zip(batch["labels_texts"], decoded)):
        if true != dec:
            # eos token is added so everything larger than 15 is cut off
            assert len(flan.tokenizer.encode(true)) >= 15
        else:
            assert true == dec

    flan = FlanModule(
        FlanConfig(
            dataset=tiny_flan_id,
            model="EleutherAI/gpt-neo-125m",
            model_family="gpt",
            max_input_length=4000,
            train_batch_size=4,
            predict_batch_size=100,
            truncation_side="left",
            include_template_type="zs_opt",
            include_task_source="CoT",
        ),
        for_generation=False,
        val_mixin=False,
    )
    dl = flan.val_dataloader()
    batch = next(iter(dl))

    batch["labels"][batch["labels"] == -100] = flan.tokenizer.pad_token_id
    decoded = flan.tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)
    decoded = [d.strip() for d in decoded]

    for i, (true, dec) in enumerate(zip(batch["labels_texts"], decoded)):
        # tokenization is not invertible, so check first 10 chars
        assert true[:10] == dec[:10]


def test_auto_module(tiny_flan_id):
    flan = FlanModule(
        FlanConfig(
            dataset=tiny_flan_id,
            model="t5-small",
            model_family="seq2seq",
            train_batch_size=4,
            predict_batch_size=4,
            include_template_type="zs_opt",
            include_task_source="CoT",
        ),
        for_generation=False,
        val_mixin=False,
    )
    assert len(flan.task_names) == 18
    assert len(flan.train_dataset) == 765
    assert len(flan.test_dataset) == 73
    assert len(flan.dev_dataset) == 111

    flan = FlanModule(
        FlanConfig(
            dataset=tiny_flan_id,
            model="t5-small",
            model_family="seq2seq",
            train_batch_size=4,
            predict_batch_size=4,
            include_template_type="*",
            include_task_source="*",
        ),
        for_generation=False,
        val_mixin=False,
    )
    assert len(flan.task_names) == 18
    assert len(flan.train_dataset) == 1440
    assert len(flan.test_dataset) == 144
    assert len(flan.dev_dataset) == 216


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


def test_multichoice_collator():
    from mttl.datamodule.base import MultipleChoiceCollator
    from mttl.datamodule.utils import get_tokenizer_with_args

    tokenizer = get_tokenizer_with_args(
        "EleutherAI/gpt-neo-125m", "gpt", "left", "left", False
    )
    collator = MultipleChoiceCollator(
        tokenizer=tokenizer,
    )
    batch = [
        {"source": "a", "target": ["a1", "a2"], "task_name": "t1", "label_index": 1},
        {"source": "b", "target": ["b1"], "task_name": "t2", "label_index": 0},
    ]
    output = collator(batch)
    assert output["sources_texts"] == ["a", "a", "b"]
    assert output["labels_texts"] == ["a1", "a2", "b1"]
    assert output["labels_index"][0] == 1
    assert output["labels_index"][1] == 0
    assert output["num_options"] == [2, 1]
    assert output["task_names"] == ["t1", "t1", "t2"]

    collator = MultipleChoiceCollator(tokenizer=tokenizer, multisource=True)
    batch = [
        {"source": ["a1", "a2"], "target": "a", "task_name": "t1", "label_index": 1},
        {"source": ["b1"], "target": "b", "task_name": "t2", "label_index": 0},
    ]
    output = collator(batch)
    assert output["sources_texts"] == ["a1", "a2", "b1"]
    assert output["labels_texts"] == ["a", "a", "b"]
    assert output["num_options"] == [2, 1]
    assert output["task_names"] == ["t1", "t1", "t2"]


def test_mbpp():
    config = MBPPDataConfig(
        model="EleutherAI/gpt-neo-125m",
        model_family="gpt",
        max_input_length=4096,
        train_batch_size=4,
        predict_batch_size=4,
    )

    module = MBPPDataModule(config, for_generation=False)
    assert len(module.train_dataset) == 120
    # must be executable so that the model trains on valid code
    for ex in module.train_dataset:
        exec(ex["source"] + ex["target"])


@pytest.mark.parametrize(
    "subsample, subsample_per_task, train_size",
    [
        (None, None, 160),
        (0.5, True, 80),
        (5, False, 5),
        (5, True, 10),
    ],
)
def test_dst_subsample(tiny_flan_id, subsample, subsample_per_task, train_size):
    common_kwargs = {
        "model": "EleutherAI/gpt-neo-125m",
        "train_batch_size": 4,
        "predict_batch_size": 4,
        "model_family": "gpt",
        "truncation_side": "left",
        "finetune_task_name": "cot_ecqa,stream_qed",
        "dataset": tiny_flan_id,
    }
    if subsample is not None:
        common_kwargs["subsample_train"] = subsample
    if subsample_per_task is not None:
        common_kwargs["subsample_per_task"] = subsample_per_task

    config = FlatMultiTaskConfig(**common_kwargs)
    dm = FlatMultiTaskModule(config)
    assert len(dm.train_dataset) == train_size
