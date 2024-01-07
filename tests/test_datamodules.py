import pytest
import numpy as np
from mttl.datamodule.base import AutoDataModule
from mttl.datamodule.mt_seq_to_seq_module import (
    FlanModule,
    FlanConfig,
    FlatMultiTaskModule,
    FlatMultiTaskConfig,
)
from mttl.datamodule.mmlu_data_module import MMLUDataModule, MMLUDataConfig
from mttl.datamodule.mbpp_datamodule import MBPPDataConfig, MBPPDataModule


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
            include_template_type="zs_noopt",
            include_task_source="P3,Flan2021",
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
    assert batch["sources_texts"][0][-1] == "\n"
    assert batch["labels_texts"][0][0] != ""
    input_ids = platy.tokenizer(
        batch["sources_texts"][0] + batch["labels_texts"][0]
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


@pytest.mark.skip(reason="deleted dataset on remote :(")
def test_t0_module():
    t0 = AutoDataModule.create(
        "sordonia/t0-10k-flat",
        model="t5-small",
        model_family="seq2seq",
        train_batch_size=4,
        predict_batch_size=4,
    )
    assert len(t0.task_names) == 38

    t0 = AutoDataModule.create(
        "sordonia/t0-10k-flat",
        model="t5-small",
        model_family="seq2seq",
        train_batch_size=4,
        predict_batch_size=4,
        use_templates_as_tasks=True,
    )
    assert len(t0.task_names) == 313


def test_truncation_side():
    flan = AutoDataModule.create(
        "sordonia/flan-debug-flat",
        model="EleutherAI/gpt-neo-125m",
        model_family="gpt",
        max_input_length=16,
        train_batch_size=4,
        predict_batch_size=100,
        truncation_side="left",
        include_template_type="zs_noopt",
        include_task_source="P3,Flan2021",
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

    flan = AutoDataModule.create(
        "sordonia/flan-debug-flat",
        model="EleutherAI/gpt-neo-125m",
        model_family="gpt",
        max_input_length=4000,
        train_batch_size=4,
        predict_batch_size=100,
        truncation_side="left",
        include_template_type="zs_noopt",
        include_task_source="P3,Flan2021",
    )
    dl = flan.val_dataloader()
    batch = next(iter(dl))

    batch["labels"][batch["labels"] == -100] = flan.tokenizer.pad_token_id
    decoded = flan.tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)
    decoded = [d.strip() for d in decoded]

    for i, (true, dec) in enumerate(zip(batch["labels_texts"], decoded)):
        # tokenization is not invertible, so check first 10 chars
        assert true[:10] == dec[:10]


def test_auto_module():
    flan = AutoDataModule.create(
        "sordonia/flan-debug-flat",
        model="t5-small",
        model_family="seq2seq",
        train_batch_size=4,
        predict_batch_size=4,
        include_template_type="zs_noopt",
        include_task_source="P3,Flan2021",
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


def test_dst_subsample():
    common_kwargs = {
        "model": "EleutherAI/gpt-neo-125m",
        "train_batch_size": 4,
        "predict_batch_size": 4,
        "model_family": "gpt",
        "truncation_side": "left",
        "finetune_task_name": "task966_ruletaker_fact_checking_based_on_given_context,task669_ambigqa_answer_generation",
        "dataset": "sordonia/flan-debug-flat",
        "subsample_train": 5,
        "subsample_per_task": False,
    }
    config = FlatMultiTaskConfig(**common_kwargs)
    dm_10 = FlatMultiTaskModule(config)
    assert len(dm_10.train_dataset) == 5

    common_kwargs = {
        "model": "EleutherAI/gpt-neo-125m",
        "train_batch_size": 4,
        "predict_batch_size": 4,
        "model_family": "gpt",
        "truncation_side": "left",
        "finetune_task_name": "task966_ruletaker_fact_checking_based_on_given_context,task669_ambigqa_answer_generation",
        "dataset": "sordonia/flan-debug-flat",
        "subsample_train": 5,
        "subsample_per_task": True,
    }
    config = FlatMultiTaskConfig(**common_kwargs)
    dm_10 = FlatMultiTaskModule(config)
    assert len(dm_10.train_dataset) == 10

    common_kwargs = {
        "model": "EleutherAI/gpt-neo-125m",
        "train_batch_size": 4,
        "predict_batch_size": 4,
        "model_family": "gpt",
        "truncation_side": "left",
        "finetune_task_name": "task966_ruletaker_fact_checking_based_on_given_context,task669_ambigqa_answer_generation",
        "dataset": "sordonia/flan-debug-flat",
        "subsample_train": 0.5,
        "subsample_per_task": True,
    }
    config = FlatMultiTaskConfig(**common_kwargs)
    dm_10 = FlatMultiTaskModule(config)
    assert len(dm_10.train_dataset) == 8
