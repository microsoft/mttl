import pytest
import numpy as np
import os

from mttl.datamodule.base import AutoDataModule
from mttl.datamodule.mt_seq_to_seq_module import FlanModule, FlanConfig
from mttl.datamodule.mmlu_data_module import MMLUDataModule, MMLUDataConfig
from mttl.datamodule.alpaca_data_module import AlpacaDataModule
from mttl.datamodule.ni_data_module import NiDataConfig
from mttl.evaluators import MMLUEvaluator
from mttl.evaluators import NIEvaluator
from mttl.evaluators import RougeEvaluator
from transformers import AutoModelForCausalLM


def test_rouge_eval(mocker):
    flan = AutoDataModule.create(
        "sordonia/flan-debug-flat",
        model="EleutherAI/gpt-neo-125m",
        model_family="gpt",
        max_input_length=1024,
        max_output_length=128,
        train_batch_size=4,
        for_generation=True,
        predict_batch_size=1,
        truncation_side="left",
    )

    evaluator = RougeEvaluator(flan)
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125m")

    generate_func = mocker.spy(model, "generate")
    rouge = evaluator.evaluate(model, num_batches=1)
    assert pytest.approx(rouge, 0.01) == 1.769
    assert generate_func.call_args[1]["max_new_tokens"] == 128

    # new kwargs have been taken into consideration
    evaluator = RougeEvaluator(flan, generation_kwargs={"max_new_tokens": 1})
    rouge = evaluator.evaluate(model, num_batches=1)
    assert generate_func.call_args[1]["max_new_tokens"] == 1


def test_mmlu_eval():
    mmlu = MMLUEvaluator(
        MMLUDataConfig(
            "mmlu",
            model="EleutherAI/gpt-neo-125m",
            model_family="gpt",
            max_input_length=1024,
            train_batch_size=1,
            predict_batch_size=1,
            finetune_task_name="high_school_government_and_politics",
        ),
    )

    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125m")
    results = mmlu.evaluate(model, subsample=80)
    assert results["all"]["mean"] == 50
    assert results["all"]["stderr"] == 50


@pytest.mark.skipif(
    os.getenv("NI_DATA_DIR") == None,
    reason="No way of currently test this locally, SNI is too big.",
)
def test_ni_eval():
    ni = NIEvaluator(
        NiDataConfig(
            "ni",
            model="EleutherAI/gpt-neo-125m",
            model_family="gpt",
            max_input_length=1024,
            train_batch_size=1,
            predict_batch_size=1,
            finetune_task_name="task893_gap_fill_the_blank_coreference_resolution",
        ),
    )

    assert ni.config.max_output_length == 128
    assert ni.config.add_task_definition
    assert ni.config.num_pos_examples == 0

    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125m")
    results = ni.evaluate(model, subsample=80)
    assert results["all"]["mean"] == pytest.approx(1.98, 0.1)


def test_loglike_eval():
    from mttl.evaluators.hellaswag_evaluator import HellaswagEvaluator
    from mttl.datamodule.hellaswag_data_module import HellaswagDataConfig

    evaluator = HellaswagEvaluator(
        HellaswagDataConfig(
            model="EleutherAI/gpt-neo-125m",
            model_family="gpt",
            max_input_length=1024,
            train_batch_size=1,
            predict_batch_size=1,
        ),
    )
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125m")
    result = evaluator.evaluate(model, num_batches=10)
    assert np.allclose(result, 0.272, rtol=0.01)


def test_code_evaluator():
    from mttl.evaluators.mbpp_evaluator import MBPPEvaluator
    from mttl.datamodule.mbpp_datamodule import MBPPDataConfig

    evaluator = MBPPEvaluator(
        MBPPDataConfig(
            model="EleutherAI/gpt-neo-125m",
            model_family="gpt",
            max_input_length=1024,
            train_batch_size=1,
            predict_batch_size=1,
            max_output_length=200,
        ),
    )

    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125m")
    result = evaluator.evaluate(model, num_batches=2)
    assert np.allclose(result, 0.0, rtol=0.01)
