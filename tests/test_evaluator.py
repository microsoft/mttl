import pytest
import numpy as np
import os

from mttl.datamodule.mmlu_data_module import MMLUDataConfig
from mttl.datamodule.ni_data_module import NiDataConfig
from mttl.evaluators import MMLUEvaluator
from mttl.evaluators import NIEvaluator
from mttl.evaluators import RougeEvaluator
from transformers import AutoModelForCausalLM


def test_rouge_eval(mocker, flan_data_module):
    evaluator = RougeEvaluator(flan_data_module)
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125m")

    generate_func = mocker.spy(model, "generate")
    rouge = evaluator.evaluate(model, num_batches=1)
    assert pytest.approx(rouge, 0.01) == 16.949
    assert generate_func.call_args[1]["max_new_tokens"] == 128

    # new kwargs have been taken into consideration
    evaluator = RougeEvaluator(
        flan_data_module, generation_kwargs={"max_new_tokens": 1}
    )
    rouge = evaluator.evaluate(model, num_batches=5)
    assert generate_func.call_args[1]["max_new_tokens"] == 1


def test_early_stopping(mocker, flan_data_module):
    def truncate_gen(generation_list):
        # iterate over the model generation and truncate to 20 chars
        return [i.generated_texts[0][:20] for i in generation_list]

    evaluator = RougeEvaluator(
        flan_data_module, generation_kwargs={"stop_tokens": ["\n\n"]}
    )
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125m")

    generate_func = mocker.spy(model, "generate")
    generate_for_batch_func = mocker.spy(evaluator, "generate_for_batch")
    rouge = evaluator.evaluate(model, num_batches=5)
    assert generate_func.call_args[1]["stopping_criteria"][0].stop == [
        "\n\n",
        "<|endoftext|>",
    ]
    assert truncate_gen(generate_for_batch_func.spy_return_list) == [
        "",
        "",
        "",  # starts with "\n\n"
        " whether the answer ",
        " The man with the ax",
    ]
    assert pytest.approx(rouge, 0.01) == 7.412

    evaluator = RougeEvaluator(
        flan_data_module, generation_kwargs={"stop_tokens": ["answer", "with"]}
    )
    generate_for_batch_func = mocker.spy(evaluator, "generate_for_batch")
    rouge = evaluator.evaluate(model, num_batches=5)
    assert truncate_gen(generate_for_batch_func.spy_return_list) == [
        "\n\nThe construction w",
        "\n\nA:\n\nThe hypothesis",
        "\n\nA:\n\nThe premise is",
        " whether the ",
        " The man ",
    ]


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
    mean_scores = mmlu.evaluate(model, subsample=80)
    assert mean_scores == 50


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
    assert results == pytest.approx(1.98, 0.1)


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
    assert np.allclose(result, 0.2, rtol=0.01)


def test_code_evaluator(mocker):
    from mttl.evaluators.mbpp_evaluator import MBPPEvaluator
    from mttl.evaluators.humaneval_evaluator import HumanEvalEvaluator
    from mttl.datamodule.mbpp_datamodule import MBPPDataConfig
    from mttl.datamodule.humaneval_module import HumanEvalConfig

    evaluator = MBPPEvaluator(
        MBPPDataConfig(
            model="EleutherAI/gpt-neo-125m",
            model_family="gpt",
            max_input_length=1024,
            train_batch_size=1,
            predict_batch_size=1,
            max_output_length=20,
        ),
    )

    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125m")
    result = evaluator.evaluate(model, num_batches=2)
    assert np.allclose(result, 0.0, rtol=0.01)

    evaluator = HumanEvalEvaluator(
        HumanEvalConfig(
            model="EleutherAI/gpt-neo-125m",
            model_family="gpt",
            max_input_length=1024,
            train_batch_size=1,
            predict_batch_size=1,
            max_output_length=20,
        ),
    )

    gen_spy = mocker.spy(model, "generate")
    result = evaluator.evaluate(model, num_batches=1)
    assert np.allclose(result, 0.0, rtol=0.01)
    assert gen_spy.call_args[1]["max_new_tokens"] == 20
    assert gen_spy.call_args[1]["stopping_criteria"] is not None


def test_setup_evaluators():
    from mttl.evaluators.base import setup_evaluators
    from mttl.evaluators.loglike_evaluator import LogLikeEvaluator

    runner = setup_evaluators(
        "EleutherAI/gpt-neo-125m",
        "gpt",
        max_input_length=1024,
        max_output_length=128,
        predict_batch_size=1,
        truncation_side="left",
        tasks="piqa,arc-easy",
    )
    assert len(runner.evaluators) == 2
    assert isinstance(runner.evaluators["piqa"], LogLikeEvaluator)


def test_runner(mocker, gpt_neo):
    from mttl.evaluators.base import setup_evaluators
    from mttl.evaluators.mmlu_evaluator import MMLUEvaluatorFast

    runner = setup_evaluators(
        "EleutherAI/gpt-neo-125m",
        "gpt",
        max_input_length=1024,
        max_output_length=128,
        predict_batch_size=1,
        truncation_side="left",
        tasks="mmlu-fast,mmlu",
    )

    obj_mmlu = mocker.patch(
        "mttl.evaluators.mmlu_evaluator.MMLUEvaluator.evaluate", return_value=2
    )
    scores = runner.run(gpt_neo)
    assert scores["mmlu-fast"] == 2
    assert scores["mmlu"] == 2
    assert scores["mean"] == 2
    assert obj_mmlu.call_count == 2
    assert "shuffle" not in obj_mmlu._mock_call_args_list[0][1]
    assert obj_mmlu._mock_call_args_list[1][1]["shuffle"]
