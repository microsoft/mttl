from transformers import AutoModelForCausalLM
import sys
import os
from copy import deepcopy

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from projects.instr_routing.finetune_llama import RoutingConfig
from projects.instr_routing.models.clm import CLM
from mttl.datamodule.alpaca_data_module import AlpacaDataModule
import os
import torch

torch.set_float32_matmul_precision("high")


def dict_to_dataclass(d):
    from dataclasses import make_dataclass

    return make_dataclass("X", d.keys())(**d)


def eval_ni(
    config,
    model,
    nshot=2,
    data_dir=None,
    subsample=-1,
    max_input_length=None,
):
    from mttl.evaluators import NIEvaluator

    config = deepcopy(config)
    out_file_name = f"ni_pred_{config.model}ni-nshot{nshot}.jsonl"
    out_file_name = out_file_name.replace("/", "_")
    out_file_name = out_file_name.strip()
    config.output_dir = os.path.join(config.output_dir, "eval/ni")
    config.out_file_name = out_file_name
    ni_evaluator = NIEvaluator(
        config,
        data_dir=data_dir or config.data_dir,
        num_pos_examples=nshot,
        max_input_length=max_input_length,
    )
    metrics = ni_evaluator.evaluate(model, subsample=subsample)

    # evaluate using the original script for evaluaitng sni
    from projects.instr_routing.eval.ni.evaluate import parse_args, eval_instances

    args = dict_to_dataclass(
        {
            "prediction_file": os.path.join(config.output_dir, out_file_name),
            "reference_file": os.environ["NI_DATA_DIR"] + "/test_references.jsonl",
            "output_file": os.path.join(
                config.output_dir, out_file_name.replace(".jsonl", "_metrics.json")
            ),
            "clean": 0,
        }
    )
    all_results_original = eval_instances(args)

    torch.cuda.empty_cache()
    return metrics, all_results_original


if __name__ == "__main__":
    from huggingface_hub import login

    login(token=os.environ["HF_TOKEN"])
    config = RoutingConfig.parse(extra_kwargs={"eval_superni": True})
    config.model = "meta-llama/Llama-2-7b-hf"
    config.load_in_8bit = True
    config.model_family = "gpt"
    config.data_dir = os.environ["NI_DATA_DIR"]
    config.predict_batch_size = 2
    config.max_input_length = 4096
    config.max_output_length = 128

    model = AutoModelForCausalLM.from_pretrained(
        config.model, load_in_8bit=config.load_in_8bit, device_map="auto"
    )
    print(eval_ni(config, model, nshot=0, subsample=-1, max_input_length=4096))
