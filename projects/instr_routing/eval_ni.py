from transformers import AutoModelForCausalLM
import sys
import os
from copy import deepcopy
from mttl.utils import remote_login

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from projects.instr_routing.finetune_llama import RoutingConfig
from projects.instr_routing.models.clm import CLM
from mttl.datamodule.alpaca_data_module import AlpacaDataModule
from mttl.dataloader.ni_metrics import eval_instances as eval_output_file

import os
import torch

torch.set_float32_matmul_precision("high")


def dict_to_dataclass(d):
    from dataclasses import make_dataclass

    return make_dataclass("args", d.keys())(**d)


def eval_ni(
    config,
    model,
    nshot=2,
    data_dir=None,
    subsample=-1,
    max_input_length=None,
):
    from mttl.evaluators import NIEvaluator

    output_file_name = f"ni_pred_{config.model}ni-nshot{nshot}.jsonl"
    output_file_name = output_file_name.replace("/", "_")
    output_file_name = output_file_name.strip()
    output_file_path = os.path.join(config.output_dir, "eval/ni", output_file_name)
    ni_evaluator = NIEvaluator(
        config,
        data_dir=data_dir or config.data_dir,
        num_pos_examples=nshot,
        max_input_length=max_input_length,
        pred_output_file_path=output_file_path,
    )
    metrics = ni_evaluator.evaluate(model, subsample=subsample)
    torch.cuda.empty_cache()
    return metrics


if __name__ == "__main__":
    remote_login()
    config = RoutingConfig.parse(extra_kwargs={"eval_superni": True})
    config.model = "meta-llama/Llama-2-13b-hf"
    config.load_in_8bit = True
    config.model_family = "gpt"
    config.data_dir = os.environ["NI_DATA_DIR"]
    config.predict_batch_size = 2
    config.max_input_length = 4096
    config.max_output_length = 128

    model = AutoModelForCausalLM.from_pretrained(
        config.model, load_in_8bit=config.load_in_8bit, device_map="auto"
    )
    print(eval_ni(config, model, nshot=0, subsample=-1))
