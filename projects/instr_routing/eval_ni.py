from transformers import AutoModelForCausalLM
import sys
import os
from copy import deepcopy
import os
import re
import glob
import wandb
import click

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from projects.instr_routing.finetune_llama import RoutingConfig
from projects.instr_routing.models.clm import CLM
from mttl.datamodule.alpaca_data_module import AlpacaDataModule
from mttl.dataloader.ni_metrics_old import eval_instances

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
    # evaluate generations file per category and write to the output directory
    args = dict_to_dataclass(
        {
            "prediction_file": output_file_path,
            "reference_file": os.environ["NI_DATA_DIR"] + "/test_references.jsonl",
            "output_file": output_file_path.replace(".jsonl", "_metrics.json"),
            "clean": 0,
        }
    )
    all_results_old = eval_instances(args)
    torch.cuda.empty_cache()
    return metrics, all_results_old


def load_hf_model(model_name):
    from mttl.datamodule.utils import get_tokenizer
    from peft import PeftModel

    config = RoutingConfig()
    config.model = "yahma/llama-7b-hf"
    config.model_family = "gpt"
    tokenizer = get_tokenizer(config)
    model = CLM(**vars(config), tokenizer=tokenizer)
    model = model.cuda()
    model = PeftModel.from_pretrained(
        model.model,
        model_name,
        device_map={"": "cuda"},
    )
    config.model_object = model
    model = CLM(**vars(config), tokenizer=tokenizer)
    config.model_object = None
    return model, config


def upcast_to_dtype(model, dtype: torch.dtype):
    # iterate over all modules and convert to dtype
    import bitsandbytes as bnb

    for name, module in dict(model.named_modules()).items():
        if isinstance(module, bnb.nn.Linear8bitLt):
            # convert to dtype
            new_layer = torch.nn.Linear(
                module.in_features, module.out_features, module.bias
            )
            new_layer.training = module.training
            state_dict = module.state_dict()
            # upcast everything to dtype
            for k, v in state_dict.items():
                if hasattr(v, "dtype") and v.dtype != dtype:
                    state_dict[k] = v.to(dtype)
            new_layer.load_state_dict(state_dict, strict=False)
            setattr(model, name, new_layer)
            print("Upcasting layer", name, "to", dtype)
    return model


@click.command()
@click.option(
    "--model_name", type=str, default="platypus_dense_er4"
)  # chainyo/alpaca-lora-7b") #alpaca_vsmear_e12[xr4,t_1]")
@click.option("--amlt_experiment_name", type=str, default="routing")  # routing")
@click.option(
    "--model_path",
    type=str,
    default="/home/v-oostapenko/dev/amlt/routing/platypus_vsmear_e8[xr4]/meta-llama_Llama-2-13b-hfpqtor60j_platypus_vsmear_e8[xr4]-val/loss=0.5533.ckpt",
    help="path to the model",
)
@click.option("--batch_size", type=int, default=2)
@click.option("--wandb_proj", type=str, default="eval")
@click.option("--n_shots", type=int, default=0)
@click.option("--use_old_gen_config", type=bool, default=False)
@click.option("--subsample", type=int, default=-1)
@click.option("--load_in_8bit", type=bool, default=False)
@click.option("--dtype", type=str, default="float16")
def run_ni_eval(
    model_name,
    amlt_experiment_name=None,
    model_path=None,
    batch_size=5,
    wandb_proj=None,
    use_old_gen_config=False,
    n_shots=0,
    subsample=-1,
    load_in_8bit=False,
    dtype="float16",
):
    if amlt_experiment_name == "hf":
        raise NotImplementedError
        model, config = load_hf_model(model_name)
    elif amlt_experiment_name == "hf_peft":
        model, config = load_hf_model(model_name)
    else:
        if model_path is None:
            if os.environ.get("AMLT_OUTPUT_DIR") is not None:  # on gcr
                base_model_path = "/mnt/default/data/models"
            else:
                base_model_path = os.path.join(
                    os.path.dirname(__file__), "..", "..", ".."
                )
                base_model_path = os.path.join(base_model_path, "amlt")
            if amlt_experiment_name:
                model_name = re.sub(r"(\[)", r"[[]", model_name)
                model_name = re.sub(r"(\])$", r"[]]", model_name)
                model_path = glob.glob(
                    f"{base_model_path}/{amlt_experiment_name}/{model_name}/yahma*/loss=*.ckpt"
                )
                if len(model_path) == 1:
                    model_path = model_path[0]
                else:
                    import numpy as np

                    # take the one with minimum loss
                    idx_min = np.argmin(
                        [
                            float(x.split("loss=")[-1].split(".ckpt")[0])
                            for x in model_path
                        ]
                    )
                    model_path = model_path[idx_min]
                model_name = model_name.replace("[]", "")
        print("#" * 100)
        print("loaded model fom ", model_path)
        print("#" * 100)
        # load state dict
        config = RoutingConfig()
        config.update_kwargs(torch.load(model_path)["hyper_parameters"])
        dm = AlpacaDataModule(config)
        dtype = torch.float32
        if dtype == "float16":
            dtype = torch.float16
        model = CLM.load_from_checkpoint(   
            model_path, tokenizer=dm.tokenizer, load_in_8bit=load_in_8bit, dtype=dtype, load_for_eval=True
        ).to("cuda")
        # model = upcast_to_dtype(model, torch.float32)
        config = model.hparams
        config.model_path = model_path
        config.model_name = model_name
    if wandb_proj:
        wandb_proj += f"_{config.dataset}"
        run_name = os.getenv("AMLT_JOB_NAME", f"{config.model}ni-nshot{0}")
        wandb.init(
            project=wandb_proj,
            name=os.environ.get("AMLT_JOB_NAME", run_name),
            config=config,
        )
    config.use_old_gen_config = use_old_gen_config
    config.data_dir = os.environ["NI_DATA_DIR"]
    config.predict_batch_size = batch_size
    config.output_dir = os.environ.get(
        "AMLT_OUTPUT_DIR",
        os.path.join(
            os.path.dirname(__file__),
            "..",
            f"../tmp/instruction_learning/{model_name}/",
        ),
    )
    rougel_ni_all, all_results_original = eval_ni(
        config, model, nshot=n_shots, subsample=subsample, max_input_length=-1
    )
    rougel_ni = rougel_ni_all["all"]["mean"]
    config.model_path = model_path
    if wandb_proj:
        wandb.log({"rouge_L_super_ni": rougel_ni_all["all"]["mean"]})
        wandb.log({"rouge_L_super_ni_stderr": rougel_ni_all["all"]["stderr"]})
        wandb.log(
            {"rouge_L_super_ni[original]": all_results_original["rougeL_default_track"]}
        )
    print(rougel_ni, all_results_original["rougeL_default_track"])


if __name__ == "__main__":
    run_ni_eval()
    # from huggingface_hub import login

    # login(token=os.environ["HF_TOKEN"])
    # config = RoutingConfig.parse(extra_kwargs={"eval_superni": True})
    # config.model = "meta-llama/Llama-2-13b-hf"
    # config.load_in_8bit = True
    # config.model_family = "gpt"
    # config.data_dir = os.environ["NI_DATA_DIR"]
    # config.predict_batch_size = 2
    # config.max_input_length = 4096
    # config.max_output_length = 128

    # model = AutoModelForCausalLM.from_pretrained(
    #     config.model, load_in_8bit=config.load_in_8bit, device_map="auto"
    # )
    # print(eval_ni(config, model, nshot=0, subsample=100))
