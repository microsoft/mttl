from transformers import AutoModelForCausalLM
import os
import torch
import sys
import os
import re
import glob
import click
import wandb    

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from projects.instr_routing.finetune_llama import RoutingConfig
from projects.instr_routing.models.clm import CLM
from mttl.datamodule.alpaca_data_module import AlpacaDataModule
from mttl.datamodule.flan100k_module import Flan100kModule
from mttl.datamodule.platypus_module import PlatypusModule
from eval_ni import load_hf_model

torch.set_float32_matmul_precision("high")


def eval_mmlu(
    config,
    model,
    data_dir=None,
    subsample=-1,
    max_input_length=None,
):
    from mttl.evaluators import MMLUEvaluator

    evaluator = MMLUEvaluator(
        config, data_dir=data_dir or config.data_dir, max_input_length=max_input_length
    )
    metrics = evaluator.evaluate(model, subsample=subsample)
    torch.cuda.empty_cache()
    return metrics


@click.command()  
@click.option("--model_name", type=str, default="platypus_13B")
@click.option("--amlt_experiment_name", type=str, default="routing")
@click.option(
    "--model_path",
    type=str,
    default="/home/v-oostapenko/results/platypus/platypus-13b-right/meta-llama_Llama-2-13b-hf_platypus-13b-right-val/loss=0.5543.ckpt",
)
# /home/v-oostapenko/dev/amlt/routing/platypus_vsmear_e8[xr4]/meta-llama_Llama-2-13b-hfpqtor60j_platypus_vsmear_e8[xr4]-val/loss=0.5533.ckpt
# /home/v-oostapenko/dev/amlt/routing/platypus_dense_er4/meta-llama_Llama-2-13b-hf284ifsm8_platypus_dense_er4-val/loss=0.5510.ckpt
#   /home/v-oostapenko/results/platypus/platypus-13b-right/meta-llama_Llama-2-13b-hf_platypus-13b-right-val/loss=0.5543.ckpt", help="path to the model")
@click.option("--batch_size", type=int, default=3)
@click.option("--wandb_proj", type=str, default="eval")
@click.option("--subsample", type=int, default=100)
@click.option("--load_in_8bit", type=bool, default=False)
@click.option("--dtype", type=str, default="float16")
def run_mmlu_eval(
    model_name,
    amlt_experiment_name=None,
    model_path=None,
    batch_size=5,
    wandb_proj=None,
    subsample=-1,
    load_in_8bit=None,
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
        # load state dict
        config = RoutingConfig()
        config.update_kwargs(torch.load(model_path)["hyper_parameters"])
        if config.dataset == "alpaca":
            dm = AlpacaDataModule(config)
        elif config.dataset == "platypus":
            dm = PlatypusModule(config)
        elif config.dataset == "flan100k":
            dm = Flan100kModule(config)
        else:
            raise NotImplementedError()
        if dtype == "float16":
            dtype = torch.float16
        else:
            dtype = torch.float32
        model = CLM.load_from_checkpoint(
            model_path, tokenizer=dm.tokenizer, load_in_8bit=load_in_8bit, dtype=dtype
        ).to("cuda")
        config = model.hparams
        config.model_path = model_path

    if wandb_proj:
        wandb_proj += f"_{config.dataset}"
        run_name = os.getenv("AMLT_JOB_NAME", f"{config.model}_mmlu")
        wandb.init(
            project=wandb_proj,
            name=os.environ.get("AMLT_JOB_NAME", run_name),
            config=config,
        )
        # log all the function arguments to wandb
        wandb.config.update(
            {
                "eval_model_path": model_path,
                "eval_model_name": model_name,
                "eval_batch_size": batch_size,
                "eval_subsample": subsample,
                "eval_load_in_8bit": load_in_8bit,
                "eval_dtype": dtype,
            }
        )

    config.predict_batch_size = batch_size
    config.max_output_length = 5
    config.data_dir = os.environ["MMLU_DATA_DIR"]
    config.output_dir = os.environ.get(
        "AMLT_OUTPUT_DIR",
        os.path.join(
            os.path.dirname(__file__),
            "..",
            f"../tmp/instruction_learning/{model_name}/",
        ),
    )
    em_mmlu_all = eval_mmlu(config, model, subsample=subsample)
    mmlu_em = em_mmlu_all["all"]["mean"]
    if wandb_proj:
        wandb.log({"mmlu_acc": mmlu_em})        
        data = [[label, val["mean"]] for (label, val) in em_mmlu_all.items()]
        table = wandb.Table(data=data, columns = ["task", "mean_acc"])
        wandb.log({"mmlu_per_task_acc" : wandb.plot.bar(table, "task", "mean_acc",
                                    title="mmlu_per_task_acc")})
    print(em_mmlu_all)


if __name__ == "__main__":
    run_mmlu_eval()
    os._exit(0)

    from huggingface_hub import login

    if "HF_TOKEN" in os.environ:
        login(token=os.environ["HF_TOKEN"])

    config = RoutingConfig.parse(extra_kwargs={"eval_superni": False})

    config.model = "meta-llama/Llama-2-7b-hf"
    config.load_in_8bit = True
    config.model_family = "gpt"
    config.data_dir = os.environ["MMLU_DATA_DIR"]
    config.predict_batch_size = 2
    config.max_input_length = 4096
    config.max_output_length = 5

    model = AutoModelForCausalLM.from_pretrained(
        config.model, load_in_8bit=config.load_in_8bit, device_map="auto"
    )
    print(eval_mmlu(config, model, subsample=10))
