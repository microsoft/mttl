import os
import sys
import json
import glob
import click
import torch
import wandb 
import re
import logging
import pytorch_lightning as pl
from huggingface_hub import login  
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from mttl.callbacks import MMLUCallback, MiniProgress
from mttl.datamodule.alpaca_data_module import AlpacaDataModule
from mttl.datamodule.platypus_module import PlatypusModule
from mttl.datamodule.flan100k_module import Flan100kModule
from mttl.utils import get_mlf_logger, setup_logging, logger
from mttl.dist_utils import is_main_process
torch.set_float32_matmul_precision('high')

# register models
import models.vsmear  # noqa: F401
import models.softmoe # noqa: F401
from models.monitors import SelectorMetricsLog, SelectorRoutingsLog
from models.clm import CLM
from config import RoutingConfig


def run_multitask(args, module):  
    seed_everything(args.seed, workers=True)

    if args.hf_token_hub:
        login(token=args.hf_token_hub)

    if args.example_to_ids_path:
        raise NotImplementedError()

    # select dataloader
    model_class = CLM
    if args.dataset == "alpaca":
        dm = AlpacaDataModule(args)
    elif args.dataset == "platypus":
        dm = PlatypusModule(args)
    elif args.dataset == "flan100k":
        dm = Flan100kModule(args)
    else:
        raise NotImplementedError()
    
    # module = CLM(**vars(args), tokenizer=dm.tokenizer)
    # # save state dict
    # import numpy as np
    # torch.save(module.state_dict(), "model.pt")
    # print("params sum", np.sum([torch.sum(p.detach().cpu()) for p in module.parameters()]))
    # module.load_state_dict(torch.load("model.pt"))
    # print("params sum after", np.sum([torch.sum(p.detach().cpu()) for p in module.parameters()]))
    
    trainer = Trainer(
        devices=-1, 
        accelerator="gpu",   
        num_sanity_val_steps=5,  
        default_root_dir=args.output_dir,
        max_epochs=0,
        max_steps=-1,       
        gradient_clip_val=args.max_grad_norm,
        log_every_n_steps=20,
        strategy=args.compute_strategy if args.compute_strategy else "auto",
        accumulate_grad_batches=args.gradient_accumulation_steps,
        precision=int(args.precision)
        if args.precision in ["16", "32"]
        else args.precision,
        fast_dev_run=0,
    ) 
    trainer.fit(module, dm)

    path_best_model = args.model_path
    ckpt_path = path_best_model  
    # trainer.validate(dataloaders=dm, model=module)
    print("Validation with checkpoint", ckpt_path)      
    trainer.validate(dataloaders=dm, ckpt_path=ckpt_path)
           
@click.command()  
@click.option("--model_name", type=str, default="alpaca_dense_r4") #alpaca_dense_r4") #alpaca_vsmear_e12[xr4,t_1]")
@click.option("--amlt_experiment_name", type=str, default="routing")
@click.option("--model_path", type=str, default="/home/v-oostapenko/dev/amlt/shared_files/results_as_sep10/platypus/platypus-13b-right/meta-llama_Llama-2-13b-hf_platypus-13b-right-val/loss=0.5543.ckpt", help="path to the model")
def run_eval(
    model_name, 
    amlt_experiment_name=None,
    model_path=None,
):      
    if model_path is None:
        if os.environ.get("AMLT_OUTPUT_DIR") is not None:  # on gcr
            base_model_path = "/mnt/default/data/models"
        else:
            base_model_path = os.path.join(os.path.dirname(__file__), "..", "..", "..")
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
                idx_min = np.argmin([float(x.split("loss=")[-1].split(".ckpt")[0]) for x in model_path])
                model_path = model_path[idx_min]
            model_name = model_name.replace("[]","")
            
    # load state dict
    config = RoutingConfig()
    config.update_kwargs(torch.load(model_path)["hyper_parameters"])
    dm = AlpacaDataModule(config)
    model = CLM.load_from_checkpoint(model_path, tokenizer=dm.tokenizer).cuda()
    config_loaded = RoutingConfig()
    config_loaded.update_kwargs(model.hparams)
    config_loaded.output_dir = os.environ.get(
        "AMLT_OUTPUT_DIR",
        os.path.join(  
            os.path.dirname(__file__),
            "..",
            f"../../tmp/instruction_learning/",
        ),
    )
    config_loaded.model_path = model_path
    run_multitask(config_loaded, model)

if __name__ == "__main__":
    run_eval()
    # args = RoutingConfig.parse()
    # run_multitask(args)
