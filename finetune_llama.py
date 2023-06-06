import os
import argparse
import json
import numpy as np
import torch
import pytorch_lightning as pl 
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from inst_follow.models.clm import CLM
from mttl.callbacks import ProgressCallback
from mttl.datamodule.alpaca_data_module import AlpacaDataModule
from mttl.datamodule.longform_data_module import LongFormDataModule
from mttl.models.encoder_decoder import EncoderDecoder
from mttl.models.t0_encoder_decoder import T0EncoderDecoder
from mttl.config import Config as MTTLConfig
from mttl.models.monitors import get_monitors
from mttl.utils import get_mlf_logger
from mttl.models.modify_model import modify_transformer
from transformers import LlamaForCausalLM, LlamaTokenizer
from mttl.dataloader.data_utils import ExampleInfo
from mttl.utils import get_ni_tasks_from_file, trim_batch, hash_example
from typing import List

# from peft import prepare_model_for_int8_training


# os.environ["AP_DATA_DIR"] = "/home/v-oostapenko/data" # temp for gcr
def remove_non_serializable(d):
    """
    Recursively remove non-JSON serializable values from a dictionary.
    """
    for k, v in d.items():
        if isinstance(v, (list, dict)):
            remove_non_serializable(v)
        elif not json.dumps(v, default=lambda x: None):
            del d[k]


class Config(MTTLConfig):
    def __init__(self, **kwargs):
        self.rank = 1  
        self.prune_unused_loras = True
        self.init_b_random = False
        self.lora_dropout = 0
        self.lora_alpha = 16
        self.load_in_8bit = False
        self.micro_batch_size = 4
        self.train_on_inputs = False
        self.padding_side = "right"
        self.adapter_modules = None
        self.poly_selector_use_distances = False
        self.adapter_layers = 0  # llama adapter
        self.adapter_len = 0  # llama adapter
        super().__init__(**kwargs)
        # to reproduce setup in https://github.com/daanelson/alpaca-lora
        self.gradient_accumulation_steps = (
            self.train_batch_size // self.micro_batch_size
        )
        self.train_batch_size = self.micro_batch_size


def parse_config(
    extra_kwargs=None, raise_error=True, parent=None, return_parser=False, c=None
):
    import itertools

    # dont do it if called from jupyter notebook
    if c is None:
        parser = (
            argparse.ArgumentParser(parents=[parent])
            if parent
            else argparse.ArgumentParser()
        )
        parser.add_argument("-c", "--config_files", required=False)
        parser.add_argument("-k", "--kwargs", nargs="*", action="append")
        args = parser.parse_args()
    else:
        args = argparse.Namespace()
        args.kwargs = None
        args.config_files = c
    kwargs = {}
    if args.kwargs:
        kwargs_opts = list(itertools.chain(*args.kwargs))
        for value in kwargs_opts:
            key, _, value = value.partition("=")
            kwargs[key] = value

    args.kwargs = kwargs
    if extra_kwargs:  
        args.kwargs.update(extra_kwargs)

    config = Config(
        filenames=args.config_files, kwargs=args.kwargs, raise_error=raise_error
    )

    print(config.to_json())
    if return_parser:
        return config, args
    return config


def run_multitask(args):
    seed_everything(args.seed, workers=True)
    # get directory of the current file
    print(os.path.dirname(os.path.realpath(__file__)))
    if args.example_to_ids_path:
        from mttl.cluster_tuning.cluster_reader import ClusterResult

        cluster_result = ClusterResult(args.example_to_ids_path)
        args.n_tasks = cluster_result.n_clusters()

        if args.poly_selector in ["cluster_soft", "cluster_hard"]:
            args.n_skills = cluster_result.n_clusters()
        else:
            raise NotImplementedError()

    # select dataloader
    model_class = CLM 
    if args.dataset == "alpaca":
        dm = AlpacaDataModule(args)
    elif args.dataset == "longform":        
        dm = LongFormDataModule(args)
    else:
        raise NotImplementedError()

    args.n_tasks = len(dm.task2id)
    args.model_object = LlamaForCausalLM.from_pretrained(
        args.model,
        # load_in_8bit=args.load_in_8bit, # this doesnt work right now with current implementatio of lora
        # torch_dtype=torch.float16,
        device_map="auto",
    )  # , load_in_8bit=True, torch_dtype=torch.float16) 
    if args.model_object.config.vocab_size != len(dm.tokenizer): #if adding [EOI] in LongForm dataset
        args.model_object.resize_token_embeddings(len(dm.tokenizer))
    args.model_object = modify_transformer(args.model_object, args)
    # if args.load_in_8bit:
    #     args.model_object = prepare_model_for_int8_training(args.model_object)

    if args.checkpoint is not None:
        from mttl.utils import get_checkpoint_path

        checkpoint_path = get_checkpoint_path(args.checkpoint)

        kwargs = vars(args)
        kwargs.pop("checkpoint")
        module = model_class.load_from_checkpoint(
            checkpoint_path, **kwargs, tokenizer=dm.tokenizer
        )
    else:
        module = model_class(**vars(args), tokenizer=dm.tokenizer)
        del args.model_object

    if args.poly_selector in ["cluster_soft", "cluster_hard"]:
        if args.n_skills > 1 and args.prune_unused_loras:
            # prune unused loras
            counts = m = np.bincount(cluster_result._instance.infos.cluster_ids)
            skill_ids_to_keep = np.where(
                np.bincount(cluster_result._instance.infos.cluster_ids) > 0
            )[0]
            module.model.remove_skills(skill_ids_to_keep)
            cluster_result.remove_skills(skill_ids_to_keep)

    # legit logging
    loggers = []
    if os.environ.get("WANDB_API_KEY"):
        # args_=args.__dict__.copy()
        # remove_non_serializable(args_)
        wandb_logger = pl.loggers.WandbLogger(
            project="alpaca_tuning",
            name=os.environ.get("AMLT_JOB_NAME", args.exp_name),  # , config=args_
        )
        wandb_logger.experiment.save("*.py")
        loggers.append(wandb_logger)
    else:
        wandb_logger = None

    mlf_logger = get_mlf_logger()
    if mlf_logger:
        loggers.append(mlf_logger)

    loggers.append(pl.loggers.CSVLogger(save_dir=args.output_dir, name="csv_metrics"))

    kwargs = {"val_check_interval": args.eval_every} if args.eval_every else {}

    # get metric monitors for models
    callbacks = get_monitors(args)
    callbacks.append(ProgressCallback())

    monitor = "val/loss"
    mode = "min"

    model_name = args.model.replace("/", "_")
    # check if wandb run exists
    if wandb_logger:
        # get run id
        run_id = wandb_logger.experiment.id
        model_name += run_id
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        monitor=monitor,
        filename=f"{model_name}" + f"_{args.exp_name}" + "-{" + monitor + ":.004f}",
        save_top_k=1,
        save_last=True,
        save_weights_only=True,  # make checkpoints smaller
        mode=mode,
    )
    callbacks.append(checkpoint_callback)

    trainer = Trainer(
        gpus=1,
        accelerator="gpu",
        logger=loggers,
        num_sanity_val_steps=5,
        amp_backend="native",
        default_root_dir=args.output_dir,
        max_epochs=args.num_train_epochs,   
        max_steps=args.total_steps + 1 if args.total_steps != -1 else -1,
        gradient_clip_val=args.max_grad_norm,
        log_every_n_steps=20,    
        strategy=args.compute_strategy if args.compute_strategy else None,
        callbacks=callbacks,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        precision=int(args.precision)
        if args.precision in ["16", "32"]
        else args.precision,
        **kwargs,
    )

    trainer.fit(module, dm)

    # try:
    trainer.validate(module, dm)
    # except:
    #     pass
    if args.dataset in ["ni", "xfit", "alpaca", "longform"]:
        best_model_path = trainer.checkpoint_callback.best_model_path
        print(f"Best model path: {best_model_path}")
        # Rename the file at best_model_path to 'best_model'
        path_best_model = args.output_dir + f"/best_model_{args.exp_name}"
        # create dir if not exists
        if not os.path.exists(path_best_model):
            os.makedirs(path_best_model)
        # copy the best model to the best_model dir
        os.system("cp " + best_model_path + " " + path_best_model + "/")


if __name__ == "__main__":
    args = parse_config()
    run_multitask(args)
