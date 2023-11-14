import os
import re

import sys
import json
import torch
import pytorch_lightning as pl
from huggingface_hub import login
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
import json

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


from projects.wiki_experts.src.expert_trainer import ExpertTrainer
from projects.wiki_experts.src.graph.module_graph import ModuleGraph
from mttl.datamodule.mmlu_data_module import MMLUDataModule
from projects.wiki_experts.src.expert_model import (
    MultiExpertModel,
    RoutedMultiExpertModel,
)
from mttl.utils import get_mlf_logger, setup_logging, logger
from projects.wiki_experts.src.config import ExpertConfig
from projects.wiki_experts.experts_merge.config import ExpertsMergeConfig
from typing import List


class SimpleLogger(pl.loggers.logger.DummyLogger):
    def __init__(self, output_dir):
        self.metrics = {}
        self.output_file = os.path.join(output_dir, "metrics.json")

    def log_metrics(self, metrics, step=None):
        for k, v in metrics.items():
            if k not in self.metrics:
                self.metrics[k] = []
            self.metrics[k].append({"step": step, "value": v})
        with open(self.output_file, "w") as f:
            json.dump(self.metrics, f)


def save_new_module(module_copy, args):
    # make Loras trainable
    module_copy.trainable_param_names = [
        n for n, p in module_copy.named_parameters() if re.match(".*lora.*", n)
    ]
    checkpoint = module_copy.save_pretrained(args.output_dir)
    return checkpoint


def finetune_expert(
    args: ExpertsMergeConfig, dm, module_dest, val_check_interval, loggers: List = []
):
    seed_everything(args.seed, workers=True)
    # get directory of the current file
    setup_logging(args.output_dir)
    logger.info("Args: {}".format(args.to_json()))

    if args.hf_token_hub:
        login(token=args.hf_token_hub)

    # legit logging
    exp_name = os.environ.get("AMLT_JOB_NAME", args.exp_name)
    if len(loggers) == 0:
        loggers = []
        if os.environ.get("WANDB_API_KEY") or args.wandb_project:
            import wandb

            project = (
                "wiki_experts" if args.wandb_project is None else args.wandb_project
            )
            args.exp_name = "dev_run" if args.exp_name is None else args.exp_name
            project = os.environ.get("WANDB_PROJECT", project)
            wandb_logger = pl.loggers.WandbLogger(
                project=project,
                name=exp_name,  # , config=args_
                settings=wandb.Settings(start_method="fork"),
            )
            wandb_logger.experiment.save("*.py")
            loggers.append(wandb_logger)

    module = ExpertTrainer(**vars(args), tokenizer=dm.tokenizer, device_map="auto")
    module.from_pretrained(module_dest)
    module.to("cuda")
    ##############################

    mlf_logger = get_mlf_logger()
    if mlf_logger:
        loggers.append(mlf_logger)

    if args.tensorboard:
        tb_logger = pl.loggers.TensorBoardLogger(save_dir=args.output_dir)
        loggers.append(tb_logger)

    loggers.append(SimpleLogger(args.output_dir))

    # get metric monitors for models
    callbacks = []

    monitor = "val/loss"
    mode = "min"
    model_name = args.model.replace("/", "_")

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        monitor=monitor,
        filename=f"{model_name}" + "-{" + monitor + ":.004f}",
        save_top_k=1,
        save_last=True,
        save_weights_only=True,  # make checkpoints smaller
        mode=mode,
    )
    callbacks.append(checkpoint_callback)

    trainer = Trainer(
        devices=-1,
        accelerator="gpu",
        logger=loggers,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        default_root_dir=args.output_dir,
        max_epochs=args.num_train_epochs,
        max_steps=args.total_steps + 1 if args.total_steps != -1 else -1,
        gradient_clip_val=args.max_grad_norm,
        strategy=args.compute_strategy if args.compute_strategy else "auto",
        callbacks=callbacks,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        precision=int(args.precision)
        if args.precision in ["16", "32"]
        else args.precision,
        val_check_interval=val_check_interval,
    )

    # initial validation!
    trainer.fit(module, dm)

    checkpoint = (
        checkpoint_callback.best_model_path or checkpoint_callback.last_model_path
    )
    del module
    torch.cuda.empty_cache()
    return checkpoint


if __name__ == "__main__":
    args = ExpertsMergeConfig.parse()
    from utils import get_module_graph

    graph, module_dict, subjects = get_module_graph(args)

    # add MMLU val data to validaiton set
    dm = MMLUDataModule(args, for_generation=False)
    args.n_tasks = len(dm.task_to_id) if hasattr(dm, "task_to_id") else 0
    args.num_train_epochs = 1
    dm.train_dataset = dm.dev_dataset

    weights, checkpoint = finetune_expert(args, dm=dm, graph=graph)
    print(weights, checkpoint)
