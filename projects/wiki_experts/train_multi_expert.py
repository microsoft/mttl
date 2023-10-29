import os
import re

import sys
import json
import wandb
import pytorch_lightning as pl

import torch
from huggingface_hub import login
from src.graph.module_graph import ModuleGraph
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
import json

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from mttl.callbacks import MMLUCallback
from mttl.evaluators import MMLUEvaluator
from mttl.datamodule.mmlu_data_module import MMLUDataModule
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from mttl.datamodule.oasst1_module import OA1Config, OA1Module
from mttl.datamodule.retrieval_lm_module import RetrievalLMDataModule
from mttl.datamodule.facts_lm_module import FactsLMConfig, FactsLMDataModule
from mttl.datamodule.platypus_module import (
    PlatypusModule,
    PlatypusConfig,
    PlatypusQAModule,
)
from projects.wiki_experts.src.expert_model import MultiExpertModel
from mttl.utils import get_mlf_logger, setup_logging, logger

from projects.wiki_experts.src.expert_trainer import ExpertTrainer
from projects.wiki_experts.src.config import ExpertConfig


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


def run_multitask(args):
    seed_everything(args.seed, workers=True)

    # get directory of the current file
    setup_logging(args.output_dir)

    logger.info("Args: {}".format(args.to_json()))

    if args.hf_token_hub:
        login(token=args.hf_token_hub)

    # select dataloader
    from mmlu_exper_merge_nevergrad import get_module_graph

    module_graph, subject_to_module = get_module_graph(args.module_graph)
    module_graph = re.sub(r"\$([a-zA-Z_][a-zA-Z0-9_]*)", "1", module_graph)
    graph = ModuleGraph.from_string(module_graph)

    # add MMLU val data to validaiton set
    dm = MMLUDataModule(args, for_generation=False, do_tokenize=True)
    args.n_tasks = len(dm.task_to_id) if hasattr(dm, "task_to_id") else 0
    dm.train_dataset = dm.dev_dataset
    # legit logging
    loggers = []
    exp_name = os.environ.get("AMLT_JOB_NAME", args.exp_name)
    # if os.environ.get("WANDB_API_KEY") or args.wandb_project:
    #     import wandb

    #     project = "wiki_experts" if args.wandb_project is None else args.wandb_project
    #     args.exp_name = "dev_run" if args.exp_name is None else args.exp_name
    #     project = os.environ.get("WANDB_PROJECT", project)
    #     wandb_logger = pl.loggers.WandbLogger(
    #         project=project,
    #         name=exp_name,  # , config=args_
    #         settings=wandb.Settings(start_method="fork"),
    #     )
    #     wandb_logger.experiment.save("*.py")
    #     loggers.append(wandb_logger)

    module = MultiExpertModel(**vars(args), tokenizer=dm.tokenizer)
    module.load_from_graph(graph, action="route")
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

    monitor = "train/loss"
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

    val_check_interval = args.eval_every
    if val_check_interval == -1:
        val_check_interval = None
    else:
        val_check_interval = args.gradient_accumulation_steps * args.eval_every
        if val_check_interval > len(dm.train_dataloader()):
            val_check_interval = len(dm.train_dataloader())
        elif val_check_interval > args.total_steps and args.total_steps != -1:
            val_check_interval = args.total_steps

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


if __name__ == "__main__":
    args = ExpertConfig.parse()
    run_multitask(args)
