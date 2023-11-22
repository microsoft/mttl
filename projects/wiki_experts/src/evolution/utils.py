import glob
import sys
import os
import re
import copy
import wandb
import numpy as np
import pandas as pd
from functools import partial
import pytorch_lightning as pl

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from mttl.utils import setup_logging, logger
from mttl.datamodule.base import AutoDataModule
from projects.wiki_experts.src.evolution.config import ExpertsMergeConfig

from projects.wiki_experts.src.evolution.evaluators import (
    TestLossEvaluator,
    ExtendedMMLUEvaluator,
    Evaluator,
    ExtendedRougeEvaluator,
)

from mttl.utils import logger


class TableLogger:
    def __init__(self):
        self.df = pd.DataFrame()

    def from_df(self, df):
        self.df = df
        self.columns = df.columns

    def log(self, row: dict):
        if self.df is None or len(self.df) == 0:
            self.df = pd.DataFrame(columns=row.keys())
        self.df.loc[len(self.df.index)] = row

    def get_table(self):
        return self.df

    def log_table_wandb(self):
        if wandb.run is not None:
            wandb.log({"table": wandb.Table(data=self.get_table())})


def get_loss(model, evaluator: Evaluator, **kwargs):
    return evaluator.get_loss(model, **kwargs)


def save_new_module(output_dir, module, task_name, postfix=""):
    module_copy = copy.deepcopy(module)
    # make Loras trainable so that they are saved
    module_copy.trainable_param_names = [
        n for n, p in module_copy.named_parameters() if re.match(".*lora.*", n)
    ]
    dest = output_dir + f"/{task_name}_{postfix}"
    os.makedirs(dest, exist_ok=True)
    ckpt_path = module_copy.save_pretrained(dest)
    del module_copy
    return ckpt_path


def prepare_evaluator(
    args: ExpertsMergeConfig, dataset, tasks, split=None, subsample=-1
):
    if args.eval_metric == "loss":
        EVAL_CLASS = TestLossEvaluator
        for_generation = False
    elif args.eval_metric == "rougeL":
        EVAL_CLASS = ExtendedRougeEvaluator
        for_generation = True
    elif args.eval_metric == "acc":
        assert "mmlu" in dataset
        EVAL_CLASS = ExtendedMMLUEvaluator
        for_generation = True
    else:
        raise ValueError(f"Unknown eval metric {args.eval_metric}")

    dm = AutoDataModule.create(
        name=dataset,
        for_generation=for_generation,
        model=args.model,
        model_family=args.model_family,
        validation_portion=0.0,
        finetune_task_name=tasks,
        train_batch_size=args.train_batch_size,
        predict_batch_size=args.predict_batch_size,
    )
    if split is not None:
        evaluator = EVAL_CLASS(
            datamodule=dm,
            subsample=subsample,
            name=tasks,
            split=split,
            use_vllm=args.use_vllm,
        )
        return evaluator
    return partial(
        EVAL_CLASS,
        datamodule=dm,
        subsample=subsample,
        name=tasks,
        use_vllm=args.use_vllm,
    )


def init_wandb_logger(args):
    if args.wandb_project is None:
        args.wandb_project = os.environ.get("WANDB_PROJECT", "MMLU_ninja_merge")
    if args.wandb_project:
        run_name = os.getenv("AMLT_JOB_NAME", f"{args.model}")
        # wandb.init(
        #     project=args.wandb_project,
        #     name=run_name,
        #     config=args,
        # )
        logger = pl.loggers.WandbLogger(
            project=args.wandb_project,
            name=run_name,
            config=args,
        )
    return logger


def log_wandb(scores, prefix):
    if wandb.run is not None:
        for t, v in scores.items():
            wandb.log({f"{prefix}_on_{t}_test_mmlu": v["mean"]})
