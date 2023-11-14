import glob
import sys
import os
import re
import copy
import wandb
import numpy as np
import pandas as pd
import pytorch_lightning as pl

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from mttl.utils import setup_logging, logger
from mttl.datamodule.base import AutoDataModule
from config import ExpertsMergeConfig

from evaluators import (
    TestLossEvaluator,
    ExtendedMMLUEvaluator,
    Evaluator,
    ExtendedRougeEvaluator,
)

from mttl.utils import logger


class TableLogger:
    """
    score_base -- score before optimization roung on the module of task
    score_train -- score after optimization round on the module of task, on the train set
    score_test -- score after optimization round on the module of task, on the test set
    """

    def __init__(self, columns: list = None):
        self.columns = (
            [
                "act_i",
                "task",
                "weights",
                "score_base_test",
                "score_base_train",
                "score_base_valid",
                "score_train",
                "score_test",
                "score_valid",
                "score_train_max",
                "score_test_max",
                "score_valid_max",
                "score_test_selected",
                "score_test_fine_tuned",
                "score_valid_fine_tuned" "score_test_selected_fine_tuned",
            ]
            if columns is None
            else columns
        )
        self.df = pd.DataFrame(columns=self.columns)

    def log(self, row: dict):
        self.df.loc[len(self.df.index)] = row

    def get_table(self):
        return self.df

    def log_table_wandb(self):
        if wandb.run is not None:
            wandb.log({"table": wandb.Table(data=self.get_table())})


def get_loss(model, evaluator: Evaluator, **kwargs):
    return evaluator.get_loss(model, **kwargs)


def save_new_module(output_dir, module, task_name, score):
    module_copy = copy.deepcopy(module)
    # make Loras trainable so that they are saved
    module_copy.trainable_param_names = [
        n for n, p in module_copy.named_parameters() if re.match(".*lora.*", n)
    ]
    dest = output_dir + f"/{task_name}_optimal_weights_{score}"
    os.makedirs(dest, exist_ok=True)
    ckpt_path = module_copy.save_pretrained(dest)
    del module_copy
    return ckpt_path


def prepare_evaluator(
    args: ExpertsMergeConfig, dataset, tasks, split="test", subsample=-1
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
    evaluator = EVAL_CLASS(datamodule=dm, subsample=subsample, name=tasks, split=split)
    return evaluator


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
