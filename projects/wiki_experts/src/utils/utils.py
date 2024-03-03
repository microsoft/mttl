import glob
import sys
import os
import re
import copy
import wandb
import prettytable
import numpy as np
import pandas as pd
from functools import partial
import pytorch_lightning as pl

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from mttl.utils import setup_logging, logger
from projects.wiki_experts.src.utils.evaluators import (
    TestLossEvaluator,
    ExtendedMMLUEvaluator,
    Evaluator,
    ExtendedRougeEvaluator,
)

from mttl.utils import logger
from mttl.models.modifiers.expert_containers.expert_library import (
    HFExpertLibrary,
    get_best_expert_for_score,
    get_best_expert_for_task,
)
from mttl.models.modifiers.expert_containers.expert import Expert


class TableLogger:
    def __init__(self):
        self.df = pd.DataFrame()

    def from_df(self, df):
        self.df = df
        self.columns = df.columns

    def log(self, row: dict):
        if self.df is None or len(self.df) == 0:
            self.df = pd.DataFrame(columns=row.keys())
        else:
            # Add new columns to the DataFrame if they don't exist
            new_columns = set(row.keys()) - set(self.df.columns)
            for column in new_columns:
                self.df[column] = np.nan
        self.df.loc[len(self.df.index)] = row

    def get_table(self):
        return self.df

    def means(self):
        # calculate mean for each row, column and diagonal of self.df
        # filter numeric columns
        df_numeric = self.df.select_dtypes(include=[np.number])
        self.df["mean"] = df_numeric.mean(axis=1)
        self.df.loc["mean"] = df_numeric.mean(axis=0)
        self.df.loc["mean", "mean"] = np.diag(df_numeric).mean()

    def log_final_table(self):
        if wandb.run is not None:
            wandb.log({"table": wandb.Table(data=self.get_table())})
        table = prettytable.PrettyTable()
        table.field_names = list(self.df.columns)
        for i, row in self.df.iterrows():
            table.add_row(list(row))
        logger.info("Results:\n" + str(table))


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


def get_svd_embedding(lib, expert_name: str):
    try:
        embeddings = lib.get_auxiliary_data(
            data_type="embeddings", expert_name=expert_name
        )
    except ValueError:
        return None
    return embeddings["svd"]["embeddings"]


def init_wandb_logger(args):
    if args.wandb_project is None:
        args.wandb_project = os.environ.get("WANDB_PROJECT", "MMLU_ninja_merge")
    if args.wandb_project:
        exp_name = os.getenv("AMLT_JOB_NAME", f"{args.exp_name}")
        # wandb.init(
        #     project=args.wandb_project,
        #     name=exp_name,
        #     config=args,
        # )
        logger = pl.loggers.WandbLogger(
            project=args.wandb_project,
            name=exp_name,
            config=args,
        )
    return logger


def log_wandb(scores, prefix):
    if wandb.run is not None:
        for t, v in scores.items():
            wandb.log({f"{prefix}_on_{t}_test_mmlu": v["mean"]})


def get_task_expert(task, expert_lib, default_score):
    """
    Get the best expert for a given task.

    Args:
        task (str): The task for which to find the expert.
        expert_lib (ExpertLibrary): The library of available experts.
        default_score (Score): Score to use for expert retrieval.

    Returns:
        Expert: The best expert for the given task according to the score.
    Raises:
        ValueError: If no default score is provided.
    """
    if default_score is None:
        raise ValueError("No default score provided")
    parent_exp: Expert = get_best_expert_for_score(expert_lib, default_score.hash)
    if parent_exp is None and task in expert_lib.tasks:
        parent_exp = get_best_expert_for_task(expert_lib, task, default_score.hash)
    return parent_exp
