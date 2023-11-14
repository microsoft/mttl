import os
import re
import sys
import glob
import copy
import torch
import wandb
import numpy as np
import pandas as pd
import seaborn as sns
import pytorch_lightning as pl
from matplotlib import pyplot as plt
from functools import partial
from huggingface_hub import login
from pytorch_lightning import seed_everything
from utils import get_module_graph
from projects.wiki_experts.experts_merge.train_multi_expert_router import (
    run_m_weights_learning,
)

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from projects.wiki_experts.experts_merge.config import ExpertsMergeConfig
from mttl.evaluators import MMLUEvaluator
from mttl.utils import setup_logging, logger
from projects.wiki_experts.src.graph.module_graph import ModuleGraph

# register models
from projects.wiki_experts.src.expert_model import (
    MultiExpertModel,
    RoutedMultiExpertModel,
)
from projects.wiki_experts.src.config import ExpertConfig
from mttl.datamodule.mmlu_data_module import MMLUDataModule
from mttl.vllm_engines.engines import LLMEngineMMLU, free_memory

from utils import (
    get_loss,
    get_module_graph,
    init_wandb_logger,
    log_wandb,
    save_new_module,
    prepare_evaluator,
)


def log_weights(weights: dict, task):
    if wandb.run is not None:
        df = pd.DataFrame.from_dict(weights, orient="index")
        tbl = wandb.Table(data=df)
        wandb.log({f"weights/{task}": tbl})

        for layer, dist in weights.items():
            plt.clf()
            ax = sns.barplot(x=list(dist.keys()), y=list(dist.values()))
            # rotate labels by 45 degrees
            ax.set_xticklabels(
                ax.get_xticklabels(), rotation=45, horizontalalignment="right"
            )
            ax.set_title(f"Task {task} Layer {layer}")
            ax.figure.tight_layout()
            wandb.log({f"weights/{task}_{layer}": wandb.Image(ax)})
            plt.clf()
        # average weights
        df_mean = df.mean(axis=0)
        ax = sns.barplot(x=list(df_mean.keys()), y=list(df_mean.values))
        ax.set_xticklabels(
            ax.get_xticklabels(), rotation=45, horizontalalignment="right"
        )
        ax.set_title(f"Task: {task} Layer: {layer}")
        ax.figure.tight_layout()
        wandb.log({f"weights/{task}_average": wandb.Image(ax)})


def init_wandb_logger(args):
    logger = None
    if args.wandb_project is None:
        args.wandb_project = os.environ.get("WANDB_PROJECT", "MMLU_ninja_merge")
    if args.wandb_project:
        run_name = os.getenv("AMLT_JOB_NAME", f"{args.model}")
        logger = pl.loggers.WandbLogger(
            project=args.wandb_project,
            name=run_name,
            config=args,
        )
    return logger


def _setup_logging(args):
    wandb_logger = None
    logger.info("Args: {}".format(args.to_json()))
    setup_logging(args.output_dir)
    wandb_logger = init_wandb_logger(args)
    return wandb_logger


def run_eval(args: ExpertsMergeConfig):
    seed_everything(args.seed, workers=True)
    wandb_logger = _setup_logging(args)
    loggers = [] if wandb_logger is None else [wandb_logger]
    if args.hf_token_hub:
        login(token=args.hf_token_hub)

    _, module_dict, subjects = get_module_graph(args)

    accuracy_matrix = {}
    for task in subjects:
        logger.info(f"Optimizing for {task} for {args.n_ng_iterations} iterations")
        # TODO: lr line search with backtracking

        config_copy = copy.deepcopy(args)
        config_copy.finetune_task_name = task
        config_copy.learning_rate = 5e-2
        config_copy.weight_decay = 0.09
        config_copy.trainable_param_names = ".*module_logits.*|.*selector.*"
        config_copy.warmup_steps = 0

        dm = MMLUDataModule(config_copy, for_generation=False)
        dm.train_dataset = dm.dev_dataset
        weights, checkpoint = run_m_weights_learning(
            config_copy, dm, module_dict=module_dict, loggers=loggers
        )
        log_weights(weights, task)

        config_copy.model_modifier = "lora"
        module = MultiExpertModel(
            **vars(config_copy), tokenizer=dm.tokenizer, device_map="cpu"
        )
        module.load_from_module_dict({task: checkpoint})
        # eval expert
        if not args.use_vllm:
            module = module.to("cuda")
        else:
            module = module.to("cpu")
        mmlu = MMLUEvaluator(
            config_copy, split=config_copy.mmlu_test_split, use_vllm=args.use_vllm
        )
        scores = mmlu.evaluate(module)
        accuracy_matrix[task] = {"best": scores["all"]["mean"]}
        scores.pop("all")
        log_wandb(scores, f"ng_optimal/{task}_optimal_graph/")
        logger.info(f"Scores on of {task}:{scores[task]['mean']}")
        free_memory()


if __name__ == "__main__":
    args = ExpertsMergeConfig.parse()
    run_eval(args)
