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
from train_multi_expert_weights import run_m_weights_learning

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from projects.wiki_experts.lora_hub import RoutingOptimizer, mmlu_get_loss
from mttl.evaluators import MMLUEvaluator
from mttl.utils import setup_logging, logger
from projects.wiki_experts.src.graph.module_graph import ModuleGraph

# register models
from projects.wiki_experts.src.expert_model import MultiExpertModel
from projects.wiki_experts.src.config import ExpertConfig
from mttl.datamodule.mmlu_data_module import MMLUDataModule
from mttl.vllm_engines.engines import LLMEngineMMLU, free_memory


def log_wandb(scores, prefix):
    if wandb.run is not None:
        for t, v in scores.items():
            wandb.log({f"{prefix}/on_{t}/test_mmlu": v["mean"]})


def log_dict(dict):
    if wandb.run is not None:
        for k, v in dict.items():
            wandb.log({f"{k}": v})


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


def produce_transfer_matrix(args, subject_to_module):
    """
    Eval each module on each subject
    """
    transfer_table = {}
    use_vllm = True
    for module_for_subject, graph_to_eval in subject_to_module.items():
        result = {}
        for subject_eval_on, _ in subject_to_module.items():
            # select dataloader
            graph = graph_to_eval.substitute({f"weight_{module_for_subject}": 1.0})
            config_copy = copy.deepcopy(args)
            config_copy.finetune_task_name = subject_eval_on
            mmlu = MMLUEvaluator(
                config_copy, split=config_copy.mmlu_test_split, use_vllm=use_vllm
            )
            module = MultiExpertModel(
                **vars(config_copy),
                tokenizer=mmlu.datamodule.tokenizer,
                device_map="cpu" if use_vllm else "auto",
            )
            module.load_from_graph_string(graph, action="merge")
            scores = mmlu.evaluate(module)

            result[subject_eval_on] = scores[subject_eval_on]["mean"]
            all = scores.pop("all")
            log_wandb(scores, f"transfer/{module_for_subject}")
            logger.info(
                f"Scores on of {module_for_subject} for {subject_eval_on}:", all["mean"]
            )
            transfer_table[module_for_subject] = result
    transfer_matrix = pd.DataFrame.from_dict(transfer_table)
    if wandb.run is not None:
        tbl = wandb.Table(data=transfer_matrix)
        wandb.log({"transfer_matrix": tbl})
    try:
        del module
        free_memory()
    except:
        pass
    return transfer_matrix


def run_eval(args: ExpertConfig):
    seed_everything(args.seed, workers=True)
    wandb_logger = _setup_logging(args)
    loggers = [] if wandb_logger is None else [wandb_logger]
    if args.hf_token_hub:
        login(token=args.hf_token_hub)

    module_graph, subject_to_module = get_module_graph(args.module_graph)

    accuracy_matrix = {}
    for task in subject_to_module.keys():
        logger.info(f"Optimizing for {task} for {args.n_ng_iterations} iterations")

        config_copy = copy.deepcopy(args)
        config_copy.finetune_task_name = task
        config_copy.learning_rate = 5e-4
        config_copy.weight_decay = 0.09
        config_copy.num_train_epochs = 5
        config_copy.precision = "32-true"
        config_copy.trainable_param_names = ".*_merging_weights.*"

        module_graph = re.sub(r"\$([a-zA-Z_][a-zA-Z0-9_]*)", "1", module_graph)
        config_copy.module_graph = module_graph

        dm = MMLUDataModule(config_copy, for_generation=False, do_tokenize=True)
        dm.train_dataset = dm.dev_dataset
        weights, checkpoint = run_m_weights_learning(config_copy, dm, loggers=loggers)
        log_weights(weights, task)
        use_vllm = False
        module = MultiExpertModel.load_from_checkpoint(checkpoint)
        if not use_vllm:
            module = module.to("cuda")
        else:
            module = module.to("cpu")
        mmlu = MMLUEvaluator(
            config_copy, split=config_copy.mmlu_test_split, use_vllm=use_vllm
        )
        scores = mmlu.evaluate(module)
        accuracy_matrix[task] = {"best": scores["all"]["mean"]}
        scores.pop("all")
        log_wandb(scores, f"ng_optimal/{task}_optimal_graph/")
        logger.info(f"Scores on of {task}:{scores[task]['mean']}")
        free_memory()


if __name__ == "__main__":
    args = ExpertConfig.parse()
    run_eval(args)
