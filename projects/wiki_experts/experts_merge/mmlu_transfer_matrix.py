import os
import sys
import glob
import copy
import torch
import wandb
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from functools import partial
from huggingface_hub import login
from collections import defaultdict
from pytorch_lightning import seed_everything
from utils import get_module_graph

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from mttl.evaluators import MMLUEvaluator
from mttl.utils import setup_logging, logger

# register models
from projects.wiki_experts.src.expert_model import MultiExpertModel
from projects.wiki_experts.src.config import ExpertConfig
from mttl.datamodule.mmlu_data_module import MMLUDataModule
from mttl.vllm_engines.engines import LLMEngineMMLU, free_memory


def log_wandb(scores, prefix):
    if wandb.run is not None:
        for t, v in scores.items():
            wandb.log({f"{prefix}_on_{t}_test_mmlu": v["mean"]})


def init_wandb_logger(args):
    if args.wandb_project is None:
        args.wandb_project = os.environ.get("WANDB_PROJECT", "MMLU_ninja_merge")
    if args.wandb_project:
        run_name = os.getenv("AMLT_JOB_NAME", f"{args.model}")
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=args,
        )


def _setup_logging(args):
    logger.info("Args: {}".format(args.to_json()))
    setup_logging(args.output_dir)
    init_wandb_logger(args)


def produce_transfer_matrix(args, subject_to_module, use_vllm=True):
    """
    Eval each module on each subject
    """
    transfer_table = {}
    for module_for_subject, module_dest in subject_to_module.items():
        result = {}
        for subject_eval_on, _ in subject_to_module.items():
            # select dataloader
            graph = f"{module_for_subject} -> linear({module_dest}:1.0)"
            config_copy = copy.deepcopy(args)
            config_copy.finetune_task_name = subject_eval_on
            config_copy.module_graph = None
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
                f"Scores on of {module_for_subject} for {subject_eval_on}: {all['mean']}"
            )
            transfer_table[module_for_subject] = result
    transfer_matrix = pd.DataFrame.from_dict(transfer_table)
    if wandb.run is not None:
        tbl = wandb.Table(data=transfer_matrix)
        wandb.log({"transfer_matrix": tbl})
        ax = sns.heatmap(transfer_matrix, annot=True, linewidth=0.5)
        ax.figure.tight_layout()
        wandb.log({"transfer_matrix_heatmap": wandb.Image(ax.get_figure())})
    try:
        del module
        free_memory()
    except:
        pass
    plt.clf()
    return transfer_matrix


def run_eval(args: ExpertConfig):
    seed_everything(args.seed, workers=True)
    _setup_logging(args)
    if args.hf_token_hub:
        login(token=args.hf_token_hub)
    _, subject_to_module = get_module_graph(args.module_graph)
    transfer_matrix: pd.DataFrame = produce_transfer_matrix(args, subject_to_module)
    print("Transfer matrix", transfer_matrix)


if __name__ == "__main__":
    args = ExpertConfig.parse()
    run_eval(args)
