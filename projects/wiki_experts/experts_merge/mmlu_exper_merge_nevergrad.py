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
from pytorch_lightning import seed_everything
from utils import get_module_graph

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


def run_eval(args: ExpertConfig):
    seed_everything(args.seed, workers=True)
    _setup_logging(args)
    if args.hf_token_hub:
        login(token=args.hf_token_hub)
    use_vllm = args.use_vllm
    _, module_2_dest = get_module_graph(args.module_graph)

    # 1. How good is the merging optimization procedure? Can we find a routing that is equivalent or better than oracle? (How does it compare to join training?)

    # we use the test-sets of each of the modules in the population and see if we can find the right routing or perform better than the oracle
    # we directly use tes-set for search, i.e. its an oracle!
    args.module_graph = None
    dm = MMLUDataModule(args, for_generation=use_vllm, do_tokenize=not use_vllm)
    module = MultiExpertModel(
        **vars(args), tokenizer=dm.tokenizer, device_map="cpu" if use_vllm else "auto"
    )
    get_loss_function = partial(mmlu_get_loss, use_vllm=use_vllm)
    best_weights_matrix = {}
    accuracy_matrix = {}
    for task in module_2_dest.keys():
        logger.info(f"Optimizing for {task} for {args.n_ng_iterations} iterations")
        config_copy = copy.deepcopy(args)
        config_copy.finetune_task_name = task
        dm = MMLUDataModule(
            config_copy, for_generation=use_vllm, do_tokenize=not use_vllm
        )

        optimizer = RoutingOptimizer(
            model=module,
            modules_2_dest=module_2_dest,
            dataloader=dm.test_dataloader(),
            get_loss=get_loss_function,
            budget=config_copy.n_ng_iterations,
        )
        best_weights, best_graph_string = optimizer.optimize()
        best_weights = best_weights.tolist()
        logger.info("Found best weights: {}".format(best_weights))
        logger.info("Found best graph: {}".format(best_graph_string))
        if wandb.run is not None:
            wandb.log(
                {
                    f"best_weight/{task}:{t}": v
                    for t, v in zip(module_2_dest.keys(), best_weights)
                }
            )
            plt.clf()
            ax = sns.barplot(x=list(module_2_dest.keys()), y=best_weights)
            # turn x labels
            ax.set_xticklabels(
                ax.get_xticklabels(), rotation=45, horizontalalignment="right"
            )
            ax.figure.tight_layout()
            ax.set_title(f"Best weights for {task}")
            wandb.log({f"best_weight_{task}": wandb.Image(ax.get_figure())})
        best_weights_matrix[task] = {
            t: v for t, v in zip(module_2_dest.keys(), best_weights)
        }
        # test model with these weights
        graph = ModuleGraph.from_string(best_graph_string)
        model_copy = copy.deepcopy(module)
        model_copy.load_from_graph(graph, action="merge")
        mmlu = MMLUEvaluator(
            config_copy, split=config_copy.mmlu_test_split, use_vllm=True
        )
        scores = mmlu.evaluate(model_copy)

        accuracy_matrix[task] = {"best": scores["all"]["mean"]}
        scores.pop("all")
        log_wandb(
            scores, f"ng_optimal/{task}_optimal_graph/"
        )  # TODO: log this also as a table and potentially as a barchart
        logger.info(
            f"Scores on of {task} with graph {best_graph_string}:{scores[task]['mean']}"
        )
        del model_copy
        plt.clf()
        free_memory()

    accuracy_matrix = pd.DataFrame.from_dict(accuracy_matrix)
    best_weights_matrix = pd.DataFrame.from_dict(best_weights_matrix)
    if wandb.run is not None:
        tbl_bw = wandb.Table(data=best_weights_matrix)
        tbl_acc = wandb.Table(data=accuracy_matrix)
        wandb.log({"best_weights_matrix": tbl_bw})
        wandb.log({"ng_optimal/accuracy_matrix": tbl_acc})
        ax = sns.heatmap(best_weights_matrix, annot=True, linewidth=0.5)
        ax.figure.tight_layout()
        wandb.log({"best_weights_matrix_heatmap": wandb.Image(ax.get_figure())})
    plt.clf()

    # We can do:
    #   - in-distribution evaluation: test sets we consider are the test sets of the tasks we have experts for
    #   - out-of-distribution evaluation: new task

    # Questions:
    # 1. How good is the merging optimization procedure?
    # On a the in-domain val-set of one of the modules in the population, can it converge to the right routing? (run this for each of the 10 test sets)
    # Does it attain perofrmance like the in-domain module? Could it find this module? if not, did it find a better combination?
    # How does it compare to join-training?

    # 2. How well can we generalize to new task? the baseline here is using jointly pre-trained model vs. merging the experts
    # If I could now tain on the new task a bit, is it bette to use as innitialization the merged pexpert vs. jointl pre-trained?

    # Given the modules lets first eval all of them on each other's test sets -> get a tansfe matix
    #

    # Then for each of the subjects for which we have the module, we optimize the merging procedure and see if we can get the right routing.
    # Can we get beyong expert performance with the right routing? The right module is there in the population.


if __name__ == "__main__":
    args = ExpertConfig.parse()
    run_eval(args)
