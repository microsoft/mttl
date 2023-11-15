import os
import sys
import re
import json
import copy
import torch
import wandb
import numpy as np
import pandas as pd
import seaborn as sns
from functools import partial
from matplotlib import pyplot as plt
from huggingface_hub import login
from pytorch_lightning import seed_everything
from utils import (
    get_loss,
    init_wandb_logger,
    log_wandb,
    save_new_module,
    prepare_evaluator,
    TableLogger,
)

from evaluators import Evaluator
from projects.wiki_experts.src.evolution.expert_library import ExpertLibrary

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from dataclasses import dataclass
from config import ExpertsMergeConfig
from projects.wiki_experts.src.evolution.lora_hub import RoutingOptimizer
from mttl.utils import setup_logging, logger
from projects.wiki_experts.src.graph.module_graph import ModuleGraph

# register models
from projects.wiki_experts.src.expert_model import MultiExpertModel

# from projects.wiki_experts.src.expert_trainer import ExpertTrainer
# from mttl.datamodule.mmlu_data_module import MMLUDataModule
from mttl.vllm_engines.engines import free_memory


class ExperimentState:
    @dataclass
    class State:
        config: ExpertsMergeConfig
        active_iteration: int
        expert_lib: ExpertLibrary
        results_table: TableLogger

    def __init__(self, **kwargs):
        self.state = self.State(**kwargs)

    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self.state, k, v)

    @property
    def path(self):
        if wandb.run is not None:
            run_name = wandb.run.name
        else:
            run_name = os.getenv("AMLT_JOB_NAME", "_some_experiment")
        run_name = run_name.replace("/", "_")
        path = self.state.config.output_dir
        path = os.path.join(path, f"exp_state_{run_name}")

        os.makedirs(path, exist_ok=True)
        return path

    def save(self, path=None):
        path = path or self.path
        if not path.endswith(".pt"):
            path = os.path.join(path, "state.pt")

        state = copy.deepcopy(self.state)
        torch.save(state, path)

    def load_from_path(self, path=None):
        path = path or self.path
        if not path.endswith(".pt"):
            path = os.path.join(path, "state.pt")
        state = torch.load(path)
        self.state = state

    def tasks_in_active_iteration(self, aci):
        return self.state.results_table.tasks_in_active_iteration(aci)


def log_best_weights(module_dict, best_weights, task, prefix=""):
    if wandb.run is not None:
        wandb.log(
            {
                f"{prefix}best_weight/{task}:{t}": v
                for t, v in zip(module_dict.keys(), best_weights)
            }
        )
        plt.clf()
        _size = 1 * len(list(module_dict.keys()))
        plt.figure(figsize=(_size, _size))
        ax = sns.barplot(x=list(module_dict.keys()), y=best_weights)
        # turn x labels
        ax.set_xticklabels(
            ax.get_xticklabels(), rotation=45, horizontalalignment="right"
        )
        ax.set_title(f"{prefix}Best weights for {task}")
        ax.figure.tight_layout()
        wandb.log({f"{prefix}best_weight_{task}": wandb.Image(ax.get_figure())})
        plt.clf()


def run_eval(args: ExpertsMergeConfig):
    seed_everything(args.seed, workers=True)
    setup_logging(args.output_dir)
    wandb_logger = init_wandb_logger(args)
    module = None
    if args.hf_token_hub:
        login(token=args.hf_token_hub)

    exp_state = ExperimentState(
        config=args,
        active_iteration=0,
        expert_lib=ExpertLibrary(model_name=args.model, modules_dir=args.modules_dir),
        results_table=TableLogger(),
    )

    if args.experiment_state_path is not None:
        exp_state.load_from_path(args.experiment_state_path)

    expert_lib = exp_state.state.expert_lib
    tablelogger = exp_state.state.results_table
    args = exp_state.state.config
    iterations_run = exp_state.state.active_iteration

    tasks = args.finetune_task_name

    expert_lib.pop("base")
    print("###### Tasks", tasks)
    exp_state.save()
    for it in range(args.n_active_iterations - iterations_run):
        a_i = it + iterations_run
        best_weights_matrix = {}
        log_prefix = f"ai_{a_i}_"

        # continue from task in the task iteration
        tasks_seen_in_active_iteration = exp_state.tasks_in_active_iteration(aci=a_i)
        tasks_left = [t for t in tasks if t not in tasks_seen_in_active_iteration]

        print("###### Active iteration", a_i, " tasks to be trained on ", tasks_left)
        for task in tasks_left:  # tasks iteration
            log_row = {c: 0 for c in tablelogger.columns}
            log_row["act_i"] = a_i
            log_row["task"] = task

            evaluator_train: Evaluator = prepare_evaluator(
                args,
                args.dataset,
                tasks=task,
                split="train",
            )

            evaluator_valid: Evaluator = prepare_evaluator(
                args,
                args.dataset,
                tasks=task,
                split="val",
            )

            evaluator_test: Evaluator = prepare_evaluator(
                args,
                args.dataset_test,
                tasks=task,
                split="test",
            )

            if module is None:
                module = MultiExpertModel(
                    **vars(args), tokenizer=evaluator_train.tokenizer, device_map="cpu"
                )
            ########################################################################
            # 1. eval base performance
            logger.info(f"Evaluating base perf for {task} using its best module sofar")

            if task in expert_lib:
                model_copy = copy.deepcopy(module)
                model_copy.load_from_module_dict(
                    {task: expert_lib[task]}, action=args.action
                )
                if args.action == "route":
                    model_copy.convert_container_to_expert(task)

                scores_base_test = evaluator_test.evaluate(model_copy)
                log_row["score_base_test"] = scores_base_test[task]["mean"]
                score_base_train = evaluator_train.evaluate(model_copy)
                log_row["score_base_train"] = score_base_train[task]["mean"]
                score_base_valid = evaluator_valid.evaluate(model_copy)
                log_row["score_base_valid"] = score_base_valid[task]["mean"]
                del model_copy
                free_memory()

            ########################################################################
            # 2. use ng optimizer
            logger.info(
                f"############ Optimizing for {task} for {args.n_ng_iterations} iterations"
            )
            get_loss_function = partial(get_loss, evaluator=evaluator_train)

            optimizer = RoutingOptimizer(
                model=module,
                modules_2_dest=expert_lib,
                get_loss=get_loss_function,
                budget=args.n_ng_iterations,
                init_one=list(expert_lib.keys()).index(task)
                if args.init_ng_oracle
                else None,
                action=args.action,
                regularizer_factor=args.regularizer_factor,
            )
            best_weights, best_graph_string = optimizer.optimize()
            best_weights = best_weights.tolist()
            logger.info("Found best weights: {}".format(best_weights))
            logger.info("Found best graph: {}".format(best_graph_string))

            log_best_weights(expert_lib, best_weights, task, prefix=log_prefix)

            best_weights_matrix[task] = {
                t: v for t, v in zip(expert_lib.keys(), best_weights)
            }
            log_row["weights"] = str(
                {t: v for t, v in zip(expert_lib.keys(), best_weights)}
            )
            ########################################################################
            # 3. get test/train/val score of the new expert
            logger.info(f"Getting test/train/val score of the new expert")

            model_optimal = copy.deepcopy(module)
            model_optimal.load_from_graph_string(best_graph_string, action=args.action)
            if args.action == "route":
                model_optimal.convert_container_to_expert("new_task")

            # train score
            optimal_train_score = evaluator_train.evaluate(model_optimal)
            log_row["score_train"] = optimal_train_score[task]["mean"]
            # test score
            optimal_test_scores = evaluator_test.evaluate(model_optimal)
            log_row["score_test"] = optimal_test_scores[task]["mean"]
            # valid score
            optimal_valid_scores = evaluator_valid.evaluate(model_optimal)
            log_row["score_valid"] = optimal_valid_scores[task]["mean"]

            # log also the optimal score: i.e. if we were only keep the best score for each task
            log_row["score_train_max"] = max(
                log_row["score_base_train"], log_row["score_train"]
            )
            log_row["score_test_max"] = max(
                log_row["score_base_test"], log_row["score_test"]
            )
            log_row["score_valid_max"] = max(
                log_row["score_base_valid"], log_row["score_valid"]
            )
            # log test score as selected with validation
            improved_on_valid = log_row["score_valid"] > log_row["score_base_valid"]
            log_row["score_test_selected"] = (
                log_row["score_test"]
                if improved_on_valid
                else log_row["score_base_test"]
            )

            ########################################################################
            # 4. save new module (keeps track of its parent)
            logger.info(f"Saving new module for {task}")
            # TODO: maybe save meta information about the module: its performance on tasks that it has been evaluated on? can we use it for search?
            model_optimal.hparams.model_modifier = "lora"
            new_momodule_path = save_new_module(
                args.output_dir,
                model_optimal,
                task,
                postfix=f"_{task}_test_score_{log_row['score_test_selected']}_aci{a_i}",
            )

            logger.info(
                f"{log_prefix} Scores on of {task} with graph {best_graph_string}:{optimal_test_scores[task]['mean']}"
            )

            ########################################################################
            # 4 We now fine-tune the new expert
            del model_optimal
            free_memory()

            if args.finetune_new_expert:
                from finetune_expert import finetune_expert
                from mttl.datamodule.base import AutoDataModule

                dm = AutoDataModule.create(
                    args.dataset,
                    model=args.model,
                    model_family=args.model_family,
                    for_generation=False,
                    validation_portion=args.validation_portion,
                    finetune_task_name=task,
                    train_batch_size=8,
                    predict_batch_size=16,
                )
                args_copy = copy.deepcopy(args)
                args_copy.num_train_epochs = 1
                args_copy.model_modifier = "lora"
                val_check_interval = args.gradient_accumulation_steps * 4
                checkpoint = finetune_expert(
                    args_copy,
                    dm=dm,
                    module_dest=new_momodule_path,
                    val_check_interval=val_check_interval,
                    loggers=[wandb_logger],
                )
                model_optimal = copy.deepcopy(module)
                model_optimal.load_from_module_dict(
                    {"new_task": checkpoint}, action=args.action
                )
                if args.action == "route":
                    model_optimal.convert_container_to_expert("new_task")

                # test score
                optimal_test_scores = evaluator_test.evaluate(model_optimal)
                log_row["score_test_fine_tuned"] = optimal_test_scores[task]["mean"]
                # valid score
                optimal_valid_scores = evaluator_valid.evaluate(model_optimal)
                log_row["score_valid_finetuned"] = optimal_valid_scores[task]["mean"]

                improved_on_valid = (
                    log_row["score_valid_finetuned"] > log_row["score_valid_max"]
                )
                log_row["score_test_selected_fine_tuned"] = (
                    log_row["score_test_fine_tuned"]
                    if improved_on_valid
                    else log_row["score_test_selected"]
                )

                model_optimal.hparams.model_modifier = "lora"
                new_momodule_path = save_new_module(
                    args.output_dir,
                    model_optimal,
                    task,
                    postfix=f"_{task}_test_score_{log_row['score_test_selected_fine_tuned']}_aci{a_i}",
                )
            ########################################################################
            tablelogger.log(log_row)

            # replace the module in the dict with the new one or add new module
            if improved_on_valid:
                if args.new_module_action == "replace":
                    logger.info(
                        f"Module {expert_lib[task]} \n for {task} is replaced in the dict with \n {new_momodule_path}"
                    )

                    expert_lib[task] = new_momodule_path
                elif args.new_module_action == "add":
                    logger.info(
                        f"Module {expert_lib[task]} \n for {task} is added to the dict with \n {new_momodule_path}"
                    )
                    old_path = expert_lib.get(task, None)
                    expert_lib[task] = new_momodule_path
                    if old_path is not None:
                        expert_lib[f"{task}_from_{a_i}"] = old_path
                else:
                    logger.info(
                        f"New module for {task} is not added to the dict (new_module_action is {args.new_module_action}), only saved to {new_momodule_path}"
                    )
            ########################################################################
            plt.clf()
            free_memory()
            tablelogger.log_table_wandb()
            exp_state.update(active_iteration=a_i)
            exp_state.save()

            # TODO: log experiment: exp. library with the table, restart from expert library checkpoint if provided

        best_weights_matrix = pd.DataFrame.from_dict(best_weights_matrix)
        if wandb.run is not None:
            tbl_bw = wandb.Table(data=best_weights_matrix)
            wandb.log({f"{log_prefix}best_weights_matrix": tbl_bw})
        plt.clf()


if __name__ == "__main__":
    args = ExpertsMergeConfig.parse()
    run_eval(args)
