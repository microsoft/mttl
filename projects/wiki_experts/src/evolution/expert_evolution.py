import os
import sys
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
    save_new_module,
    prepare_evaluator,
    TableLogger,
)
from projects.wiki_experts.src.evolution.train_router import train_router
from evaluators import Evaluator
from projects.wiki_experts.src.expert_library import LocalExpertLibrary
from dataclasses import dataclass
from config import ExpertsMergeConfig
from projects.wiki_experts.src.evolution.nevergrad_opt import NGRoutingOptimizer
from mttl.utils import setup_logging, logger
from projects.wiki_experts.src.expert_model import MultiExpertModel
from projects.wiki_experts.src.evolution.experiment_state import ExperimentState
from mttl.vllm_engines.engines import free_memory

torch.set_float32_matmul_precision("medium")
a_i = 0
log_prefix = None
wandb_logger = None
log_row = {}


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


def evaluate_expert_on_task(
    task,
    module,
    checkpoint,
    evaluator_train=None,
    evaluator_valid=None,
    evaluator_test=None,
):
    logger.info(f"Evaluating perf for {task}")
    if checkpoint is not None:
        model_copy = copy.deepcopy(module)
        model_copy.load_from_module_dict({task: checkpoint}, action=args.action)
        if args.action == "route":
            model_copy.convert_container_to_expert(task)
        module = model_copy

    result = {}
    if evaluator_train is not None:
        scores_base_test = evaluator_test.evaluate(module)
        result["train"] = scores_base_test[task]["mean"]
    if evaluator_valid is not None:
        score_base_train = evaluator_train.evaluate(module)
        result["valid"] = score_base_train[task]["mean"]
    if evaluator_test is not None:
        score_base_valid = evaluator_valid.evaluate(module)
        result["test"] = score_base_valid[task]["mean"]
    return result


def optimize_expert_routing(
    args: ExpertsMergeConfig, task, module, expert_lib, evaluator_train
):
    if args.expert_routing == "nevergrad":
        logger.info(
            f"############ Optimizing with nevergrad for {task} for {args.n_ng_iterations} iterations"
        )
        get_loss_function = partial(get_loss, evaluator=evaluator_train)

        optimizer = NGRoutingOptimizer(
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
        log_best_weights(expert_lib, best_weights, task, prefix=log_prefix)

        model_optimal = copy.deepcopy(module)
        model_optimal.load_from_graph_string(best_graph_string, action=args.action)
        if args.action == "route":
            model_optimal.convert_container_to_expert("new_task")
        logger.info("Found best graph: {}".format(best_graph_string))
        logger.info("Found best weights: {}".format(best_weights))
        log_row["weights"] = str(
            {t: v for t, v in zip(expert_lib.keys(), best_weights)}
        )

    elif args.expert_routing == "sgd_router":
        config_copy = copy.deepcopy(args)
        config_copy.finetune_task_name = task
        config_copy.output_dir = os.path.join(
            args.output_dir, f"sgd_router_{task}_ai{a_i}"
        )
        config_copy.trainable_param_names = ".*module_logits.*|.*selector.*"
        config_copy.warmup_steps = 0
        eval_every = args.gradient_accumulation_steps * 4
        loggers = [] if wandb_logger is None else [wandb_logger]
        best_weights, expert_checkpoint = train_router(
            config_copy,
            evaluator_train.datamodule,
            module_dict=expert_lib,
            val_check_interval=eval_every,
            loggers=loggers,
            logging_prefix=log_prefix,
        )
        config_copy.model_modifier = None
        model_optimal = MultiExpertModel(
            **vars(config_copy),
            tokenizer=evaluator_train.datamodule.tokenizer,
            device_map="cpu",
        )
        model_optimal.load_from_module_dict({task: expert_checkpoint})

        logger.info("Found best weights: {}".format(best_weights))
        log_row["weights"] = str(best_weights)

    else:
        raise ValueError(
            f"Optimizer {args.expert_routing} not supported. Choose from 'nevergrad' or 'sgd_router'"
        )
    return model_optimal


def maybe_finetune_new_module(
    args: ExpertsMergeConfig,
    task,
    new_momodule_path,
):
    module_path_fine_tuned = None
    if args.finetune_new_expert:
        raise NotImplementedError("Finetuning new expert not implemented yet")
        from projects.wiki_experts.src.evolution._finetune_expert import finetune_expert
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
        module_path_fine_tuned = finetune_expert(
            args_copy,
            dm=dm,
            module_dest=new_momodule_path,
            val_check_interval=val_check_interval,
        )
    return module_path_fine_tuned


def setup(args: ExpertsMergeConfig):
    seed_everything(args.seed, workers=True)
    setup_logging(args.output_dir)
    global wandb_logger
    wandb_logger = init_wandb_logger(args)
    if args.hf_token_hub:
        login(token=args.hf_token_hub)

    exp_state = ExperimentState(
        config=args,
        active_iteration=0,
        expert_lib=LocalExpertLibrary(
            model_name=args.model, modules_dir=args.modules_dir
        ),
        results_table=TableLogger(),
    )

    if args.experiment_state_path is not None:
        exp_state.load_from_path(args.experiment_state_path)

    tasks = args.finetune_task_name

    print("###### Tasks", tasks)
    return exp_state, tasks


def main(args: ExpertsMergeConfig):
    exp_state, tasks = setup(args)
    tablelogger, expert_lib, iterations_run = (
        exp_state.state.results_table,
        exp_state.state.expert_lib,
        exp_state.state.active_iteration,
    )
    expert_lib.pop("base", None)
    module = None
    global a_i, log_prefix, log_row

    for it in range(args.n_active_iterations - iterations_run + 1):
        a_i = it + iterations_run

        df = tablelogger.df
        tasks_seen_in_active_iteration = (
            [] if len(df) == 0 else df[df["act_i"] == a_i]["task"].tolist()
        )
        tasks_left = [t for t in tasks if t not in tasks_seen_in_active_iteration]

        print(
            "#" * 10,
            "\n",
            "Active iteration",
            a_i,
            " tasks to be trained on",
            tasks_left,
            "\n",
            "#" * 10,
        )

        for task in tasks_left:
            log_row = {"act_i": a_i}
            log_row["task"] = task
            log_prefix = f"act_it:{a_i}_t:{task}/"

            evaluator_constructor = prepare_evaluator(args, args.dataset, tasks=task)
            evaluator_train = evaluator_constructor(split="train")
            evaluator_valid = evaluator_constructor(split="val")
            evaluator_test = evaluator_constructor(split="test")

            if module is None:
                module = MultiExpertModel(
                    **vars(args), tokenizer=evaluator_train.tokenizer, device_map="cpu"
                )

            if task in expert_lib:
                logger.info(f"Evaluating base perf for {task}")
                base_perf: dict = evaluate_expert_on_task(
                    task,
                    module,
                    expert_lib[task],
                    evaluator_train,
                    evaluator_valid,
                    evaluator_test,
                )
                log_row[f"score_base_test"] = base_perf["test"]
                log_row[f"score_base_train"] = base_perf["train"]
                log_row[f"score_base_valid"] = base_perf["valid"]

            module_optimal = optimize_expert_routing(
                args, task, module, expert_lib, evaluator_train
            )

            optimized_perf: dict = evaluate_expert_on_task(
                task,
                module_optimal,
                None,
                evaluator_train,
                evaluator_valid,
                evaluator_test,
            )
            log_row[f"score_test"] = optimized_perf["test"]
            log_row[f"score_train"] = optimized_perf["train"]
            log_row[f"score_valid"] = optimized_perf["valid"]

            ########################################################################
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
            # log test score as selected with early stopping
            improved_on_valid = log_row["score_valid"] > log_row["score_base_valid"]
            log_row["score_test_selected"] = (
                log_row["score_test"]
                if improved_on_valid
                else log_row["score_base_test"]
            )
            ########################################################################
            logger.info(f"Saving new module for {task}")
            module_optimal.hparams.model_modifier = "lora"
            new_module_path = save_new_module(
                args.output_dir,
                module_optimal,
                task,
                postfix=f"_{task}_test_score_{log_row['score_test_selected']}_aci{a_i}",
            )
            del module_optimal
            free_memory()

            logger.info(
                f"{log_prefix} Scores on of {task} :{log_row['score_test_selected']}"
            )

            ########################################################################
            module_path_fine_tuned = maybe_finetune_new_module(
                args, task, new_module_path
            )
            if module_path_fine_tuned is not None:
                fine_tuned_perf = evaluate_expert_on_task(
                    task,
                    module,
                    module_path_fine_tuned,
                    None,
                    evaluator_valid,
                    evaluator_test,
                )
                log_row["score_test_fine_tuned"] = fine_tuned_perf["test"]
                log_row["score_valid_finetuned"] = fine_tuned_perf["valid"]

                improved_on_valid_ft = (
                    log_row["score_valid_finetuned"] > log_row["score_valid_max"]
                )
                log_row["score_test_selected_fine_tuned"] = (
                    log_row["score_test_fine_tuned"]
                    if improved_on_valid_ft
                    else log_row["score_test_selected"]
                )
                if improved_on_valid_ft:
                    new_module_path = module_path_fine_tuned
                    improved_on_valid = improved_on_valid_ft

            ########################################################################
            # replace the module in the dict with the new one or add new module
            if improved_on_valid:
                if args.new_module_action == "replace":
                    logger.info(
                        f"Module {expert_lib[task]} \n for {task} is replaced in the dict with \n {new_module_path}"
                    )

                    expert_lib[task] = new_module_path
                elif args.new_module_action == "add":
                    logger.info(
                        f"Module {expert_lib[task]} \n for {task} is added to the dict with \n {new_module_path}"
                    )
                    old_path = expert_lib.get(task, None)
                    expert_lib[task] = new_module_path
                    if old_path is not None:
                        expert_lib[f"{task}_from_ai{a_i}"] = old_path
                else:
                    logger.info(
                        f"New module for {task} is not added to the dict (new_module_action is {args.new_module_action}), only saved to {new_module_path}"
                    )
            ########################################################################
            plt.clf()
            free_memory()
            tablelogger.log(log_row)
            tablelogger.log_table_wandb()
            exp_state.update(active_iteration=a_i)
            exp_state.save()


if __name__ == "__main__":
    args = ExpertsMergeConfig.parse()
    main(args)
