import os
import sys
import copy
import torch
import wandb
import numpy as np
import seaborn as sns
from dataclasses import replace
from functools import partial
from matplotlib import pyplot as plt
from tempfile import TemporaryDirectory
from pytorch_lightning import seed_everything

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from projects.wiki_experts.src.evolution.utils import (
    get_loss,
    init_wandb_logger,
    TableLogger,
)

from mttl.models.modifiers.expert_containers.expert_library import (
    LocalExpertLibrary,
    HFExpertLibrary,
    ExpertLibrary,
    Score,
)
from projects.wiki_experts.src.evolution.train_router import train_router
from projects.wiki_experts.src.evolution.evaluators import Evaluator, prepare_evaluator


from mttl.models.modifiers.expert_containers.module_graph import Expert

from projects.wiki_experts.src.evolution.config import (
    EvolExpertConfig,
    increase_version,
)
from projects.wiki_experts.src.evolution.nevergrad_opt import NGRoutingOptimizer
from mttl.utils import setup_logging, logger
from projects.wiki_experts.src.expert_model import MultiExpertModel
from projects.wiki_experts.src.evolution.experiment_state import ExperimentState
from mttl.vllm_engines.engines import free_memory
from projects.wiki_experts.src.evolution.transfer_matrix import (
    eval_all_experts_on_task,
    eval_expert_on_task,
)
from projects.wiki_experts.src.evolution.expert_evolution import *
from huggingface_hub import create_repo, login

# this script evolves a single task for 1 active iteration and commits it to the library


class ParallelEvolutionConfig(EvolExpertConfig):
    def _set_defaults(self):
        super()._set_defaults()
        self.to_repo_id: str = None
        self.n_active_iterations = 1


def setup(args: ParallelEvolutionConfig):
    seed_everything(args.seed, workers=True)
    setup_logging(args.output_dir)
    global wandb_logger
    login(token=os.environ.get("HF_TOKEN", args.hf_token_hub))
    if not DEBUG:
        create_repo(args.to_repo_id, token=args.hf_token_hub, exist_ok=True)
        wandb_logger = init_wandb_logger(args)
        local_lib_location = os.path.join(args.output_dir, args.to_repo_id)
    else:
        global temp_dir
        temp_dir = TemporaryDirectory(dir=args.output_dir + "/")
        local_lib_location = temp_dir.name

    os.makedirs(local_lib_location, exist_ok=True)
    expert_lib = LocalExpertLibrary.from_remote(
        HFExpertLibrary(args.hf_repo_id), local_lib_location
    )
    expert_lib.ignore_sliced = True

    exper_state = ExperimentState(
        config=args,
        active_iteration=0,
        expert_lib=expert_lib,
        results_table=TableLogger(),
    )
    # dont want to overwrite the exp lib from which we start here for now
    if args.experiment_state_path is not None:
        exper_state.load_from_path(args.experiment_state_path)

    tasks = args.finetune_task_name

    # filter only experts that are in the task list
    expert_lib = exper_state.state.expert_lib
    expert_lib.filter_with_tasks(args.finetune_task_name)
    # remove tasks for which we dont have experts
    tasks = [t for t in tasks if t in expert_lib.tasks]

    print("###### Tasks", tasks)
    return exper_state, tasks


def active_iteration(task: str, expert_lib: HFExpertLibrary):
    log_row = {"act_i": a_i}
    log_row["task"] = task
    log_prefix = f"act_it:{a_i}/t:{task}"
    default_score = Score(name=args.eval_metric, task=task, split="valid")

    evaluator_constructor = prepare_evaluator(args, args.dataset, tasks=task)
    evaluator_train = evaluator_constructor(
        split="train", subsample=args.subsample_ng_train_set
    )
    evaluator_valid = evaluator_constructor(split="valid", subsample=-1)
    evaluator_test = evaluator_constructor(split="test", subsample=-1)

    if module is None:
        module = MultiExpertModel(
            **vars(args), tokenizer=evaluator_train.tokenizer, device_map="cpu"
        )

    assert task in expert_lib.tasks
    parent_exp: Expert = get_best_expert_for_task(expert_lib, task, default_score.hash)
    base_perf = {
        "train": expert_lib.get_score(
            expert_name=parent_exp.name,
            hash=Score(name=args.eval_metric, task=task, split="train").hash,
        ),
        "valid": expert_lib.get_score(
            expert_name=parent_exp.name,
            hash=Score(name=args.eval_metric, task=task, split="valid").hash,
        ),
        "test": expert_lib.get_score(
            expert_name=parent_exp.name,
            hash=Score(name=args.eval_metric, task=task, split="test").hash,
        ),
    }

    if (
        base_perf["train"] is None
        or base_perf["valid"] is None
        or base_perf["valid"] is None
    ):
        logger.info(f"Evaluating base perf for {task}")
        if DEBUG:
            base_perf = {
                "test": np.random.random(),
                "train": np.random.random(),
                "valid": np.random.random(),
            }

        else:
            base_perf: dict = eval_expert_on_task(
                task,
                module,
                parent_exp,
                evaluator_train,
                evaluator_valid,
                evaluator_test,
            )

        expert_lib.add_score(
            expert_name=parent_exp.name,
            score=Score(
                name=args.eval_metric,
                task=task,
                value=base_perf["test"],
                split="test",
            ),
        )
        expert_lib.add_score(
            expert_name=parent_exp.name,
            score=Score(
                name=args.eval_metric,
                task=task,
                value=base_perf["train"],
                split="train",
            ),
        )
        expert_lib.add_score(
            expert_name=parent_exp.name,
            score=Score(
                name=args.eval_metric,
                task=task,
                value=base_perf["valid"],
                split="valid",
            ),
        )

    log_row[f"{args.eval_metric}_base_test"] = base_perf["test"]
    log_row[f"{args.eval_metric}_base_train"] = base_perf["train"]
    log_row[f"{args.eval_metric}_base_valid"] = base_perf["valid"]

    # optinally subset the expert library
    retrieved_expert_lib = retrieve_experts_for_task(
        args.sk,
        args.retrieve_with,
        module,
        expert_lib,
        task,
        evaluator=evaluator_valid,
    )
    # log retrieved experts
    log_row["retrieved_experts"] = str(list(retrieved_expert_lib.keys()))

    optimal_expert: Expert = optimize_evol_expert_routing(
        args,
        task,
        module,
        retrieved_expert_lib,
        evaluator_train,
        evaluator_valid,
    )

    optimized_perf: dict = eval_expert_on_task(
        task,
        module,
        optimal_expert,
        evaluator_train,
        evaluator_valid,
        evaluator_test,
    )
    log_row[f"{args.eval_metric}_test"] = optimized_perf["test"]
    log_row[f"{args.eval_metric}_train"] = optimized_perf["train"]
    log_row[f"{args.eval_metric}_valid"] = optimized_perf["valid"]

    ########################################################################
    # log also the optimal score: i.e. if we were only keep the best score for each task
    log_row[f"{args.eval_metric}_train_max"] = max(
        log_row[f"{args.eval_metric}_base_train"],
        log_row[f"{args.eval_metric}_train"],
    )
    log_row[f"{args.eval_metric}_test_max"] = max(
        log_row[f"{args.eval_metric}_base_test"],
        log_row[f"{args.eval_metric}_test"],
    )
    log_row[f"{args.eval_metric}_valid_max"] = max(
        log_row[f"{args.eval_metric}_base_valid"],
        log_row[f"{args.eval_metric}_valid"],
    )
    # log test score as selected with early stopping
    improved_on_valid = (
        log_row[f"{args.eval_metric}_valid"] > log_row[f"{args.eval_metric}_base_valid"]
    )
    log_row[f"{args.eval_metric}_test_selected"] = (
        log_row[f"{args.eval_metric}_test"]
        if improved_on_valid
        else log_row[f"{args.eval_metric}_base_test"]
    )
    ########################################################################
    logger.info(f"Saving new module for {task}")
    # change name
    optimal_expert.expert_info = replace(
        optimal_expert.expert_info,
        expert_name=increase_version(parent_exp.name),
    )

    logger.info(
        f"{log_prefix} Scores on of {task} :{log_row[f'{args.eval_metric}_test_selected']}"
    )

    ########################################################################
    module_path_fine_tuned = maybe_finetune_module(args, task, optimal_expert)
    if module_path_fine_tuned is not None:
        raise NotImplementedError("Fine tuning new expert not implemented yet")
        fine_tuned_perf = eval_expert_on_task(
            task,
            module,
            module_path_fine_tuned,
            None,
            evaluator_valid,
            evaluator_test,
        )
        log_row[f"{args.eval_metric}_test_fine_tuned"] = fine_tuned_perf["test"]
        log_row[f"{args.eval_metric}_valid_finetuned"] = fine_tuned_perf["valid"]

        improved_on_valid_ft = (
            log_row[f"{args.eval_metric}_valid_finetuned"]
            > log_row[f"{args.eval_metric}_valid_max"]
        )
        log_row[f"{args.eval_metric}_test_selected_fine_tuned"] = (
            log_row[f"{args.eval_metric}_test_fine_tuned"]
            if improved_on_valid_ft
            else log_row[f"{args.eval_metric}_test_selected"]
        )
        if improved_on_valid_ft:
            new_module_path = module_path_fine_tuned
            improved_on_valid = improved_on_valid_ft

    ########################################################################
    # replace the module in the expertlib with the new one or add new module
    if improved_on_valid:
        # make sure the library is on hf
        if args.new_module_action == "replace":
            logger.info(
                f"!!!!!!!!!!!!! Module {parent_exp.name} \n for {task} is replaced in the dict with \n {optimal_expert.name}"
            )
            expert_lib.replace_expert(parent_exp, optimal_expert)

        elif args.new_module_action == "add":
            logger.info(
                f"!!!!!!!!!!!!! Module {optimal_expert.name} \n for {task} is added to the library."
            )
            assert optimal_expert.name != parent_exp.name
            expert_lib.add_expert(optimal_expert)
        else:
            # new_module_path = optimal_expert.save(args.output_dir)
            logger.info(
                f"!!!!!!!!!!!!! New module for {task} is not added to the dict (new_module_action is {args.new_module_action})."
            )

        if optimal_expert in expert_lib:
            assert args.new_module_action in ["replace", "add"]
            expert_lib.add_score(
                expert_name=optimal_expert.name,
                score=Score(
                    name=args.eval_metric,
                    task=task,
                    value=optimized_perf["test"],
                    split="test",
                ),
            )
            expert_lib.add_score(
                expert_name=optimal_expert.name,
                score=Score(
                    name=args.eval_metric,
                    task=task,
                    value=optimized_perf["test"],
                    split="train",
                ),
            )
            expert_lib.add_score(
                expert_name=optimal_expert.name,
                score=Score(
                    name=args.eval_metric,
                    task=task,
                    value=optimized_perf["train"],
                    split="train",
                ),
            )
    ########################################################################
    plt.clf()
    free_memory()
    return expert_lib


def main(args: EvolExpertConfig):
    exper_state, tasks = setup(args)
    tablelogger, expert_lib, iterations_run = (
        exper_state.state.results_table,
        exper_state.state.expert_lib,
        exper_state.state.active_iteration,
    )
    expert_lib: ExpertLibrary = expert_lib
    global a_i, log_prefix, log_row, default_score_name
    default_score_name = f"{args.eval_metric}_valid"

    for task in tasks:
        print("Evolving on task", task)
        updated_expert_lib: LocalExpertLibrary = active_iteration(task, expert_lib)
        expert_lib = update(updated_expert_lib)

    # save the expert lib
    remote_lib = HFExpertLibrary.from_local(expert_lib, args.to_repo_id)


if __name__ == "__main__":
    args = ParallelEvolutionConfig.parse()
    main(args)
