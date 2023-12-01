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
from huggingface_hub import login
from tempfile import TemporaryDirectory
from pytorch_lightning import seed_everything

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from projects.wiki_experts.src.evolution.utils import (
    get_loss,
    init_wandb_logger,
    TableLogger,
)

from mttl.models.modifiers.expert_containers.expert_library import (
    get_best_expert_for_task,
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

DEBUG = True
if "AMLT_OUTPUT_DIR" in os.environ:
    DEBUG = False
if DEBUG:
    print("!!!!!!!!!!!!!!!!!!!!!! DEBUG MODE")

torch.set_float32_matmul_precision("medium")
a_i = 0
log_prefix = None
wandb_logger = None
log_row = {}
temp_dir = None
default_score: Score = None


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


def optimize_evol_expert_routing(
    args: EvolExpertConfig,
    task,
    module: MultiExpertModel,
    expert_lib: ExpertLibrary,
    evaluator_train: Evaluator,
    evaluator_valid: Evaluator,
) -> Expert:
    if args.evol_expert_routing == "nevergrad":
        logger.info(
            f"############ Optimizing with nevergrad for {task} for {args.n_ng_iterations} iterations"
        )
        get_loss_function = partial(get_loss, evaluator=evaluator_train)

        optimizer = NGRoutingOptimizer(
            model=module,
            expert_lib=expert_lib,
            get_loss=get_loss_function,
            budget=args.n_ng_iterations,
            base_module_name=get_best_expert_for_task(
                expert_lib, task, default_score.hash
            ).name
            if args.init_router_best
            else None,
            action="route",
            regularizer_factor=args.regularizer_factor,
        )
        best_weights, best_graph_string = optimizer.optimize()
        best_weights = best_weights.tolist()
        # log_best_weights(expert_lib, best_weights, task, prefix=log_prefix)

        model_optimal = copy.deepcopy(module)
        model_optimal.load_from_graph_string(
            best_graph_string, "route", expert_library=expert_lib
        )
        expert = model_optimal.replace_container_with_expert("new_task")

        logger.info("Found best graph: {}".format(best_graph_string))
        logger.info("Found best weights: {}".format(best_weights))
        log_row["weights"] = str(
            {t: v for t, v in zip(expert_lib.keys(), best_weights)}
        )
        expert.expert_info.parent_node = best_graph_string

    elif args.evol_expert_routing in ["sgd", "sgd_full_ft"]:
        config_copy = copy.deepcopy(args)
        config_copy.finetune_task_name = task
        config_copy.output_dir = os.path.join(
            args.output_dir, f"sgd_router_{task}_ai{a_i}"
        )
        if args.evol_expert_routing == "sgd_full_ft":
            # we also train the loras
            config_copy.trainable_param_names += "|.*module_logits.*|.*selector.*"
        elif args.evol_expert_routing == "sgd":
            config_copy.trainable_param_names = "|.*module_logits.*|.*selector.*"

        config_copy.warmup_steps = 0

        # dm_train must have for generation = False here
        dm_train = evaluator_train.datamodule
        dm_train = (
            prepare_evaluator(
                config_copy,
                config_copy.dataset,
                tasks=task,
                split="train",
                for_generation=False,
            ).datamodule
            if dm_train.for_generation
            else dm_train
        )
        assert dm_train.for_generation == False
        dm_eval = evaluator_valid.datamodule

        # eval 10 times but with at least 10 updates interval
        total_updtes = (
            len(dm_train.train_dataloader())
            * args.num_train_epochs
            // args.gradient_accumulation_steps
        )
        eval_every = max(10, total_updtes // 10)
        loggers = [] if wandb_logger is None else [wandb_logger]
        if DEBUG:
            eval_every = 300
            from mttl.datamodule.base import subsample_dst

            dm_train.train_dataset = subsample_dst(dm_train.train_dataset, 100)

        best_weights, expert = train_router(
            config_copy,
            dm_train,
            dm_eval,
            expert_lib=expert_lib,
            val_check_interval=eval_every,
            loggers=loggers,
            logging_prefix=log_prefix,
            silent=not DEBUG,
        )

        logger.info("Found best weights: {}".format(best_weights))
        log_row["weights"] = str(best_weights)
        expert.expert_info.expert_task_name = task

    else:
        raise ValueError(
            f"Optimizer {args.evol_expert_routing} not supported. Choose from 'nevergrad' or 'sgd' or 'sgd_full_ft"
        )

    return expert


def maybe_finetune_module(
    args: EvolExpertConfig,
    task,
    new_module,
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


def setup(args: EvolExpertConfig):
    seed_everything(args.seed, workers=True)
    setup_logging(args.output_dir)
    global wandb_logger
    if DEBUG:
        global temp_dir
        temp_dir = TemporaryDirectory(dir=args.output_dir + "/")
        local_lib_location = temp_dir.name
    else:
        wandb_logger = init_wandb_logger(args)
        local_lib_location = os.path.join(args.output_dir, args.hf_repo_id)

    if args.hf_token_hub:
        login(token=args.hf_token_hub)

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
    expert_lib = exper_state.state.expert_lib
    # remove tasks for which we dont have experts
    tasks = [t for t in tasks if t in expert_lib.tasks]

    print("###### Tasks", tasks)
    return exper_state, tasks


def retrieve_experts_for_task(
    sk,
    metric,
    module: MultiExpertModel,
    expert_lib: ExpertLibrary,
    task,
    evaluator=None,
) -> ExpertLibrary:
    """
    Retrieves a set of sk experts that are most likley to be useful for the current task
    """

    if sk <= 0 or sk >= len(expert_lib):
        return expert_lib
    # silent the logger
    logger.disabled = True
    if metric not in ["random", "lora_sim", "loss", "rougeL"]:
        return expert_lib
    expert_lib_copy = copy.deepcopy(expert_lib)
    task_module: Expert = get_best_expert_for_task(
        expert_lib_copy, task, default_score.hash
    )
    task_module_name = task_module.name

    module = copy.deepcopy(module)
    if metric == "random":
        keys = list(set(expert_lib.keys()) - {task_module_name})
        sel_exp_names = np.random.choice(keys, sk, replace=False)

    if metric == "lora_sim":
        module.load_from_module_dict(expert_lib_copy)
        from torch.nn.functional import cosine_similarity

        # compute cosine similarity between each expert and current task's expert, keep top sk
        emb_tasks = module.get_task_embeddings()
        # compare this task's embed with  other
        if task_module_name not in emb_tasks:
            return expert_lib_copy
        task_emb = emb_tasks[task_module_name]
        similarities = []
        t_names = []
        for t, emb in emb_tasks.items():
            if t != task_module_name:
                similarities.append(
                    cosine_similarity(task_emb.unsqueeze(0), emb.unsqueeze(0)).item()
                )
                t_names.append(t)
        similarities = {k: v for k, v in zip(t_names, similarities)}
        sel_exp_names = sorted(similarities, key=similarities.get, reverse=True)[:sk]

    elif metric == "loss":
        split = evaluator.split
        _expert_lib = copy.deepcopy(expert_lib_copy)
        scores = {}
        for k, exp in _expert_lib.items():
            score = expert_lib.get_score(
                expert_name=k, hash=Score(name=f"loss", task=task, split=split).hash
            )
            if score is not None:
                scores[k] = score
                _expert_lib.remove_expert(k)
        if DEBUG:
            neg_losses = {k: np.random.random() for k in _expert_lib.keys()}
        else:
            neg_losses: dict = eval_all_experts_on_task(
                task, module, _expert_lib, evaluator=evaluator
            )
        neg_losses = {**scores, **neg_losses}
        for k, v in neg_losses.items():
            expert_lib.add_score(
                expert_name=k,
                score=Score(name=f"loss", task=task, split=split, value=v),
            )
        sel_exp_names = sorted(neg_losses, key=neg_losses.get, reverse=True)[:sk]

    elif metric == "rougeL":
        split = evaluator.split
        _expert_lib = copy.deepcopy(expert_lib_copy)
        scores = {}
        for k, exp in _expert_lib.items():
            score = expert_lib.get_score(
                expert_name=k, hash=Score(name=f"rougeL", task=task, split=split).hash
            )
            if score is not None:
                scores[k] = score
                _expert_lib.remove_expert(k)
        if DEBUG:
            rouge = {k: np.random.random() for k in _expert_lib.keys()}
        else:
            rouge: dict = eval_all_experts_on_task(
                task, module, _expert_lib, evaluator=evaluator
            )
        scores = {**scores, **rouge}
        # save into experts
        for k, v in rouge.items():
            expert_lib.add_score(
                expert_name=k,
                score=Score(name=f"rougeL", task=task, split=split, value=v),
            )
        sel_exp_names = sorted(scores, key=scores.get, reverse=True)[:sk]

    # create expert_library only with selected modules + the current task's module
    for m in list(expert_lib_copy.keys()):
        if m not in sel_exp_names and m != task_module_name:
            expert_lib_copy.remove_expert(m)
    logger.info(
        "Retrieved experts: {} with metric {}".format(
            list(expert_lib_copy.keys()), metric
        )
    )
    return expert_lib_copy


def main(args: EvolExpertConfig):
    exper_state, tasks = setup(args)
    tablelogger, expert_lib, iterations_run = (
        exper_state.state.results_table,
        exper_state.state.expert_lib,
        exper_state.state.active_iteration,
    )
    expert_lib: ExpertLibrary = expert_lib
    module = None
    global a_i, log_prefix, log_row, default_score

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
            parent_exp: Expert = get_best_expert_for_task(
                expert_lib, task, default_score.hash
            )
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
                log_row[f"{args.eval_metric}_valid"]
                > log_row[f"{args.eval_metric}_base_valid"]
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
                log_row[f"{args.eval_metric}_valid_finetuned"] = fine_tuned_perf[
                    "valid"
                ]

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
            tablelogger.log(log_row)
            tablelogger.log_table_wandb()
            exper_state.update(active_iteration=a_i)
            exper_state.save()


if __name__ == "__main__":
    args = EvolExpertConfig.parse()
    main(args)
