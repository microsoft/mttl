import os
import sys
import copy
import torch
import wandb
import re
import numpy as np
import seaborn as sns
from dataclasses import replace
from functools import partial
from matplotlib import pyplot as plt
from tempfile import TemporaryDirectory
from pytorch_lightning import seed_everything
from huggingface_hub import create_repo, HfApi

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from projects.wiki_experts.src.evolution.utils import (
    get_loss,
    init_wandb_logger,
    TableLogger,
    get_task_expert,
    get_svd_embedding,
    remove_outdated_experts_from_library,
)

from projects.wiki_experts.src.expert_trainer import ExpertTrainer
from mttl.models.modifiers.expert_containers.expert_library import (
    LocalExpertLibrary,
    HFExpertLibrary,
    ExpertLibrary,
    Score,
    get_expert_library,
)
from projects.wiki_experts.src.evolution.train_router import train_module
from projects.wiki_experts.src.evolution.evaluators import (
    Evaluator,
    prepare_evaluator,
    EvalCallback,
)


from mttl.models.modifiers.expert_containers.expert import Expert

from projects.wiki_experts.src.evolution.config import (
    EvolExpertConfig,
    increase_version,
)
from projects.wiki_experts.src.evolution.nevergrad_opt import NGRoutingOptimizer
from mttl.utils import remote_login, setup_logging, logger
from projects.wiki_experts.src.expert_model import (
    MultiExpertModel,
)
from projects.wiki_experts.src.evolution.experiment_state import ExperimentState
from mttl.vllm_engines.engines import free_memory
from projects.wiki_experts.src.evolution.transfer_matrix import (
    eval_all_experts_on_task,
    eval_expert_on_task,
)
from mttl.models.modifiers.expert_containers.library_transforms import (
    SVDEmbeddingTransform,
    SVDEmbeddingTransformConfig,
)
from projects.wiki_experts.src.evolution.evolvers import EVOL_FUNCTIONS


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
to_repo_id = None


def load_expert_into_model(module, expert, task):
    if expert is not None:
        model_copy = copy.deepcopy(module)
        if isinstance(expert, str):
            model_copy.load_from_module_dict({task: expert}, action="route")
        elif isinstance(expert, Expert):
            model_copy.add_expert_instance(expert, task, action="route")
        else:
            raise ValueError(f"Checkpoint type {type(expert)} not supported")
        if len(model_copy.experts) == 1:
            model_copy.replace_container_with_expert(task, get_expert_instance=False)
        module = model_copy
    return module


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


def setup(args: EvolExpertConfig):
    seed_everything(args.seed, workers=True)
    setup_logging(args.output_dir)

    global wandb_logger
    wandb_logger = init_wandb_logger(args)
    if DEBUG:
        global temp_dir
        temp_dir = TemporaryDirectory(dir=args.output_dir + "/")
        local_lib_location = temp_dir.name
    else:
        local_lib_location = os.path.join(args.output_dir, args.hf_repo_id)

    remote_login(token=args.remote_token)

    os.makedirs(local_lib_location, exist_ok=True)
    print("Local lib location", local_lib_location)
    expert_lib = LocalExpertLibrary.from_expert_library(
        get_expert_library(args.hf_repo_id), local_lib_location
    )
    expert_lib.ignore_sliced = True

    # make sure we only consider modules of the latest version
    remove_outdated_experts_from_library(expert_lib)

    exper_state = ExperimentState(
        config=args,
        active_iteration=0,
        expert_lib=expert_lib,
        results_table=TableLogger(),
    )

    if args.experiment_state_path is not None:
        exper_state.load_from_path(args.experiment_state_path)

    tasks = args.finetune_task_name
    expert_lib = exper_state.state.expert_lib
    # remove tasks for which we dont have experts
    tasks = [t for t in tasks if t in expert_lib.tasks]

    if args.upload_lib_to_hub:
        global to_repo_id
        token = os.environ.get("HF_TOKEN", args.remote_token)
        user_name = HfApi().whoami(token=token)["name"]

        exp_name: str = os.getenv("AMLT_JOB_NAME", "_test_evol")
        # if args.to_repo_id is None:
        #     args.to_repo_id = args.hf_repo_id + exp_name
        to_repo_id = user_name + "/" + exp_name
        create_repo(to_repo_id, token=token, exist_ok=True)

    print("###### Tasks", tasks)
    return exper_state, tasks


def retrieve_experts_for_task(
    sk,
    metric,
    module: MultiExpertModel,
    expert_lib: ExpertLibrary,
    task,
    evaluator=None,
    task_expert=None,
    use_only_modules_for_tasks=None,
) -> ExpertLibrary:
    """
    Retrieves a set of sk experts that are most likley to be useful for the current task
    """
    expert_lib_copy = copy.deepcopy(expert_lib)
    if use_only_modules_for_tasks is not None:
        for k, expert in list(expert_lib_copy.items()):
            if expert.expert_info.expert_task_name not in use_only_modules_for_tasks:
                expert_lib_copy.remove_expert(k)

    if sk <= 0 or sk >= len(expert_lib):
        return expert_lib_copy
    # silent the logger
    if metric not in ["random", "lora_sim", "loss", "rougeL", "lib_embeddings"]:
        return expert_lib_copy

    logger.disabled = True

    if task_expert is None:
        task_expert: Expert = get_task_expert(task, expert_lib, default_score)

    module = copy.deepcopy(module)
    if metric == "random":
        if task_expert is None:
            keys = list(set(expert_lib.keys()))
        else:
            keys = list(set(expert_lib.keys()) - {task_expert.name})
        sel_exp_names = np.random.choice(keys, sk, replace=False)

    if metric == "lora_sim":
        # TODO: check that we have embeddings in the expert library
        if task_expert is None:
            return expert_lib_copy
        module.load_from_module_dict(expert_lib_copy)
        from torch.nn.functional import cosine_similarity

        task_module_name = task_expert.name
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

    elif metric == "lib_embeddings":
        if task_expert is None:
            return expert_lib_copy
        module.load_from_module_dict(expert_lib_copy)
        from torch.nn.functional import cosine_similarity

        task_module_name = task_expert.name
        # compute cosine similarity between each expert and current task's expert, keep top sk
        emb_tasks = {}
        emb_tasks[task_module_name] = get_svd_embedding(
            expert_lib_copy, task_module_name
        )
        for key, metadatum in expert_lib_copy.data.items():
            emb_tasks[key] = get_svd_embedding(expert_lib_copy, metadatum.expert_name)
            emb_tasks[key] = torch.tensor(emb_tasks[metadatum.expert_name])

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
    task_module_name = task_expert.name if task_expert is not None else None
    for m in list(expert_lib_copy.keys()):
        if m not in sel_exp_names and m != task_module_name:
            expert_lib_copy.remove_expert(m)
    logger.info(
        "Retrieved experts: {} with metric {}".format(
            list(expert_lib_copy.keys()), metric
        )
    )
    logger.disabled = False
    return expert_lib_copy


def eval_callbacks(
    args, callbacks: list[EvalCallback], module, expert: Expert, task, sufix=""
):
    log_row = {}
    for cb in callbacks:
        eval_module = load_expert_into_model(module, expert, task)
        if isinstance(cb, partial):
            local_config = copy.deepcopy(args)
            local_config.finetune_task_name = task
            cb = cb(config=local_config, subsample=100 if DEBUG else -1)

        score = cb.evaluate_model(model=eval_module)
        log_row[f"{cb.name}_{sufix}"] = score
        del eval_module
        free_memory()
    return log_row


def active_task_iteration(
    args: EvolExpertConfig,
    task: str,
    expert_lib: HFExpertLibrary,
    module: MultiExpertModel,
    ai=None,
    callbacks: list[EvalCallback] = [],
    update_library: bool = True,
    wandb_logger_local=None,
):
    global a_i, log_prefix, log_row, default_score, wandb_logger
    wandb_logger = (
        wandb_logger_local if wandb_logger_local is not None else wandb_logger
    )
    a_i = ai if ai is not None else a_i
    log_row = {"act_i": a_i}
    log_row["task"] = task
    log_prefix = f"act_it_{a_i}/t_{task}"
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

    # assert task in expert_lib.tasks
    parent_exp: Expert = get_task_expert(task, expert_lib, default_score)
    if parent_exp is not None:
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

        log: dict = eval_callbacks(
            args, callbacks, module, parent_exp, task, sufix="base_expert"
        )
        log_row = {**log_row, **log}

    log_row[f"{args.eval_metric}_base_test"] = (
        base_perf["test"] if parent_exp is not None else -10e10
    )
    log_row[f"{args.eval_metric}_base_train"] = (
        base_perf["train"] if parent_exp is not None else -10e10
    )
    log_row[f"{args.eval_metric}_base_valid"] = (
        base_perf["valid"] if parent_exp is not None else -10e10
    )

    # optinally subset the expert library
    retrieved_expert_lib = retrieve_experts_for_task(
        args.sk,
        args.retrieve_with,
        module,
        expert_lib,
        task,
        evaluator=evaluator_valid,
        task_expert=parent_exp,
        use_only_modules_for_tasks=args.finetune_task_name
        if args.use_only_modules_for_tasks
        else None,
    )
    # log retrieved experts
    log_row["retrieved_experts"] = str(list(retrieved_expert_lib.keys()))

    # prepare hps for evolution step
    config_copy = copy.deepcopy(args)
    config_copy.finetune_task_name = task
    config_copy.output_dir = os.path.join(
        args.output_dir, f"{args.evol_expert_routing}_router_{task}_ai{a_i}"
    )
    config_copy.warmup_steps = (
        args.evolution_warmup_steps
    )  # doubled parameter due historiucal reasons

    optimal_expert, log = EVOL_FUNCTIONS[config_copy.evol_expert_routing](
        args=config_copy,
        task=task,
        module=module,
        expert_lib=retrieved_expert_lib,
        evaluator_train=evaluator_train,
        evaluator_valid=evaluator_valid,
        log_prefix=log_prefix,
        wandb_logger=wandb_logger,
        debug=DEBUG,
        default_score=default_score,
    )
    optimal_expert: Expert = optimal_expert
    log_row = {**log_row, **log}

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
    if parent_exp is not None:
        optimal_expert.expert_info = replace(
            optimal_expert.expert_info,
            expert_name=increase_version(parent_exp.name),
        )
    else:
        optimal_expert.expert_info = replace(
            optimal_expert.expert_info,
            expert_name=f"expert_{task}",
        )

    optimal_expert.expert_info.expert_task_name = task
    logger.info(
        f"{log_prefix} Scores on of {task} :{log_row[f'{args.eval_metric}_test_selected']}"
    )

    ########################################################################
    # log callbacks
    selected_expert = (
        optimal_expert if improved_on_valid or parent_exp is None else parent_exp
    )
    log: dict = eval_callbacks(
        args, callbacks, module, selected_expert, task, sufix="selected_expert"
    )
    log_row = {**log_row, **log}
    ########################################################################
    # replace the module in the expertlib with the new one or add new module
    if (improved_on_valid or DEBUG) and update_library:
        parent_exp_name = "None" if parent_exp is None else parent_exp.name
        # make sure the library is on hf
        if args.new_module_action == "replace":
            logger.info(
                f"!!!!!!!!!!!!! Module {parent_exp_name} \n for {task} is replaced in the dict with \n {optimal_expert.name}"
            )
            expert_lib.replace_expert(parent_exp, optimal_expert)

        elif args.new_module_action == "add":
            logger.info(
                f"!!!!!!!!!!!!! Module {optimal_expert.name} \n for {task} is added to the library."
            )
            assert optimal_expert.name != parent_exp_name
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
    return log_row


def main(args: EvolExpertConfig):
    exper_state, tasks = setup(args)
    tablelogger, expert_lib, iterations_run = (
        exper_state.state.results_table,
        exper_state.state.expert_lib,
        exper_state.state.active_iteration,
    )
    expert_lib: ExpertLibrary = expert_lib
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
            if args.retrieve_with == "lib_embeddings":
                svd_embedder = SVDEmbeddingTransform(
                    SVDEmbeddingTransformConfig(sparsity_threshold=0.5)
                )
                svd_embedder.transform(expert_lib, persist=True)

            log_row = active_task_iteration(
                args, task, expert_lib, module=module
            )  # finds a better module for the task, and eds/replaces it into the library
            tablelogger.log(log_row)
            tablelogger.log_table_wandb()
            exper_state.update(active_iteration=a_i)
            logger.disabled = False
            # exper_state.save()

        # send updates to remote
        if args.upload_lib_to_hub:
            try:
                remote_lib = HFExpertLibrary.from_expert_library(
                    expert_lib,
                    to_repo_id,
                    force=True,
                    upload_aux_data=True,
                    only_tasks=tasks,
                )
                logger.info(f"Done, saving to repo {to_repo_id}")
            except Exception as e:
                logger.info(f"Saving to repo {to_repo_id} failed with {e}")


if __name__ == "__main__":
    args = EvolExpertConfig.parse()
    main(args)
