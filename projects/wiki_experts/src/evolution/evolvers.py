import os
import sys
import copy
import re
from functools import partial
from typing import Callable

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from mttl.datamodule.base import get_datamodule
from mttl.models.modifiers.expert_containers.expert import Expert
from mttl.models.modifiers.expert_containers.expert_library import ExpertLibrary
from mttl.utils import logger

from projects.wiki_experts.src.evolution.utils import get_loss, get_task_expert
from projects.wiki_experts.src.expert_trainer import ExpertTrainer
from projects.wiki_experts.src.evolution.train_router import train_module
from projects.wiki_experts.src.expert_model import (
    MultiExpertModel,
    RoutedMultiExpertModel,
)
from projects.wiki_experts.src.evolution.evaluators import Evaluator
from projects.wiki_experts.src.evolution.nevergrad_opt import NGRoutingOptimizer
from projects.wiki_experts.src.evolution.config import (
    EvolExpertConfig,
)

EVOL_FUNCTIONS: dict[str, Callable] = {}


def register_evol_funcs(name):
    def decorator(func):
        if name not in EVOL_FUNCTIONS:
            EVOL_FUNCTIONS[name] = func
        else:
            raise ValueError(f"Duplicate name {name} in evol functions")
        return func

    return decorator


def evolve_with_sgd(
    args: EvolExpertConfig,
    task,
    evaluator_valid: Evaluator,
    module_to_train=None,
    **kwargs,
) -> (Expert, dict):
    log_row = {}
    dm_eval = evaluator_valid.datamodule
    args.finetune_task_name = task
    dm_train = get_datamodule(
        args,
        for_generation=False,
    )
    assert dm_train.for_generation == False

    # eval 10 times but with at least 10 updates interval
    total_updtes = (
        len(dm_train.train_dataloader())
        * args.num_train_epochs
        // args.gradient_accumulation_steps
    )

    eval_every = max(args.evol_n_eval_times, total_updtes // args.evol_n_eval_times)
    wandb_logger = kwargs.get("wandb_logger", None)
    log_prefix = kwargs.get("log_prefix", "")
    debug = kwargs.get("debug", False)

    loggers = [] if wandb_logger is None else [wandb_logger]
    if debug:
        eval_every = 300
        from mttl.datamodule.base import subsample_dst

        dm_train.train_dataset = subsample_dst(dm_train.train_dataset, 1000)

    best_weights, expert = train_module(
        args,
        dm_train,
        dm_eval,
        module=module_to_train,
        val_check_interval=eval_every,
        loggers=loggers,
        logging_prefix=log_prefix,
        silent=not debug,
    )
    # cleanup: remove config_copy.output_dir stuff, as we have out expert already
    if os.path.exists(args.output_dir):
        try:
            os.system(f"rm -rf {args.output_dir}")
        except Exception as e:
            logger.error(e)

    logger.info("Found best weights: {}".format(best_weights))
    log_row["weights"] = str(best_weights)
    expert.expert_info.expert_task_name = task
    del module_to_train
    return expert, log_row


@register_evol_funcs("nevergrad")
def evolve_nevergrad(
    args: EvolExpertConfig,
    task,
    module: MultiExpertModel,
    expert_lib: ExpertLibrary,
    evaluator_train: Evaluator,
    **kwargs,
) -> (Expert, dict):
    """
    Learns merging weights for the experts in the library Nevergrad optimization algorithm.

    Args:
        args (EvolExpertConfig): Configuration for the evolution process.
        task: The task for which the expert is being evolved.
        module (MultiExpertModel): The initial expert model.
        expert_lib (ExpertLibrary): The library of available experts.
        evaluator_train (Evaluator): The evaluator for training data.
        **kwargs: Additional keyword arguments.

    Returns:
        Tuple[Expert, dict]: The evolved expert and a dictionary containing log information.
    """
    log_row = {}
    logger.info(
        f"############ Optimizing with nevergrad for {task} for {args.n_ng_iterations} iterations"
    )
    get_loss_function = partial(get_loss, evaluator=evaluator_train)
    default_score = kwargs.get("default_score", None)
    base_module = get_task_expert(task, expert_lib, default_score)
    base_module_name = base_module.name if base_module is not None else None

    optimizer = NGRoutingOptimizer(
        model=module,
        expert_lib=expert_lib,
        get_loss=get_loss_function,
        budget=args.n_ng_iterations,
        base_module_name=base_module_name,
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
    expert.expert_weights = {
        k: v
        for k, v in expert.expert_weights.items()
        if re.match(args.trainable_param_names, k)
    }  # make sure the checkpoint is not >5G

    logger.info("Found best graph: {}".format(best_graph_string))
    logger.info("Found best weights: {}".format(best_weights))
    log_row["weights"] = str({t: v for t, v in zip(expert_lib.keys(), best_weights)})
    expert.expert_info.parent_node = best_graph_string
    return expert, log_row


@register_evol_funcs("sgd_full_ft")
def evolve_selector_and_experts(
    args: EvolExpertConfig,
    task,
    expert_lib: ExpertLibrary,
    evaluator_valid: Evaluator,
    **kwargs,
) -> (Expert, dict):
    """
    Performs one training iteration of the selector and experts using SGD optimization.

    Args:
        args (EvolExpertConfig): Configuration for evolution.
        task: The task for which the experts are being evolved.
        expert_lib (ExpertLibrary): The library of available experts.
        evaluator_valid (Evaluator): The evaluator for validation.
        **kwargs: Additional keyword arguments.

    Returns:
        Tuple containing the evolved expert and a dictionary of additional information.
    """
    args = copy.deepcopy(args)
    args.trainable_param_names = (
        args.trainable_param_names
        + "|.*module_logits.*|.*selector.*"  # adds selector params to trainable params
    )
    log_prefix = kwargs.get("log_prefix", "")
    module = RoutedMultiExpertModel(
        **vars(args),
        device_map="auto",
        logging_prefix=log_prefix,
    )

    module.load_from_module_dict(expert_lib)
    module.to("cuda")
    return evolve_with_sgd(
        args,
        task,
        evaluator_valid,
        module_to_train=module,
        **kwargs,
    )


@register_evol_funcs("selector")
def evolve_selector_only(
    args: EvolExpertConfig,
    task,
    expert_lib: ExpertLibrary,
    evaluator_valid: Evaluator,
    **kwargs,
) -> (Expert, dict):
    """
    Performs one training iteration using SGD optimization applied only to the selector's params only.

    Args:
        args (EvolExpertConfig): The configuration for evolution.
        task: The task for which the selector is being evolved.
        expert_lib (ExpertLibrary): The library of available experts.
        evaluator_valid (Evaluator): The evaluator for validation.
        **kwargs: Additional keyword arguments.

    Returns:
        Tuple[Expert, dict]: The evolved selector and additional information.
    """
    args = copy.deepcopy(args)
    args.trainable_param_names = (
        "|.*module_logits.*|.*selector.*"  # only trains selector
    )

    log_prefix = kwargs.get("log_prefix", "")
    module = RoutedMultiExpertModel(
        **vars(args),
        device_map="auto",
        logging_prefix=log_prefix,
    )

    module.load_from_module_dict(expert_lib)
    module.to("cuda")

    return evolve_with_sgd(
        args,
        task,
        evaluator_valid,
        module_to_train=module,
        **kwargs,
    )


def load_lora_expert_in_experttrainer(module: ExpertTrainer, expert: Expert):
    keys = module.model.load_state_dict(expert.expert_weights, strict=False)
    assert sum(["lora" in k for k in keys.missing_keys]) == 0, "Some keys are missing"


@register_evol_funcs("no_transfer")
def independent_expert_fine_tuning(
    args: EvolExpertConfig,
    task,
    expert_lib: ExpertLibrary,
    evaluator_valid: Evaluator,
    **kwargs,
) -> (Expert, dict):
    """
    Continues trainining the task's expert independentnly. Hense, no transfer here.
    """
    default_score = kwargs.get("default_score", None)
    args = copy.deepcopy(args)
    task_expert: Expert = copy.deepcopy(
        get_task_expert(task, expert_lib, default_score)
    )
    args.model_modifier = task_expert.training_config.model_modifier
    args.modify_layers = task_expert.expert_config.modify_layers
    args.modify_modules = task_expert.expert_config.modify_modules

    args.trainable_param_names = task_expert.training_config.trainable_param_names
    module_to_train = ExpertTrainer(**vars(args))
    load_lora_expert_in_experttrainer(module_to_train, task_expert)

    return evolve_with_sgd(
        args,
        task,
        evaluator_valid,
        module_to_train=module_to_train,
        **kwargs,
    )


@register_evol_funcs("scratch")
def evolve_scratch(
    args: EvolExpertConfig,
    task,
    expert_lib: ExpertLibrary,
    evaluator_valid: Evaluator,
    **kwargs,
) -> (Expert, dict):
    """
    Evolves an expert from scratch using the given configuration.

    Args:
        args (EvolExpertConfig): The configuration for evolving the expert.
        task: The task for which the expert is being evolved.
        expert_lib (ExpertLibrary): The library of available experts.
        evaluator_valid (Evaluator): The evaluator for validating the expert.
        **kwargs: Additional keyword arguments.

    Returns:
        Tuple[Expert, dict]: The evolved expert and additional information.

    """
    module_to_train = ExpertTrainer(**vars(args))
    return evolve_with_sgd(
        args,
        task,
        evaluator_valid,
        module_to_train=module_to_train,
        **kwargs,
    )


@register_evol_funcs("from_joint")
def evolve_from_joint(
    args: EvolExpertConfig,
    task,
    expert_lib: ExpertLibrary,
    evaluator_valid: Evaluator,
    **kwargs,
) -> (Expert, dict):
    """
    Evolves an expert from a joint expert.

    Args:
        args (EvolExpertConfig): The configuration for the evolution process.
        task: The task for which the expert is being evolved.
        expert_lib (ExpertLibrary): The library of available experts. Assumes llibrary contains a joint expert.
        evaluator_valid (Evaluator): The evaluator for the validation set.
        **kwargs: Additional keyword arguments.

    Returns:
        Tuple[Expert, dict]: The evolved expert and additional information.
    """
    module_to_train: ExpertTrainer = ExpertTrainer(**vars(args))
    assert "joint" in expert_lib, "No joint expert in library"
    joint_expert = expert_lib["joint"]
    load_lora_expert_in_experttrainer(module_to_train, joint_expert)

    return evolve_with_sgd(
        args,
        task,
        evaluator_valid,
        module_to_train=module_to_train,
        **kwargs,
    )


@register_evol_funcs("mope_only_router")
def evolve_with_mope(
    args: EvolExpertConfig,
    task,
    expert_lib: ExpertLibrary,
    evaluator_valid: Evaluator,
    **kwargs,
) -> (Expert, dict):
    """
    Evolves an expert from with MOPE, i.e. an x-conditioned router. We learn only router here.

    Args:
        args (EvolExpertConfig): The configuration for the evolution process.
        task: The task for which the expert is being evolved.
        expert_lib (ExpertLibrary): The library of available experts. Assumes llibrary contains a joint expert.
        evaluator_valid (Evaluator): The evaluator for the validation set.
        **kwargs: Additional keyword arguments.

    Returns:
        Tuple[Expert, dict]: The evolved expert and additional information.
    """
    log_prefix = kwargs.get("log_prefix", "")
    module = RoutedMultiExpertModel(
        **vars(args),
        device_map="auto",
        logging_prefix=log_prefix,
    )
    module.load_from_module_dict(expert_lib)
    module.to("cuda")
    args.router_granularity = "finegrained"
    args.trainable_param_names = ".*selector.*"  # only trains selector
    return evolve_with_sgd(
        args,
        task,
        evaluator_valid,
        module_to_train=module,
        **kwargs,
    )
