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
from huggingface_hub import login
from tempfile import TemporaryDirectory
from pytorch_lightning import seed_everything
from huggingface_hub import create_repo, login, HfApi

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from projects.wiki_experts.utils import get_datamodule
from projects.wiki_experts.src.evolution.utils import get_loss, get_task_expert

from projects.wiki_experts.src.expert_trainer import ExpertTrainer
from mttl.models.modifiers.expert_containers.expert_library import (
    get_best_expert_for_task,
    get_best_expert_for_score,
    LocalExpertLibrary,
    HFExpertLibrary,
    ExpertLibrary,
    Score,
)
from projects.wiki_experts.src.evolution.train_router import train_module
from projects.wiki_experts.src.evolution.evaluators import Evaluator
from mttl.models.modifiers.expert_containers.module_graph import Expert

from projects.wiki_experts.src.evolution.nevergrad_opt import NGRoutingOptimizer
from mttl.utils import setup_logging, logger
from projects.wiki_experts.src.expert_model import (
    MoETrainer,
    MultiExpertModel,
    RoutedMultiExpertModel,
)
from typing import Callable, Dict, Any

from projects.wiki_experts.src.evolution.config import (
    EvolExpertConfig,
    increase_version,
)

FT_FUNCS: dict[str, Callable] = {}


def register_evol_funcs(name):
    def decorator(func):
        if name not in FT_FUNCS:
            FT_FUNCS[name] = func
        else:
            raise ValueError(f"Duplicate name {name} in evol functions")
        return func

    return decorator


def finetune(
    args: EvolExpertConfig,
    dm_train,
    dm_eval,
    module_to_train=None,
    **kwargs,
) -> (Expert, dict):
    assert dm_train.for_generation == False

    # eval 10 times but with at least 10 updates interval
    total_updtes = (
        len(dm_train.train_dataloader())
        * args.num_train_epochs
        // args.gradient_accumulation_steps
    )

    eval_every = max(args.evol_n_eval_times, total_updtes // args.evol_n_eval_times)
    wandb_logger = kwargs.get("wandb_logger", None)
    debug = kwargs.get("debug", False)
    loggers = [] if wandb_logger is None else [wandb_logger]

    best_weights, expert = train_module(
        args,
        dm_train,
        dm_eval,
        module=module_to_train,
        val_check_interval=eval_every,
        loggers=loggers,
        silent=not debug,
    )
    del module_to_train
    return expert


@register_evol_funcs("poly_full")
def fine_tune_w_poly(
    args: EvolExpertConfig,
    dm_train,
    dm_eval,
    expert_lib: ExpertLibrary,
    **kwargs,
) -> (Expert, dict):
    args = copy.deepcopy(args)
    args.trainable_param_names = (
        args.trainable_param_names
        + "|.*module_logits.*|.*selector.*"  # adds selector params to trainable params
    )
    module = RoutedMultiExpertModel(**vars(args), device_map="auto")

    module.load_from_module_dict(expert_lib)
    module.to("cuda")
    return finetune(
        args,
        dm_train,
        dm_eval,
        module_to_train=module,
        **kwargs,
    )


@register_evol_funcs("poly_selector")
def fine_tune_w_poly(
    args: EvolExpertConfig,
    dm_train,
    dm_eval,
    expert_lib: ExpertLibrary,
    **kwargs,
) -> (Expert, dict):
    args = copy.deepcopy(args)

    args.trainable_param_names = (
        "|.*module_logits.*|.*selector.*"  # adds selector params to trainable params
    )
    module = RoutedMultiExpertModel(**vars(args), device_map="auto")

    module.load_from_module_dict(expert_lib)
    module.to("cuda")
    return finetune(
        args,
        dm_train,
        dm_eval,
        module_to_train=module,
        **kwargs,
    )
