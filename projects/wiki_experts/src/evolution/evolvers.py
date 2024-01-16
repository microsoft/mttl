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

from projects.wiki_experts.train_experts_main import get_datamodule
from mttl.models.utils import get_global_batch_size
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
from projects.wiki_experts.src.evolution.evol_train import train_module, EvolModuleOut
from projects.wiki_experts.src.evolution.evaluators import Evaluator
from mttl.models.modifiers.expert_containers.module_graph import Expert

from projects.wiki_experts.src.evolution.nevergrad_opt import NGRoutingOptimizer
from mttl.utils import setup_logging, logger
from projects.wiki_experts.src.expert_model import (
    MoETrainer,
    MultiExpertModel,
    RoutedMultiExpertModel,
)
from typing import Callable, Dict, Any, Union, List

from projects.wiki_experts.src.evolution.config import (
    EvolExpertConfig,
    increase_version,
    find_version,
)
from projects.wiki_experts.src.evolution.retrievers import RETRIEVERS, Retriever
from abc import ABC, abstractmethod


class Evolver(ABC):
    def __init__(self, retriever: Retriever = None):
        self.last_checkpoints: Dict[
            str, Union[ExpertTrainer, Expert]
        ] = (
            {}
        )  # this is here to save last checkpoint (as opposed to best) that we can continue training with
        self.ai = 0
        self.retriever = retriever

    @staticmethod
    def evolve_with_sgd(
        args: EvolExpertConfig,
        task,
        evaluator_valid: Evaluator,
        module_to_train=None,
        dm_train=None,
        **kwargs,
    ) -> EvolModuleOut:
        args = copy.deepcopy(args)
        dm_eval = evaluator_valid.datamodule
        args.finetune_task_name = task
        dm_train = dm_train or get_datamodule(
            args,
            for_generation=False,
            subsample_train=1 / args.subsample_train_set
            if args.subsample_train_set
            else None,
        )
        assert dm_train.for_generation == False

        # eval 10 times but with at least 10 updates interval
        total_updtes = (
            len(dm_train.train_dataloader()) * 1 // args.gradient_accumulation_steps
        )
        eval_every = max(args.evol_n_eval_times, total_updtes // args.evol_n_eval_times)
        wandb_logger = kwargs.get("wandb_logger", None)
        log_prefix = kwargs.get("log_prefix", "")
        debug = kwargs.get("debug", False)

        loggers = [] if wandb_logger is None else [wandb_logger]
        if debug:
            eval_every = 3000
        kwargs["debug"] = debug
        out: EvolModuleOut = train_module(
            args,
            dm_train,
            dm_eval,
            module_to_train=module_to_train,
            val_check_interval=eval_every,
            loggers=loggers,
            logging_prefix=log_prefix,
            **kwargs,
        )

        out.expert.expert_info.expert_task_name = task
        del module_to_train
        return out

    @abstractmethod
    def evolve(self, task, **kwargs) -> EvolModuleOut:
        pass

    def maybe_apply_retriever(self, task, expert_lib, module, task_expert, **kwargs):
        if self.retriever is not None:
            default_score = kwargs.get("default_score", None)
            task_expert = (
                task_expert
                if task_expert is not None
                else get_task_expert(task, expert_lib, default_score)
            )
            expert_lib = self.retriever.transform(
                expert_lib=expert_lib,
                current_task=task,
                task_expert=task_expert,
                module=module,
            )
        return expert_lib


def get_latest_expert(task, expert_lib):
    version = -1
    expert = None
    for name, metadatum in expert_lib.data.items():
        if metadatum.expert_task_name == task and not name.endswith("_last"):
            v = find_version(name)
            if v > version:
                version = v
                expert = expert_lib[name]
    return expert


EVOLVERS: dict[str, Evolver] = {}


def register_evolvers(name):
    def decorator(func):
        if name not in EVOLVERS:
            EVOLVERS[name] = func
        else:
            raise ValueError(f"Duplicate name {name} in EVOLVERSs")
        return func

    return decorator


@register_evolvers("nevergrad")
class NevergradEvolver(Evolver):
    def evolve(
        self,
        args: EvolExpertConfig,
        task,
        module: MultiExpertModel,
        expert_lib: ExpertLibrary,
        evaluator_train: Evaluator,
        **kwargs,
    ) -> EvolModuleOut:
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
        log_row["weights"] = str(
            {t: v for t, v in zip(expert_lib.keys(), best_weights)}
        )
        expert.expert_info.parent_node = best_graph_string
        self.ai += 1
        return EvolModuleOut(expert=expert, log_row=log_row)


@register_evolvers("sgd_full_ft")
class SGD_Full_FT_Evolver(Evolver):
    def evolve(
        self,
        args,
        task,
        expert_lib: ExpertLibrary,
        evaluator_valid: Evaluator,
        module,
        parent_exp,
        **kwargs,
    ) -> EvolModuleOut:
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
        expert_lib = self.maybe_apply_retriever(
            task, expert_lib, module, task_expert=parent_exp, **kwargs
        )

        args = copy.deepcopy(args)
        args.trainable_param_names = (
            args.trainable_param_names
            + "|.*module_logits.*|.*selector.*"  # adds selector params to trainable params
        )
        args.finetune_task_name = task

        dm_train = get_datamodule(args, for_generation=False)

        module = self.setup_module(
            args, task, expert_lib, dm_train, **kwargs
        )  # <- changes args in place

        out: EvolModuleOut = self.evolve_with_sgd(
            args,
            task,
            evaluator_valid,
            dm_train=dm_train,
            module_to_train=module,
            save_last_expert=args.simulate_normal_training,  # if True, returns last model checkpoint as expert. Set it to True here, as we want to get access to optimizer state
            **kwargs,
        )

        if args.simulate_normal_training:
            # keep around the reference to module for future training
            module.optimizer_state_dict = out.expert_last.expert_optimizer_state
            self.last_checkpoints[task] = module.cpu()

        torch.cuda.empty_cache()
        self.ai += 1
        return out

    def setup_module(
        self,
        args: EvolExpertConfig,
        task,
        expert_lib,
        dm_train,
        **kwargs,
    ) -> ExpertTrainer:
        log_prefix = kwargs.get("log_prefix", "")

        if args.simulate_normal_training:
            if task in self.last_checkpoints:
                # we have a module pointer stored to continue training
                module: ExpertTrainer = self.last_checkpoints[task]
                module._log_pref = log_prefix
                module.to("cuda")
                return module

        module = None

        if args.simulate_normal_training:
            # we make sure that optimizers and scheduler are initialized as if we were training for args.n_active_iterations epochs.
            global_bs = get_global_batch_size(
                args.train_batch_size, args.gradient_accumulation_steps
            )
            args.total_steps = (
                len(dm_train.train_dataset) // global_bs
            ) * args.n_active_iterations

            for name in list(expert_lib.keys()):
                if name.endswith("_last"):
                    expert_lib.remove_expert(name)

        module = RoutedMultiExpertModel(
            **vars(args), device_map="auto", logging_prefix=log_prefix
        )

        module.load_from_module_dict(expert_lib)
        module.to("cuda")

        return module


@register_evolvers("selector")
class SGD_Selector_Evolver(Evolver):
    def evolve(
        self,
        args,
        task,
        expert_lib: ExpertLibrary,
        evaluator_valid: Evaluator,
        **kwargs,
    ) -> EvolModuleOut:
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
        raise NotImplementedError("TODO: Check the implementation")
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

        return self.evolve_with_sgd(
            args,
            task,
            evaluator_valid,
            module_to_train=module,
            **kwargs,
        )


def load_lora_expert_in_experttrainer(module: ExpertTrainer, expert: Expert):
    keys = module.model.load_state_dict(expert.expert_weights, strict=False)
    assert sum(["lora" in k for k in keys.missing_keys]) == 0, "Some keys are missing"


@register_evolvers("fine_tune")
class FineTune_Evolver(Evolver):
    def evolve(
        self,
        args,
        task,
        expert_lib: ExpertLibrary,
        evaluator_valid: Evaluator,
        module,
        parent_exp,
        **kwargs,
    ) -> EvolModuleOut:
        """
        Continues training the task's expert independently. Hence, no transfer here.
        If not task expert exist, will initialize one.

        So this evolver allows to also run normal expert training procedure essentially.

        """
        expert_lib = self.maybe_apply_retriever(
            task, expert_lib, module, task_expert=parent_exp, **kwargs
        )

        args = copy.deepcopy(args)
        args.finetune_task_name = task
        dm_train = get_datamodule(args, for_generation=False)

        module = self.setup_module(args, task, expert_lib, dm_train, **kwargs)

        # performs one epoch of training
        out: EvolModuleOut = self.evolve_with_sgd(
            args,
            task,
            evaluator_valid,
            module_to_train=module,
            save_last_expert=args.simulate_normal_training,
            dm_train=dm_train,
            **kwargs,
        )

        if args.simulate_normal_training:
            # keep around the reference to module for future training
            module.optimizer_state_dict = out.expert_last.expert_optimizer_state
            self.last_checkpoints[task] = module.cpu()

        return out

    def setup_module(
        self,
        args: EvolExpertConfig,
        task,
        expert_lib,
        dm_train,
        **kwargs,
    ) -> ExpertTrainer:
        log_prefix = kwargs.get("log_prefix", "")
        default_score = kwargs.get("default_score", None)

        if args.simulate_normal_training:
            if task in self.last_checkpoints:
                # we have a module pointer stored to continue training
                module: ExpertTrainer = self.last_checkpoints[task]
                module._log_pref = log_prefix
                module.to("cuda")
                return module

        task_expert: Expert = copy.deepcopy(
            get_task_expert(task, expert_lib, default_score)
        )
        if args.simulate_normal_training:
            # we make sure that optimizers and scheduler are initialized as if we were training for args.n_active_iterations epochs.
            global_bs = get_global_batch_size(
                args.train_batch_size, args.gradient_accumulation_steps
            )
            args.total_steps = (
                len(dm_train.train_dataset) // global_bs
            ) * args.n_active_iterations

        if task_expert is None:
            # initialize one from scratch
            task_expert: Expert = ExpertTrainer(**vars(args)).to_expert()

        args.model_modifier = task_expert.training_config.model_modifier
        args.modify_layers = task_expert.expert_config.modify_layers
        args.modify_modules = task_expert.expert_config.modify_modules
        args.trainable_param_names = task_expert.training_config.trainable_param_names

        expert_optimizer_state = None
        if args.simulate_normal_training:
            # we will load the optimizer state
            expert_optimizer_state = task_expert.expert_optimizer_state

        module_to_train = ExpertTrainer(
            **vars(args),
            optimizer_state_dict=expert_optimizer_state,
            logging_prefix=log_prefix,
        )
        load_lora_expert_in_experttrainer(module_to_train, task_expert)
        logger.info(
            f"#####################  Fine-tuning on {task}, with expert {task_expert.name}"
        )
        return module_to_train


def parce_mixed_schedule(schedule):
    schedule = schedule.split("+")
    schedule = [s.split(":") for s in schedule]
    schedule = [
        {"name": s[0], "retriever": s[1], "n_iterations": int(s[2])} for s in schedule
    ]
    return schedule


@register_evolvers("mixed")
class MixedEvolver(Evolver):
    def __init__(self, args: EvolExpertConfig, **kwargs):
        super().__init__(**kwargs)
        self.args = args
        self.schedule: List = parce_mixed_schedule(args.mixed_evolver_schedule)
        # [["sgd_full_ft", "none", 2], ["fine_tune", "lora_sim", 5]]

        assert args.n_active_iterations == sum(
            [s["n_iterations"] for s in self.schedule]
        )

        self.current_evolver: Dict = None

    def next_current_evolver(self):
        if len(self.schedule) == 0:
            raise ValueError("Evolution schedule is empty")

        self.current_evolver: Dict = self.schedule.pop(0)
        assert self.current_evolver["n_iterations"] > 0
        retriever = (
            RETRIEVERS[self.current_evolver["retriever"]](config=self.args)
            if self.current_evolver["retriever"] in RETRIEVERS
            else None
        )
        self.current_evolver["instance"] = EVOLVERS[self.current_evolver["name"]](
            retriever=retriever
        )

    def evolve(self, *args, **kwargs):
        if self.current_evolver is None or self.current_evolver["n_iterations"] == 0:
            self.next_current_evolver()

        out = self.current_evolver["instance"].evolve(*args, **kwargs)
        self.current_evolver["n_iterations"] -= 1

        return out


@register_evolvers("scratch")
class Scratch_Evolver(Evolver):
    def evolve(
        self,
        args,
        task,
        expert_lib: ExpertLibrary,
        evaluator_valid: Evaluator,
        **kwargs,
    ) -> EvolModuleOut:
        """
        Evolves an expert from scratch using the given configuration.

        This is mainly for new task addition experiment.

        Args:
            args (EvolExpertConfig): The configuration for evolving the expert.
            task: The task for which the expert is being evolved.
            expert_lib (ExpertLibrary): The library of available experts.
            evaluator_valid (Evaluator): The evaluator for validating the expert.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple[Expert, dict]: The evolved expert and additional information.

        """
        raise NotImplementedError("TODO: Check the implementation")
        args = copy.deepcopy(args)
        log_prefix = kwargs.get("log_prefix", "")
        module_to_train = ExpertTrainer(**vars(args), logging_prefix=log_prefix)
        return self.evolve_with_sgd(
            args,
            task,
            evaluator_valid,
            module_to_train=module_to_train,
            **kwargs,
        )


@register_evolvers("joint")
class Joint_Evolver(Evolver):
    """
    This one creates a joint dataset from retrieved experts. It assumes sharing the data is allowed.
    It inits a joint expert as uniform combination of retrieved experts.
    """

    def evolve(
        self,
        args,
        task,
        expert_lib: ExpertLibrary,
        evaluator_valid: Evaluator,
        module,
        parent_exp,
        **kwargs,
    ) -> EvolModuleOut:
        default_score = kwargs.get("default_score", None)
        task_expert = get_task_expert(task, expert_lib, default_score=default_score)
        assert task_expert is not None, "No expert for task"
        assert self.retriever is not None, "No retriever"

        expert_lib = self.maybe_apply_retriever(
            task, expert_lib, module, task_expert=parent_exp, **kwargs
        )

        args = copy.deepcopy(args)
        args.finetune_task_name = [
            ex.expert_task_name for ex in expert_lib.data.values()
        ]
        dm_train = get_datamodule(args, for_generation=False)

        module = self.setup_module(args, task, expert_lib, dm_train, **kwargs)

        # performs one epoch of training
        out: EvolModuleOut = self.evolve_with_sgd(
            args,
            task,
            evaluator_valid,
            module_to_train=module,
            save_last_expert=args.simulate_normal_training,
            dm_train=dm_train,
            **kwargs,
        )

        if args.simulate_normal_training:
            # keep around the reference to module for future training
            module.optimizer_state_dict = out.expert_last.expert_optimizer_state
            self.last_checkpoints[task] = module.cpu()

        return out

    def setup_module(
        self,
        args: EvolExpertConfig,
        task,
        expert_lib,
        dm_train,
        **kwargs,
    ) -> ExpertTrainer:
        log_prefix = kwargs.get("log_prefix", "")

        if args.simulate_normal_training:
            if task in self.last_checkpoints:
                # we have a module pointer stored to continue training
                module: ExpertTrainer = self.last_checkpoints[task]
                module._log_pref = log_prefix
                module.to("cuda")
                return module

        if args.simulate_normal_training:
            # we make sure that optimizers and scheduler are initialized as if we were training for args.n_active_iterations epochs.
            global_bs = get_global_batch_size(
                args.train_batch_size, args.gradient_accumulation_steps
            )
            args.total_steps = (
                len(dm_train.train_dataset) // global_bs
            ) * args.n_active_iterations

            for name in list(expert_lib.keys()):
                if name.endswith("_last"):
                    expert_lib.remove_expert(name)

        # we init as uniform combination of retrieved experts
        args_copy = copy.deepcopy(args)
        args_copy.router_selector = "uniform"
        module = RoutedMultiExpertModel(
            **vars(args_copy), device_map="auto", logging_prefix=log_prefix
        )

        module.load_from_module_dict(expert_lib)
        expert: Expert = module.to_expert()

        module = ExpertTrainer(
            **vars(args),
            logging_prefix=log_prefix,
        )
        load_lora_expert_in_experttrainer(module, expert)

        module.to("cuda")

        return module


@register_evolvers("from_joint")
class FromJoint_Evolver(Evolver):
    def evolve(
        self,
        args,
        task,
        expert_lib: ExpertLibrary,
        evaluator_valid: Evaluator,
        **kwargs,
    ) -> EvolModuleOut:
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
        raise NotImplementedError("TODO: Check the implementation")
        args = copy.deepcopy(args)
        log_prefix = kwargs.get("log_prefix", "")
        module_to_train: ExpertTrainer = ExpertTrainer(
            **vars(args), logging_prefix=log_prefix
        )
        assert "joint" in expert_lib, "No joint expert in library"
        joint_expert = expert_lib["joint"]
        load_lora_expert_in_experttrainer(module_to_train, joint_expert)

        return self.evolve_with_sgd(
            args,
            task,
            evaluator_valid,
            module_to_train=module_to_train,
            **kwargs,
        )


@register_evolvers("mope_only_router")
class MOPE_Only_Router_Evolver(Evolver):
    def evolve(
        self,
        args,
        task,
        expert_lib: ExpertLibrary,
        evaluator_valid: Evaluator,
        **kwargs,
    ) -> (Expert, dict):
        raise NotImplementedError("Not implemented yet")
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
        return self.evolve_with_sgd(
            args,
            task,
            evaluator_valid,
            module_to_train=module,
            **kwargs,
        )
