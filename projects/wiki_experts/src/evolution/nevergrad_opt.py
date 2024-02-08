import sys
import os
import tqdm
import copy
import torch
import hashlib
import nevergrad as ng

from torch.utils.data import DataLoader
from functools import partial

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from typing import Union, Callable, List, Dict
from mttl.dataloader.ni_metrics import compute_metrics
from mttl.evaluators.base import compute_task_aggregation

from projects.wiki_experts.src.expert_model import MultiExpertModel
from mttl.utils import logger, setup_logging
from mttl.vllm_engines.engines import LLMEngineMMLU, free_memory
from mttl.evaluators import MMLUEvaluator
from mttl.models.modifiers.expert_containers.expert_library import ExpertLibrary
from projects.wiki_experts.src.evolution.config import EvolExpertConfig as ExpertConfig
import wandb

# from mttl.models.modifiers.expert_containers.module_graph import ModuleGraph
ModuleGraph = None


def mmlu_get_loss(
    model: MultiExpertModel,
    tokenizer,
    dataloader: DataLoader,
    use_vllm=True,
    use_loss=False,
):
    # use gpu if available
    train_loss = 0
    if use_vllm:
        # use vllm
        generation_config = model.generation_config
        model_hash = hashlib.sha256()
        model_hash.update(str([p for p in model.parameters()]).encode())
        model = LLMEngineMMLU(
            model,
            temp_path=f"{os.environ.get('MTTL_TEMP','/tmp/merged')}/{model_hash.hexdigest()}/",
        )
        all_predictions, all_references, all_task_names = model.eval(
            dataloader,
            generation_config=generation_config,
            tokenizer=tokenizer,
        )
        del model
        free_memory()

        eval_metrics = compute_metrics(
            all_predictions, [[r] for r in all_references], reduction="none"
        )
        return (
            compute_task_aggregation(all_task_names, eval_metrics["exact_match"])[
                "all"
            ]["mean"]
            * -1.0
        )

    else:
        if use_loss:
            with torch.no_grad():
                device = "cuda" if torch.cuda.is_available() else "cpu"
                for _, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
                    batch = {
                        k: v.to(device)
                        for k, v in batch.items()
                        if isinstance(v, torch.Tensor)
                    }
                    with torch.no_grad():
                        loss = model(batch)
                    train_loss += loss.detach().float()
            loss = train_loss.float()
            # average loss over the number of examples
            return float(loss) / len(dataloader.dataset)
        else:
            # using accuracy
            mmlu_evaluator = MMLUEvaluator(model.hparams, split="test", use_vllm=False)
            scores = mmlu_evaluator.evaluate(model, dataloader=dataloader)
            return scores["all"]["mean"] * -1.0


def default_l1_regularization(weights):
    """
    Get the L1 regularization term for the weights
    """
    sum_of_squares = sum([abs(x) for x in weights]) / len(weights)
    return 0.05 * sum_of_squares


class NGRoutingOptimizer:
    def __init__(
        self,
        model: MultiExpertModel,
        expert_lib: ExpertLibrary,
        get_loss: Callable,  # function that takes model as input and returns loss
        budget=5,
        task_name="new_task",
        base_module_name=None,
        regularizer_factor=0.0,
        action="route",
        log=True,
    ) -> None:
        self.action = action
        self.log = log
        self.regularizer_factor = regularizer_factor
        self.task_name = task_name
        self.model: MultiExpertModel = model
        self.module_graph: ModuleGraph = self.construct_graph(expert_lib)
        self.varaibles = self.module_graph.get_variables()
        self.K = len(self.varaibles)
        # vars ordered in the same order as data in expert_lib
        init = [0] * self.K
        if base_module_name is not None:
            init_one = list(expert_lib.keys()).index(base_module_name)
            init[init_one] = 1

        self.parametrization = ng.p.Array(
            init=init,
            upper=[1.5] * self.K,
            lower=[-1.5] * self.K,
        )
        self.optimizer = ng.optimizers.NGOpt(
            parametrization=self.parametrization, budget=budget
        )
        self.get_loss = get_loss

        self._iteration = 0

    def construct_graph(self, modules_to_dest: Union[Dict, ExpertLibrary]):
        return ModuleGraph.from_expert_dict(modules_to_dest, module_name="new_task")

    def get_graph_vars(self, weights: list):
        s = {}
        for p, w in zip(self.varaibles, list(weights)):
            s[p] = w
        return s

    def optimize(
        self,
    ):
        def get_score(
            weights,
            basemodel: MultiExpertModel,
            get_loss,
            get_regular,
            get_vars,
            action,
        ):
            graph_vars = get_vars(weights)
            logger.info(f"Testing weights {graph_vars.values()} into the model")
            model = basemodel.deepcopy()
            model.load_from_graph(self.module_graph, action=action, **graph_vars)
            # import numpy as np
            # np.sum([p.detach().cpu().sum().item() for p in model.parameters()])
            if action == "route":
                model.replace_container_with_expert("new_task")

            # minimize the metric
            loss = get_loss(
                model=model,
            )
            if self.log:
                wandb.log(
                    {
                        "loss": loss,
                        "iteration": self._iteration,
                    }
                )

            # L1 regularization term
            metric_val = loss + self.regularizer_factor * get_regular(weights)
            del model
            free_memory()
            self._iteration += 1
            return metric_val

        _get_score = partial(
            get_score,
            get_loss=self.get_loss,
            basemodel=self.model,
            get_regular=default_l1_regularization,
            get_vars=self.get_graph_vars,
            action=self.action,
        )
        recommendation = self.optimizer.minimize(_get_score)
        logger.info(recommendation.value)

        best_vars = self.get_graph_vars(recommendation.value)
        best_graph = self.module_graph.dumps(**best_vars)
        return recommendation.value, best_graph
