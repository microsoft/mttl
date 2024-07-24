import hashlib
import os
import sys
from functools import partial

import nevergrad as ng
import torch
import tqdm
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from typing import Callable

import wandb

from mttl.dataloader.ni_metrics import compute_metrics
from mttl.evaluators import MMLUEvaluator
from mttl.evaluators.base import compute_task_aggregation
from mttl.logging import logger
from mttl.models.expert_model import ExpertModel, MultiExpertModel
from mttl.models.library.expert_library import ExpertLibrary
from mttl.models.library.library_transforms import (
    WeightedLinearMerge,
    WeightedLinearMergeConfig,
)
from mttl.vllm_engines.engines import LLMEngineMMLU, free_memory


def mmlu_get_loss(
    model: ExpertModel,
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
        log=True,
    ) -> None:
        self.log = log
        self.regularizer_factor = regularizer_factor
        self.task_name = task_name
        self.model: MultiExpertModel = model
        self.K = len(expert_lib)
        # vars ordered in the same order as data in expert_lib
        init = [0] * self.K
        self.library = expert_lib
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

    def optimize(
        self,
    ):
        def get_score(weights, basemodel: MultiExpertModel, get_loss, get_regular):
            config = WeightedLinearMergeConfig(
                weights={
                    exp_name: w for exp_name, w in zip(self.library.keys(), weights)
                }
            )
            weighted_merge = WeightedLinearMerge(config)
            logger.info(f"Testing weights {weights}")
            expert = weighted_merge.transform(self.library)
            basemodel.add_expert_instance(expert, is_default=True)
            # minimize the metric
            loss = get_loss(
                model=basemodel,
            )
            if self.log and wandb.run is not None:
                wandb.log(
                    {
                        "ng_loss": loss,
                        "iteration": self._iteration,
                    }
                )

            # L1 regularization term
            metric_val = loss + self.regularizer_factor * get_regular(weights)
            self._iteration += 1
            return metric_val

        _get_score = partial(
            get_score,
            get_loss=self.get_loss,
            basemodel=self.model,
            get_regular=default_l1_regularization,
        )
        recommendation = self.optimizer.minimize(_get_score)
        logger.info(recommendation.value)

        best_combo = {
            expert_name: w
            for expert_name, w in zip(self.library.keys(), recommendation.value)
        }
        return recommendation.value, best_combo
