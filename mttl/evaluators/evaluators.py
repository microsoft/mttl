import copy
from abc import ABC, abstractmethod
from functools import partial

import torch

from mttl.datamodule.base import DefaultDataModule, get_datamodule
from mttl.evaluators import MMLUEvaluator, RougeEvaluator
from mttl.models.expert_config import ExpertConfig


def prepare_evaluator(
    args: ExpertConfig,
    dataset,
    tasks,
    split=None,
    subsample=-1,
    for_generation=None,
):
    from mttl.callbacks import TestLossEvaluator

    if args.eval_metric == "loss":
        EVAL_CLASS = TestLossEvaluator
        for_generation = for_generation if for_generation is not None else False
    elif args.eval_metric == "rougeL":
        EVAL_CLASS = ExtendedRougeEvaluator
        for_generation = for_generation if for_generation is not None else True
    elif args.eval_metric == "acc":
        assert "mmlu" in dataset
        EVAL_CLASS = ExtendedMMLUEvaluator
        for_generation = for_generation if for_generation is not None else True
    else:
        raise ValueError(f"Unknown eval metric {args.eval_metric}")

    args_copy = copy.deepcopy(args)
    args_copy.dataset = dataset
    args_copy.finetune_task_name = tasks
    args_copy.validation_portion = 0.0
    dm = get_datamodule(args_copy, for_generation=for_generation)

    if split is not None:
        evaluator = EVAL_CLASS(
            datamodule=dm,
            subsample=subsample,
            name=tasks,
            split=split,
            use_vllm=args.use_vllm,
        )
        return evaluator
    return partial(
        EVAL_CLASS,
        datamodule=dm,
        name=tasks,
        use_vllm=args.use_vllm,
    )


class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, model):
        pass

    @abstractmethod
    def get_loss(self, model):
        pass

    @property
    @abstractmethod
    def tasks(self):
        pass


class ExtendedRougeEvaluator(RougeEvaluator, Evaluator):
    def __init__(
        self, datamodule, name="test", split="test", subsample=-1, use_vllm=False
    ):
        super().__init__(datamodule, use_vllm=use_vllm)
        self.name = name
        self.split = split
        self.subsample = subsample
        self.datamodule = datamodule

    def evaluate(self, model):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        rougeL = super().evaluate(model, self.split, self.subsample, verbose=False)
        return {"all": {"mean": rougeL}, f"{self.name}": {"mean": rougeL}}

    def get_loss(self, model, **kwargs):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        rougeL = self.evaluate(model, **kwargs)
        return rougeL * -1.0

    @property
    def tasks(self):
        return self.datamodule.task_names


class ExtendedMMLUEvaluator(MMLUEvaluator, Evaluator):
    def __init__(
        self,
        datamodule: DefaultDataModule,
        name="test",
        split="test",
        subsample=-1,
        use_vllm=False,
    ):
        self.split = split
        assert split in ["test"]
        self.use_vllm = use_vllm
        super().__init__(datamodule.config, use_vllm=use_vllm)
        self.subsample = subsample
        self.datamodule = datamodule
        self.name = name

    def get_loss(self, model, **kwargs):
        return self.evaluate(model, subsample=self.subsample, **kwargs) * -1.0

    @property
    def tokenizer(self):
        return self.datamodule.tokenizer
