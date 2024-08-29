from abc import ABC, abstractmethod

import torch

from mttl.datamodule.base import DataModule
from mttl.evaluators import MMLUEvaluator, RougeEvaluator


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
        datamodule: DataModule,
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
