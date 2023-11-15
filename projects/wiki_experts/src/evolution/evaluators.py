import sys
import os
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from mttl.evaluators import MMLUEvaluator
from mttl.callbacks import LossCallback
from abc import ABC, abstractmethod, abstractproperty
from mttl.datamodule.base import DefaultDataModule
from mttl.evaluators import RougeEvaluator


class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, model):
        pass

    @abstractmethod
    def get_loss(self, model):
        pass

    @abstractproperty
    def tasks(self):
        pass


class ExtendedRougeEvaluator(RougeEvaluator, Evaluator):
    def __init__(self, datamodule, name="test", split="test", subsample=-1):
        super().__init__(datamodule)
        self.name = name
        self.split = split
        self.n_samples = len(self.dm.test_dataloader(subsample=subsample).dataset)
        self.datamodule = self.dm

    def evaluate(self, model):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        rougeL = super().evaluate(
            model, split=self.split, num_batches=self.n_samples, verbose=False
        )
        return {"all": {"mean": rougeL}, f"{self.name}": {"mean": rougeL}}

    def get_loss(self, model, **kwargs):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        rougeL = self.evaluate(model, **kwargs)["all"]["mean"]
        return rougeL * -1.0

    @property
    def tasks(self):
        return self.datamodule.task_names


class TestLossEvaluator(LossCallback, Evaluator):
    def __init__(
        self,
        datamodule: DefaultDataModule,
        name="test",
        split="test",
        subsample=-1,
    ):
        if split == "test":
            dataloader = datamodule.test_dataloader(subsample=subsample)
        elif split == "val":
            dataloader = datamodule.val_dataloader(subsample=subsample)
        elif split == "train":
            dataloader = datamodule.train_dataloader(subsample=subsample)
        super().__init__(
            dataloader=dataloader,
            name=name,
            output_dir=None,
            eval_every_opt_step=0,
            checkpoint_oracle=False,
        )
        self.datamodule = datamodule

    @property
    def tasks(self):
        return self.datamodule.task_names

    def evaluate(self, model):
        # return something that should be maximized
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        loss = self.test(model)
        loss *= -1.0
        return {"all": {"mean": loss.item()}, f"{self.name}": {"mean": loss.item()}}

    def get_loss(self, model, **kwargs):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        loss = self.test(model, **kwargs)
        return loss.item()

    @property
    def tokenizer(self):
        return self.datamodule.tokenizer


class ExtendedMMLUEvaluator(MMLUEvaluator, Evaluator):
    def __init__(
        self,
        datamodule: DefaultDataModule,
        name="test",
        split="test",
        subsample=-1,
    ):
        assert split in ["test"]
        super().__init__(datamodule.config)
        self.subsample = subsample
        self.datamodule = datamodule
        self.name = name

    def get_loss(self, model, **kwargs):
        return (
            self.evaluate(model, subsample=self.subsample, **kwargs)["all"]["mean"]
            * -1.0
        )

    @property
    def tokenizer(self):
        return self.datamodule.tokenizer
