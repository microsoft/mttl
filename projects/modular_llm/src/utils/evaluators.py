import sys
import os
import copy
import torch
import wandb
from functools import partial

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from mttl.evaluators import MMLUEvaluator
from mttl.callbacks import LossCallback
from abc import ABC, abstractmethod, abstractproperty
from mttl.datamodule.base import DefaultDataModule
from mttl.evaluators import RougeEvaluator
from mttl.datamodule.base import get_datamodule
from mttl.models.expert_config import ExpertConfig


class EvalCallback(ABC):
    @abstractmethod
    def evaluate_model(self, model, prefix=""):
        pass


class MMLUEvalCallback(MMLUEvaluator, EvalCallback):
    def __init__(
        self,
        config,
        name="mmlu_test_callback",
        split="test",
        subsample=-1,
        use_vllm=False,
    ):
        self.split = split
        from mttl.datamodule.mmlu_data_module import MMLUDataConfig

        assert split in ["test"]
        self.use_vllm = use_vllm
        mmlu_config = MMLUDataConfig(
            **{
                k: v
                for k, v in config.__dict__.items()
                if k in MMLUDataConfig.__dataclass_fields__.keys()
            }
        )
        super().__init__(mmlu_config, use_vllm=use_vllm)
        self.subsample = subsample
        self.name = name

    def evaluate_model(self, model, prefix=""):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        score = self.evaluate(model, self.split, self.subsample)["all"]["mean"]
        # log
        if wandb.run is not None:
            wandb.log({f"{prefix}{self.name}_{self.split}": score})
        return score


def prepare_evaluator(
    args: ExpertConfig,
    dataset,
    tasks,
    split=None,
    subsample=-1,
    for_generation=None,
):
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

    @abstractproperty
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


class TestLossEvaluator(LossCallback, Evaluator):
    def __init__(
        self,
        datamodule: DefaultDataModule,
        name="test",
        split="test",
        subsample=-1,
        **kwargs,
    ):
        self.split = split
        if split == "test":
            dataloader = datamodule.test_dataloader(subsample=subsample)
        elif split in ["val", "valid", "validation"]:
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
