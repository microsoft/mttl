from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
import math
import torch

from transformers import StoppingCriteria


def mean(arr):
    return sum(arr) / len(arr)


def pop_stddev(arr):
    mu = mean(arr)
    return math.sqrt(sum([(x - mu) ** 2 for x in arr]) / len(arr))


def sample_stddev(arr):
    mu = mean(arr)
    if len(arr) == 1:
        return 0
    return math.sqrt(sum([(x - mu) ** 2 for x in arr]) / (len(arr) - 1))


def mean_stderr(arr):
    return sample_stddev(arr) / math.sqrt(len(arr))


def compute_task_aggregation(task_names, metric_values):
    """Aggregates metric values per task and computes mean and stderr."""
    aggregation = defaultdict(list)

    for task_name, metric_value in zip(task_names, metric_values):
        aggregation[task_name] += [metric_value]
        aggregation["all"] += [metric_value]

    aggregation = {
        task_name: {
            "mean": mean(values),
            "stderr": mean_stderr(values),
        }
        for task_name, values in aggregation.items()
    }
    return aggregation


def switch_to_eval_mode(fn):
    def _switch_to_eval_mode(*args, **kwargs):
        if not hasattr(args[1], "training"):
            raise ValueError(
                "Wrapping the wrong func. The first argument must be a PyTorch module."
            )

        training = args[1].training
        args[1].eval()
        output = fn(*args, **kwargs)
        if training:
            args[1].train()
        return output

    return _switch_to_eval_mode


class Evaluator(ABC):
    def __init__(
        self,
        datamodule=None,
        config=None,
        device="cuda",
        use_vllm=False,
        generation_kwargs=None,
    ):
        if config is None and datamodule is None:
            raise ValueError("Either config or datamodule must be provided.")

        self.config = deepcopy(config)
        self.datamodule = datamodule
        self.generation_kwargs = generation_kwargs or {}
        self.use_vllm = use_vllm
        self.device = device

    def get_dataloader(self, split, subsample, shuffle):
        if self.datamodule is None:
            raise ValueError("No datamodule initialized!")

        if split in ["test", "testing"]:
            dataloader = self.datamodule.test_dataloader(subsample, shuffle)
        elif split in ["train", "training"]:
            dataloader = self.datamodule.train_dataloader(subsample)
        else:
            dataloader = self.datamodule.val_dataloader(subsample, shuffle)
        return dataloader

    @abstractmethod
    def evaluate(self, model, split="test", shuffle=False, subsample=-1, **kwargs):
        pass

    def evaluate_with_vllm(self, model, dataloader, num_batches=None, verbose=True):
        raise NotImplementedError()

    @property
    def tasks(self):
        self.datamodule.task_names

    @property
    def tokenizer(self):
        return self.datamodule.tokenizer
