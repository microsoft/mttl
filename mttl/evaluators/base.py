from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
import math
from typing import List
import torch

from transformers import StoppingCriteriaList, StoppingCriteria
from mttl.models.utils import EfficientCheckpointModule, transfer_batch_to_device


def decode(preds, tokenizer):
    preds[preds == -100] = tokenizer.pad_token_id
    preds = tokenizer.batch_decode(
        preds, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    preds = [pred.strip() for pred in preds]
    return preds


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
        use_vllm=False,
        generation_kwargs=None,
    ):
        if config is None and datamodule is None:
            raise ValueError("Either config or datamodule must be provided.")

        self.datamodule = datamodule
        if config is None:
            config = datamodule.config
        self.config = deepcopy(config)
        self.generation_kwargs = generation_kwargs or {}
        self.use_vllm = use_vllm

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


@dataclass
class GenerationOutput:
    scores: torch.Tensor
    sequences: torch.Tensor  # everything that generate() returns in ids
    sequences_texts: List[str]  # everything that generate() returns in text
    sources_texts: List[str]  # the input source texts
    generated_texts: List[str] = None  # only the generated portion


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[]):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for i, stop in enumerate(self.stops):
            stop = self.stops[i] = stop.to(input_ids.device)

            if torch.all((stop[None, :] == input_ids[:, -stop.shape[0] :])).item():
                return True
        return False


class GenerationMixin:
    """Applied to an evaluator handles generation logic for a given batch."""

    def postprocess_generation_output(self, generation_output):
        """Postprocesses the generation output."""
        return generation_output

    def _create_stopping_criteria(self):
        """Create stopping criteria if needed."""
        stop_tokens = self.generation_kwargs.pop("stop_tokens", None)
        if stop_tokens:
            stop_words_ids = [
                # tokenize stop word and return tensors
                self.tokenizer(
                    stop_word, add_special_tokens=False, return_tensors="pt"
                ).input_ids
                for stop_word in stop_tokens
            ]
            stopping_criteria = StoppingCriteriaList(
                [StoppingCriteriaSub(stops=stop_words_ids)]
            )
            self.generation_kwargs["stopping_criteria"] = stopping_criteria

    def generate_for_batch(self, model, batch):
        if hasattr(model, "module"):
            model = model.module

        self._create_stopping_criteria()

        extra_kwargs = {}
        extra_kwargs["pad_token_id"] = self.tokenizer.pad_token_id
        extra_kwargs["eos_token_id"] = self.tokenizer.eos_token_id
        extra_kwargs["max_new_tokens"] = self.config.max_output_length
        extra_kwargs.update(self.generation_kwargs)

        device = next(model.parameters()).device
        batch = transfer_batch_to_device(batch, device)

        with torch.no_grad():
            if isinstance(model, EfficientCheckpointModule):
                predictions = model.generate(
                    batch,
                    generation_config=model.generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    **extra_kwargs,
                )
            else:
                predictions = model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    generation_config=model.generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    **extra_kwargs,
                )

        # take only the prediction part
        # we cannot do cut at the token id level due to token healing problems
        sources_texts = batch.get("sources_texts", None)
        sequences_texts = decode(predictions.sequences, self.tokenizer)

        if self.config.model_family != "gpt":
            generated_texts = sequences_texts
        else:
            generated_texts = []
            generated_texts_fallback = decode(
                predictions.sequences[:, batch["input_ids"].shape[1] :], self.tokenizer
            )
            if sources_texts is not None:
                for i, sequence_text in enumerate(sequences_texts):
                    if sources_texts[i] in sequence_text:
                        # we split based on sources texts
                        generated = sequence_text.split(sources_texts[i])[1]
                    else:
                        # we fallback to the standard way of getting the output
                        generated = generated_texts_fallback[i]
                    generated_texts.append(generated)

        generated_texts = [text.strip() for text in generated_texts]

        return self.postprocess_generation_output(
            GenerationOutput(
                scores=predictions.scores,
                sequences=predictions.sequences,
                sequences_texts=sequences_texts,
                sources_texts=sources_texts,
                generated_texts=generated_texts,
            )
        )
