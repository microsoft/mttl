from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
import math
import os
from typing import List
import numpy as np
import torch

from transformers import StoppingCriteriaList, StoppingCriteria

from mttl.utils import logger
from mttl.models.utils import EfficientCheckpointModule, transfer_batch_to_device


def decode(preds, tokenizer, clean_up_tokenization_spaces=True):
    preds[preds == -100] = tokenizer.pad_token_id
    preds = tokenizer.batch_decode(
        preds,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=clean_up_tokenization_spaces,
    )
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
        **_,
    ):
        if config is None and datamodule is None:
            raise ValueError("Either config or datamodule must be provided.")

        self.datamodule = datamodule
        if config is None:
            config = datamodule.config

        self.config = deepcopy(config)
        self.use_vllm = use_vllm
        self._last_metrics = None

    def get_dataloader(self, split, subsample, shuffle):
        if self.datamodule is None:
            raise ValueError("No datamodule initialized!")

        if split in ["test", "testing"]:
            dataloader = self.datamodule.test_dataloader(subsample, shuffle)
        elif split in ["train", "training"]:
            dataloader = self.datamodule.train_dataloader(subsample)
        elif split in ["val", "valid", "validation", "dev"]:
            dataloader = self.datamodule.val_dataloader(subsample, shuffle)
        else:
            raise ValueError("Unknown split: {}".format(split))
        return dataloader

    @property
    def last_metrics(self):
        return self._last_metrics

    def save_metrics(self, metrics, output_path, predictions=None):
        import json

        class JsonCustomEncoder(json.JSONEncoder):
            """<cropped for brevity>"""

            def default(self, obj):
                if isinstance(obj, (np.ndarray, np.number)):
                    return obj.tolist()
                elif isinstance(obj, set):
                    return list(obj)
                elif isinstance(obj, bytes):  # pragma: py3
                    return obj.decode()
                return json.JSONEncoder.default(self, obj)

        self._last_metrics = metrics

        if output_path is None:
            return

        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)

        with open(output_path + "/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2, cls=JsonCustomEncoder)

        if predictions is not None:
            with open(output_path + "/predictions.json", "w", encoding="utf-8") as f:
                json.dump(predictions, f, ensure_ascii=False, indent=2)

    @abstractmethod
    def evaluate(
        self,
        model,
        split=None,
        shuffle=False,
        subsample=-1,
        output_path=None,
        **kwargs,
    ):
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
    """A stopping criteria that stops on matching token strings (and not ids).

    We decide to stop on strings rather than stopping on particular ids is a bit difficult to do,
    i.e. \n\n might be tokenized differently if it is preceeded by a particular token or followed
    by a particular token, i.e. the model can generate \n\na, which is tokenized as a whole.
    """

    def __init__(self, stop_tokens=[], tokenizer=None):
        super().__init__()
        self.stop = stop_tokens
        self.max_length = max([len(s) for s in stop_tokens])
        self.tokenizer = tokenizer
        self.finished = None
        self.num_tokens = 1

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        """Stops on matching token strings and not ids."""
        if self.finished is None:
            self.finished = [None for _ in range(input_ids.shape[0])]

        batch_size = input_ids.shape[0]
        # must look as far as the number of generated tokens
        decoded = self.tokenizer.batch_decode(
            input_ids[:, -min(self.max_length, self.num_tokens) :]
        )
        self.num_tokens += 1

        for j in range(batch_size):
            # fill the rest of input ids with pad tokens, the generation finished!
            if self.finished[j]:
                input_ids[j, self.finished[j][1] :] = self.tokenizer.pad_token_id
                continue
            # check which stop token is in the decoded text
            for stop in self.stop:
                pos = decoded[j].find(stop)
                if pos != -1:
                    self.finished[j] = (stop, input_ids.shape[1])
                if self.finished[j]:
                    break
        return all(self.finished)


class GenerativeEvaluator(Evaluator):
    """Applied to an evaluator handles generation logic for a given batch."""

    def __init__(
        self,
        datamodule=None,
        config=None,
        use_vllm=False,
        generation_kwargs=None,
    ):
        super().__init__(datamodule, config, use_vllm)

        self.generation_kwargs = generation_kwargs or {}

        if self.generation_kwargs.pop("auto_max_new_tokens", None):
            self.generation_kwargs["max_new_tokens"] = self._detect_max_new_tokens()

    def _detect_max_new_tokens(self) -> int:
        """Tries to detect the max_new_tokens automatically based on the length of the test / valid set answers."""
        logger.warn(
            "Trying to auto detect max_new_tokens. This functionality should only be used for training and not for reporting results (as it assumes access to test labels)."
        )

        length = -1
        try:
            for batch in iter(self.datamodule.val_dataloader()):
                if "labels" in batch:
                    length = max(length, batch["labels"].shape[1])
            for batch in iter(self.datamodule.test_dataloader()):
                if "labels" in batch:
                    length = max(length, batch["labels"].shape[1])
        except Exception as e:
            logger.warn("Exception: {}", e)

        if length == -1:
            logger.warn("Could not auto detect max_new_tokens.")
            return self.config.max_output_length
        else:
            logger.info("Auto detected max_new_tokens: %d", length)
            return length

    def postprocess_generation_output(self, generation_output):
        """Postprocesses the generation output."""
        return generation_output

    def generate_for_batch(self, model, batch):
        if hasattr(model, "module"):
            model = model.module

        extra_kwargs = {}
        extra_kwargs["pad_token_id"] = self.tokenizer.pad_token_id
        extra_kwargs["eos_token_id"] = self.tokenizer.eos_token_id
        extra_kwargs["max_new_tokens"] = self.config.max_output_length
        extra_kwargs.update(self.generation_kwargs)

        stop_tokens = extra_kwargs.pop("stop_tokens", None)
        if stop_tokens:
            stop_tokens = stop_tokens + [self.tokenizer.eos_token]
            stopping_criteria = StoppingCriteriaList(
                [StoppingCriteriaSub(stop_tokens, tokenizer=self.tokenizer)]
            )
            extra_kwargs["stopping_criteria"] = stopping_criteria

        if extra_kwargs.get("temperature", 0.0) == 0.0:
            # stop hf from complaining
            extra_kwargs["do_sample"] = False

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
        sources_texts = batch.get("sources_texts")
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

        # if stop tokens have been specified, then we need to remove them from the generated text
        # we search the surface for of the first matching stop token from the end of the generated
        # and the sequence text. We then remove everything after that and keep only the first part
        # (before the stopping sequence has been produced)
        if stop_tokens:
            finished_with = extra_kwargs["stopping_criteria"][0].finished

            # we strip the finished with token
            for i in range(len(generated_texts)):
                if (
                    finished_with[i] is not None
                    and finished_with[i][0] != self.tokenizer.eos_token
                ):
                    assert finished_with[i][0] in generated_texts[i]
                    generated_texts[i] = generated_texts[i].rpartition(
                        finished_with[i][0]
                    )[0]
                    sequences_texts[i] = sequences_texts[i].rpartition(
                        finished_with[i][0]
                    )[0]

        return self.postprocess_generation_output(
            GenerationOutput(
                scores=predictions.scores,
                sequences=predictions.sequences,
                sequences_texts=sequences_texts,
                sources_texts=sources_texts,
                generated_texts=generated_texts,
            )
        )


class EvaluatorRunner:
    def __init__(self, output_path=None):
        self.evaluators = {}
        self.output_path = output_path

    def add_evaluator(self, name, evaluator):
        self.evaluators[name] = evaluator

    def run(self, module, verbose=False):
        import json
        import prettytable
        from mttl.utils import logger

        if self.output_path:
            os.makedirs(self.output_path, exist_ok=True)

        scores = {}
        for name in sorted(self.evaluators.keys()):
            logger.info("Evaluating %s", name)

            if self.output_path:
                task_output_path = os.path.join(self.output_path, name)
                os.makedirs(self.output_path, exist_ok=True)
            else:
                task_output_path = None

            scores[name] = self.evaluators[name].evaluate(
                module,
                verbose=verbose,
                output_path=task_output_path,
            )

            if self.output_path:
                with open(self.output_path + "/metrics.json", "w") as f:
                    json.dump(scores, f, indent=2)

        scores["mean"] = np.array(list(scores.values())).mean()

        if self.output_path:
            with open(self.output_path + "/metrics.json", "w") as f:
                json.dump(scores, f, indent=2)

        table = prettytable.PrettyTable()
        table.field_names = list(scores.keys())
        table.add_row(["{:.3f}".format(v) for v in list(scores.values())])
        logger.info("Results:\n" + str(table))
        return scores


def setup_evaluators(
    model_type,
    model_family,
    max_input_length,
    max_output_length,
    predict_batch_size,
    truncation_side,
    instruct_template_for_code=False,
    output_path=None,
    tasks=None,
) -> EvaluatorRunner:
    import copy
    from mttl.datamodule.mmlu_data_module import MMLUDataConfig
    from mttl.evaluators.mmlu_evaluator import MMLUEvaluator, MMLUEvaluatorFast
    from mttl.evaluators.piqa_evaluator import PiqaEvaluator
    from mttl.evaluators.hellaswag_evaluator import HellaswagEvaluator
    from mttl.evaluators.humaneval_evaluator import HumanEvalEvaluator
    from mttl.evaluators.mbpp_evaluator import MBPPEvaluator
    from mttl.evaluators.bbh_evaluator import DirectBBHEvaluator, DirectBBHEvaluatorFast
    from mttl.evaluators.superglue_evaluators import BoolQEvaluator
    from mttl.evaluators.arc_evaluator import ArcEvaluator
    from mttl.evaluators.openbookqa_evaluator import OpenbookQAEvaluator
    from mttl.evaluators.winogrande_evaluator import WinograndeEvaluator
    from mttl.datamodule.winogrande_data_module import WinograndeDataConfig
    from mttl.datamodule.openbookqa_data_module import OpenbookQADataConfig
    from mttl.datamodule.arc_data_module import ArcDataConfig
    from mttl.datamodule.piqa_data_module import PiqaDataConfig
    from mttl.datamodule.hellaswag_data_module import HellaswagDataConfig
    from mttl.datamodule.superglue_data_module import SuperGLUEDataConfig
    from mttl.datamodule.bbh_data_module import BBHConfig
    from mttl.datamodule.mbpp_datamodule import MBPPDataConfig
    from mttl.datamodule.humaneval_module import HumanEvalConfig

    evaluators = {}
    common_kwargs_ = {
        "model": model_type,
        "model_family": model_family,
        "max_input_length": max_input_length,
        "max_output_length": max_output_length,
        "predict_batch_size": predict_batch_size,
        "truncation_side": truncation_side,
    }
    generation_kwargs_ = {
        "temperature": 0.0,
        "do_sample": False,
    }

    if type(tasks) == str:
        tasks = tasks.split(",")

    for task in tasks or [
        "humaneval",
        "mbpp",
        "boolq",
        "arc-easy",
        "arc-challenge",
        "piqa",
        "hellaswag",
        "winogrande",
        "openbookqa",
        "bbh-fast",
        "mmlu-fast",
    ]:
        common_kwargs = copy.deepcopy(common_kwargs_)
        generation_kwargs = copy.deepcopy(generation_kwargs_)

        if task == "humaneval":
            generation_kwargs.update(
                {
                    "temperature": 0.05,
                    "top_p": 0.95,
                    "do_sample": True,
                    "max_new_tokens": 300,
                }
            )
            config = HumanEvalConfig(
                **common_kwargs,
                use_instruct_template=instruct_template_for_code,
            )
            evaluators["humaneval"] = HumanEvalEvaluator(
                config, generation_kwargs=generation_kwargs, split="test"
            )
        elif task == "mbpp":
            generation_kwargs.update(
                {
                    "temperature": 0.05,
                    "top_p": 0.95,
                    "do_sample": True,
                    "max_new_tokens": 300,
                }
            )
            evaluators["mbpp"] = MBPPEvaluator(
                MBPPDataConfig(
                    **common_kwargs,
                    use_instruct_template=instruct_template_for_code,
                ),
                generation_kwargs=generation_kwargs,
            )
        elif task == "mbpp-train":
            generation_kwargs.update(
                {
                    "temperature": 0.05,
                    "top_p": 0.95,
                    "do_sample": True,
                    "max_new_tokens": 300,
                }
            )
            evaluators["mbpp-train"] = MBPPEvaluator(
                MBPPDataConfig(
                    **common_kwargs,
                    use_instruct_template=instruct_template_for_code,
                ),
                generation_kwargs=generation_kwargs,
                split="train",
            )
        elif task in ["boolq", "bool_q_1_0_0"]:
            config = SuperGLUEDataConfig(
                **common_kwargs,
            )
            evaluators["boolq"] = BoolQEvaluator(
                config, generation_kwargs=generation_kwargs
            )
        elif task == "bbh":
            generation_kwargs["max_new_tokens"] = 128
            config = BBHConfig(
                **common_kwargs,
                augment_few_shot=5,
            )
            evaluators["bbh"] = DirectBBHEvaluator(
                config, generation_kwargs=generation_kwargs
            )
        elif task == "bbh-fast":
            generation_kwargs["max_new_tokens"] = 128
            config = BBHConfig(
                **common_kwargs,
                augment_few_shot=5,
            )
            evaluators["bbh-fast"] = DirectBBHEvaluatorFast(
                config, generation_kwargs=generation_kwargs
            )
        elif task in ["arc-easy", "ai2_arc_ARC_Easy_1_0_0"]:
            config = ArcDataConfig(
                **common_kwargs,
                arc_type="ARC-Easy",
            )
            evaluators["arc-easy"] = ArcEvaluator(
                config, generation_kwargs=generation_kwargs
            )
        elif task in ["arc-challenge", "ai2_arc_ARC_Challenge_1_0_0"]:
            config = ArcDataConfig(
                **common_kwargs,
                arc_type="ARC-Challenge",
            )
            evaluators["arc-challenge"] = ArcEvaluator(
                config, generation_kwargs=generation_kwargs
            )
        elif task in ["piqa", "piqa_1_0_0"]:
            config = PiqaDataConfig(
                **common_kwargs,
            )
            evaluators["piqa"] = PiqaEvaluator(
                config, generation_kwargs=generation_kwargs
            )
        elif task in ["hellaswag", "hellaswag_1_1_0", "hswag"]:
            evaluators["hellaswag"] = HellaswagEvaluator(
                HellaswagDataConfig(**common_kwargs),
                generation_kwargs=generation_kwargs,
            )
        elif task in ["winogrande", "winogrande_1_1_0"]:
            evaluators["winogrande"] = WinograndeEvaluator(
                WinograndeDataConfig(**common_kwargs),
                generation_kwargs=generation_kwargs,
            )
        elif task in ["openbookqa", "openbookqa_0_1_0"]:
            evaluators["openbookqa"] = OpenbookQAEvaluator(
                OpenbookQADataConfig(**common_kwargs),
                generation_kwargs=generation_kwargs,
            )
        elif task == "mmlu":
            evaluators["mmlu"] = MMLUEvaluator(
                MMLUDataConfig(**common_kwargs),
                generation_kwargs=generation_kwargs,
            )
        elif task == "mmlu-fast":
            evaluators["mmlu-fast"] = MMLUEvaluatorFast(
                MMLUDataConfig(**common_kwargs),
                generation_kwargs=generation_kwargs,
            )
        else:
            raise ValueError("No active tasks")

    runner = EvaluatorRunner(output_path)
    for name, evaluator in evaluators.items():
        runner.add_evaluator(name, evaluator)
    return runner
