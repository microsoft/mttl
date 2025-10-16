from typing import Dict

import numpy as np
import torch
import tqdm
from transformers import DynamicCache

from mttl.arguments import create_config_class_from_args
from mttl.dataloader.ni_metrics import compute_metrics
from mttl.datamodule.base import DataModule, DatasetConfig
from mttl.datamodule.utils import get_tokenizer, maybe_filter_hf_dataset_by_task
from mttl.dist_utils import (
    distributed_mean,
    get_local_rank,
    get_world_size,
    is_main_process,
)
from mttl.evaluators.base import (
    GenerationOutput,
    GenerativeEvaluator,
    switch_to_eval_mode,
)
from mttl.evaluators.loglike_evaluator import LogLikeEvaluator
from mttl.evaluators.rouge_evaluator import RougeEvaluator
from mttl.logging import logger, warn_once
from mttl.models.base_model import BaseExpertModel
from mttl.models.expert_model import ExpertModel
from projects.kms.utils.nqa_datamodule import NQADatamodule, NQADatasetConfig


class NQAZeroShotEvaluator(RougeEvaluator):
    def __init__(self, dataset_args: "DataArgs", generation_kwargs: Dict = {}):
        import copy

        from mttl.datamodule.base import get_datamodule

        copy_args = copy.deepcopy(dataset_args)
        copy_args.dataset_type = "narrativeqa"

        datamodule = get_datamodule(copy_args, for_generation=True)
        super().__init__(datamodule, generation_kwargs=generation_kwargs)

    def evaluate(self, model, split=None, **kwargs):

        # splits in order of preference
        splits = ["test", "dev", "train"]

        if split is not None:
            assert split in splits, f"Split {split} not found in {splits}"
            dataset = getattr(self.datamodule, f"{split}_dataset")
            if dataset is None or len(dataset) == 0:
                warn_once(
                    f"invalid dataset for split {split}, \n{dataset}. Using default split."
                )
                split = None

        if split is None:
            for split in splits:
                dataset = getattr(self.datamodule, f"{split}_dataset")
                if dataset and len(dataset):
                    break

        return super().evaluate(model, split=split, **kwargs)


class SharedNQAEvaluator(NQAZeroShotEvaluator):
    """A bit more ad-hoc evaluator for NQA, where many contexts share the same questions.

    Speeds-up generation by a lot.
    """

    def __init__(self, dataset_args: "DataArgs", generation_kwargs: Dict = {}):
        # don't expand questions for this evaluator
        self.config = dataset_args
        self.config.expand_questions = False
        self.datamodule = NQADatamodule(self.config, for_generation=True)
        self.ctx_lengths = []
        self.cmp_lengths = []

    def encode_context(self, context, questions):
        context_prompt = f"{self.config.prompt}"

        if self.config.include_context:
            if isinstance(context, list):
                # this is for RAG
                passages = " ".join(
                    [
                        f"Passage {k+1}: {context[k]}\n\n"
                        for k in range(min(self.config.topk_context, len(context)))[
                            ::-1
                        ]
                    ]
                )
                context_prompt = (
                    f"Consider the following passages:\n{context}\n{context_prompt}"
                )
            else:
                # this is for standard ICL
                context_prompt = (
                    f"Consider the following paragraph:\n{context}\n{context_prompt}"
                )

        separator = "\n" + "#" * len(context_prompt)
        context = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": context_prompt + separator}],
            add_generation_prompt=True,
            tokenize=False,
        )
        context, question_suffix = context.split(separator)

        # Add question_suffix and answer prefix
        # e.g. for llama3.1, question_suffix="<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
        questions = [question + question_suffix for question in questions]

        # Tokenize the context and questions
        context_ids = self.tokenizer.encode(
            context, return_tensors="pt", add_special_tokens=False
        )
        question_ids = [
            self.tokenizer.encode(
                question, return_tensors="pt", add_special_tokens=False
            )
            for question in questions
        ]

        # Truncate context
        if context_ids.shape[1] > self.config.max_input_length:
            context_ids = context_ids[:, -self.config.max_input_length :]

        return context_ids, question_ids

    @torch.no_grad()
    def generate_answer(
        self,
        model,
        cache,
        context_length,
        question_ids,
        max_new_tokens,
        forward_kwargs,
    ):
        do_sample = True
        top_p = model.generation_config.top_p
        temperature = model.generation_config.temperature

        cache_seq_lengths = [
            cache.get_seq_length(layer_idx) for layer_idx in range(len(cache))
        ]
        position_ids = torch.arange(
            context_length,
            context_length + question_ids.shape[1],
            device=model.device,
        ).unsqueeze(0)

        # if the user doesn't provide a question, skip forward pass
        outputs = model(
            input_ids=question_ids.to(model.device),
            past_key_values=cache,
            position_ids=position_ids,
            **forward_kwargs,
        )

        position_ids = position_ids[:, -1:] + 1
        generated_ids = [outputs.logits[0, -1].argmax()]

        should_stop_token_ids = model.generation_config.eos_token_id
        if not isinstance(should_stop_token_ids, list):
            should_stop_token_ids = [should_stop_token_ids]

        for i in range(max_new_tokens - 1):
            outputs = model(
                input_ids=generated_ids[-1].unsqueeze(0).unsqueeze(0),
                past_key_values=cache,
                position_ids=position_ids + i,
                **forward_kwargs,
            )
            if do_sample:
                logits = outputs.logits[0, -1]

                if temperature is not None:
                    logits = logits / temperature
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=False)
                    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

                    # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
                    sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
                    # Keep at least min_tokens_to_keep
                    sorted_indices_to_remove[-1:] = 0

                    # scatter sorted tensors to original indexing
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        0, sorted_indices, sorted_indices_to_remove
                    )
                    logits = logits.masked_fill(indices_to_remove, -float("Inf"))

                    probs = torch.nn.functional.softmax(logits, 0)
                else:
                    probs = torch.nn.functional.softmax(logits, 0)
                new_id = torch.multinomial(probs, num_samples=1)[0]
            else:
                new_id = outputs.logits[0, -1].argmax()

            generated_ids.append(new_id)
            del outputs
            torch.cuda.empty_cache()
            if new_id.item() in should_stop_token_ids:
                break

        answer = self.tokenizer.decode(
            torch.stack(generated_ids), skip_special_tokens=True
        )

        # Remove the generated tokens from the cache
        cache.key_cache = [
            cache.key_cache[layer_idx][:, :, :sequence_length]
            for layer_idx, sequence_length in enumerate(cache_seq_lengths)
        ]
        cache.value_cache = [
            cache.value_cache[layer_idx][:, :, :sequence_length]
            for layer_idx, sequence_length in enumerate(cache_seq_lengths)
        ]
        if hasattr(cache, "_quantized_key_cache"):
            cache._quantized_key_cache = [
                cache._quantized_key_cache[layer_idx][:, :, :sequence_length]
                for layer_idx, sequence_length in enumerate(cache_seq_lengths)
            ]
            cache._quantized_value_cache = [
                cache._quantized_value_cache[layer_idx][:, :, :sequence_length]
                for layer_idx, sequence_length in enumerate(cache_seq_lengths)
            ]

        torch.cuda.empty_cache()
        return answer

    def shard_by_local_rank(self, dataset):
        import math

        num_procs = get_world_size()
        total_len = len(dataset)
        shard_size = math.ceil(total_len / num_procs)

        start = shard_size * get_local_rank()
        end = min(start + shard_size, total_len)

        dataset = dataset.select(range(start, end))
        return dataset

    @switch_to_eval_mode
    def evaluate(
        self,
        model,
        split="val",
        shuffle=False,
        verbose=True,
        return_predictions=False,
        output_path=None,
    ):
        if shuffle:
            logger.info("Shuffle is not supported for this evaluator.")

        dataset = getattr(self.datamodule, f"{split}_dataset")

        # shard the dataset across processes manually
        dataset = self.shard_by_local_rank(dataset)

        pbar = tqdm.tqdm(
            enumerate(dataset),
            total=len(dataset),
            disable=not is_main_process(),
        )

        all_rougeL = []
        all_predictions = []
        all_references = []
        all_sources = []
        all_times = []
        all_tokens = []

        for num_example, example in pbar:
            context = example["text"]
            questions = example["questions"]
            gt_answers = example["answers"]

            context_ids, questions_ids = self.encode_context(context, questions)
            context_ids = context_ids.to(model.device)
            context_length = context_ids.shape[1]

            cache = DynamicCache()

            if isinstance(model, BaseExpertModel):
                # build task names manually
                forward_kwargs = {"task_names": [example[self.config.task_name_field]]}
            else:
                forward_kwargs = {}

            with torch.no_grad():
                model(input_ids=context_ids, past_key_values=cache, **forward_kwargs)

            self.ctx_lengths.append(context_length)
            self.cmp_lengths.append(cache.get_seq_length())
            logger.info("Encoded context length: %s", context_length)

            # Greedy decoding for each question
            gen_answers = []
            for question_ids in questions_ids:
                answer = self.generate_answer(
                    model,
                    cache,
                    context_length,
                    question_ids,
                    max_new_tokens=self.config.max_output_length,
                    forward_kwargs=forward_kwargs,
                )
                gen_answers.append(answer)

            eval_metrics = compute_metrics(gen_answers, gt_answers, reduction="none")
            all_rougeL.extend(eval_metrics["rougeL"])

            if verbose:
                logger.info(
                    "Source:\n%s", self.datamodule.tokenizer.decode(context_ids[0])
                )
                logger.info("Question:\n%s", questions[0])
                logger.info("Label:\n%s", gt_answers[0])
                logger.info("Prediction:\n%s", gen_answers[0])

            pbar.set_description(f"RougeL: {np.mean(all_rougeL):.4f}")
            all_predictions.extend(gen_answers)
            all_references.extend(gt_answers)
            all_tokens.append(cache.get_seq_length())
            all_sources.extend(questions)
            torch.cuda.empty_cache()

        rouge_L = distributed_mean(all_rougeL, model.device)
        tokens_per_request = distributed_mean(all_tokens, model.device)

        if output_path:
            metrics = {
                "rouge_L": rouge_L,
                "tokens_per_request": tokens_per_request,
            }
            self.save_metrics(metrics, output_path)

        if return_predictions:
            return rouge_L, GenerationOutput(
                predictions=all_predictions,
                references=all_references,
                sources=questions,
            )
        return rouge_L
