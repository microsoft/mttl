from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist
from tqdm.auto import tqdm

from mttl.dist_utils import (
    distributed_mean,
    is_dist_avail_and_initialized,
    is_main_process,
)
from mttl.evaluators.base import GenerativeEvaluator, switch_to_eval_mode
from mttl.evaluators.ni_evaluator import compute_metrics
from mttl.logging import logger


@dataclass
class GenerationOutput:
    predictions: list[str]
    references: list[str]
    sources: list[str]

    def __iter__(self):
        for prediction, reference, source in zip(
            self.predictions, self.references, self.sources
        ):
            yield prediction, reference, source


class RougeEvaluator(GenerativeEvaluator):
    def __init__(self, datamodule, use_vllm=False, generation_kwargs=None):
        super().__init__(
            datamodule=datamodule,
            use_vllm=use_vllm,
            generation_kwargs=generation_kwargs,
        )

    def evaluate_with_vllm(self, model, dataloader, num_batches=None, verbose=True):
        raise NotImplementedError()

    @switch_to_eval_mode
    def evaluate(
        self,
        model,
        split="val",
        subsample=-1,
        num_batches=None,
        verbose=True,
        shuffle=False,
        return_predictions=False,
    ):
        dataloader = self.get_dataloader(split, subsample, shuffle=shuffle)

        if self.use_vllm:
            return self.evaluate_with_vllm(model, dataloader, num_batches, verbose)

        pbar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            disable=not is_main_process(),
        )

        all_rougeL = []
        all_predictions = []
        all_references = []
        all_sources = []
        for num_batch, batch in pbar:
            if num_batches is not None and num_batch >= num_batches:
                break

            labels_texts = batch["labels_texts"]
            sources_texts = batch["sources_texts"]

            predictions = self.generate_for_batch(model, batch).generated_texts

            # if we only have one label per prediction, wrap each label in a list
            if not isinstance(labels_texts[0], (list, tuple)):
                references = [[l] for l in labels_texts]
            else:
                references = labels_texts

            eval_metrics = compute_metrics(predictions, references, reduction="none")
            all_rougeL.extend(eval_metrics["rougeL"])

            if verbose:
                logger.info("Sources:\n%s", sources_texts[0])
                logger.info("Label:\n%s", labels_texts[0])
                logger.info("Prediction:\n%s", predictions[0])

            pbar.set_description(f"rougeL: {np.mean(all_rougeL):.4f}")
            all_predictions.extend(predictions)
            all_references.extend(labels_texts)
            all_sources.extend(sources_texts)

        rouge_L = distributed_mean(all_rougeL, model.device)

        if return_predictions:
            return rouge_L, GenerationOutput(
                predictions=all_predictions,
                references=all_references,
                sources=all_sources,
            )
        return rouge_L
