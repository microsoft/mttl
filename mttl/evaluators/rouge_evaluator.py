import hashlib
import os

import numpy as np
import tqdm

from mttl.evaluators.base import GenerativeEvaluator, switch_to_eval_mode
from mttl.evaluators.mmlu_evaluator import swap_model
from mttl.evaluators.ni_evaluator import compute_metrics
from mttl.logging import logger
from mttl.vllm_engines.engines import LLMEngineRouge, free_memory


class RougeEvaluator(GenerativeEvaluator):
    def __init__(self, datamodule, use_vllm=False, generation_kwargs=None):
        super().__init__(
            datamodule=datamodule,
            use_vllm=use_vllm,
            generation_kwargs=generation_kwargs,
        )

    def evaluate_with_vllm(self, model, dataloader, num_batches=None, verbose=True):
        model_hash = hashlib.sha256()
        model_hash.update(f"{model.hparams}_{model.model.__class__}".encode())

        # move the model to CPU as VLLM loads its own version of the model
        state = swap_model(model)

        vllm_model = LLMEngineRouge(
            model,
            temp_path=f"{os.environ.get('MTTL_TEMP', '/tmp/merged')}/{model_hash.hexdigest()}/",
        )

        all_predictions, all_references = vllm_model.eval(
            dataloader, model.generation_config, self.max_output_length
        )

        free_memory()
        del vllm_model

        # move the model back to GPU
        swap_model(model, state)
        eval_metrics = compute_metrics(
            all_predictions, all_references, reduction="none"
        )
        all_rougeL = eval_metrics["rougeL"]

        return np.mean(all_rougeL)

    @switch_to_eval_mode
    def evaluate(
        self,
        model,
        split="val",
        subsample=-1,
        num_batches=None,
        verbose=True,
        shuffle=False,
    ):
        dataloader = self.get_dataloader(split, subsample, shuffle=shuffle)

        if self.use_vllm:
            return self.evaluate_with_vllm(model, dataloader, num_batches, verbose)

        pbar = tqdm.tqdm(
            enumerate(dataloader),
            total=len(dataloader),
        )

        all_rougeL = []
        for num_batch, batch in pbar:
            if num_batches is not None and num_batch >= num_batches:
                break

            labels_texts = batch["labels_texts"]
            sources_texts = batch["sources_texts"]

            predictions = self.generate_for_batch(model, batch).generated_texts
            references = [[l] for l in labels_texts]

            eval_metrics = compute_metrics(predictions, references, reduction="none")
            all_rougeL.extend(eval_metrics["rougeL"])

            if verbose:
                logger.info("Sources:\n%s", sources_texts[0])
                logger.info("Label:\n%s", labels_texts[0])
                logger.info("Prediction:\n%s", predictions[0])

            pbar.set_description(f"rougeL: {np.mean(all_rougeL):.4f}")

        return np.mean(all_rougeL)
