import numpy as np
import tqdm

from mttl.dataloader.ni_metrics import compute_metrics
from mttl.evaluators.base import (
    GenerationOutput,
    GenerativeEvaluator,
    switch_to_eval_mode,
)
from mttl.utils import logger


class EMEvaluator(GenerativeEvaluator):
    def postprocess_generation_output(self, generation_output):
        """Usually EM evaluator is insensitive to this kind of spaces."""
        generation_output.generated_texts = [
            t.strip() for t in generation_output.generated_texts
        ]
        return generation_output

    def __init__(self, datamodule, use_vllm=False, generation_kwargs=None):
        super().__init__(
            datamodule=datamodule,
            use_vllm=use_vllm,
            generation_kwargs=generation_kwargs,
        )

        self.max_output_length = datamodule.config.max_output_length

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
        output_path=None,
    ):
        dataloader = self.get_dataloader(split, subsample, shuffle=shuffle)

        if self.use_vllm:
            return self.evaluate_with_vllm(model, dataloader, num_batches, verbose)

        pbar = tqdm.tqdm(
            enumerate(dataloader),
            total=len(dataloader),
        )

        extra_kwargs = {}
        extra_kwargs["pad_token_id"] = self.tokenizer.pad_token_id
        extra_kwargs["eos_token_id"] = self.tokenizer.eos_token_id

        all_em = []
        all_predictions = []

        for num_batch, batch in pbar:
            if num_batches is not None and num_batch >= num_batches:
                break

            labels_texts = batch["labels_texts"]
            sources_texts = batch["sources_texts"]

            predictions: GenerationOutput = self.generate_for_batch(model, batch)
            predictions = predictions.generated_texts
            references = [[l] for l in labels_texts]

            eval_metrics = compute_metrics(predictions, references, reduction="none")
            all_em.extend(eval_metrics["exact_match"])
            all_predictions.extend(predictions)

            if verbose:
                logger.info("Sources:\n%s", sources_texts[0])
                logger.info("Label:\n%s", labels_texts[0])
                logger.info("Prediction:\n%s", predictions[0])

            pbar.set_description(f"exact_match: {np.mean(all_em):.4f}")

        self.save_metrics(
            {"exact_match": np.mean(all_em)}, output_path, predictions=all_predictions
        )
        return np.mean(all_em)
