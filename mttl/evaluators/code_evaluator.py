import tqdm
import os
from evaluate import load

from mttl.evaluators.base import Evaluator, GenerationMixin, switch_to_eval_mode
from mttl.utils import logger


# reference: https://github.com/declare-lab/instruct-eval/blob/main/human_eval/main.py#L35
def filter_code(completion: str) -> str:
    # The program tends to overwrite, we only take the first function
    completion = completion.lstrip("\n")
    return completion.split("\n\n")[0]


class CodeEvaluator(Evaluator, GenerationMixin):
    def __init__(self, datamodule, use_vllm=False, generation_kwargs=None):
        super().__init__(
            datamodule=datamodule,
            use_vllm=use_vllm,
            generation_kwargs=generation_kwargs,
        )
        os.environ["HF_ALLOW_CODE_EVAL"] = "1"

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

        metric = load("code_eval")
        for num_batch, batch in pbar:
            labels_texts = batch["labels_texts"]

            predictions = self.generate_for_batch(model, batch)

            # take only the prediction part, and sanitize it with filter code,
            # we cannot do cut at the token id level due to token healing problems
            for i, prediction in enumerate(predictions.generated_texts):
                predictions.sequences_texts[i] = batch["sources_texts"][
                    i
                ] + filter_code(prediction)

            predictions = [[p] for p in predictions.sequences_texts]

            if verbose:
                logger.info("Prediction:")
                logger.info(predictions[0][0])

            metric.add_batch(predictions=predictions, references=labels_texts)

            if num_batches is not None and num_batch >= num_batches:
                break

        metrics, _ = metric.compute(k=[1])
        return metrics["pass@1"]
