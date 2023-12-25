import tqdm
import os
from evaluate import load

from mttl.evaluators.base import Evaluator, GenerationMixin, switch_to_eval_mode
from mttl.utils import logger


# reference: https://github.com/declare-lab/instruct-eval/blob/main/human_eval/main.py#L35
def count_indent(text: str) -> int:
    count = 0
    for char in text:
        if char == " ":
            count += 1
        else:
            break
    return count


def fix_indents(text: str, multiple: int = 2):
    outputs = []
    for line in text.split("\n"):
        while count_indent(line) % multiple != 0:
            line = " " + line
        outputs.append(line)
    return "\n".join(outputs)


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
            sources_texts = batch["sources_texts"]
            labels_texts = batch["labels_texts"]

            predictions = self.generate_for_batch(model)
            predictions = [
                [s + p]
                for s, p in zip(
                    sources_texts,
                    map(
                        lambda x: filter_code(fix_indents(x)),
                        predictions.generated_texts,
                    ),
                )
            ]

            if verbose:
                logger.info("Prediction:")
                logger.info(predictions[0][0])

            metric.add_batch(predictions=predictions, references=labels_texts)

            if num_batches is not None and num_batch >= num_batches:
                break

        metrics, _ = metric.compute(k=[1])
        return metrics["pass@1"]
