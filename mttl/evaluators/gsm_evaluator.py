import os

from tqdm.auto import tqdm
from mttl.evaluators.base import GenerativeEvaluator, switch_to_eval_mode
import re


class GsmEvaluator(GenerativeEvaluator):
    def __init__(
        self,
        datamodule,
        use_vllm=False,
        generation_kwargs=None,
        prepend_source=True,
        split="test",
    ):
        super().__init__(
            datamodule=datamodule,
            use_vllm=use_vllm,
            generation_kwargs=generation_kwargs,
        )

        self.split = split
        self.prepend_source = prepend_source
        os.environ["HF_ALLOW_CODE_EVAL"] = "1"

    @switch_to_eval_mode
    def evaluate(
        self,
        model,
        split=None,
        subsample=-1,
        num_batches=None,
        verbose=True,
        shuffle=False,
        output_path=None,
    ):
        dataloader = self.get_dataloader(split, subsample, shuffle=shuffle)

        pbar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
        )

        all_predictions = []
        all_targets = []

        for num_batch, batch in pbar:
            predictions = self.generate_for_batch(model, batch)
            all_predictions.extend(predictions.sequences_texts)
            all_targets.extend(batch["labels_texts"])

        metrics = self.compute_metrics(all_predictions, all_targets)
        return metrics

    def compute_metrics(self, predictions, targets):
        # compute the accuracy
        correct = 0

        for pred, target in tqdm(zip(predictions, targets)):

            sentence = pred
            predictions = sentence.replace(",", "")
            # code is from https://github.com/aksh555/LoRA-Soups/blob/main/utils.py#L224
            pred = [s for s in re.findall(r"-?\d+\.?\d*", predictions)]
            if not pred:
                return float("inf")
            pred_answer = float(pred[-1])
            if pred_answer == float(target):
                correct += 1
            print(pred_answer, float(target))

        accuracy = correct / len(predictions)
        return accuracy
