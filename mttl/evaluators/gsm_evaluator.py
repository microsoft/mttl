import os

from tqdm.auto import tqdm
from mttl.evaluators.base import GenerativeEvaluator, switch_to_eval_mode
import re
from mttl.logging import logger
import json


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

        if self.config.gsm_template == "python":
            self.save_file = f"experiment/{self.config.dataset}.jsonl"

            if not os.path.exists("experiment"):
                os.mkdir("experiment")

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

        # https://colab.research.google.com/github/google-deepmind/gemma/blob/main/colabs/gsm8k_eval.ipynb
        def find_numbers(x: str) -> list[str]:
            """Finds all numbers in a string."""
            # Search for number, possibly negative (hyphen), with thousand separators
            # (comma), and with a decimal point (period inbetween digits).
            numbers = re.compile(
                r"-?[\d,]*\.?\d+",
                re.MULTILINE | re.DOTALL | re.IGNORECASE,
            ).findall(x)
            return numbers

        def get_predictions(predictions_texts, batch, all_predictions, all_targets):
            # iterate over the predictions and targets

            for i, (pred, source, target) in enumerate(
                zip(predictions_texts, batch["sources_texts"], batch["labels_texts"])
            ):
                pred = pred[len(source) :]
                fields = pred.split("The answer is")
                if len(fields) != 0:
                    pred_item = fields[0]
                    predictions = pred_item.replace(",", "")
                    # code is from https://github.com/aksh555/LoRA-Soups/blob/main/utils.py#L224
                    pred = find_numbers(predictions)
                    if not pred:
                        all_predictions.append(float("inf"))
                    else:
                        pred_answer = float(pred[-1])
                        all_predictions.append(pred_answer)
                        logger.info(f"Predictions: {pred_answer}, Targets: {target}")
                        print(f"Predictions: {pred_answer}, Targets: {target}")
                else:
                    all_predictions.append(float("inf"))
            all_targets.extend(batch["labels_texts"])

        def print_python_code(predictions_texts, batch, file):

            for i, (pred, source, target) in enumerate(
                zip(predictions_texts, batch["sources_texts"], batch["labels_texts"])
            ):
                outputs = (
                    pred.split("### Response:")[-1].strip().split("### Instruction:")[0]
                )

                data = {}
                data["answer"] = float(target)
                data["output_pred"] = outputs
                file.write(json.dumps(data) + "\n")
                file.flush()

        with open(self.save_file, "w") as f:
            for num_batch, batch in pbar:
                predictions = self.generate_for_batch(model, batch)
                predictions_texts = predictions.sequences_texts
                if self.config.gsm_template == "cot":
                    get_predictions(
                        predictions_texts, batch, all_predictions, all_targets
                    )
                elif self.config.gsm_template == "python":
                    print_python_code(predictions_texts, batch, f)
                else:
                    raise ValueError("Invalid templete")

        metrics = self.compute_metrics(all_predictions, all_targets)
        return metrics

    def compute_metrics(self, predictions, targets):
        # compute the accuracy based on the cot prompt
        correct = 0

        for pred_answer, target in tqdm(zip(predictions, targets)):
            if pred_answer == float(target):
                correct += 1

        accuracy = correct / len(predictions)
        return accuracy
