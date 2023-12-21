from typing import Optional
import tqdm
import torch
import os
from evaluate import load

from mttl.datamodule.humaneval_module import HumanEvalDataModule
from mttl.evaluators.base import Evaluator, switch_to_eval_mode
from mttl.models.utils import transfer_batch_to_device, EfficientCheckpointModule
from mttl.utils import logger


def decode(preds, tokenizer):
    preds[preds == -100] = tokenizer.pad_token_id
    preds = tokenizer.batch_decode(
        preds, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    preds = [pred.strip() for pred in preds]
    return preds


# reference: https://github.com/declare-lab/instruct-eval/blob/main/human_eval/main.py#L35
def filter_code(completion: str) -> str:
    # The program tends to overwrite, we only take the first function
    completion = completion.lstrip("\n")
    return completion.split("\n\n")[0]


class CodeEvaluator(Evaluator):
    def __init__(
        self, datamodule, device="cuda", use_vllm=False, generation_kwargs=None
    ):
        super().__init__(
            datamodule=datamodule,
            device=device,
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

        extra_kwargs = {}
        extra_kwargs["pad_token_id"] = self.tokenizer.pad_token_id
        extra_kwargs["eos_token_id"] = self.tokenizer.eos_token_id

        metric = load("code_eval")
        for num_batch, batch in pbar:
            labels_texts = batch["labels_texts"]

            batch = transfer_batch_to_device(batch, self.device)
            with torch.no_grad():
                if isinstance(model, EfficientCheckpointModule):
                    predictions = model.generate(
                        batch,
                        max_new_tokens=self.config.max_output_length,
                        generation_config=model.generation_config,
                        return_dict_in_generate=True,
                        output_scores=True,
                        **extra_kwargs,
                        **self.generation_kwargs,
                    )
                else:
                    predictions = model.generate(
                        batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        max_new_tokens=self.config.max_output_length,
                        generation_config=model.generation_config,
                        return_dict_in_generate=True,
                        output_scores=True,
                        **extra_kwargs,
                        **self.generation_kwargs,
                    )

            predictions = predictions.sequences
            predictions = decode(predictions, self.tokenizer)

            # take only the prediction part, and sanitize it with filter code,
            # we cannot do cut at the token id level due to token healing problems
            for i in range(len(predictions)):
                source = batch["sources_texts"][i].lstrip("\n")
                prediction = predictions[i].strip("\n")
                prediction = filter_code(prediction.split(source)[1])
                predictions[i] = source + prediction
            predictions = [[p] for p in predictions]

            if verbose:
                logger.info("Prediction:")
                logger.info(predictions[0][0])

            metric.add_batch(predictions=predictions, references=labels_texts)

            if num_batches is not None and num_batch >= num_batches:
                break

        metrics, _ = metric.compute(k=[1])
        return metrics["pass@1"]
