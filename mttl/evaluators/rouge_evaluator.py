import tqdm
import torch
import numpy as np

from mttl.evaluators.ni_evaluator import compute_metrics
from mttl.models.utils import transfer_batch_to_device, EfficientCheckpointModule
from mttl.utils import logger


def decode(preds, tokenizer):
    preds[preds == -100] = tokenizer.pad_token_id
    preds = tokenizer.batch_decode(
        preds, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    preds = [pred.strip() for pred in preds]
    return preds


class RougeEvaluator:
    def __init__(self, datamodule, device="cuda"):
        super().__init__()
        self.device = device
        self.dm = datamodule
        self.tokenizer = datamodule.tokenizer
        self.max_output_length = datamodule.config.max_output_length

    def evaluate(self, model, split="val", num_batches=None, verbose=True):
        extra_kwargs = {}
        extra_kwargs["pad_token_id"] = self.tokenizer.pad_token_id
        extra_kwargs["eos_token_id"] = self.tokenizer.eos_token_id

        all_rougeL = []

        if split == "val":
            dataloader = self.dm.val_dataloader()
        else:
            dataloader = self.dm.test_dataloader()
        pbar = tqdm.tqdm(
            enumerate(dataloader),
            total=len(dataloader),
        )

        for _, batch in pbar:
            labels_texts = batch["labels_texts"]
            sources_texts = batch["sources_texts"]

            max_length = self.max_output_length
            if self.dm.config.model_family == "gpt":
                max_length += batch["input_ids"].shape[-1]

            batch = transfer_batch_to_device(batch, self.device)
            with torch.no_grad():
                if isinstance(model, EfficientCheckpointModule):
                    predictions = model.generate(
                        batch,
                        max_length=max_length,
                        generation_config=model.generation_config,
                        return_dict_in_generate=True,
                        output_scores=True,
                        **extra_kwargs,
                    )
                else:
                    predictions = model.generate(
                        batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        max_length=max_length,
                        generation_config=model.generation_config,
                        return_dict_in_generate=True,
                        output_scores=True,
                        **extra_kwargs,
                    )

            predictions = predictions.sequences
            if self.dm.config.model_family == "gpt":
                predictions = predictions[:, batch["input_ids"].shape[-1] :]

            predictions = decode(predictions, self.tokenizer)
            references = [[l] for l in labels_texts]

            eval_metrics = compute_metrics(predictions, references, reduction="none")
            all_rougeL.extend(eval_metrics["rougeL"])
            if verbose:
                logger.info("Sources:\n%s", sources_texts[0])
                logger.info("Label:\n%s", labels_texts[0])
                logger.info("Prediction:\n%s", predictions[0])

            pbar.set_description(f"rougeL: {np.mean(all_rougeL):.4f}")

            if num_batches is not None and len(all_rougeL) >= num_batches:
                break

        return np.mean(all_rougeL)
