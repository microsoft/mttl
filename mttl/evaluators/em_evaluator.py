import tqdm
import torch
import numpy as np

from mttl.evaluators.base import Evaluator, switch_to_eval_mode
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


class EMEvaluator(Evaluator):
    def __init__(
        self, datamodule, device="cuda", use_vllm=False, generation_kwargs=None
    ):
        super().__init__(
            datamodule=datamodule,
            device=device,
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
        max_length=None,
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
        all_em = []

        for _, batch in pbar:
            labels_texts = batch["labels_texts"]
            sources_texts = batch["sources_texts"]

            max_length = max_length or self.max_output_length

            batch = transfer_batch_to_device(batch, self.device)
            with torch.no_grad():
                if isinstance(model, EfficientCheckpointModule):
                    predictions = model.generate(
                        batch,
                        max_new_tokens=max_length,
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
                        max_new_tokens=max_length,
                        generation_config=model.generation_config,
                        return_dict_in_generate=True,
                        output_scores=True,
                        **extra_kwargs,
                        **self.generation_kwargs,
                    )

            predictions = predictions.sequences
            if self.datamodule.config.model_family == "gpt":
                predictions = predictions[:, batch["input_ids"].shape[-1] :]

            predictions = decode(predictions, self.tokenizer)
            references = [[l] for l in labels_texts]

            eval_metrics = compute_metrics(predictions, references, reduction="none")
            all_em.extend(eval_metrics["exact_match"])
            if verbose:
                logger.info("Sources:\n%s", sources_texts[0])
                logger.info("Label:\n%s", labels_texts[0])
                logger.info("Prediction:\n%s", predictions[0])

            pbar.set_description(f"exact_match: {np.mean(all_em):.4f}")

            if num_batches is not None and len(all_em) >= num_batches:
                break

        return np.mean(all_em)
