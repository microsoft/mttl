import tqdm
import torch
import hashlib
import numpy as np

import os
from mttl.evaluators.ni_evaluator import compute_metrics
from mttl.evaluators.mmlu_evaluator import swap_model
from mttl.models.utils import transfer_batch_to_device, EfficientCheckpointModule
from mttl.utils import logger
from mttl.vllm_engines.engines import LLMEngineRouge, free_memory


def decode(preds, tokenizer):
    preds[preds == -100] = tokenizer.pad_token_id
    preds = tokenizer.batch_decode(
        preds, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    preds = [pred.strip() for pred in preds]
    return preds


class RougeEvaluator:
    def __init__(
        self, datamodule, device="cuda", use_vllm=False, split="val", subsample=-1
    ):
        super().__init__()
        self.device = device
        self.dm = datamodule
        self.use_vllm = use_vllm
        self.tokenizer = datamodule.tokenizer
        self.max_output_length = datamodule.config.max_output_length

        if split in ["val", "valid", "validation"]:
            self.dataloader = self.dm.val_dataloader(subsample=subsample)
        elif split in ["test", "testing"]:
            self.dataloader = self.dm.test_dataloader(subsample=subsample)
        else:
            self.dataloader = self.dm.train_dataloader(subsample=subsample)

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

    def evaluate(self, model, num_batches=None, verbose=True):
        extra_kwargs = {}
        extra_kwargs["pad_token_id"] = self.tokenizer.pad_token_id
        extra_kwargs["eos_token_id"] = self.tokenizer.eos_token_id

        all_rougeL = []
        if model.training:
            was_training = True
            model.eval()

        with torch.no_grad():
            dataloader = self.dataloader

            if self.use_vllm:
                return self.evaluate_with_vllm(model, dataloader, num_batches, verbose)

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

                eval_metrics = compute_metrics(
                    predictions, references, reduction="none"
                )
                all_rougeL.extend(eval_metrics["rougeL"])
                if verbose:
                    logger.info("Sources:\n%s", sources_texts[0])
                    logger.info("Label:\n%s", labels_texts[0])
                    logger.info("Prediction:\n%s", predictions[0])

                pbar.set_description(f"rougeL: {np.mean(all_rougeL):.4f}")

                if num_batches is not None and len(all_rougeL) >= num_batches:
                    break
                if was_training:
                    model.train()

        del pbar, dataloader
        return np.mean(all_rougeL)
