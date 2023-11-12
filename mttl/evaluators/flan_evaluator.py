from copy import deepcopy
import re
import tqdm
import torch
import numpy as np
import pytorch_lightning as pl
from pathlib import Path
import rouge
import os
import json

from mttl.models.utils import transfer_batch_to_device


def decode(preds, tokenizer):
    preds[preds == -100] = tokenizer.pad_token_id
    preds = tokenizer.batch_decode(
        preds, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    preds = [pred.strip() for pred in preds]
    return preds


class FlanEvaluator(object):
    def __init__(
        self,
        config,
        max_input_length=None,
        pred_output_file_path=None,
        device="cuda",
    ):
        self.config = deepcopy(config)
        self.device = device
        self.pred_output_file_path = (
            pred_output_file_path  # if not None, will trute generations into it
        )

    def load_data_module(self, for_generation=True):
        from mttl.datamodule.flan10k_module import Flan10kModule

        self.for_generation = for_generation
        self.datamodule = Flan10kModule(
            self.config,
            for_generation=self.for_generation,
        )

    def evaluate(self, model, subsample=-1):
        was_train = model.training
        if was_train:
            model.eval()

        tokenizer = self.datamodule.tokenizer

        # DDP
        if hasattr(model, "module"):
            model = model.module

        all_rougeL = []

        dataloader = self.datamodule.test_dataloader()
        if not self.pred_output_file_path is None:
            path = re.sub(r"/[^/]*$", "", self.pred_output_file_path)
            Path(path).mkdir(parents=True, exist_ok=True)

        pbar = tqdm.tqdm(
            enumerate(dataloader),
            total=len(dataloader),
        )

        # get all val loss
        all_val_loss = []
        if not os.path.exists(self.pred_output_file_path):
            os.mkdir(self.pred_output_file_path)
        f = open(self.pred_output_file_path + "/predict.jsonl", "w")
        for _, batch in pbar:
            task_names = batch.get("task_names", None)
            batch.pop("input_texts", None)
            # we use labels texts here for evaluation, because some tokenizers do not skip
            # pad token when decoding, even if skip_special_tokens=True
            labels_texts = batch.pop("labels_texts", None)

            extra_kwargs = {}
            max_length = self.config.max_output_length

            if self.config.model_family == "gpt":
                max_length += batch["input_ids"].shape[-1]

                extra_kwargs["pad_token_id"] = tokenizer.pad_token_id

            batch = transfer_batch_to_device(batch, self.device)
            with torch.no_grad():
                if isinstance(model, pl.LightningModule):
                    if not self.for_generation:
                        # get loss ()
                        loss = model.validation_step(batch, batch_idx=0)
                        all_val_loss.extend(loss.cpu())
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
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        max_length=max_length,
                        generation_config=model.generation_config,
                        return_dict_in_generate=True,
                        output_scores=True,
                        **extra_kwargs,
                    )
            predictions_texts = tokenizer.batch_decode(predictions.sequences)
            predictions = predictions.sequences
            if self.config.model_family == "gpt":
                predictions = predictions[:, batch["input_ids"].shape[-1] :]

            predictions = decode(predictions, tokenizer)

            # write the inputs_texts, predictions, labels_texts to json file
            if not self.pred_output_file_path is None:
                for i in range(len(predictions)):
                    result_str = (
                        json.dumps(
                            {
                                "input_texts": batch["sources_texts"][i],
                                "predictions": predictions[i],
                                "labels_texts": labels_texts[i],
                            }
                        )
                        + "\n"
                    )
                    f.write(result_str)  # write to file

            references = labels_texts
            try:
                rouge_score = rouge.Rouge().get_scores(
                    predictions, references, avg=True
                )
                all_rougeL.extend([rouge_score["rouge-1"]["f"]])
            except ValueError:
                all_rougeL.extend([0.0])

            pbar.set_description(
                f"Task: {task_names[0] if task_names else None}, rougeL: {np.mean(all_rougeL):.4f}, val_loss: {np.mean(all_val_loss):.4f}"
            )

        return all_rougeL
