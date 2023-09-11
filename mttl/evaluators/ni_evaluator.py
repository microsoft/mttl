from collections import defaultdict
from copy import deepcopy
import os
import json
import tqdm
import torch
import numpy as np
import pytorch_lightning as pl

from mttl.dataloader.ni_metrics import compute_metrics
from mttl.models.utils import transfer_batch_to_device
from mttl.evaluators.base import compute_task_aggregation


def decode(preds, tokenizer):
    preds[preds == -100] = tokenizer.pad_token_id
    preds = tokenizer.batch_decode(
        preds, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    preds = [pred.strip() for pred in preds]
    return preds


class NIEvaluator(object):
    def __init__(
        self,
        config,
        data_dir=None,
        num_pos_examples=0,
        max_input_length=None,
        device="cuda",
    ):
        from mttl.datamodule.ni_original_data_module import NIOriginalDataModule

        self.config = deepcopy(config)
        self.device = device

        # unrestricted input length for SNI pass -1
        if max_input_length is not None:
            self.config.max_input_length = max_input_length
        self.config.max_output_length = 128
        self.config.num_pos_examples = num_pos_examples
        self.config.use_task_descriptions = True

        if data_dir is None:
            data_dir = config.data_dir

        self.data_dir = data_dir
        self.datamodule = NIOriginalDataModule(
            self.config,
            data_dir=data_dir,
            for_generation=True,
        )
        self.datamodule.setup("test")

    def evaluate(self, model, subsample=-1):
        was_train = model.training
        if was_train:
            model.eval()

        tokenizer = self.datamodule.tokenizer
        samples_seen = 0

        # DDP
        if hasattr(model, "module"):
            model = model.module

        all_predictions = []
        all_references = []
        task_names = []
        all_rougeL = []

        dataloader = self.datamodule.test_dataloader(subsample)
        output_path = self.config.output_dir  
        out_file_name = self.config.out_file_name

        # write results to a file
        if not os.path.exists(output_path):
            # create
            os.makedirs(output_path)

        output_dir = os.path.join(output_path, out_file_name)

        pbar = tqdm.tqdm(
            enumerate(dataloader),
            total=len(dataloader),
        )
        for step, batch in pbar:
            task_name = batch.pop("task_names", None)
            batch.pop("input_texts", None)
            # TODO: add some logic to remove examples from the batch if they ae already in the generated file?
            # we use labels texts here for evaluation, because some tokenizers do not skip
            # pad token when decoding, even if skip_special_tokens=True
            labels_texts = batch.pop("labels_texts", None)

            extra_kwargs = {}
            max_length = self.config.max_output_length  # default output length for NI

            if self.config.model_family == "gpt":
                max_length += batch["input_ids"].shape[-1]

                extra_kwargs["pad_token_id"] = tokenizer.pad_token_id

            batch = transfer_batch_to_device(batch, self.device)
            with torch.no_grad():
                if isinstance(model, pl.LightningModule):
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

            predictions = predictions.sequences
            if self.config.model_family == "gpt":
                predictions = predictions[:, batch["input_ids"].shape[-1] :]
            predictions = decode(predictions, tokenizer)

            references = labels_texts

            # If we are in a multiprocess environment, the last batch has duplicates
            if step == len(dataloader) - 1:
                predictions = predictions[: len(dataloader.dataset) - samples_seen]
                references = references[: len(dataloader.dataset) - samples_seen]
                task_name = task_name[: len(dataloader.dataset) - samples_seen]
            else:
                samples_seen += len(references)

            all_predictions += predictions
            all_references += references
            task_names += task_name

            eval_metrics = compute_metrics(
                predictions, [[r] for r in references], reduction="none"
            )
            all_rougeL.append(np.mean(eval_metrics["rougeL"]))
            pbar.set_description(
                f"Task: {task_name[0] if task_name else None}, rougeL: {np.mean(all_rougeL):.4f}"
            )

            # save generations to a file
            with open(output_dir, "a") as f:
                for p, id_, tn, r, rouge in zip(
                    predictions,
                    batch["instance_ids"],
                    task_name,
                    references,
                    eval_metrics["rougeL"],
                ):
                    l = {
                        "id": id_,
                        "task_name": tn,
                        "prediction": p,
                        "reference": r,
                        "rougeL": rouge,
                    }
                    f.write(json.dumps(l) + "\n")

        eval_metrics = compute_metrics(
            all_predictions, [[r] for r in all_references], reduction="none"
        )

        if was_train:
            model.train()
        return compute_task_aggregation(task_names, eval_metrics["rougeL"])
