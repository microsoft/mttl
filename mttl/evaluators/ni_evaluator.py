from collections import defaultdict
from copy import deepcopy
import re
import json
import tqdm
import torch
import numpy as np
import pytorch_lightning as pl
from pathlib import Path

from mttl.dataloader.ni_metrics import compute_metrics, compute_grouped_metrics
from mttl.models.utils import transfer_batch_to_device
from mttl.evaluators.base import mean_stderr


def decode(preds, tokenizer):
    preds[preds == -100] = tokenizer.pad_token_id
    preds = tokenizer.batch_decode(
        preds, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    preds = [pred.strip() for pred in preds]
    return preds


def compute_aggregation_and_maybe_save(
    results, predictions, references, eval_instances, track="default", output_file=None
):
    instance_ids = [
        id for id, instance in eval_instances.items() if instance["track"] == track
    ]
    all_results = {}
    print("======== Overall Metrics ========")
    for metric, value in results.items():
        print(f"{metric}: {value}")
        all_results[f"{metric}"] = value

    if "task_category" in eval_instances[instance_ids[0]]:
        all_results["per_category"] = {}
        categories = [
            "_".join(eval_instances[id]["task_category"].lower().split())
            for id in instance_ids
        ]
        results_per_category = compute_grouped_metrics(
            predictions, references, categories, xlingual=(track == "xlingual")
        )
        print("======== Metrics per Category ========")
        for metric, value in results_per_category.items():
            print(f"{metric}: {value}")
            all_results["per_category"][f"{metric}"] = value

    if "task_id" in eval_instances[instance_ids[0]]:
        all_results["per_task"] = {}
        tasks = [eval_instances[id]["task_id"] for id in instance_ids]
        results_per_task = compute_grouped_metrics(
            predictions, references, tasks, xlingual=(track == "xlingual")
        )
        print("======== Metrics per Task ========")
        for metric, value in results_per_task.items():
            print(f"{metric}: {value}")
            all_results["per_task"][f"{metric}"] = value

    if output_file:
        with open(output_file, "w") as fout:
            json.dump(all_results, fout, indent=2)
    return all_results


class NIEvaluator(object):
    def __init__(
        self,
        config,
        data_dir=None,
        num_pos_examples=0,
        max_input_length=None,
        pred_output_file_path=None,
        device="cuda",
    ):
        from mttl.datamodule.ni_original_data_module import NIOriginalDataModule

        self.config = deepcopy(config)
        self.device = device
        self.pred_output_file_path = (
            pred_output_file_path  # if not None, will trute generations into it
        )

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

        # DDP
        if hasattr(model, "module"):
            model = model.module

        all_predictions = []
        all_task_names = []
        all_rougeL = []

        dataloader = self.datamodule.test_dataloader(subsample)
        if not self.pred_output_file_path is None:
            path = re.sub(r"/[^/]*$", "", self.pred_output_file_path)
            Path(path).mkdir(parents=True, exist_ok=True)

        pbar = tqdm.tqdm(
            enumerate(dataloader),
            total=len(dataloader),
        )
        eval_instances = {}
        for step, batch in pbar:
            task_names = batch.pop("task_names", None)
            batch.pop("input_texts", None)
            # we use labels texts here for evaluation, because some tokenizers do not skip
            # pad token when decoding, even if skip_special_tokens=True
            labels_texts = batch.pop("labels_full_seq", None)
            task_ids = batch.pop("task_identifiers", None)
            task_categories = batch.pop("task_categories", None)

            extra_kwargs = {}
            max_length = self.config.max_output_length  # default output length for NI

            for id, category, label_text, task_id in zip(
                batch["instance_ids"], task_categories, labels_texts, task_ids
            ):
                eval_instances[id] = {
                    "id": id,
                    "track": "default",
                    "references": label_text,
                    "task_category": category[0],
                    "task_id": task_id,
                }

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

            all_predictions += predictions
            all_task_names += task_names

            eval_metrics = compute_metrics(predictions, references, reduction="none")
            all_rougeL.extend(eval_metrics["rougeL"])

            pbar.set_description(
                f"Task: {task_names[0] if task_names else None}, rougeL: {np.mean(all_rougeL):.4f}"
            )

            # save generations to a file
            if self.pred_output_file_path is not None:
                with open(self.pred_output_file_path, "a") as f:
                    for p, id_, tn, r, rouge in zip(
                        predictions,
                        batch["instance_ids"],
                        task_names,
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

        instance_ids = [id for id, instance in eval_instances.items()]
        all_references = [eval_instances[id]["references"] for id in instance_ids]
        eval_metrics = compute_metrics(all_predictions, all_references)

        if was_train:
            model.train()

        all_results = compute_aggregation_and_maybe_save(
            eval_metrics,
            all_predictions,
            all_references,
            eval_instances,
            track="default",
        )
        all_results["all"] = {}
        all_results["all"]["mean"] = all_results["rougeL"]
        if "per_task" in all_results:
            all_results["all"]["stderr"] = mean_stderr(
                [v for k, v in all_results["per_task"].items() if "rougeL" in k]
            )
        return all_results
