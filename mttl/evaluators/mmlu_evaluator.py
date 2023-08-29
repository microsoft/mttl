from collections import defaultdict
from copy import deepcopy

from mttl.dataloader.ni_metrics import compute_metrics

import tqdm
import torch
import re
import string
import datasets
import numpy as np


class MMLUEvaluator(object):
    def __init__(self, config, data_dir=None, num_pos_examples=0, device="cuda"):
        from mttl.datamodule.mmlu_data_module import MMLUDataModule

        self.config = deepcopy(config)
        self.device = device
        self.config.num_pos_examples = num_pos_examples

        if data_dir is None:
            data_dir = config.data_dir

        self.data_dir = data_dir
        self.datamodule = MMLUDataModule(self.config, data_dir=self.data_dir)
        self.datamodule.setup("test")

    def evaluate(self, model, metric_per_task=True, eval_batches=-1):
        tokenizer = self.datamodule.tokenizer
        samples_seen = 0

        # DDP
        if hasattr(model, "module"):
            model = model.module

        def decode(preds):
            preds[preds == -100] = tokenizer.pad_token_id
            preds = tokenizer.batch_decode(
                preds, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            preds = [pred.strip() for pred in preds]
            return preds

        all_predictions = []
        all_references = []
        task_names = []
        all_EM = []

        dataloader = self.datamodule.test_dataloader()
        pbar = tqdm.tqdm(
            enumerate(dataloader),
            total=len(dataloader),
        )
        for step, batch in pbar:
            task_name = batch.pop("task_names", None)
            batch.pop("input_texts", None)
            labels_text = batch.pop("labels_texts", None)

            extra_kwargs = {}

            max_length = self.config.max_output_length
            if self.config.model_family == "gpt":
                max_length += batch["input_ids"].shape[-1]
                extra_kwargs["pad_token_id"] = tokenizer.pad_token_id

            with torch.no_grad():
                try:
                    batch["input_ids"] = batch["input_ids"].to(self.device)
                    batch["attention_mask"] = batch["attention_mask"].to(self.device)
                    predictions = model.generate(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        max_length=max_length,
                        generation_config=model.generation_config,
                        return_dict_in_generate=True,
                        output_scores=True,
                        **extra_kwargs,
                    )
                except:
                    batch = model.transfer_batch_to_device(batch, self.device, 0)
                    predictions = model.generate(
                        batch,
                        max_length=max_length,
                        generation_config=model.generation_config,
                        return_dict_in_generate=True,
                        output_scores=True,
                        **extra_kwargs,
                    )

            logits = predictions.scores[0]
            probs = (
                torch.stack(
                    [
                        logits[:, tokenizer(" A").input_ids[0]],
                        logits[:, tokenizer(" B").input_ids[0]],
                        logits[:, tokenizer(" C").input_ids[0]],
                        logits[:, tokenizer(" D").input_ids[0]],
                    ],
                    dim=-1,
                )
                .max(dim=-1)[1]
                .detach()
                .cpu()
                .numpy()
            )
            predictions = [{0: "A", 1: "B", 2: "C", 3: "D"}[p] for p in probs]
            references = labels_text

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
                predictions, [[r] for r in references], reduction="mean"
            )

            all_EM.append(eval_metrics["exact_match"])
            pbar.set_description(f"EM: {np.mean(all_EM):.4f}")

            if step == eval_batches:
                break

        eval_metrics = compute_metrics(
            all_predictions, [[r] for r in all_references], reduction="none"
        )
        mean_metrics = {}
        for metric_name, metric_value in eval_metrics.items():
            metric_value = sum(eval_metrics[metric_name]) / len(
                eval_metrics[metric_name]
            )
            mean_metrics[metric_name] = metric_value

        if metric_per_task:
            metric_values_all = {}

            for metric_name in eval_metrics.keys():
                metric_values = defaultdict(list)
                for task_name, v in zip(task_names, eval_metrics[metric_name]):
                    metric_values[task_name] += [v]
                metric_values["all"] = [mean_metrics[metric_name]]
                metric_values = {
                    task_name: sum(vs) / len(vs)
                    for task_name, vs in metric_values.items()
                }
                metric_values_all[metric_name] = metric_values
            metric_values = metric_values_all
        else:
            metric_values = mean_metrics
        return metric_values
