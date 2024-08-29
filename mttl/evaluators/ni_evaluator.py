import json
import re
from pathlib import Path

import numpy as np
import tqdm

from mttl.dataloader.ni_metrics import compute_grouped_metrics, compute_metrics
from mttl.evaluators.base import GenerativeEvaluator, mean_stderr, switch_to_eval_mode


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


class NIEvaluator(GenerativeEvaluator):
    def __init__(
        self,
        config,
        num_pos_examples=0,
        max_input_length=None,
    ):
        from mttl.datamodule.ni_data_module import NiDataModule

        super().__init__(config=config)

        # unrestricted input length for SNI pass -1
        if max_input_length is not None:
            self.config.max_input_length = max_input_length

        self.config.max_output_length = 128
        self.config.num_pos_examples = num_pos_examples
        self.config.use_task_descriptions = True

        self.datamodule = NiDataModule(
            self.config,
            for_generation=True,
        )

    @switch_to_eval_mode
    def evaluate(
        self,
        model,
        split="test",
        subsample=-1,
        num_batches=None,
        shuffle=False,
        output_path=None,
    ):
        dataloader = self.get_dataloader(split, subsample, shuffle)

        all_predictions = []
        all_task_names = []
        all_rougeL = []

        if output_path:
            path = re.sub(r"/[^/]*$", "", output_path)
            Path(path).mkdir(parents=True, exist_ok=True)

        pbar = tqdm.tqdm(
            enumerate(dataloader),
            total=len(dataloader),
        )
        eval_instances = {}

        for num_batch, batch in pbar:
            task_names = batch.pop("task_names", None)
            batch.pop("input_texts", None)
            # we use labels texts here for evaluation, because some tokenizers do not skip
            # pad token when decoding, even if skip_special_tokens=True
            labels_texts = batch.pop("labels_full_seq", None)
            task_ids = batch.pop("task_identifiers", None)
            task_categories = batch.pop("task_categories", None)

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

            outputs = self.generate_for_batch(model, batch)
            references = labels_texts

            all_predictions += outputs.generated_texts
            all_task_names += task_names

            eval_metrics = compute_metrics(
                outputs.generated_texts, references, reduction="none"
            )
            all_rougeL.extend(eval_metrics["rougeL"])

            pbar.set_description(
                f"Task: {task_names[0] if task_names else None}, rougeL: {np.mean(all_rougeL):.4f}"
            )

            # save generations to a file
            if output_path is not None:
                with open(output_path + "/generations.jsonl", "a") as f:
                    for p, id_, tn, r, rouge in zip(
                        outputs.generated_texts,
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

            if num_batches and num_batch >= num_batches:
                break

        instance_ids = [id for id, _ in eval_instances.items()]
        all_references = [eval_instances[id]["references"] for id in instance_ids]
        eval_metrics = compute_metrics(all_predictions, all_references)

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

        self.save_metrics(all_results, output_path)
        return all_results["all"]["mean"]
