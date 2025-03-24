from typing import Dict

import numpy as np

from mttl.arguments import create_config_class_from_args
from mttl.dataloader.ni_metrics import compute_metrics
from mttl.datamodule.base import DataModule, DatasetConfig
from mttl.datamodule.utils import maybe_filter_hf_dataset_by_task
from mttl.evaluators.base import (
    GenerativeEvaluator,
    compute_task_aggregation,
    switch_to_eval_mode,
)
from mttl.evaluators.loglike_evaluator import LogLikeEvaluator
from mttl.logging import logger, warn_once
from projects.kms.utils.nqa_datamodule import NQADatamodule, NQADatasetConfig
from projects.kms.utils.quality_datamodule import (
    GenQualityDataModule,
    GenQualityDatasetConfig,
    QualityDatamodule,
    QualityDatasetConfig,
)


class QualityEvaluator(LogLikeEvaluator):
    def __init__(self, dataset_args: "DataArgs", generation_kwargs: Dict = {}):
        import copy

        from mttl.datamodule.base import get_datamodule

        dataset_args = copy.deepcopy(dataset_args)
        dataset_args.dataset_type = "quality"
        dataset_args.add_eos_to_targets = False

        datamodule = get_datamodule(dataset_args, for_generation=False)
        super().__init__(
            datamodule, generation_kwargs=generation_kwargs, length_normalization=True
        )

    def evaluate(self, model, split=None, **kwargs):

        # splits in order of preference
        splits = ["dev", "train", "test"]

        if split is not None:
            assert split in splits, f"Split {split} not found in {splits}"
            dataset = getattr(self.datamodule, f"{split}_dataset")
            if dataset is None or len(dataset) == 0:
                warn_once(
                    f"invalid dataset for split {split}, \n{dataset}. Using default split."
                )
                split = None

        if split is None:
            for split in splits:
                dataset = getattr(self.datamodule, f"{split}_dataset")
                if dataset and len(dataset):
                    break

        return super().evaluate(model, split=split, **kwargs)


class GenQualityEvaluator(GenerativeEvaluator):
    def __init__(self, dataset_args: "DataArgs", generation_kwargs: Dict = {}):
        import copy

        from mttl.datamodule.base import get_datamodule

        dataset_args = copy.deepcopy(dataset_args)
        dataset_args.dataset_type = "gen_quality"
        dataset_args.add_eos_to_targets = False

        generation_kwargs["max_new_tokens"] = 4
        generation_kwargs["do_sample"] = False

        datamodule = get_datamodule(dataset_args, for_generation=True)
        super().__init__(
            datamodule,
            generation_kwargs=generation_kwargs,  # length_normalization=True
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
        verbose=False,
        **kwargs,
    ):
        all_references = []
        all_predictions = []
        all_task_names = []
        all_EM = []

        dataloader = self.get_dataloader(split, subsample, shuffle)

        from tqdm.auto import tqdm

        pbar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
        )
        for num_batch, batch in pbar:
            if num_batches and num_batch >= num_batches:
                break

            task_names = batch.get("task_names", None)
            raw_predictions = self.generate_for_batch(model, batch)
            predictions = [pred.strip()[0] for pred in raw_predictions.generated_texts]

            sources_texts = batch.pop("sources_texts", None)
            labels_texts = batch.pop("labels_texts", None)

            if verbose:
                logger.info("Sources:\n%s", sources_texts[0])
                logger.info("Predictions:\n%s", predictions[0])
                logger.info("Label:\n%s", labels_texts[0])

            invalid = [
                i
                for i in range(len(predictions))
                if predictions[i] not in ["A", "B", "C", "D"]
            ]
            if len(invalid) > 0:
                print(
                    f"Invalid predictions: {[raw_predictions.generated_texts[i] for i in invalid]}"
                )

            all_references += labels_texts
            all_predictions += predictions
            all_task_names += task_names

            eval_metrics = compute_metrics(
                predictions, [[r] for r in labels_texts], reduction="none"
            )

            all_EM.extend(eval_metrics["exact_match"])
            pbar.set_description(
                f"Task: {task_names[0] if task_names else None}, EM: {np.mean(all_EM):.4f}"
            )

        eval_metrics = compute_metrics(
            all_predictions, [[r] for r in all_references], reduction="none"
        )
        metrics = compute_task_aggregation(all_task_names, eval_metrics["exact_match"])

        self.save_metrics(metrics, output_path)
        return metrics["all"]["mean"]
