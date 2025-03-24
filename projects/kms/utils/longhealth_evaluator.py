import re
from typing import Dict

import numpy as np
import torch

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
from mttl.evaluators.rouge_evaluator import RougeEvaluator
from mttl.logging import warn_once
from projects.kms.utils.longhealth_datamodule import (
    LonghealthDatamodule,
    LonghealthDatasetConfig,
)
from projects.kms.utils.nqa_datamodule import NQADatamodule, NQADatasetConfig
from projects.kms.utils.quality_evaluator import QualityEvaluator


class LonghealthEvaluator(GenerativeEvaluator):
    def __init__(self, dataset_args: "DataArgs", generation_kwargs: Dict = {}):
        import copy

        from mttl.datamodule.base import get_datamodule

        dataset_args = copy.deepcopy(dataset_args)
        dataset_args.dataset_type = "longhealth"
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

            """
            '<|eot_id|><|eot_id|><|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\nYou are a highly skilled and detail-oriented assistant, specifically trained to assist medical professionals in interpreting and extracting key information from medical documents. Your primary responsibility will be to analyze discharge letters from hospitals. When you receive one or more of these letters, you will be expected to carefully review the contents and accurately answer multiple-choice questions related to these documents. \n\nYour answers should be:\n1. Accurate: Make sure your answers are based on the information provided in the letters.\n2. Concise: Provide brief and direct answers without unnecessary elaboration.\n3. Contextual: Consider the context and specifics of each question to provide the most relevant information.\n\nRemember, your job is to streamline the physician\'s decision-making process by providing them with accurate and relevant information from discharge summaries. Efficiency and reliability are key.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWhat was Mr. Davies\' bone density T-score in the lumbar spine (L2-4) in his last measurement?\nA: -0.8\nB: -0.5\nC: -0.2\nD: -1.5\nE: -2.5\n\nPlease answer using the following format:\n1. Begin your answer with the phrase "The correct answer is".\n2. State the letter of the correct option (e.g., A, B, C, D, E).\n3. Follow the letter with a colon and the exact text of the option you chose.\n4. Make sure your answer is a single, concise sentence.\n\nFor example, if the correct answer to a question is option C, and the text for C is \'Acute Bronchitis\', your answer should be: \n\'The correct answer is C: Acute bronchitis.\'<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
            """
            task_names = batch.get("task_names", None)
            labels_text = batch.pop("labels_texts", None)

            raw_predictions = self.generate_for_batch(model, batch)
            predictions = [pred.strip()[0] for pred in raw_predictions.generated_texts]

            invalid = [
                i
                for i in range(len(predictions))
                if predictions[i] not in ["A", "B", "C", "D", "E"]
            ]
            if len(invalid) > 0:
                print(
                    f"Invalid predictions: {[raw_predictions.generated_texts[i] for i in invalid]}"
                )

            all_references += labels_text
            all_predictions += predictions
            all_task_names += task_names

            eval_metrics = compute_metrics(
                predictions, [[r] for r in labels_text], reduction="none"
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
