import os

from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from mttl.evaluators.base import EvaluatorRunner, setup_evaluators
from mttl.evaluators.rouge_evaluator import RougeEvaluator
from mttl.logging import maybe_wandb_log
from projects.kms.utils.nqa_datamodule import NQADatamodule, NQADatasetConfig
from projects.kms.utils.nqa_evaluator import NQAZeroShotEvaluator


class NQAZeroShotCallback(TrainerCallback):
    METRIC_KEY = "nqa"

    def __init__(self, model, args) -> None:
        super().__init__()

        self.model = model
        self.last_log = None

        # setup evaluator
        self.evaluator = NQAZeroShotEvaluator(args, generation_kwargs={})

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics={},
        **kwargs,
    ):
        if state.is_world_process_zero:
            rougeL, predictions = self.evaluator.evaluate(
                self.model, split="test", return_predictions=True
            )
            metrics_ = {f"{self.METRIC_KEY}_test/rougeL": rougeL}

            # record in log_history
            state.log_history.append({**metrics_, **{"step": state.global_step}})
            maybe_wandb_log(metrics_)
            metrics.update(metrics_)
        return control

    def on_predict(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics={},
        **kwargs,
    ) -> None:
        return self.on_evaluate(args, state, control, metrics=metrics, **kwargs)
