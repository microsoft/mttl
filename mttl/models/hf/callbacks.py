import os

from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from mttl.evaluators.base import EvaluatorRunner, setup_evaluators
from mttl.logging import maybe_wandb_log


class DownstreamEvalCallback(TrainerCallback):
    METRIC_KEY = "downstream"

    def __init__(self, model, args) -> None:
        super().__init__()

        self.model = model
        self.args = args
        self.last_log = None
        self.runner: EvaluatorRunner = setup_evaluators(
            model_type=args.model,
            model_family=args.model_family,
            max_input_length=args.max_input_length,
            max_output_length=args.max_output_length,
            predict_batch_size=args.predict_batch_size,
            truncation_side=args.truncation_side,
            tasks=args.pipeline_eval_tasks,
            output_path=os.path.join(args.output_dir, self.METRIC_KEY),
            add_eos_to_targets=args.add_eos_to_downstream_targets,
        )

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics={},
        **kwargs,
    ):
        if state.is_world_process_zero:
            metrics_ = self.runner.run(self.model)

            all_metrics = {}
            for task, metric in metrics_.items():
                all_metrics.update({f"{self.METRIC_KEY}/{task}": metric})

            # record in log_history
            state.log_history.append({**all_metrics, **{"step": state.global_step}})

            maybe_wandb_log(all_metrics)
            metrics_.update(all_metrics)

        return control

    def on_predict(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics={},
        **kwargs,
    ) -> None:
        if state.is_world_process_zero:
            metrics_ = self.runner.run(self.model)

            all_metrics = {}
            for task, metric in metrics_.items():
                all_metrics.update({f"{self.METRIC_KEY}_last/{task}": metric})

            state.log_history.append({**all_metrics, **{"step": state.global_step}})

            maybe_wandb_log(all_metrics)
            metrics_.update(all_metrics)

        return control
