import os

from transformers import (
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from mttl.evaluators.base import EvaluatorRunner, setup_evaluators


class DownstreamEvalCallback(TrainerCallback):
    METRIC_KEY = "downstream"

    def __init__(self, model, args) -> None:
        super().__init__()

        self.model = model
        self.args = args
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

    def on_log():
        pass

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs=None,
        **kwargs,
    ):
        if (
            self.args.eval_every_n_epoch is not None
            and state.epoch is not None
            and state.epoch % self.args.eval_every_n_epoch != 0
        ):
            return

        metrics = self.runner.run(self.model)
        for task, metric in metrics.items():
            state.log_history.append(
                {f"{self.METRIC_KEY}/{task}": metric, "step": state.global_step}
            )
            control.should_log = True

    def on_predict(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs=None,
        **kwargs,
    ) -> None:
        metrics = self.runner.run(self.model)
        for task, metric in metrics.items():
            state.log_history.append(
                {f"{self.METRIC_KEY}_last/{task}": metric, "step": state.global_step}
            )
            control.should_log = True
