import os

from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from mttl.evaluators.base import EvaluatorRunner, setup_evaluators
from mttl.logging import maybe_wandb_log


class UpdateSparseMask(pl.Callback):
    def __init__(self, update_interval=5, dm=None, save_mask_dir=None, task_name=None, parameter_selection_procedure='per_layer'):
        super().__init__()
        self.update_interval = update_interval
        self.update_counter = 0
        self.dm = dm
        self.save_mask_dir = save_mask_dir 
        self.task_name = task_name
        assert parameter_selection_procedure in ['model','per_layer'], "choose the right `parameter_selection_procedure`"
        self.parameter_selection_procedure=parameter_selection_procedure


    def update_mask(self, pl_module, batch):
        make_sparse_model_during_training(pl_module, batch, parameter_selection_procedure=self.parameter_selection_procedure)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """
        only updates the mask on epoch=0
        """
        if trainer.current_epoch==0:
            self.update_counter += 1
            if self.update_counter % self.update_interval == 0:
                # Update mask
                self.update_mask(pl_module, batch)
                self.update_counter = 0  # Reset counter for next interval
                # save mask: if needed, if need to upload to hf, use `on_train_epoch_end` or `on_train_end` to reduce number of request to hf server
                # f_name = f'{self.save_mask_dir}/{self.task_name}_mask'
                # # f_name = f'{self.save_mask_dir}/mask'
                # save_mask(pl_module, f_name)
                
                # TODO: don't use this during training, use only required during debugging
                #self.on_train_end(trainer, pl_module) 

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        f_name = f'{self.save_mask_dir}/{self.task_name}_mask'
        save_mask(pl_module, f_name)
        
        #from mttl.models.modifiers.sparse_mask import compress_sparse_2D_weight
        #compress_sparse_2D_weight(pl_module)


    # def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
    #     f_name = f'{self.save_mask_dir}/{self.task_name}_mask'
    #     # f_name = f'{self.save_mask_dir}/mask'
    #     save_mask(pl_module, f_name)
    

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
