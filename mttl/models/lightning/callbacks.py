import copy
import os
import shutil
import sys
from abc import ABC, abstractmethod

import pytorch_lightning as pl
import torch
import tqdm
import wandb
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning import callbacks as cb
from pytorch_lightning.callbacks.progress.tqdm_progress import Tqdm
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from torch.optim import Optimizer
from transformers.utils import ModelOutput

from mttl.datamodule.base import DataModule
from mttl.evaluators import MMLUEvaluator
from mttl.evaluators.base import EvaluatorRunner, setup_evaluators
from mttl.evaluators.evaluators import Evaluator
from mttl.logging import logger
from mttl.models.utils import transfer_batch_to_device

DEBUG = False


class LiveCheckpointCallback(pl.Callback):
    """A better model checkpoint callback, that works in synchrony with LiveLogMixin."""

    def __init__(
        self,
        dirpath,
        monitor=None,
        mode: str = "min",
        save_last: bool = True,
        save_weights_only: bool = True,
        save_each_epoch: bool = False,
    ):
        if not monitor and not save_last:
            raise ValueError(
                "Must specify a monitor metric to track if save_last is False."
            )

        self.dirpath = dirpath
        self.monitor = monitor
        self.mode = mode
        self.best_model_path = None
        self.last_model_path = None
        self.save_last = save_last
        self._last_step = -1
        self._last_value = None
        self.save_weights_only = save_weights_only
        self.save_each_epoch = save_each_epoch

    def _store_checkpoint(self, trainer, checkpoint_path):
        """Saves the checkpoint and pushes to the ExpertLibrary if one is available."""
        trainer.save_checkpoint(checkpoint_path, weights_only=self.save_weights_only)

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Saves the last checkpoint."""
        if self.save_last:
            self.last_model_path = os.path.join(f"{self.dirpath}", "last.ckpt")
            self._store_checkpoint(trainer, self.last_model_path)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Saves each checkpoint after each epoch"""
        if self.save_each_epoch:
            checkpoint_name = f"epoch_{trainer.current_epoch}"
            save_model_path = os.path.join(f"{self.dirpath}", f"{checkpoint_name}.ckpt")
            self._store_checkpoint(trainer, save_model_path)

    @rank_zero_only
    def _maybe_delete_best_path(self):
        if self.best_model_path and os.path.exists(self.best_model_path):
            # added this to enable training with multiple GPUS (with deepspeed)
            # deepspeed seem to create a checkpoint per device, resulting in self.best_model_path being a directory
            if os.path.isdir(self.best_model_path):
                shutil.rmtree(self.best_model_path)
            else:
                os.remove(self.best_model_path)

    def _save_best(self, trainer, this_value):
        if this_value is None:
            raise ValueError("No value to save.. Something has gone wrong!")

        monitor = self.monitor.replace("/", "-")
        monitor = monitor.replace("_", "-")

        self._maybe_delete_best_path()
        this_filename = os.path.join(
            f"{self.dirpath}",
            f"best_mode_{self.mode}_metric_{monitor}_value_{this_value:.004f}_step_{self._last_step}.ckpt",
        )

        logger.info("Saving new best model to %s", this_filename)
        self._store_checkpoint(trainer, this_filename)
        self.best_model_path = this_filename

    def on_log(self, trainer, pl_module, metric_name, metric_value, **kwargs):
        """Dummy callback called by LiveLogMixin. Every time a metric is logged,
        we call this function to check if we should save a checkpoint.
        """
        if not self.monitor:
            return

        if metric_name != self.monitor:
            return

        if trainer.global_step == 0:
            # skip checkpoint before training (deepspeed fails checkpointing if optimizer non-existent
            return

        last_step = trainer.global_step

        if last_step == self._last_step:
            return

        old_value = metric_value.clone()
        sync_dist = kwargs.get("sync_dist", False)
        if sync_dist:
            is_dist_initialized = (
                torch.distributed.is_available() and torch.distributed.is_initialized()
            )
            world_size = (
                torch.distributed.get_world_size() if is_dist_initialized else 1
            )
            if is_dist_initialized and world_size > 1:
                assert isinstance(
                    metric_value, torch.Tensor
                ), "sync_dist=True requires a scalar value"
                metric_value = metric_value.to(torch.float32)
                torch.distributed.all_reduce(metric_value)
                metric_value = metric_value / world_size

        # compare last_value and _last_value wrt self.mode
        do_save = False
        self._last_step = last_step

        if self.mode == "min":
            do_save = self._last_value is None or metric_value < self._last_value
        else:
            do_save = self._last_value is None or metric_value > self._last_value

        if do_save:
            self._save_best(trainer, metric_value)
            self._last_value = metric_value


class LossCallback(cb.Callback):
    def __init__(
        self,
        dataloader,
        output_dir,
        name="test",
        eval_every_opt_step=1,
        checkpoint_oracle=True,
    ):
        self.name = name
        self.output_dir = output_dir
        self.dataloader = dataloader
        self.eval_every_opt_step = eval_every_opt_step
        # save best perf
        self._best_loss = None
        # checkpointing
        self.do_checkpoint = checkpoint_oracle
        self._checkpoint_now = False
        self._prev_checkpoint = None

    @property
    def last_model_path(self):
        return self._prev_checkpoint

    @property
    def best_model_path(self):
        return self._prev_checkpoint

    @property
    def last_chkpt(self):
        return self._prev_checkpoint

    @property
    def best_loss(self):
        return self._best_loss

    @best_loss.setter
    def best_loss(self, value):
        if self._best_loss is None:
            self._best_loss = value
            self._checkpoint_now = True
        else:
            if value < self._best_loss:
                self._checkpoint_now = True
                self._best_loss = value

    def on_before_optimizer_step(
        self, trainer: Trainer, pl_module: LightningModule, optimizer: Optimizer
    ) -> None:
        if trainer.global_step % self.eval_every_opt_step == 0:
            metrics = self.test(pl_module)
            self.best_loss = copy.deepcopy(metrics)
            self.maybe_checkpoint_now(trainer)
            self.log_metrics(metrics, pl_module)
            # checksum of parameters
            # p_sum = np.sum([p.detach().cpu().sum() for p in pl_module.parameters()])
        return super().on_before_optimizer_step(trainer, pl_module, optimizer)

    def maybe_checkpoint_now(self, trainer):
        if self.do_checkpoint and self._checkpoint_now:
            try:
                dir_name = trainer.checkpoint_callback.dirpath
                filename = (
                    self.output_dir + f"{self.name}/" + f"{self.best_loss:.004f}.ckpt"
                )
                ckpt_path = os.path.join(dir_name, filename)
                trainer.save_checkpoint(ckpt_path)
                if (
                    self._prev_checkpoint is not None
                    and ckpt_path != self._prev_checkpoint
                ):
                    os.remove(self._prev_checkpoint)
                self._prev_checkpoint = ckpt_path
            except Exception as e:
                logger.error(e)
        self._checkpoint_now = False

    def test(self, pl_module: LightningModule):
        outputs = []
        was_train = pl_module.training
        if was_train:
            pl_module.eval()

        total_loss, deno = 0.0, 0.0
        with torch.no_grad():
            for i, batch in tqdm.tqdm(
                enumerate(self.dataloader),
                total=len(self.dataloader),
                desc=f"Test {self.name}",
            ):
                batch = transfer_batch_to_device(batch, pl_module.device)
                loss = pl_module.forward(**batch, reduction="none")

                if isinstance(loss, ModelOutput):
                    loss = loss.loss

                total_loss += loss.detach().cpu()
                deno += 1

        if was_train:
            pl_module.train()

        return total_loss / deno

    def log_metrics(self, metrics, pl_module: pl.LightningModule, on_step=True):
        pl_module.log(
            f"downstream/{self.name}",
            metrics,
            on_step=on_step,
        )


class RougeCallback(cb.Callback):
    def __init__(
        self,
        datamodule,
        every_n_epochs=1,
        subsample=-1,
        generation_kwargs=None,
    ):
        super().__init__()

        from mttl.evaluators.rouge_evaluator import RougeEvaluator

        generation_kwargs = generation_kwargs or {"auto_max_new_tokens": True}
        self.evaluator = RougeEvaluator(
            datamodule=datamodule,
            generation_kwargs=generation_kwargs,
        )

        self.every_n_epochs = every_n_epochs
        self.verbose = False
        self.subsample = subsample

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if self.every_n_epochs > 0 and trainer.current_epoch % self.every_n_epochs == 0:
            rouge = self.evaluator.evaluate(
                pl_module,
                split="val",
                verbose=self.verbose,
                subsample=self.subsample,
            )
            pl_module.log("val/rougeL", rouge, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        rouge = self.evaluator.evaluate(
            pl_module,
            split="test",
            verbose=self.verbose,
            subsample=self.subsample,
        )
        pl_module.log("test/rougeL", rouge, on_epoch=True, prog_bar=True)


class NanoMMLUCallback(cb.Callback):
    def __init__(
        self,
        datamodule,
        every_n_epochs=1,
        subsample=-1,
    ):
        super().__init__()

        from mttl.evaluators import MMLUEvaluator

        self.evaluator = MMLUEvaluator(datamodule=datamodule)
        self.every_n_epochs = every_n_epochs
        self.verbose = False
        self.subsample = subsample
        self.first_eval = False

    def on_validation_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if self.every_n_epochs > 0 and trainer.current_epoch % self.every_n_epochs == 0:
            em = self.evaluator.evaluate(
                pl_module,
                split="test",
                verbose=self.verbose,
                subsample=self.subsample,
            )
            pl_module.log("val/mmlu", em, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        em = self.evaluator.evaluate(
            pl_module,
            split="test",
            verbose=self.verbose,
            subsample=self.subsample,
        )
        pl_module.log("test/mmlu", em, on_epoch=True, prog_bar=True)


class MMLUCallback(cb.Callback):
    def __init__(
        self,
        eval_every_opt_step=1,
        split="test",
        max_input_length=None,
        checkpoint_oracle=True,
    ):
        super().__init__()

        self.eval_every_opt_step = eval_every_opt_step
        self.max_input_length = max_input_length
        self.evaluator = None
        self.split = split
        # save best perf
        self._best_perf = None
        # checkpointing
        self.do_checkpoint = checkpoint_oracle
        self._checkpoint_now = False
        self._prev_checkpoint = None
        # debug
        self.eval_mmlu_count = 0

    @property
    def last_chkpt(self):
        return self._prev_checkpoint

    @property
    def best_perf(self):
        return self._best_perf

    @best_perf.setter
    def best_perf(self, value):
        if self._best_perf is None:
            self._best_perf = value
            self._checkpoint_now = True
        else:
            if value["all"]["mean"] > self._best_perf["all"]["mean"]:
                self._checkpoint_now = True
            for (k1, v1), (k2, v2) in zip(self._best_perf.items(), value.items()):
                if v2["mean"] > v1["mean"]:
                    self._best_perf[k1] = v2

    def on_before_optimizer_step(
        self, trainer: Trainer, pl_module: LightningModule, optimizer: Optimizer
    ) -> None:
        if trainer.global_step % self.eval_every_opt_step == 0:
            metrics = self.eval_mmlu(pl_module)
            self.best_perf = copy.deepcopy(metrics)
            self.maybe_checkpoint_now(trainer)
            self.log_metrics(metrics, pl_module)

        return super().on_before_optimizer_step(trainer, pl_module, optimizer)

    def maybe_checkpoint_now(self, trainer):
        if self.do_checkpoint and self._checkpoint_now:
            dir_name = trainer.checkpoint_callback.dirpath
            filename = (
                trainer.checkpoint_callback.filename.split("{")[0]
                + f"mmlu_{self.split}_oracle/"
                + f"{self.best_perf['all']['mean']:.004f}.ckpt"
            )
            ckpt_path = os.path.join(dir_name, filename)
            trainer.save_checkpoint(ckpt_path)
            # if self._prev_checkpoint is not None and ckpt_path != self._prev_checkpoint:
            #     os.remove(self._prev_checkpoint)
            self._prev_checkpoint = ckpt_path
        self._checkpoint_now = False

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        # log best perf
        if self.best_perf is not None:
            for t, v in self.best_perf.items():
                pl_module.log(
                    f"downstream_best/{self.split}/oracle/mmlu_{t}",
                    v["mean"],
                    on_epoch=True,
                )
        return super().on_validation_epoch_end(trainer, pl_module)

    def log_metrics(self, metrics, pl_module: pl.LightningModule, on_step=True):
        for t, v in metrics.items():
            pl_module.log(
                f"downstream/{self.split}/mmlu_{t}",
                v["mean"],
                on_step=on_step,
            )

    def eval_mmlu(self, pl_module):
        from mttl.datamodule.mmlu_data_module import MMLUDataConfig
        from mttl.evaluators import MMLUEvaluator

        if DEBUG:
            self.eval_mmlu_count += 1
            return {"all": {"mean": self.eval_mmlu_count}}

        if self.evaluator is None:
            mmlu_data_config = MMLUDataConfig(
                model=pl_module.hparams.model,
                predict_batch_size=pl_module.hparams.predict_batch_size,
                max_input_length=(
                    pl_module.hparams.max_input_length
                    if self.max_input_length is None
                    else self.max_input_length
                ),
                max_output_length=pl_module.hparams.max_input_length,  # not necessary
                model_family=pl_module.hparams.model_family,
                finetune_task_name=pl_module.hparams.finetune_task_name,
            )
            self.evaluator = MMLUEvaluator(
                config=mmlu_data_config,
            )

        metrics = self.evaluator.evaluate(pl_module, split=self.split)
        return metrics

    def on_after_backward(self, *args, **kwargs):
        self.val_epoch += 1

        trainer, pl_module = args

        if trainer.global_step < 1:
            return

        if self.val_epoch % self.every_val_epochs != 0:
            return

        from mttl.evaluators import MMLUEvaluator

        evaluator = MMLUEvaluator(
            pl_module.hparams,
            **self.eval_kwargs,
        )
        metrics = evaluator.evaluate(
            pl_module,
            split=self.split,
            subsample=10,
        )
        pl_module.log(
            "val/mmlu",
            metrics["all"]["mean"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )


class NICallback(cb.Callback):
    def __init__(self, every_val_epochs=1, **kwargs):
        super().__init__()

        self.val_epoch = 0
        self.every_val_epochs = every_val_epochs
        self.eval_kwargs = kwargs

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        self.val_epoch += 1

        if trainer.global_step == 0:
            return

        if self.val_epoch % self.every_val_epochs != 0:
            return

        from mttl.evaluators import NIEvaluator

        evaluator = NIEvaluator(
            config=pl_module.hparams,
            num_pos_examples=2,
            **self.eval_kwargs,
        )
        metrics = evaluator.evaluate(
            pl_module,
            split="val",
            eval_batches=50,
        )
        pl_module.log(
            "val/sni",
            metrics["all"]["mean"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )


class ProgressCallback(cb.TQDMProgressBar):
    def init_sanity_tqdm(self) -> Tqdm:
        """Override this to customize the tqdm bar for the validation sanity run."""
        bar = Tqdm(
            desc="Validation sanity check",
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=False,
            dynamic_ncols=True,
            file=sys.stderr,
        )
        return bar

    def init_train_tqdm(self) -> Tqdm:
        """Override this to customize the tqdm bar for training."""
        bar = Tqdm(
            desc="Training",
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stderr,
            smoothing=0,
        )
        return bar

    def init_predict_tqdm(self) -> Tqdm:
        """Override this to customize the tqdm bar for predicting."""
        bar = Tqdm(
            desc="Predicting",
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stderr,
            smoothing=0,
        )
        return bar

    def init_validation_tqdm(self) -> Tqdm:
        """Override this to customize the tqdm bar for validation."""
        # The main progress bar doesn't exist in `trainer.validate()`
        has_main_bar = self.trainer.state.fn != "validate"
        bar = Tqdm(
            desc="Validating",
            position=(2 * self.process_position + has_main_bar),
            disable=self.is_disabled,
            leave=False,
            dynamic_ncols=True,
            file=sys.stderr,
        )
        return bar

    def init_test_tqdm(self) -> Tqdm:
        """Override this to customize the tqdm bar for testing."""
        bar = Tqdm(
            desc="Testing",
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stderr,
        )
        return bar


class EvalCallback(ABC):
    @abstractmethod
    def evaluate_model(self, model, prefix=""):
        pass


class MMLUEvalCallback(MMLUEvaluator, EvalCallback):
    def __init__(
        self,
        config,
        name="mmlu_test_callback",
        split="test",
        subsample=-1,
        use_vllm=False,
    ):
        self.split = split
        from mttl.datamodule.mmlu_data_module import MMLUDataConfig

        assert split in ["test"]
        self.use_vllm = use_vllm
        mmlu_config = MMLUDataConfig(
            **{
                k: v
                for k, v in config.__dict__.items()
                if k in MMLUDataConfig.__dataclass_fields__.keys()
            }
        )
        super().__init__(mmlu_config, use_vllm=use_vllm)
        self.subsample = subsample
        self.name = name

    def evaluate_model(self, model, prefix=""):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        score = self.evaluate(model, self.split, self.subsample)["all"]["mean"]
        # log
        if wandb.run is not None:
            wandb.log({f"{prefix}{self.name}_{self.split}": score})
        return score


class TestLossEvaluator(LossCallback, Evaluator):
    def __init__(
        self,
        datamodule: DataModule,
        name="test",
        split="test",
        subsample=-1,
        **kwargs,
    ):
        self.split = split
        if split == "test":
            dataloader = datamodule.test_dataloader(subsample=subsample)
        elif split in ["val", "valid", "validation"]:
            dataloader = datamodule.val_dataloader(subsample=subsample)
        elif split == "train":
            dataloader = datamodule.train_dataloader(subsample=subsample)
        super().__init__(
            dataloader=dataloader,
            name=name,
            output_dir=None,
            eval_every_opt_step=0,
            checkpoint_oracle=False,
        )
        self.datamodule = datamodule

    @property
    def tasks(self):
        return self.datamodule.task_names

    def evaluate(self, model):
        # return something that should be maximized
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        loss = self.test(model)
        loss *= -1.0
        return {"all": {"mean": loss.item()}, f"{self.name}": {"mean": loss.item()}}

    def get_loss(self, model, **kwargs):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        loss = self.test(model, **kwargs)
        return loss.item()

    @property
    def tokenizer(self):
        return self.datamodule.tokenizer


class DownstreamEvalCallback(cb.Callback):
    METRIC_KEY = "downstream"

    def __init__(self, args) -> None:
        super().__init__()

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

    def on_validation_epoch_start(
        self, trainer: Trainer, pl_module: pl.LightningModule
    ) -> None:
        if trainer.global_step == 0 and not self.args.eval_before_training:
            return

        if self.args.eval_every_n_epoch is None or (
            self.args.eval_every_n_epoch
            and trainer.current_epoch % self.args.eval_every_n_epoch != 0
        ):
            return

        metrics = self.runner.run(pl_module)
        for task, metric in metrics.items():
            pl_module.log(
                f"{self.METRIC_KEY}/{task}",
                metric,
                on_epoch=True,
                prog_bar=True,
            )

    def on_test_epoch_end(
        self, trainer: Trainer, pl_module: pl.LightningModule
    ) -> None:
        metrics = self.runner.run(pl_module)
        for task, metric in metrics.items():
            pl_module.log(
                f"{self.METRIC_KEY}_last/{task}",
                metric,
                on_epoch=True,
                prog_bar=True,
            )
