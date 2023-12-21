import datetime
import time
import sys, os
import copy
import torch
import tqdm
from typing import Any

import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer, callbacks as cb
from pytorch_lightning.callbacks.progress.tqdm_progress import Tqdm
from torch.optim import Optimizer
from mttl.utils import Averager, logger
from mttl.models.utils import transfer_batch_to_device


DEBUG = False


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
        with torch.no_grad():
            for i, batch in tqdm.tqdm(
                enumerate(self.dataloader),
                total=len(self.dataloader),
                desc=f"Test {self.name}",
            ):
                batch = transfer_batch_to_device(batch, pl_module.device)
                loss = pl_module.forward(batch, reduction="none")
                outputs += [(loss.detach().cpu(),)]
        losses = torch.cat([out[0] for out in outputs], 0)

        if was_train:
            pl_module.train()
        return losses.mean()

    def log_metrics(self, metrics, pl_module: pl.LightningModule, on_step=True):
        pl_module.log(
            f"downstream/{self.name}",
            metrics,
            on_step=on_step,
        )


class RougeCallback(cb.Callback):
    def __init__(
        self, datamodule, device="cuda", every_n_epochs=1, subsample=-1, max_length=None
    ):
        super().__init__()

        from mttl.evaluators.rouge_evaluator import RougeEvaluator

        self.evaluator = RougeEvaluator(datamodule=datamodule)
        self.every_n_epochs = every_n_epochs
        self.max_length = max_length
        self.verbose = False
        self.subsample = subsample
        self.first_eval = False

    def on_after_backward(self, trainer, pl_module):
        if not self.first_eval:
            rouge = self.evaluator.evaluate(
                pl_module,
                split="val",
                verbose=self.verbose,
                subsample=self.subsample,
            )

            pl_module.log("val/rougeL", rouge, on_step=True, prog_bar=True)
            self.first_eval = True

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

        return super().on_validation_epoch_end(trainer, pl_module)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        rouge = self.evaluator.evaluate(
            pl_module,
            split="test",
            verbose=self.verbose,
            subsample=self.subsample,
        )

        pl_module.log("test/rougeL", rouge, on_epoch=True, prog_bar=True)

        return super().on_test_epoch_end(trainer, pl_module)


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
        from mttl.evaluators import MMLUEvaluator
        from mttl.datamodule.mmlu_data_module import MMLUDataConfig

        if DEBUG:
            self.eval_mmlu_count += 1
            return {"all": {"mean": self.eval_mmlu_count}}

        if self.evaluator is None:
            mmlu_data_config = MMLUDataConfig(
                model=pl_module.hparams.model,
                predict_batch_size=pl_module.hparams.predict_batch_size,
                max_input_length=pl_module.hparams.max_input_length
                if self.max_input_length is None
                else self.max_input_length,
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
