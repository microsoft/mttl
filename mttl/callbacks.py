import datetime
import time
import sys, os
from typing import Any

from pytorch_lightning import callbacks as cb
from pytorch_lightning.callbacks.progress.tqdm_progress import Tqdm
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.utils.data import DataLoader


from mttl.utils import logger


class MMLUCallback(cb.Callback):
    def __init__(self, every_val_epochs=1):
        super().__init__()

        self.val_epoch = 0
        self.every_val_epochs = every_val_epochs

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        self.val_epoch += 1

        if trainer.global_step == 0:
            return

        if self.val_epoch % self.every_val_epochs != 0:
            return

        from mttl.evaluators import MMLUEvaluator

        evaluator = MMLUEvaluator(
            pl_module.hparams,
            data_dir=os.environ["MMLU_DATA_DIR"],
        )
        metrics = evaluator.evaluate(
            pl_module,
            metric_per_task=True,
            eval_batches=200,
        )
        pl_module.log(
            "val/mmlu",
            metrics["exact_match"]["all"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )


class NICallback(cb.Callback):
    def __init__(self, every_val_epochs=1):
        super().__init__()

        self.val_epoch = 0
        self.every_val_epochs = every_val_epochs

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        self.val_epoch += 1

        if trainer.global_step == 0:
            return

        if self.val_epoch % self.every_val_epochs != 0:
            return

        from mttl.evaluators import NIEvaluator

        evaluator = NIEvaluator(
            pl_module.hparams,
            data_dir=os.environ["NI_DATA_DIR"],
            num_pos_examples=2,
        )
        metrics = evaluator.evaluate(
            pl_module,
            metric_per_task=True,
            eval_batches=50,
        )
        pl_module.log(
            "val/sni",
            metrics["rougeL"]["all"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )


class MiniProgress(cb.ProgressBar):
    def on_train_batch_start(
        self, trainer, pl_module, batch: Any, batch_idx: int
    ) -> None:
        self.time_start = time.time()

    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
    ) -> None:
        self.time_end = time.time()
        metrics = self.get_metrics(trainer, pl_module)
        metrics = {
            k.replace("_step", "").replace("_epoch", ""): v for k, v in metrics.items()
        }
        metrics["it/s"] = "{:.1f}".format(1 / (self.time_end - self.time_start))
        seconds_to_go = (
            (trainer.num_training_batches - trainer.global_step)
            * trainer.accumulate_grad_batches
            / ((1.0 / (self.time_end - self.time_start)) * 3600)
        )
        metrics["ETA"] = "~{}".format(str(datetime.timedelta(seconds=seconds_to_go)))

        for k, v in metrics.items():
            metrics[k] = "{:.2f}".format(v) if isinstance(v, float) else v

        msg_start = (
            f"Trn - Epc {trainer.current_epoch} / {trainer.global_step} / {trainer.num_training_batches}"
            + " | "
        )
        dict_msg = " | ".join([f"{k} -> {v}" for k, v in metrics.items()]) + " | "
        msg = msg_start + dict_msg
        logger.info(msg)

    def on_validation_batch_start(
        self, trainer, pl_module, batch: Any, batch_idx: int
    ) -> None:
        self.time_start = time.time()

    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
    ) -> None:
        self.time_end = time.time()
        metrics = self.get_metrics(trainer, pl_module)
        metrics = {
            k.replace("_step", "").replace("_epoch", ""): v for k, v in metrics.items()
        }
        metrics["it/s"] = 1.0 / (self.time_end - self.time_start)
        for k, v in metrics.items():
            metrics[k] = "{:.2f}".format(v) if isinstance(v, float) else v

        msg_start = (
            f"Val - Epc {trainer.current_epoch} / {batch_idx} / {trainer.num_val_batches[0]}"
            + " | "
        )
        dict_msg = " | ".join([f"{k} -> {v}" for k, v in metrics.items()]) + " | "
        msg = msg_start + dict_msg
        logger.info(msg)


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
