import sys, os

from pytorch_lightning import callbacks as cb
from pytorch_lightning.callbacks.progress.tqdm_progress import Tqdm
from torch.utils.data import DataLoader


class MMLUCallback(cb.Callback):
    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        from mttl.evaluators import MMLUEvaluator

        evaluator = MMLUEvaluator(
            pl_module.hparams,
            data_dir=os.environ["MMLU_DATA_DIR"],
        )
        metrics = evaluator.evaluate(
            pl_module, metric_per_task=True, eval_batches=150,
        )
        pl_module.log(
            "val/mmlu",
            metrics["exact_match"]["all"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )


class NICallback(cb.Callback):
    def son_validation_epoch_end(self, trainer, pl_module) -> None:
        from mttl.evaluators import NIEvaluator

        evaluator = NIEvaluator(
            pl_module.hparams,
            data_dir=os.environ["NI_DATA_DIR"],
            num_pos_examples=2,
        )
        metrics = evaluator.evaluate(
            pl_module, metric_per_task=True, eval_batches=50,
        )
        pl_module.log(
            "val/sni",
            metrics["rougeL"]["all"],
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
