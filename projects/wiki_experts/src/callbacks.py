import os
import tqdm
import copy
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer, callbacks as cb
from torch.optim import Optimizer
from mttl.evaluators.base import EvaluatorRunner, setup_evaluators
from projects.wiki_experts.src.config import ExpertConfig
from projects.wiki_experts.src.expert_trainer import ExpertTrainer

from mttl.utils import logger
from mttl.evaluators.rouge_evaluator import RougeEvaluator
from pytorch_lightning.utilities.types import LRSchedulerConfig


DEBUG = False


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
        )

    def on_validation_epoch_start(
        self, trainer: Trainer, pl_module: ExpertTrainer
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

    def on_test_epoch_end(self, trainer: Trainer, pl_module: ExpertTrainer) -> None:
        metrics = self.runner.run(pl_module)
        for task, metric in metrics.items():
            pl_module.log(
                f"{self.METRIC_KEY}_last/{task}",
                metric,
                on_epoch=True,
                prog_bar=True,
            )


class RougeCallbackTestPerEpoch(cb.Callback):
    def __init__(
        self,
        datamodule,
        checkpointing_callback: cb.Callback,
        name="test_rougeL_per_epoch",
    ):
        self.name = name
        self.datamodule = datamodule
        self.evaluator = RougeEvaluator(datamodule)
        self.checkpointing_callback = checkpointing_callback
        self.epoch = 0

    def on_train_epoch_end(self, trainer: Trainer, pl_module: ExpertTrainer) -> None:
        # test best model sofar
        pl_module_device = pl_module.device
        pl_module.to("cpu")
        copy_pl_module = copy.deepcopy(pl_module)
        copy_pl_module.to(pl_module_device)

        best_model_path = (
            self.checkpointing_callback.best_model_path
            or self.checkpointing_callback.last_model_path
        )

        copy_pl_module.from_pretrained(best_model_path)
        metrics_test = self.test(copy_pl_module, split="test")

        pl_module.log(f"test/{self.name}", metrics_test, on_epoch=True, prog_bar=True)

        del copy_pl_module
        pl_module.to(pl_module_device)
        self.epoch += 1
        return super().on_train_epoch_end(trainer, pl_module)

        if self.do_checkpoint and self._checkpoint_now:
            try:
                filename = (
                    self.output_dir
                    + f"/{self.name}_val_epoch{self.epoch}/"
                    + f"{self.best_loss:.004f}.ckpt"
                )
                ckpt_path = os.path.join(filename)
                trainer.save_checkpoint(ckpt_path)
                self._prev_checkpoint = ckpt_path
            except Exception as e:
                logger.error("Error in checkpointing with RougeLCallback: " + str(e))
        self._checkpoint_now = False

    def test(self, pl_module: LightningModule, split="val"):
        rougeL = self.evaluator.evaluate(pl_module, verbose=False, split=split)
        return rougeL


class RougeLCallback(cb.Callback):
    def __init__(
        self,
        datamodule,
        output_dir,
        name="rougeL",
        eval_every_opt_step=1,
        checkpoint_oracle=False,
        split="val",
    ):
        self.name = name
        self.output_dir = output_dir
        self.datamodule = datamodule
        self.eval_every_opt_step = eval_every_opt_step
        # save best perf
        self._best_rouge = None
        # checkpointing
        self.do_checkpoint = checkpoint_oracle
        self._checkpoint_now = False
        self._prev_checkpoint = None
        self.split = split
        self.evaluator = RougeEvaluator(datamodule)

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
        return self._best_rouge

    @best_loss.setter
    def best_loss(self, value):
        if self._best_rouge is None:
            self._best_rouge = value
            self._checkpoint_now = True
        else:
            if value > self._best_rouge:
                self._checkpoint_now = True
                self._best_rouge = value

    def on_before_optimizer_step(
        self, trainer: Trainer, pl_module: LightningModule, optimizer: Optimizer
    ) -> None:
        if trainer.global_step % self.eval_every_opt_step == 0:
            metrics = self.test(pl_module)
            self.best_loss = copy.deepcopy(metrics)
            self.log_metrics(metrics, pl_module)
            self.maybe_checkpoint_now(trainer)
            # checksum of parameters
            # p_sum = np.sum([p.detach().cpu().sum() for p in pl_module.parameters()])
        return super().on_before_optimizer_step(trainer, pl_module, optimizer)

    def maybe_checkpoint_now(self, trainer: Trainer):
        if self.do_checkpoint and self._checkpoint_now:
            try:
                filename = (
                    self.output_dir + f"/{self.name}/" + f"{self.best_loss:.004f}.ckpt"
                )
                ckpt_path = os.path.join(filename)
                trainer.save_checkpoint(ckpt_path)
                if (
                    self._prev_checkpoint is not None
                    and ckpt_path != self._prev_checkpoint
                ):
                    os.remove(self._prev_checkpoint)
                self._prev_checkpoint = ckpt_path
            except Exception as e:
                logger.error("Error in checkpointing with RougeLCallback: " + str(e))
        self._checkpoint_now = False

    def test(self, pl_module: LightningModule):
        rougeL = self.evaluator.evaluate(pl_module, split=self.split, verbose=False)
        return rougeL

    def log_metrics(self, metrics, pl_module: pl.LightningModule, on_step=True):
        pl_module.log(
            f"downstream/{self.name}",
            metrics,
            on_step=on_step,
        )

    def remove_checkpoints(self):
        if self._prev_checkpoint is not None and os.path.exists(self._prev_checkpoint):
            os.remove(self._prev_checkpoint)


class ValLossCheckpointCallback(cb.Callback):
    def __init__(self, args: ExpertConfig) -> None:
        super().__init__()
        self.args = args
        self.best_val_loss = None
        self._prev_checkpoint = None
        self.criteria = "val_loss"

    @property
    def best_model_path(self):
        return self._prev_checkpoint

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: ExpertTrainer
    ) -> None:
        if self.best_val_loss is None or pl_module.best_val_result < self.best_val_loss:
            self.best_val_loss = pl_module.best_val_result
            logger.info(f"New best val loss: {self.best_val_loss}")
            self.save_best_val_model(trainer, pl_module)

    def save_best_val_model(self, trainer: Trainer, pl_module: LightningModule):
        dir_name = self.args.output_dir
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        filename = self.criteria + f"/{self.best_val_loss:.004f}.ckpt"
        ckpt_path = os.path.join(dir_name, filename)
        if (
            self._prev_checkpoint is not None
            and ckpt_path != self._prev_checkpoint
            and os.path.exists(self._prev_checkpoint)
        ):
            os.remove(self._prev_checkpoint)
        trainer.save_checkpoint(ckpt_path)
        self._prev_checkpoint = ckpt_path
