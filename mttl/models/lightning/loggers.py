import json
import os

import pytorch_lightning as pl
import torch

from mttl.logging import logger


class SimpleLogger(pl.loggers.logger.DummyLogger):
    def __init__(self, output_dir):
        self.output_file = os.path.join(output_dir, "metrics.json")
        os.makedirs(output_dir, exist_ok=True)

        if os.path.exists(self.output_file):
            os.remove(self.output_file)

    def log_metrics(self, metrics, step=None):
        lines = []
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            lines.append({"name": k, "value": v, "step": step})

        try:
            with open(self.output_file, "a+") as f:
                for l in lines:
                    f.write(json.dumps(l) + "\n")
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")


def init_wandb_logger(args):
    if args.wandb_project is None:
        args.wandb_project = os.environ.get("WANDB_PROJECT", "MMLU_ninja_merge")
    if args.wandb_project:
        exp_name = os.getenv("AMLT_JOB_NAME", f"{args.exp_name}")
        pl_logger = pl.loggers.WandbLogger(
            project=args.wandb_project,
            name=exp_name,
            config=args,
        )
    return pl_logger


def get_pl_loggers(args):
    loggers = []

    add_simple_logger(loggers, args)
    add_tb_logger(loggers, args)
    add_wandb_logger(loggers, args)
    add_mlf_logger(loggers)
    return loggers


def add_simple_logger(loggers, args):
    loggers.append(SimpleLogger(args.output_dir))


def add_tb_logger(loggers, args):
    if args.tensorboard:
        tb_logger = pl.loggers.TensorBoardLogger(save_dir=args.output_dir)
        loggers.append(tb_logger)


def add_wandb_logger(loggers, args) -> pl.loggers.WandbLogger:
    if not os.environ.get("WANDB_API_KEY"):
        return

    import wandb

    if not args.exp_name:
        args.exp_name = os.environ.get("AMLT_JOB_NAME")
    if not args.wandb_project:
        args.wandb_project = os.environ.get("WANDB_PROJECT")

    wandb_logger = pl.loggers.WandbLogger(
        project=args.wandb_project,
        name=args.exp_name,
        settings=wandb.Settings(start_method="fork"),
    )
    wandb_logger.experiment.save("*.py")
    loggers.append(wandb_logger)


def add_mlf_logger(loggers):
    import pytorch_lightning as pl
    from pytorch_lightning.utilities.rank_zero import rank_zero_only

    class MLFlowLoggerCustom(pl.loggers.MLFlowLogger):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        @rank_zero_only
        def log_hyperparams(self, *args, **kwargs) -> None:
            try:
                super().log_hyperparams(*args, **kwargs)
            except:
                pass

    try:
        from azureml.core.run import Run

        run = Run.get_context()

        mlf_logger = MLFlowLoggerCustom(
            experiment_name=run.experiment.name,
        )
        mlf_logger._run_id = run.id
    except:
        logger.warning("Couldn't instantiate MLFlowLogger!")
        mlf_logger = None

    if mlf_logger is not None:
        loggers.append(mlf_logger)
