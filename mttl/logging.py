import logging
import os
from functools import lru_cache

import numpy as np
import pandas as pd
import prettytable
import pytorch_lightning as pl
import wandb

logger = logging.getLogger("mttl")


@lru_cache
def warn_once(msg: str):
    logger.warning(msg)


def setup_logging(log_dir: str = None):
    logging.basicConfig(
        format="%(asctime)s %(levelname)s --> %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.setLevel(logging.INFO)
    logging.getLogger("openai").setLevel(logging.WARNING)

    if log_dir:
        log_file_path = os.path.join(log_dir, "log.txt")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        handler_exists = any(
            isinstance(handler, logging.FileHandler)
            and handler.baseFilename == log_file_path
            for handler in logger.handlers
        )

        if not handler_exists:
            logger.addHandler(logging.FileHandler(log_file_path))
            logger.info(
                "New experiment, log will be at %s",
                log_file_path,
            )


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
    from mttl.models.utils import SimpleLogger

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


class TableLogger:
    def __init__(self):
        self.df = pd.DataFrame()

    def from_df(self, df):
        self.df = df
        self.columns = df.columns

    def log(self, row: dict):
        if self.df is None or len(self.df) == 0:
            self.df = pd.DataFrame(columns=row.keys())
        else:
            # Add new columns to the DataFrame if they don't exist
            new_columns = set(row.keys()) - set(self.df.columns)
            for column in new_columns:
                self.df[column] = np.nan
        self.df.loc[len(self.df.index)] = row

    def get_table(self):
        return self.df

    def means(self):
        # calculate mean for each row, column and diagonal of self.df
        # filter numeric columns
        df_numeric = self.df.select_dtypes(include=[np.number])
        self.df["mean"] = df_numeric.mean(axis=1)
        self.df.loc["mean"] = df_numeric.mean(axis=0)
        self.df.loc["mean", "mean"] = np.diag(df_numeric).mean()

    def log_final_table(self):
        if wandb.run is not None:
            wandb.log({"table": wandb.Table(data=self.get_table())})
        table = prettytable.PrettyTable()
        table.field_names = list(self.df.columns)
        for i, row in self.df.iterrows():
            table.add_row(list(row))
        logger.info("Results:\n" + str(table))
