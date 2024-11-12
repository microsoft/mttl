import logging
import os
from functools import lru_cache

import numpy as np
import pandas as pd
import prettytable

import wandb

# warning if logger is not initialized
logger = logging.getLogger("mttl")
logger.setLevel(logging.WARNING)
logging.getLogger("datasets.arrow_dataset").setLevel(logging.CRITICAL + 1)


def maybe_wandb_log(logs: dict):
    if wandb.run is not None:
        wandb.log(logs)


@lru_cache
def warn_once(msg: str, **kwargs):
    logger.warning(msg, **kwargs)


@lru_cache
def debug_once(msg: str, **kwargs):
    logger.debug(msg, **kwargs)


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
        maybe_wandb_log({"table": wandb.Table(data=self.get_table())})
        table = prettytable.PrettyTable()
        table.field_names = list(self.df.columns)
        for i, row in self.df.iterrows():
            table.add_row(list(row))
        logger.info("Results:\n" + str(table))
