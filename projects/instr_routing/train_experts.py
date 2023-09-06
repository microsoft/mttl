import os
import sys
import json
import torch
import wandb 
import logging
import pytorch_lightning as pl
from huggingface_hub import login
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from mttl.callbacks import MMLUCallback, MiniProgress
from mttl.datamodule.wiki_mmlu_module import WikiMMLUDataModule
from mttl.utils import get_mlf_logger, setup_logging, logger
from mttl.dist_utils import is_main_process


# register models
import models.vsmear  # noqa: F401
import models.softmoe # noqa: F401
from models.monitors import SelectorMetricsLog, SelectorRoutingsLog
from models.lm_trainer import ExpertCard, ExpertTrainer
from config import RoutingConfig


def remove_non_serializable(d):
    """
    Recursively remove non-JSON serializable values from a dictionary.
    """
    for k, v in d.items():
        if isinstance(v, (list, dict)):
            remove_non_serializable(v)
        elif not json.dumps(v, default=lambda x: None):
            del d[k]


def run_multitask(args):
    seed_everything(args.seed, workers=True)

    # get directory of the current file
    setup_logging(args.output_dir)

    logger.info("Args: {}".format(args.to_json()))

    if args.hf_token_hub:
        login(token=args.hf_token_hub)

    # select dataloader
    model_class = ExpertTrainer
    dm = WikiMMLUDataModule(args)

    args.n_tasks = len(dm.task_to_id)
    module = model_class(**vars(args), tokenizer=dm.tokenizer)

    # legit logging
    loggers = []

    mlf_logger = get_mlf_logger()
    if mlf_logger:
        loggers.append(mlf_logger)

    if args.tensorboard:
        tb_logger = pl.loggers.TensorBoardLogger(save_dir=args.output_dir)
        loggers.append(tb_logger)

    loggers.append(pl.loggers.CSVLogger(save_dir=args.output_dir, name="csv_metrics"))

    kwargs = {"val_check_interval": args.eval_every} if args.eval_every else {}

    # get metric monitors for models
    callbacks = []
    callbacks.append(MiniProgress())

    monitor = "val/loss"
    mode = "min"

    model_name = args.model.replace("/", "_")
    exp_name = os.environ.get("AMLT_JOB_NAME", args.exp_name)

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        monitor=monitor,
        filename=f"{model_name}" + f"_{exp_name}" + "-{" + monitor + ":.004f}",
        save_top_k=1,
        save_last=True,
        save_weights_only=True,  # make checkpoints smaller
        mode=mode,
    )
    callbacks.append(checkpoint_callback)

    trainer = Trainer(
        devices=-1, 
        accelerator="gpu",
        logger=loggers,
        num_sanity_val_steps=5,
        default_root_dir=args.output_dir,
        max_epochs=args.num_train_epochs,
        max_steps=args.total_steps + 1 if args.total_steps != -1 else -1,
        gradient_clip_val=args.max_grad_norm,
        log_every_n_steps=20,
        strategy=args.compute_strategy if args.compute_strategy else "auto",
        callbacks=callbacks,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        precision=int(args.precision)
        if args.precision in ["16", "32"]
        else args.precision,
        fast_dev_run=args.fast_dev_run,
        **kwargs,
    )
    trainer.fit(module, dm)

    path_best_model = trainer.checkpoint_callback.best_model_path
    ckpt_path = "best" if path_best_model else "last"

    trainer.validate(dataloaders=dm, ckpt_path=ckpt_path)


if __name__ == "__main__":
    args = RoutingConfig.parse()
    run_multitask(args)
