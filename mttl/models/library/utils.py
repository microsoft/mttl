import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from pytorch_lightning import Trainer

from mttl.arguments import ExpertConfig
from mttl.callbacks import LiveCheckpointCallback
from mttl.logging import get_pl_loggers
from mttl.models.expert_model import ExpertModel as ExpertTrainer
from mttl.models.monitors import get_monitors


def train_module(args: ExpertConfig, module: ExpertTrainer, dm):
    loggers = get_pl_loggers(args)
    callbacks = get_monitors(args)

    monitor = "val/loss"
    mode = "min"

    checkpoint_callback = LiveCheckpointCallback(
        dirpath=args.output_dir,
        monitor=monitor,
        save_last=True,
        mode=mode,
    )
    callbacks.append(checkpoint_callback)

    val_check_interval = args.eval_every
    if val_check_interval == -1 or val_check_interval is None:
        val_check_interval = None
    else:
        val_check_interval = args.gradient_accumulation_steps * args.eval_every
        if val_check_interval > len(dm.train_dataloader()):
            val_check_interval = len(dm.train_dataloader())
        elif val_check_interval > args.total_steps and args.total_steps != -1:
            val_check_interval = args.total_steps

    val_check_interval = args.eval_every
    if val_check_interval == -1 or val_check_interval is None:
        val_check_interval = None
    else:
        val_check_interval = args.gradient_accumulation_steps * args.eval_every
        if val_check_interval > len(dm.train_dataloader()):
            val_check_interval = len(dm.train_dataloader())
        elif val_check_interval > args.total_steps and args.total_steps != -1:
            val_check_interval = args.total_steps

    trainer = Trainer(
        devices=-1,
        accelerator="gpu",
        logger=loggers,
        num_sanity_val_steps=0,
        default_root_dir=args.output_dir,
        max_epochs=args.num_train_epochs,
        max_steps=args.total_steps + 1 if args.total_steps != -1 else -1,
        gradient_clip_val=args.max_grad_norm,
        strategy=args.compute_strategy if args.compute_strategy else "auto",
        callbacks=callbacks,
        enable_checkpointing=False,
        log_every_n_steps=args.gradient_accumulation_steps,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        precision=(
            int(args.precision) if args.precision in ["16", "32"] else args.precision
        ),
        val_check_interval=val_check_interval,
    )
    trainer.fit(module, dm)
    checkpoint = (
        checkpoint_callback.best_model_path or checkpoint_callback.last_model_path
    )
    return checkpoint


def get_svd_embedding(lib, expert_name: str):
    try:
        embeddings = lib.get_auxiliary_data(
            data_type="embeddings", expert_name=expert_name
        )
    except ValueError:
        return None
    return embeddings["svd"]["embeddings"]
