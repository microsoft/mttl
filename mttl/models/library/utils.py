from datetime import time
from functools import wraps

from pytorch_lightning import Trainer

from mttl.arguments import ExpertConfig
from mttl.logging import get_pl_loggers
from mttl.models.lightning.callbacks import LiveCheckpointCallback
from mttl.models.lightning.expert_module import ExpertModule
from mttl.models.monitors import get_monitors


def retry(max_retries=10, wait_seconds=60):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_retries + 1):
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:  # requests.exceptions.HTTPError as e:
                    print(e, type(e), "retrying...")
                    if attempt < max_retries:
                        print(f"Waiting {wait_seconds} seconds before retrying...")
                        time.sleep(wait_seconds)
            raise RuntimeError(
                f"Function {wrapper.__name__} failed after {max_retries} attempts."
            )

        return wrapper

    return decorator


def train_module(args: ExpertConfig, module: ExpertModule, dm):
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
