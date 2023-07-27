import os
import json
import hydra
import pytorch_lightning as pl 
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from mttl.callbacks import ProgressCallback
from mttl.models.monitors import get_monitors

from mttl.datamodule.alpaca_data_module import AlpacaDataModule
from lm import LM
from alpaca_config import AlpacaConfig


def remove_non_serializable(d):
    """
    Recursively remove non-JSON serializable values from a dictionary.
    """
    for k, v in d.items():
        if isinstance(v, (list, dict)):
            remove_non_serializable(v)
        elif not json.dumps(v, default=lambda x: None):
            del d[k]


def run(args):
    seed_everything(args.seed, workers=True)

    print(os.path.dirname(os.path.realpath(__file__)))

    dm = AlpacaDataModule(args)
    module = LM(**vars(args), tokenizer=dm.tokenizer)

    loggers = []
    if os.environ.get("WANDB_API_KEY"):
        wandb_logger = pl.loggers.WandbLogger(
            project="alpaca_tuning",
            name=os.environ.get("AMLT_JOB_NAME", args.exp_name),
        )
        wandb_logger.experiment.save("*.py")
        loggers.append(wandb_logger)
    else:
        wandb_logger = None

    loggers.append(pl.loggers.CSVLogger(save_dir=args.output_dir, name="csv_metrics"))

    kwargs = {"val_check_interval": args.eval_every} if args.eval_every else {}
    # get metric monitors for models
    callbacks = get_monitors(args)
    callbacks.append(ProgressCallback())

    monitor = "val/loss"
    mode = "min"
    model_name = args.model.replace("/", "_")

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        monitor=monitor,
        filename=f"{model_name}" + f"_{args.exp_name}" + "-{" + monitor + ":.004f}",
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
        strategy="auto",
        callbacks=callbacks,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        precision=int(args.precision)
        if args.precision in ["16", "32"]
        else args.precision,
        **kwargs,
    )

    trainer.fit(module, dm)
    trainer.validate(module, dm)


if __name__ == "__main__":
    args = AlpacaConfig.parse(raise_error=False)
    run(args)
