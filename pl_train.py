import os
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

from mttl.online_eval import NIOnlineZeroShot, T0OnlineZeroShot
from mttl.config import parse_config
from mttl.callbacks import ProgressCallback
from mttl.datamodule.ni_data_module import NIPretrainDataModule
from mttl.datamodule.xfit_data_module import XFitPretrainDataModule
from mttl.datamodule.t0_data_module import T0PretrainDataModule
from mttl.models.encoder_decoder import EncoderDecoder
from mttl.models.t0_encoder_decoder import T0EncoderDecoder
from mttl.models.monitors import get_monitors
from mttl.utils import get_mlf_logger


def run_multitask(args):
    seed_everything(args.seed, workers=True)

    # select dataloader
    if args.dataset == "xfit":
        model_class = EncoderDecoder
        dm = XFitPretrainDataModule(args)
    elif args.dataset == "ni":
        model_class = EncoderDecoder
        dm = NIPretrainDataModule(args)
    elif args.dataset == "t0":
        model_class = T0EncoderDecoder
        dm = T0PretrainDataModule(args)
    else:
        raise NotImplementedError()

    args.n_tasks = len(dm.task2id)

    if args.checkpoint is not None:
        from mttl.utils import get_checkpoint_path

        checkpoint_path = get_checkpoint_path(args.checkpoint)

        kwargs = vars(args)
        kwargs.pop("checkpoint")
        module = model_class.load_from_checkpoint(
            checkpoint_path, **kwargs, tokenizer=dm.tokenizer
        )
    else:
        module = model_class(**vars(args), tokenizer=dm.tokenizer)

    # legit logging
    loggers = []
    if os.environ.get("WANDB_API_KEY"):
        wandb_logger = pl.loggers.WandbLogger(
            project=args.wandb_project,
            name=args.exp_name,
        )
        wandb_logger.experiment.save("*.py")
        loggers.append(wandb_logger)
    else:
        wandb_logger = None

    mlf_logger = get_mlf_logger()
    if mlf_logger:
        loggers.append(mlf_logger)

    loggers.append(pl.loggers.CSVLogger(save_dir=args.output_dir, name="csv_metrics"))

    kwargs = {"val_check_interval": args.eval_every * args.gradient_accumulation_steps} if args.eval_every else {}

    # get metric monitors for models
    callbacks = get_monitors(args)
    callbacks.append(ProgressCallback())

    monitor = "val/loss"
    mode = "min"

    if args.dataset in ["ni", "xfit"]:
        if args.early_stop_on_zero_shot and not args.ni_online_eval:
            raise NotImplementedError("Specify online zero-shot if early stopping on zero shot.")

        if args.ni_online_eval:
            callbacks.append(NIOnlineZeroShot(args.eval_every))

            if args.early_stop_on_zero_shot:
                monitor = "val/zero_shot_perf"
                mode = "max"

        checkpoint_callback = ModelCheckpoint(
            dirpath=args.output_dir,
            monitor=monitor,
            filename=f"{args.model}" + "-{" + monitor + ":.004f}",
            save_top_k=1,
            save_last=True,
            save_weights_only=True,  # make checkpoints smaller
            mode=mode,
        )
        callbacks.append(checkpoint_callback)
    else:
        # no need for checkpointing in t0 as we checkpoint manually in the module    
        if args.t0_online_eval:
            callbacks.append(T0OnlineZeroShot(args.eval_every))

            if args.early_stop_on_zero_shot:
                raise NotImplementedError()

        kwargs["enable_checkpointing"] = False

    trainer = Trainer(
        devices=-1,
        accelerator="gpu",
        logger=loggers,
        num_sanity_val_steps=5,
        default_root_dir=args.output_dir,
        max_epochs=args.num_train_epochs,
        max_steps=args.total_steps + 1 if args.total_steps != -1 else -1,
        gradient_clip_val=args.max_grad_norm,
        log_every_n_steps=50,
        strategy=args.compute_strategy if args.compute_strategy else "auto",
        callbacks=callbacks,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        precision=int(args.precision)
        if args.precision in ["16", "32"]
        else args.precision,
        **kwargs,
    )

    trainer.fit(module, dm)

    try:
        trainer.validate(module, dm)
    except:
        pass


if __name__ == "__main__":
    args = parse_config()
    run_multitask(args)
