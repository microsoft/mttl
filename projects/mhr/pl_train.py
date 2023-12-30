import os
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

from mttl.online_eval import NIOnlineZeroShot, T0OnlineZeroShot
from mttl.callbacks import ProgressCallback
from mttl.datamodule.ni_data_module import NiDataModule
from mttl.datamodule.t0_data_module import T0PretrainDataModule
from mttl.datamodule.mt_seq_to_seq_module import T0FlatModule, T0FlatConfig
from mttl.models.encoder_decoder import EncoderDecoder
from mttl.models.t0_encoder_decoder import T0EncoderDecoder
from mttl.models.monitors import get_monitors
from mttl.utils import add_mlf_logger, add_wandb_logger, setup_logging
from mttl.config import Config


def run_multitask(args):
    seed_everything(args.seed, workers=True)
    setup_logging(args.output_dir)

    # select dataloader
    if args.dataset == "ni":
        model_class = EncoderDecoder
        dm = NiDataModule(args)
    elif args.dataset == "t0":
        model_class = T0EncoderDecoder
        dm = T0PretrainDataModule(args)
    elif args.dataset == "sordonia/t0-1.6M-flat":
        model_class = T0EncoderDecoder
        dm = T0FlatModule(
            T0FlatConfig(
                dataset=args.dataset,
                model=args.model,
                train_batch_size=args.train_batch_size,
                predict_batch_size=args.train_batch_size,
                max_input_length=args.max_input_length,
                max_output_length=args.max_output_length,
                validation_portion=0.05,
                padding_side="right",
                model_family="seq2seq",
                use_templates_as_tasks=args.use_t0_templates_as_tasks,
            )
        )
    else:
        raise NotImplementedError()

    args.n_tasks = len(dm.task_to_id)

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
    add_wandb_logger(args, loggers)
    add_mlf_logger(loggers)
    loggers.append(pl.loggers.CSVLogger(save_dir=args.output_dir, name="csv_metrics"))

    kwargs = (
        {"val_check_interval": args.eval_every * args.gradient_accumulation_steps}
        if args.eval_every
        else {}
    )

    # get metric monitors for models
    callbacks = get_monitors(args)
    callbacks.append(ProgressCallback())

    monitor = "val/loss"
    mode = "min"

    if args.dataset in ["ni"]:
        if args.early_stop_on_zero_shot and not args.ni_online_eval:
            raise NotImplementedError(
                "Specify online zero-shot if early stopping on zero shot."
            )

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
    args = Config.parse()
    run_multitask(args)
