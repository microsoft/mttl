import os
import sys
import json
import pytorch_lightning as pl

from huggingface_hub import login
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from mttl.datamodule.wiki_mmlu_module import WikiMMLUDataModule
from mttl.datamodule.platypus_module import PlatypusModule
from mttl.callbacks import MiniProgress
from mttl.utils import get_mlf_logger, setup_logging, logger

from projects.wiki_experts.expert_trainer import ExpertTrainer
from config import ExpertConfig


def run_multitask(args):
    seed_everything(args.seed, workers=True)

    # get directory of the current file
    setup_logging(args.output_dir)

    logger.info("Args: {}".format(args.to_json()))

    if args.hf_token_hub:
        login(token=args.hf_token_hub)

    # select dataloader
    model_class = ExpertTrainer
    if args.dataset == "platypus":
        dm = PlatypusModule(args)
    elif "wiki_mmlu" in args.dataset:
        dm = WikiMMLUDataModule(args)

    args.n_tasks = len(dm.task_to_id) if hasattr(dm, "task_to_id") else 0
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

    val_check_interval = args.eval_every
    if val_check_interval > len(dm.train_dataloader()):
        val_check_interval = len(dm.train_dataloader())
    if val_check_interval > args.total_steps and args.total_steps != -1:
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
        log_every_n_steps=20,
        strategy=args.compute_strategy if args.compute_strategy else "auto",
        callbacks=callbacks,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        precision=int(args.precision)
        if args.precision in ["16", "32"]
        else args.precision,
        val_check_interval=val_check_interval,
    )

    # initial validation!
    trainer.validate(module, dm)[0]
    trainer.fit(module, dm)

    # reload best model before pushing!
    checkpoint = checkpoint_callback.best_model_path or checkpoint_callback.last_model_path

    if args.hf_repo_id and checkpoint:
        from mttl.models.utils import convert_and_push_to_hub

        convert_and_push_to_hub(
            checkpoint,
            "{}/experts-{}-{}".format(
                args.hf_repo_id, args.model.replace("/", "_").lower(), args.expert_name
            ),
            auto_search=False,
        )


if __name__ == "__main__":
    args = ExpertConfig.parse()
    run_multitask(args)
