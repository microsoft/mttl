import os
import sys
import json
import pytorch_lightning as pl

from mttl.models.modifiers.expert_containers.expert_library import HFExpertLibrary

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
from huggingface_hub import login
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
import json

from mttl.callbacks import LiveCheckpointCallback, RougeCallback
from mttl.utils import (
    add_mlf_logger,
    add_simple_logger,
    add_tb_logger,
    add_wandb_logger,
    setup_logging,
    logger,
)

from projects.wiki_experts.train_experts_main import get_datamodule
from projects.wiki_experts.src.expert_model import MoETrainer
from projects.wiki_experts.src.config import ExpertConfig


def run_multitask(args: ExpertConfig):
    seed_everything(args.seed, workers=True)

    # get directory of the current file
    setup_logging(args.output_dir)

    logger.info("Args: {}".format(args.to_json()))

    if args.hf_token_hub:
        login(token=args.hf_token_hub)

    # select dataloader
    dm = get_datamodule(args)
    args.n_tasks = len(dm._task_names)
    gen_dm = get_datamodule(args, for_generation=True)

    # legit logging
    loggers = []
    add_mlf_logger(loggers)
    add_wandb_logger(loggers, args)
    add_tb_logger(loggers, args)
    add_simple_logger(loggers, args)

    args.trainable_param_names = args.trainable_param_names + "|.*rkhs.*"

    module = MoETrainer(**vars(args), tokenizer=dm.tokenizer)

    # get metric monitors for models
    callbacks = []

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
        val_check_interval = args.eval_every
        if val_check_interval > len(dm.train_dataloader()):
            val_check_interval = len(dm.train_dataloader())
        elif val_check_interval > args.total_steps and args.total_steps != -1:
            val_check_interval = args.total_steps

    logger.warn("Validitating every {} steps!".format(val_check_interval))

    # evaluate 500 batches generation
    num_batches = min(len(gen_dm.val_dataloader()), 500)
    subsample = len(gen_dm.val_dataloader()) / num_batches

    callbacks.append(RougeCallback(gen_dm, every_n_epochs=3, subsample=int(subsample)))

    trainer = Trainer(
        devices=-1,
        accelerator="auto",
        strategy="ddp_find_unused_parameters_true",
        logger=loggers,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        default_root_dir=args.output_dir,
        max_epochs=args.num_train_epochs if not args.total_steps else None,
        max_steps=args.total_steps + 1 if args.total_steps != -1 else -1,
        gradient_clip_val=args.max_grad_norm,
        callbacks=callbacks,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        precision=int(args.precision)
        if args.precision in ["16", "32"]
        else args.precision,
        limit_val_batches=500,
        val_check_interval=val_check_interval,
    )

    # initial validation!
    trainer.fit(module, dm)
    trainer.test(module, dm)

    del module
    torch.cuda.empty_cache()

    # reload best model before pushing!
    checkpoint = (
        checkpoint_callback.best_model_path or checkpoint_callback.last_model_path
    )

    if args.hf_lib_id and checkpoint:
        library = HFExpertLibrary(args.hf_lib_id, create=True)
        library.add_expert_from_ckpt(checkpoint)

    if args.hf_repo_id and checkpoint:
        from projects.wiki_experts.src.expert_model import push_expert_to_hub

        push_expert_to_hub(checkpoint, args.hf_repo_id, auto_search=False)


if __name__ == "__main__":
    args = ExpertConfig.parse()
    run_multitask(args)
