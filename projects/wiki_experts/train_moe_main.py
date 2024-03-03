import os
import sys
import torch
from pytorch_lightning import Trainer, seed_everything

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from mttl.models.modifiers.expert_containers.expert_library import get_expert_library
from mttl.callbacks import LiveCheckpointCallback
from mttl.models.monitors import get_monitors
from mttl.callbacks import NanoMMLUCallback, RougeCallback
from mttl.utils import (
    get_pl_loggers,
    remote_login,
    setup_logging,
    logger,
)
from mttl.datamodule.base import get_datamodule

from projects.wiki_experts.src.callbacks import DownstreamEvalCallback
from mttl.models.expert_model import MoEModel
from mttl.models.expert_config import ExpertConfig


def run_multitask(args: ExpertConfig):
    seed_everything(args.seed, workers=True)

    # get directory of the current file
    setup_logging(args.output_dir)
    logger.info("Args: {}".format(args.to_json()))

    remote_login(args.remote_token)
    # select dataloader
    model_class = MoEModel
    dm = get_datamodule(args)
    args.n_tasks = len(dm._task_names)
    args.task_names = dm._task_names

    loggers = get_pl_loggers(args)
    module = model_class(**vars(args), tokenizer=dm.tokenizer)

    # get metric monitors for models
    callbacks = get_monitors(args)
    checkpoint_callback = LiveCheckpointCallback(
        dirpath=args.output_dir,
        monitor="val/loss",
        save_last=True,
        mode="min",
    )
    callbacks.append(checkpoint_callback)

    if args.eval_rouge_flag:
        rouge = RougeCallback(
            get_datamodule(args, for_generation=True),
            every_n_epochs=3 if args.num_train_epochs > 3 else 1,
        )
        callbacks.append(rouge)
    else:
        logger.warn(
            "Deactivating rouge callback as it is not enabled in the config. Please set `eval_rouge_flag=True`."
        )

    if args.eval_mmlu_flag:
        mmlu = NanoMMLUCallback(
            get_datamodule(args, dataset_override="mmlu", for_generation=True),
            every_n_epochs=3 if args.num_train_epochs > 3 else 1,
        )
        callbacks.append(mmlu)
    else:
        logger.warn(
            "Deactivating mmlu callback as it is not enabled in the config. Please set `eval_mmlu_flag=True`."
        )

    if args.pipeline_eval_tasks:
        if args.pipeline_eval_tasks == "all":
            args.pipeline_eval_tasks = "arc-challenge,arc-easy,boolq,hellaswag,humaneval,mbpp,openbookqa,piqa,bbh-fast,winogrande"

        eval = DownstreamEvalCallback(args)
        callbacks.append(eval)
    else:
        logger.warn(
            "Deactivating downstream eval callback as it is not enabled in the config. Please set `pipeline_eval_tasks`."
        )

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
        strategy="auto",
        num_sanity_val_steps=0,
        default_root_dir=args.output_dir,
        max_epochs=args.num_train_epochs,
        max_steps=args.total_steps + 1 if args.total_steps != -1 else -1,
        gradient_clip_val=args.max_grad_norm,
        callbacks=callbacks,
        enable_checkpointing=False,
        log_every_n_steps=args.gradient_accumulation_steps,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        precision=int(args.precision)
        if args.precision in ["16", "32"]
        else args.precision,
        val_check_interval=val_check_interval,
    )

    # initial validation only for a bunch of datasets... ?
    trainer.validate(module, dm)

    if args.do_train:
        trainer.fit(module, dm)

        torch.cuda.empty_cache()

        # reload best model before pushing!
        checkpoint = (
            checkpoint_callback.best_model_path or checkpoint_callback.last_model_path
        )
        module.load_state_dict(torch.load(checkpoint)["state_dict"])
        trainer.test(module, dm)

        if args.library_id and checkpoint:
            library = get_expert_library(args.library_id, create=True)
            library.add_expert_from_ckpt(checkpoint)


if __name__ == "__main__":
    args = ExpertConfig.parse()
    run_multitask(args)
