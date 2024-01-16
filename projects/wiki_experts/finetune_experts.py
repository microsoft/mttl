import os
import sys
import pytorch_lightning as pl
import glob

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from mttl.datamodule.mbpp_datamodule import MBPPDataConfig, MBPPDataModule
from mttl.datamodule.mmlu_data_module import MMLUDataConfig, MMLUDataModule

from mttl.models.modifiers.expert_containers.expert_library import HFExpertLibrary
from mttl.callbacks import LiveCheckpointCallback

from mttl.models.monitors import get_monitors
from projects.wiki_experts.src.callbacks import DownstreamEvalCallback
from projects.wiki_experts.src.expert_model import MoETrainer
from mttl.models.modifiers.expert_containers.module_graph import load_expert


import torch
from huggingface_hub import login
from pytorch_lightning import Trainer, seed_everything

from projects.wiki_experts.utils import get_datamodule
from mttl.callbacks import NanoMMLUCallback, RougeCallback
from mttl.utils import (
    get_checkpoint_path,
    get_pl_loggers,
    setup_logging,
    logger,
)

from projects.wiki_experts.src.expert_trainer import ExpertTrainer
from projects.wiki_experts.src.config import ExpertConfig


def run_multitask(args: ExpertConfig):
    seed_everything(args.seed, workers=True)

    # get directory of the current file
    setup_logging(args.output_dir)
    logger.info("Args: {}".format(args.to_json()))

    if args.hf_token_hub:
        login(token=args.hf_token_hub)

    loggers = get_pl_loggers(args)
    # select dataloader
    dm = get_datamodule(args)
    args.n_tasks = len(dm._task_names)

    if args.checkpoint is not None:
        # Passing a checkpoint assumes the use of `ExpertTrainer`
        # e.g. for poly-μ and MHR-μ
        ckpt_path = get_checkpoint_path(args.checkpoint)
        expert = load_expert(ckpt_path)
        module = ExpertTrainer(**vars(expert.training_config))

        ckpt = torch.load(ckpt_path)
        result = module.load_state_dict(ckpt["state_dict"], strict=False)
        assert len(result.unexpected_keys) == 0, result.unexpected_keys

        # For Poly and MHR, apply potential averaging, or resizing
        if args.finetune_type and args.finetune_type == "MuZ":
            module.model.switch_selector_to_average()
        elif expert.training_config.model_modifier == "poly":
            module.model.resize_module_logits(1)
    elif args.hf_lib_id is not None:
        # load a MultiExpertModel
        if args.library_to_expert_transform:
            # load the appropriate transform
            # load library and pass to transform
            # retrieve expert, and add as previously done.
            raise NotImplementedError()
        else:
            module = MoETrainer(**vars(args), tokenizer=dm.tokenizer)
    else:
        raise ValueError("please specify a library, or a checkpoint")

    # get metric monitors for models
    callbacks = get_monitors(args)
    if "mbpp" in args.dataset:
        monitor = "downstream/mbpp"
        mode = "max"
    else:
        monitor = "val/loss"
        mode = "min"

    checkpoint_callback = LiveCheckpointCallback(
        dirpath=args.output_dir,
        monitor=monitor,
        save_last=True,
        mode=mode,
    )
    callbacks.append(checkpoint_callback)

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
        precision=int(args.precision)
        if args.precision in ["16", "32"]
        else args.precision,
        val_check_interval=val_check_interval,
    )

    # initial validation only for a bunch of datasets... ?
    trainer.validate(module, dm)
    trainer.fit(module, dm)


if __name__ == "__main__":
    args = ExpertConfig.parse()
    run_multitask(args)
