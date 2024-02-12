import os
import sys
import pytorch_lightning as pl
import glob
import copy

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from mttl.models.modifiers.expert_containers.expert_library import get_expert_library
from mttl.callbacks import LiveCheckpointCallback

from mttl.models.monitors import get_monitors
from projects.wiki_experts.src.callbacks import DownstreamEvalCallback


import torch
from pytorch_lightning import Trainer, seed_everything

from projects.wiki_experts.utils import get_datamodule
from mttl.callbacks import NanoMMLUCallback, RougeCallback
from mttl.utils import (
    get_pl_loggers,
    remote_login,
    setup_logging,
    logger,
)
from huggingface_hub import whoami
from projects.wiki_experts.src.expert_trainer import ExpertTrainer
from projects.wiki_experts.src.expert_model import MultiExpertModel
from projects.wiki_experts.src.config import ExpertConfig
from projects.wiki_experts.src.evolution.transfer_matrix import (
    TransferMatrixConfig,
    run_eval as produce_transfer_matrix,
)


def parse_libname(libname):
    parts = libname.split("/")
    if len(parts) == 2:
        return libname, None
    else:
        return "/".join(parts[:-1]), parts[-1]


def main(args: ExpertConfig):
    # load expert from the library
    seed_everything(args.seed, workers=True)

    # get directory of the current file
    setup_logging(args.output_dir)
    logger.info("Args: {}".format(args.to_json()))

    remote_login(args.remote_token)

    library_id, expert_name = parse_libname(args.library_id)
    assert (
        expert_name is not None
    ), "Please provide the library_id in the format user/name/expert_name"

    library = get_expert_library(library_id, create=False)
    expert = library[expert_name]

    training_config = expert.training_config
    training_config.include_task_source = args.include_task_source
    training_config.output_dir = args.output_dir
    training_config.router_selector = "phatgoose_selector"
    training_config.trainable_param_names = ".*selector.*"
    training_config.dataset = expert.expert_info.dataset
    training_config.subsample_train = args.subsample_train
    if expert.expert_info.expert_task_name:
        train_tasks = expert.expert_info.expert_task_name.split(",")
        training_config.finetune_task_name = ",".join(train_tasks)

    model_class = MultiExpertModel
    module: MultiExpertModel = model_class(**vars(training_config))
    module.add_expert_instance(expert, is_default=True)

    # we will push the nex experts to a new dedicated library
    hf_user = whoami()["name"]
    training_config.library_id = (
        f"{hf_user}/{library_id.split('/')[1]}/_phatgoose_selector"
    )
    run_multitask(training_config, module)


def run_multitask(args: ExpertConfig, module: ExpertTrainer):
    loggers = get_pl_loggers(args)
    dm = get_datamodule(args)
    args.n_tasks = len(dm._task_names)

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

    if args.do_train:
        trainer.fit(module, dm)

        torch.cuda.empty_cache()

        # reload best model before pushing!
        checkpoint = (
            checkpoint_callback.best_model_path or checkpoint_callback.last_model_path
        )
        module.load_state_dict(torch.load(checkpoint)["state_dict"])

        if args.library_id and checkpoint:
            library = get_expert_library(args.library_id, create=True)
            library.add_expert_from_ckpt(checkpoint)

        if args.hf_repo_id and checkpoint:
            from projects.wiki_experts.src.expert_model import push_expert_to_hub

            push_expert_to_hub(checkpoint, args.hf_repo_id, auto_search=False)


if __name__ == "__main__":
    args = ExpertConfig.parse()
    main(args)
