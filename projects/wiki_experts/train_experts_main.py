import os
import sys
import pytorch_lightning as pl
import glob

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from mttl.datamodule.mmlu_data_module import MMLUDataConfig, MMLUDataModule

from mttl.models.modifiers.expert_containers.expert_library import HFExpertLibrary
from mttl.callbacks import LiveCheckpointCallback

from mttl.models.monitors import get_monitors
from projects.wiki_experts.src.callbacks import DownstreamEvalCallback


import torch
from huggingface_hub import login
from pytorch_lightning import Trainer, seed_everything

from mttl.datamodule.mt_seq_to_seq_module import (
    FlanConfig,
    FlanModule,
    FlatMultiTaskConfig,
    FlatMultiTaskModule,
)

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from mttl.callbacks import NanoMMLUCallback, RougeCallback
from mttl.utils import (
    get_pl_loggers,
    setup_logging,
    logger,
)

from projects.wiki_experts.src.expert_trainer import ExpertTrainer
from projects.wiki_experts.src.config import ExpertConfig
from projects.wiki_experts.src.evolution.transfer_matrix import (
    TransferMatrixConfig,
    produce_transfer_matrix,
)


def create_transfer_matrix(args, checkpoint):
    ########################
    # create transfer matrix
    config = TransferMatrixConfig()
    for k, v in vars(args).items():
        if k in vars(config):
            setattr(config, k, v)
    config.eval_metric = "rougeL"
    config.hf_repo_id = checkpoint
    config.finetune_task_name = (
        args.finetune_task_name.split(",")
        if not isinstance(args.finetune_task_name, list)
        else args.finetune_task_name
    )
    produce_transfer_matrix(config, debug=False)
    ########################


def get_datamodule(args, for_generation=False, dataset_override=None):
    # refactor all the common arguments below into a dict common kwargs
    dataset = args.dataset if not dataset_override else dataset_override

    common_kwargs = {
        "model": args.model,
        "train_batch_size": args.train_batch_size,
        "predict_batch_size": args.predict_batch_size,
        "max_input_length": args.max_input_length,
        "max_output_length": args.max_output_length,
        "validation_portion": args.validation_portion,
        "model_family": args.model_family,
        "finetune_task_name": args.finetune_task_name,
        "truncation_side": args.truncation_side,
        "dataset": dataset,
        "train_on_inputs": False,
        "add_eos_to_targets": "qamc"
        not in args.dataset,  # do not add eos for mmlu stuff (for now)
        "subsample_train": args.subsample_train,
        "subsample_dev": args.subsample_dev,
        "subsample_test": args.subsample_test,
    }
    if "flan" in dataset:
        config = FlanConfig(
            **common_kwargs,
            remove_phi_eval_tasks=args.remove_phi_eval_tasks,
        )
        dm = FlanModule(config, for_generation=for_generation)
    elif "flat" in dataset:
        config = FlatMultiTaskConfig(
            **common_kwargs,
            source_template=args.source_template,
            augment_few_shot=args.augment_few_shot,
        )
        dm = FlatMultiTaskModule(config, for_generation=for_generation)
    elif "mmlu" in dataset:
        config = MMLUDataConfig(
            **common_kwargs,
        )
        dm = MMLUDataModule(config, for_generation=for_generation)
    else:
        raise ValueError(f"Unknown dataset {args.dataset}")
    return dm


def run_multitask(args: ExpertConfig):
    seed_everything(args.seed, workers=True)

    # get directory of the current file
    setup_logging(args.output_dir)
    logger.info("Args: {}".format(args.to_json()))

    if args.hf_token_hub:
        login(token=args.hf_token_hub)

    # select dataloader
    model_class = ExpertTrainer
    dm = get_datamodule(args)
    args.n_tasks = len(dm._task_names)

    loggers = get_pl_loggers(args)

    if len(best_checkpoints) > 0:
        best_checkpoint = best_checkpoints[0]
        create_transfer_matrix(best_checkpoint)
        # end program
        sys.exit(0)

    module = model_class(**vars(args), tokenizer=dm.tokenizer)

    best_checkpoints = glob.glob(os.path.join(args.output_dir, "best_mode*", "*.ckpt"))

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
        num_sanity_val_steps=0,
        default_root_dir=args.output_dir,
        max_epochs=args.num_train_epochs,
        max_steps=args.total_steps + 1 if args.total_steps != -1 else -1,
        gradient_clip_val=args.max_grad_norm,
        strategy=args.compute_strategy if args.compute_strategy else "auto",
        callbacks=callbacks,
        enable_checkpointing=False,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        precision=int(args.precision)
        if args.precision in ["16", "32"]
        else args.precision,
        val_check_interval=val_check_interval,
    )

    # initial validation only for a bunch of datasets... ?
    trainer.validate(module, dm)
    trainer.fit(module, dm)

    torch.cuda.empty_cache()

    # reload best model before pushing!
    checkpoint = (
        checkpoint_callback.best_model_path or checkpoint_callback.last_model_path
    )
    module.load_state_dict(torch.load(checkpoint)["state_dict"])
    trainer.test(module, dm)

    if args.hf_lib_id and checkpoint:
        library = HFExpertLibrary(args.hf_lib_id, create=True)
        library.add_expert_from_ckpt(checkpoint)

    if args.hf_repo_id and checkpoint:
        from projects.wiki_experts.src.expert_model import push_expert_to_hub

        push_expert_to_hub(checkpoint, args.hf_repo_id, auto_search=False)

    create_transfer_matrix(checkpoint)


if __name__ == "__main__":
    args = ExpertConfig.parse()
    run_multitask(args)
