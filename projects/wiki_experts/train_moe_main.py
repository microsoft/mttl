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

from mttl.datamodule.mt_seq_to_seq_module import (
    FlatMultiTaskConfig,
    FlatMultiTaskModule,
)

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from mttl.callbacks import RougeCallback
from mttl.utils import get_mlf_logger, setup_logging, logger
from projects.wiki_experts.src.expert_model import MoETrainer
from projects.wiki_experts.src.config import ExpertConfig


class SimpleLogger(pl.loggers.logger.DummyLogger):
    def __init__(self, output_dir):
        self.metrics = {}
        self.output_file = os.path.join(output_dir, "metrics.json")

    def log_metrics(self, metrics, step=None):
        for k, v in metrics.items():
            if k not in self.metrics:
                self.metrics[k] = []
            self.metrics[k].append({"step": step, "value": v})
        with open(self.output_file, "w") as f:
            json.dump(self.metrics, f)


def get_datamodule(args, for_generation=False):
    # refactor all the common arguments below into a dict common kwargs
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
        "dataset": args.dataset.replace("qa:", "").replace("raw_docs:", ""),
        "train_on_inputs": False,
    }
    if "flat" in args.dataset:
        config = FlatMultiTaskConfig(
            **common_kwargs,
            source_template=args.source_template,
            augment_few_shot=args.augment_few_shot,
        )
        dm = FlatMultiTaskModule(config, for_generation=for_generation)
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
    dm = get_datamodule(args)
    args.n_tasks = len(dm._task_names)
    gen_dm = get_datamodule(args, for_generation=True)

    # legit logging
    loggers = []
    exp_name = os.environ.get("AMLT_JOB_NAME", args.exp_name)
    if os.environ.get("WANDB_API_KEY") or args.wandb_project:
        import wandb

        project = "wiki_experts" if args.wandb_project is None else args.wandb_project
        args.exp_name = "dev_run" if args.exp_name is None else args.exp_name
        project = os.environ.get("WANDB_PROJECT", project)
        wandb_logger = pl.loggers.WandbLogger(
            project=project,
            name=exp_name,  # , config=args_
            settings=wandb.Settings(start_method="fork"),
        )
        wandb_logger.experiment.save("*.py")
        loggers.append(wandb_logger)

    args.trainable_param_names = args.trainable_param_names + "|.*rkhs.*"

    module = MoETrainer(**vars(args), tokenizer=dm.tokenizer)

    mlf_logger = get_mlf_logger()
    if mlf_logger:
        loggers.append(mlf_logger)

    if args.tensorboard:
        tb_logger = pl.loggers.TensorBoardLogger(save_dir=args.output_dir)
        loggers.append(tb_logger)

    loggers.append(SimpleLogger(args.output_dir))

    # get metric monitors for models
    callbacks = []

    monitor = "val/loss"
    mode = "min"

    model_name = args.model.replace("/", "_")
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        monitor=monitor,
        filename=f"{model_name}" + "-{" + monitor + ":.004f}",
        save_top_k=1,
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

    callbacks.append(
        RougeCallback(gen_dm, every_n_epochs=3, subsample=int(subsample), max_length=3)
    )

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
