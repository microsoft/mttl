import os
import sys
import json
import pytorch_lightning as pl

from huggingface_hub import login
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, Callback

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from mttl.callbacks import MMLUCallback
from mttl.datamodule.oasst1_module import OA1Config, OA1Module
from mttl.datamodule.retrieval_lm_module import RetrievalLMDataModule
from mttl.datamodule.platypus_module import PlatypusModule, PlatypusConfig, PlatypusQAModule
from mttl.utils import get_mlf_logger, setup_logging, logger

from projects.wiki_experts.expert_trainer import ExpertTrainer
from projects.wiki_experts.config import ExpertConfig


def run_multitask(args):
    seed_everything(args.seed, workers=True)

    # get directory of the current file
    setup_logging(args.output_dir)

    logger.info("Args: {}".format(args.to_json()))

    if args.hf_token_hub:
        login(token=args.hf_token_hub)

    # select dataloader
    model_class = ExpertTrainer

    if args.dataset.startswith("qa-"):
        args.dataset = args.dataset.replace("qa-", "")
        config = PlatypusConfig(
            model=args.model,
            padding_side=args.padding_side,
            train_batch_size=args.train_batch_size,
            predict_batch_size=args.predict_batch_size,
            max_input_length=args.max_input_length,
            max_output_length=args.max_output_length,
            validation_portion=args.validation_portion,
            model_family=args.model_family,
            train_on_inputs=False,
            finetune_task_name=args.finetune_task_name,
            dataset=args.dataset,
        )
        dm = PlatypusQAModule(config)
    elif "platypus" in args.dataset:
        config = PlatypusConfig(
            model=args.model,
            padding_side=args.padding_side,
            train_batch_size=args.train_batch_size,
            predict_batch_size=args.predict_batch_size,
            max_input_length=args.max_input_length,
            max_output_length=args.max_output_length,
            validation_portion=args.validation_portion,
            model_family=args.model_family,
            train_on_inputs=False,
            train_on_reverse=args.dataset == "inverse-platypus",
        )
        dm = PlatypusModule(config)
    elif "oa1" in args.dataset:
        config = OA1Config(
            model=args.model,
            padding_side=args.padding_side,
            train_batch_size=args.train_batch_size,
            predict_batch_size=args.predict_batch_size,
            max_input_length=args.max_input_length,
            max_output_length=args.max_output_length,
            validation_portion=args.validation_portion,
            model_family=args.model_family,
            train_on_inputs=False,
            train_on_reverse=args.dataset == "inverse-oa1",
        )
        dm = OA1Module(config)
    else:
        dm = RetrievalLMDataModule(args)

    args.n_tasks = len(dm.task_to_id) if hasattr(dm, "task_to_id") else 0
    module = model_class(**vars(args), tokenizer=dm.tokenizer)

    # legit logging
    loggers = []
    if os.environ.get("WANDB_API_KEY") or args.wandb_project:
        project = (
            "wiki_experts" if args.wandb_project is None else args.wandb_project
        )
        args.exp_name = "dev_run" if args.exp_name is None else args.exp_name
        project = os.environ.get("WANDB_PROJECT", project)
        exp_name = os.environ.get("AMLT_JOB_NAME", args.exp_name)
        exp_name += f"_{args.finetune_task_name}"
        wandb_logger = pl.loggers.WandbLogger(
            project=project,
            name=exp_name,  # , config=args_
        )
        wandb_logger.experiment.save("*.py")
        loggers.append(wandb_logger)

    mlf_logger = get_mlf_logger()
    if mlf_logger:
        loggers.append(mlf_logger)

    if args.tensorboard:
        tb_logger = pl.loggers.TensorBoardLogger(save_dir=args.output_dir)
        loggers.append(tb_logger)

    loggers.append(pl.loggers.CSVLogger(save_dir=args.output_dir, name="csv_metrics"))

    # get metric monitors for models
    callbacks = []
    
    monitor = "downstream_val/mmlu"
    mode = "max"

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
    if val_check_interval == -1:
        val_check_interval = None
    elif val_check_interval > len(dm.train_dataloader()):
        val_check_interval = len(dm.train_dataloader())
    elif val_check_interval > args.total_steps and args.total_steps != -1:
        val_check_interval = args.total_steps

    callbacks.append(MMLUCallback(eval_every=val_check_interval))
    
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
    trainer.fit(module, dm)

    # reload best model before pushing!
    checkpoint = (
        checkpoint_callback.best_model_path or checkpoint_callback.last_model_path
    )

    if args.hf_repo_id and checkpoint:
        from expert_model import push_expert_to_hub

        push_expert_to_hub(checkpoint, args.hf_repo_id, auto_search=False)


if __name__ == "__main__":
    args = ExpertConfig.parse()
    run_multitask(args)
