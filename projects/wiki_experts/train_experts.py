import os
import sys
import json
import wandb
import pytorch_lightning as pl

from huggingface_hub import login
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
import json

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from mttl.callbacks import MMLUCallback
from mttl.evaluators import MMLUEvaluator
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from mttl.datamodule.oasst1_module import OA1Config, OA1Module
from mttl.datamodule.retrieval_lm_module import RetrievalLMDataModule
from mttl.datamodule.facts_lm_module import FactsLMConfig, FactsLMDataModule
from mttl.datamodule.platypus_module import (
    PlatypusModule,
    PlatypusConfig,
    PlatypusQAModule,
)
from mttl.utils import get_mlf_logger, setup_logging, logger

from projects.wiki_experts.src.expert_trainer import ExpertTrainer
from projects.wiki_experts.src.config import ExpertConfig


class SimpleLogger(pl.loggers.logger.DummyLogger):
    def __init__(self, output_dir):
        self.output_file = os.path.join(output_dir, "metrics.json")
        self.metrics = []

    def log_metrics(self, metrics, step=None):
        metrics["step"] = step
        self.metrics.append(metrics)

        with open(self.output_file, "w") as f:
            for metric in self.metrics:
                f.write(json.dumps(metric) + "\n")


def eval_mmlu(module, args):
    mmlu = MMLUEvaluator(
        args,
        split=args.mmlu_test_split,
    )
    scores = mmlu.evaluate(module)
    logger.info("MMLU Accuracy: {}".format(scores["all"]["mean"]))
    for t, v in scores.items():
        logger.info("MMLU Accuracy {}: {}".format(t, v["mean"]))
    # super hard to log with pllogger here
    if wandb.run is not None:
        wandb.log({"downstream/mmlu_test_best_model": scores["all"]["mean"]})


def run_multitask(args):
    seed_everything(args.seed, workers=True)

    # get directory of the current file
    setup_logging(args.output_dir)

    logger.info("Args: {}".format(args.to_json()))

    if args.hf_token_hub:
        login(token=args.hf_token_hub)

    # select dataloader
    model_class = ExpertTrainer

    if "qa" in args.dataset:
        args.dataset = (
            args.dataset.split("/")[0].replace("qa-", "")
            + "/"
            + args.dataset.split("/")[1]
        )
        config = PlatypusConfig(
            model=args.model,
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
    elif "facts" in args.dataset or "id" in args.dataset:
        config = FactsLMConfig(
            model=args.model,
            train_batch_size=args.train_batch_size,
            predict_batch_size=args.predict_batch_size,
            max_input_length=args.max_input_length,
            validation_portion=args.validation_portion,
            finetune_task_name=args.finetune_task_name,
            dataset=args.dataset,
            text_field="facts" if "facts" in args.dataset else "text",
        )
        dm = FactsLMDataModule(config)
    elif "platypus" in args.dataset:
        config = PlatypusConfig(
            model=args.model,
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
        raise ValueError(f"Unknown dataset {args.dataset}")

    args.n_tasks = len(dm.task_to_id) if hasattr(dm, "task_to_id") else 0
    module = model_class(**vars(args), tokenizer=dm.tokenizer)

    # legit logging
    loggers = []
    if os.environ.get("WANDB_API_KEY") or args.wandb_project:
        import wandb

        project = "wiki_experts" if args.wandb_project is None else args.wandb_project
        args.exp_name = "dev_run" if args.exp_name is None else args.exp_name
        project = os.environ.get("WANDB_PROJECT", project)
        exp_name = os.environ.get("AMLT_JOB_NAME", args.exp_name)
        exp_name += f"_{args.finetune_task_name}"
        wandb_logger = pl.loggers.WandbLogger(
            project=project,
            name=exp_name,  # , config=args_
            settings=wandb.Settings(start_method="fork"),
        )
        wandb_logger.experiment.save("*.py")
        loggers.append(wandb_logger)

    mlf_logger = get_mlf_logger()
    if mlf_logger:
        loggers.append(mlf_logger)

    if args.tensorboard:
        tb_logger = pl.loggers.TensorBoardLogger(save_dir=args.output_dir)
        loggers.append(tb_logger)

    loggers.append(SimpleLogger(args.output_dir))

    # get metric monitors for models
    callbacks = []

    criteria = {
        "downstream/val/mmlu": "max",
        "val/loss": "min",
    }
    monitor = args.selection_criteria
    mode = criteria[monitor]

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
    else:
        val_check_interval = args.gradient_accumulation_steps * args.eval_every
        if val_check_interval > len(dm.train_dataloader()):
            val_check_interval = len(dm.train_dataloader())
        elif val_check_interval > args.total_steps and args.total_steps != -1:
            val_check_interval = args.total_steps

    callbacks.append(MMLUCallback(split="test"))
    callbacks.append(MMLUCallback(split="val"))

    if args.es_patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor=monitor, patience=args.es_patience, mode=mode, verbose=True
            )
        )

    trainer = Trainer(
        devices=-1,
        accelerator="gpu",
        logger=loggers,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        default_root_dir=args.output_dir,
        max_epochs=args.num_train_epochs,
        max_steps=args.total_steps + 1 if args.total_steps != -1 else -1,
        gradient_clip_val=args.max_grad_norm,
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

    # perform final eval on MMLU
    if checkpoint:
        module = model_class.load_from_checkpoint(checkpoint).to("cuda")
        eval_mmlu(module, args)

    if args.hf_repo_id and checkpoint:
        from projects.wiki_experts.src.expert_model import push_expert_to_hub

        push_expert_to_hub(checkpoint, args.hf_repo_id, auto_search=False)


if __name__ == "__main__":
    args = ExpertConfig.parse()
    run_multitask(args)
