import os
import sys
import json
import pytorch_lightning as pl
import torch
from huggingface_hub import login
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
import json

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from mttl.datamodule.mt_seq_to_seq_module import (
    FlanConfig,
    FlanModule,
    FlatMultiTaskConfig,
    FlatMultiTaskModule,
)

from mttl.datamodule.oasst1_module import OA1Config, OA1Module
from mttl.datamodule.facts_lm_module import FactsLMConfig, FactsLMDataModule
from mttl.datamodule.platypus_module import (
    PlatypusModule,
    PlatypusConfig,
    PlatypusQAModule,
)
from mttl.utils import get_mlf_logger, setup_logging, logger

from mttl.models.expert_model import ExpertModel as ExpertTrainer
from mttl.models.expert_config import ExpertConfig
from projects.wiki_experts.src.callbacks import (
    RougeCallbackTestPerEpoch,
    OptimResetCallback,
)
from projects.wiki_experts.src.evolution.transfer_matrix import (
    TransferMatrixConfig,
    run_eval as create_transfer_matrix,
)

DEBUG = True
if "AMLT_OUTPUT_DIR" in os.environ:
    DEBUG = False
if DEBUG:
    print("!!!!!!!!!!!!!!!!!!!!!! DEBUG MODE")


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


def get_datamodule(args, for_generation=False, subsample=-1):
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
        "subsample": subsample,
    }
    if args.dataset.startswith("qa:"):
        config = PlatypusConfig(**common_kwargs)
        dm = PlatypusQAModule(config, for_generation=for_generation)
    elif args.dataset.startswith("raw_docs:"):
        config = FactsLMConfig(
            **common_kwargs,
            text_field="facts" if "facts" in args.dataset else "text",
        )
        dm = FactsLMDataModule(config, for_generation=for_generation)
    elif "oa1" in args.dataset:
        config = OA1Config(
            **common_kwargs,
            train_on_reverse=args.dataset == "inverse-oa1",
        )
        dm = OA1Module(config, for_generation=for_generation)
    elif "cot:flan" in args.dataset:
        common_kwargs["dataset"] = common_kwargs["dataset"].replace("cot:", "")
        config = FlanConfig(
            **common_kwargs,
            include_template_type="*",
            include_task_source="CoT",
        )
        dm = FlanModule(config, for_generation=for_generation)
    elif "flan" in args.dataset:
        config = FlanConfig(
            **common_kwargs,
            include_template_type="*",
        )
        dm = FlanModule(config, for_generation=for_generation)
    elif "flat" in args.dataset:
        config = FlatMultiTaskConfig(
            **common_kwargs,
            source_template=args.source_template,
            augment_few_shot=args.augment_few_shot,
        )
        dm = FlatMultiTaskModule(config, for_generation=for_generation)
    elif "flat" in args.dataset:
        config = FlatMultiTaskConfig(
            **common_kwargs,
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
    model_class = ExpertTrainer
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

    module = model_class(**vars(args), tokenizer=dm.tokenizer).to("cuda")
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
        val_check_interval = args.gradient_accumulation_steps * args.eval_every
        if val_check_interval > len(dm.train_dataloader()):
            val_check_interval = len(dm.train_dataloader())
        elif val_check_interval > args.total_steps and args.total_steps != -1:
            val_check_interval = args.total_steps

    # callbacks.append(RougeCallback(gen_dm))
    callbacks.append(RougeCallbackTestPerEpoch(gen_dm, checkpoint_callback))

    callbacks.append(
        OptimResetCallback(reset_lr=args.reset_lr, reset_optim=args.reset_optim)
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
    trainer.test(module, dm)

    del module
    torch.cuda.empty_cache()

    # reload best model before pushing!
    checkpoint = (
        checkpoint_callback.best_model_path or checkpoint_callback.last_model_path
    )
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
    create_transfer_matrix(config, debug=False)
    ########################

    if args.hf_repo_id and checkpoint:
        from projects.wiki_experts.src.expert_model import push_expert_to_hub

        push_expert_to_hub(checkpoint, args.hf_repo_id, auto_search=False)


if __name__ == "__main__":
    args = ExpertConfig.parse()
    run_multitask(args)
