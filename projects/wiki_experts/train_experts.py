import os
import sys
import json
import wandb
import pytorch_lightning as pl

import torch
from huggingface_hub import login
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
import json

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from mttl.callbacks import MMLUCallback
from mttl.evaluators import MMLUEvaluator
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# from mttl.datamodule.oasst1_module import OA1Config, OA1Module
from mttl.datamodule.retrieval_lm_module import RetrievalLMDataModule
from mttl.datamodule.facts_lm_module import FactsLMConfig, FactsLMDataModule
from mttl.datamodule.platypus_module import (
    PlatypusModule,
    PlatypusConfig,
    PlatypusQAModule,
)
from mttl.datamodule.flan10k_module import Flan10kModule, Flan10kConfig
from mttl.utils import get_mlf_logger, setup_logging, logger

from projects.wiki_experts.src.expert_trainer import ExpertTrainer
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


def eval_mmlu(module, args, base_perf=None, chkpt_criteria=None):
    mmlu = MMLUEvaluator(
        args,
        split=args.mmlu_test_split,
    )
    scores = mmlu.evaluate(module)
    print(f"Evaluating final checkpoint with selection criteria {chkpt_criteria}")
    logger.info("Final MMLU Accuracy: {}".format(scores["all"]["mean"]))
    for t, v in scores.items():
        logger.info("MMLU Accuracy {}: {}".format(t, v["mean"]))
    # super hard to log with pllogger here
    improvement = None
    if base_perf is not None:
        improvement = {
            m: scores[m]["mean"] - base_perf[m]["mean"]
            for m in scores
            if m in base_perf
        }
    if wandb.run is not None:
        for t, v in scores.items():
            wandb.log(
                {f"downstream_estoped/crit_{chkpt_criteria}/test_mmlu_" + t: v["mean"]}
            )
        if improvement is not None:
            for t, v in improvement.items():
                wandb.log(
                    {
                        f"downstream_estoped/crit_{chkpt_criteria}/test_mmlu_improvement_"
                        + t: improvement[t]
                    }
                )
    return scores


def run_multitask(args):
    seed_everything(args.seed, workers=True)
    # get directory of the current file
    setup_logging(args.output_dir)

    logger.info("Args: {}".format(args.to_json()))

    if args.hf_token_hub:
        login(token=args.hf_token_hub)

    # select dataloader
    model_class = ExpertTrainer
    # add MMLU val data to validaiton set
    val_mixin = None
    if args.expand_val_set_w_downstream:
        from mttl.datamodule.mmlu_data_module import MMLUDataModule

        val_mixin = MMLUDataModule(args).dev_dataset

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
        dm = PlatypusQAModule(config, val_mixin=val_mixin)
    elif args.dataset.startswith("raw_docs:"):
        args.dataset = args.dataset.replace("raw_docs:", "")
        config = FactsLMConfig(
            model=args.model,
            train_batch_size=args.train_batch_size,
            predict_batch_size=args.predict_batch_size,
            max_input_length=args.max_input_length,
            validation_portion=args.validation_portion,
            finetune_task_name=args.finetune_task_name,
            dataset=args.dataset,
            text_field="text",
        )
        dm = FactsLMDataModule(config)
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
    elif "flan10k" in args.dataset:
        config = Flan10kConfig(
            model=args.model,
            train_batch_size=args.train_batch_size,
            predict_batch_size=args.predict_batch_size,
            max_input_length=args.max_input_length,
            max_output_length=args.max_output_length,
            validation_portion=args.validation_portion,
            model_family=args.model_family,
            train_on_inputs=False,
            category=args.category,
        )
        dm = Flan10kModule(config)
        dm.setup()
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

    module = model_class(**vars(args), tokenizer=dm.tokenizer)
    module.to("cuda")

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
    if args.eval_mmlu_flag is True:
        mmlu_test_cb = MMLUCallback(
            args.eval_every, split="test", checkpoint_oracle=True
        )
        mmmlu_val_cb = MMLUCallback(
            args.eval_every, split="val", checkpoint_oracle=True
        )
        callbacks += [mmlu_test_cb, mmmlu_val_cb]

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

    # perform final evals on MMLU
    # for oracle
    del module
    torch.cuda.empty_cache()

    if args.eval_mmlu_flag is True:
        module_test_oracle = model_class.load_from_checkpoint(
            mmlu_test_cb.last_chkpt
        ).to("cuda")
        eval_mmlu(
            module_test_oracle,
            args,
            mmlu_test_cb.base_perf,
            chkpt_crit="test_mmlu_oracle",
        )
        del module_test_oracle
        torch.cuda.empty_cache()

        # for best model selected with mmlu/val
        module_valid_oracle = model_class.load_from_checkpoint(
            mmmlu_val_cb.last_chkpt
        ).to("cuda")
        eval_mmlu(
            module_valid_oracle, args, mmlu_test_cb.base_perf, chkpt_crit="val_mmlu"
        )
        del module_valid_oracle
        torch.cuda.empty_cache()
    # reload best model before pushing!
    checkpoint = (
        checkpoint_callback.best_model_path or checkpoint_callback.last_model_path
    )

    if checkpoint:
        module = model_class.load_from_checkpoint(checkpoint).to("cuda")
        if args.eval_mmlu_flag is True:
            eval_mmlu(module, args, mmlu_test_cb.base_perf, chkpt_crit="val_mmlu")

    if args.hf_repo_id and checkpoint:
        from projects.wiki_experts.src.expert_model import push_expert_to_hub

        push_expert_to_hub(checkpoint, args.hf_repo_id, auto_search=False)


if __name__ == "__main__":
    args = ExpertConfig.parse()
    run_multitask(args)
