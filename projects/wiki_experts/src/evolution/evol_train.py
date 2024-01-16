import os
import re

import sys
import json
import torch
from tempfile import TemporaryDirectory
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from dataclasses import dataclass
from mttl.models.modifiers.expert_containers.module_graph import Expert, load_expert
from mttl.utils import get_mlf_logger, setup_logging, logger
from projects.wiki_experts.src.config import ExpertConfig
from projects.wiki_experts.src.evolution.config import EvolExpertConfig
from typing import List
from projects.wiki_experts.src.expert_trainer import ExpertTrainer


def save_new_module(module_copy, args):
    # make Loras trainable
    module_copy.trainable_param_names = [
        n for n, p in module_copy.named_parameters() if re.match(".*lora.*", n)
    ]
    checkpoint = module_copy.save_pretrained(args.output_dir)
    return checkpoint


@dataclass
class EvolModuleOut:
    expert: Expert
    expert_last: Expert = None
    log_row: dict = None


def load_expert_from_checkpoint(checkpoint):
    ckpt = torch.load(checkpoint)
    if "expert_dumps" in ckpt:
        expert_dumps = ckpt["expert_dumps"]
        weights = None
        expert: Expert = Expert.fromdict(expert_dumps)
    else:
        expert: Expert = load_expert(checkpoint)
        weights = []

    # make sure expert also keeps the state of optimizer and scheduler, but at last checkpoint
    expert.expert_optimizer_state = {
        "optimizer_states": ckpt.get("optimizer_states", None),
        "lr_schedulers": ckpt.get("lr_schedulers", None),
    }

    return expert, weights


def train_module(
    args: EvolExpertConfig,
    dm,
    dm_eval,
    module_to_train: ExpertTrainer,
    loggers: List = [],
    val_check_interval=None,
    logging_prefix="",
    save_last_expert=False,
    debug=False,
    **kwargs,
) -> EvolModuleOut:
    silent = not debug
    """
    Performs one epoch of training.
    """
    module = module_to_train
    seed_everything(args.seed, workers=True)

    mlf_logger = get_mlf_logger()
    if mlf_logger:
        loggers.append(mlf_logger)

    if args.tensorboard:
        tb_logger = pl.loggers.TensorBoardLogger(save_dir=args.output_dir)
        loggers.append(tb_logger)

    if silent:
        logger.disabled = True
    # get metric monitors for models
    callbacks = []
    if "rouge" in args.eval_metric:  # early stop on Rouge
        from projects.wiki_experts.src.callbacks import RougeLCallback

        checkpoint_callback = RougeLCallback(
            datamodule=dm_eval,
            output_dir=args.output_dir,
            eval_every_opt_step=val_check_interval,
            name=f"{logging_prefix}rougeL_val_es",
            checkpoint_oracle=True,
        )
        val_check_interval = 0.2 if not debug else 1.0
    else:
        monitor = f"{logging_prefix}val/loss"
        mode = "min"
        model_name = args.model.replace("/", "_")

        checkpoint_callback = ModelCheckpoint(
            dirpath=args.output_dir,
            monitor=monitor,
            filename=f"{model_name}" + "-{" + monitor + ":.004f}",
            save_top_k=1,
            save_last=True,
            save_weights_only=False,  # save also optimizer etc.
            mode=mode,
        )
        val_check_interval = val_check_interval * args.gradient_accumulation_steps

    callbacks.append(checkpoint_callback)

    trainer = Trainer(
        devices=-1,
        accelerator="gpu",
        logger=loggers,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        enable_checkpointing=False,
        default_root_dir=args.output_dir,
        max_epochs=1,
        max_steps=-1,
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
    logger.disabled = False

    checkpoint = (
        checkpoint_callback.best_model_path or checkpoint_callback.last_model_path
    )
    logger.info(f"Loading best model from {checkpoint}")
    torch.cuda.empty_cache()
    expert, weights = load_expert_from_checkpoint(checkpoint)

    expert_last = None
    if save_last_expert:
        temp_dir = TemporaryDirectory()
        last_model_path = temp_dir.name + "/last.ckpt"
        trainer.save_checkpoint(last_model_path)
        expert_last, _ = load_expert_from_checkpoint(last_model_path)
        os.remove(last_model_path)

    del module
    if hasattr(checkpoint_callback, "remove_checkpoints"):
        # cleanup
        checkpoint_callback.remove_checkpoints()
    try:
        os.remove(checkpoint)
    except:
        pass

    return EvolModuleOut(
        expert=expert, expert_last=expert_last, log_row={"weights": weights}
    )


if __name__ == "__main__":
    from mttl.models.modifiers.expert_containers.module_graph import Expert, load_expert
    from mttl.datamodule.base import AutoDataModule

    def create_dummy_expert(config: ExpertConfig, exp_name) -> Expert:
        # create random Lora
        exp_trainer = ExpertTrainer(
            tokenizer=None,
            expert_info={},
            **vars(config),
        )
        dir = f"{config.output_dir}/{exp_name}"
        os.makedirs(dir, exist_ok=True)
        checkpoint = exp_trainer.save_pretrained(dir)
        expert: Expert = load_expert(checkpoint)
        return expert

    tmp_path = TemporaryDirectory().name

    class SimpleConfig(ExpertConfig):
        def _set_defaults(self):
            super()._set_defaults()
            self.model = "EleutherAI/gpt-neo-125m"
            self.model_family = "gpt"
            self.max_input_length = 1024
            self.max_output_length = 4
            self.train_batch_size = 1
            self.predict_batch_size = 1
            self.model_modifier = "lora"
            self.modify_layers = "q_proj|v_proj|k_proj"
            self.modify_modules = ".*"
            self.trainable_param_names = ".*lora_[ab].*"
            self.output_dir = tmp_path
            self.router_selector = "poly_router"
            self.router_granularity = "coarsegrained"
            self.dataset = "sordonia/flan-debug-flat"
            self.action = "route"
            self.finetune_task_name = "ai2_arc_ARC_Challenge_1_0_0"

    args = SimpleConfig()
    # add MMLU val data to validaiton set
    dm = AutoDataModule.create(
        name=args.dataset,
        for_generation=False,
        model=args.model,
        model_family=args.model_family,
        validation_portion=0.0,
        finetune_task_name=args.finetune_task_name,
        train_batch_size=args.train_batch_size,
        predict_batch_size=args.predict_batch_size,
    )
    args.n_tasks = len(dm.task_to_id) if hasattr(dm, "task_to_id") else 0
    args.num_train_epochs = 1

    exp_1: Expert = create_dummy_expert(args, "exp1")
    exp_2: Expert = create_dummy_expert(args, "exp2")
    exp_3: Expert = create_dummy_expert(args, "exp3")

    module_dict = {
        "ai2_arc_ARC_Challenge_1_0_0": exp_1,
        "best_performing_task": exp_2,
        "default": exp_3,
    }

    weights, checkpoint = train_module(args, dm=dm, module_dict=module_dict)
    print(weights, checkpoint)
