import os
import re

import sys
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from mttl.models.modifiers.expert_containers.expert import Expert, load_expert
from mttl.models.expert_model import ExpertModel as ExpertTrainer
from mttl.models.expert_config import ExpertConfig

from config import EvolExpertConfig
from typing import List


def save_new_module(module_copy, args):
    # make Loras trainable
    module_copy.trainable_param_names = [
        n for n, p in module_copy.named_parameters() if re.match(".*lora.*", n)
    ]
    checkpoint = module_copy.save_pretrained(args.output_dir)
    return checkpoint


def train_module(
    args: EvolExpertConfig,
    dm,
    dm_eval,
    module: ExpertTrainer,
    loggers: List = [],
    val_check_interval=None,
    logging_prefix="",
    silent=False,
):
    seed_everything(args.seed, workers=True)

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
        val_check_interval = None
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
            save_weights_only=True,  # make checkpoints smaller
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
    logger.disabled = False

    checkpoint = (
        checkpoint_callback.best_model_path or checkpoint_callback.last_model_path
    )
    logger.info(f"Loading best model from {checkpoint}")
    torch.cuda.empty_cache()
    ckpt = torch.load(checkpoint)
    if "expert_dumps" in ckpt:
        expert_dumps = ckpt["expert_dumps"]
        weights = ckpt["merging_weights"]
        expert = Expert.fromdict(expert_dumps)
    else:
        expert = load_expert(checkpoint)
        weights = []

    del module
    if hasattr(checkpoint_callback, "remove_checkpoints"):
        # cleanup
        checkpoint_callback.remove_checkpoints()
    try:
        os.remove(checkpoint)
    except:
        pass

    return weights, expert


if __name__ == "__main__":
    from tempfile import TemporaryDirectory
    from mttl.models.modifiers.expert_containers.expert import Expert, load_expert
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
