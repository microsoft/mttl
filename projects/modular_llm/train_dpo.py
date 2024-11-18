import copy
import os
import shutil
import sys
from tempfile import TemporaryDirectory

import torch
from pytorch_lightning import Trainer, seed_everything


from mttl.models.lightning.callbacks import LiveCheckpointCallback
from mttl.datamodule.base import DatasetConfig
from mttl.datamodule.preference_data_module import Preferencemodule
from mttl.datamodule.ultrafeedback_data_module import UltrafeedbackDPOmodule

# from mttl.datamodule.base import get_datamodule
from mttl.arguments import ExpertConfig, MultiExpertConfig
from mttl.models.expert_model import ExpertModelConfig
from mttl.models.expert_model import (
    ExpertModel,
    ExpertModelDPO,
    MoEModel,
    ExpertModelSimPO,
)
from mttl.models.library.expert import Expert, load_expert
from mttl.models.library.expert_library import ExpertLibrary, LocalExpertLibrary
from mttl.models.monitors import get_monitors
from mttl.models.lightning.loggers import get_pl_loggers
from mttl.logging import logger, setup_logging
from mttl.utils import generate_random_string, rank_zero_only_and_wait, remote_login


def run_multitask(args: ExpertConfig):
    seed_everything(args.seed, workers=True)

    # get directory of the current file
    setup_logging(args.output_dir)
    logger.info("Args: {}".format(args.to_json()))

    remote_login(args.remote_token)
    expert_library = None
    if args.library_id:

        @rank_zero_only_and_wait(before=False, after=True)
        def create_library(args):
            expert_library = ExpertLibrary.get_expert_library(
                repo_id=args.library_id,
                create=True,
                destination_id=args.destination_library_id,
            )
            return expert_library

        expert_library = create_library(args)

    loggers = get_pl_loggers(args)
    # select dataloader
    if args.model_modifier == "poly":
        model_class = MoEModel
    else:
        model_class = ExpertModel
    config = DatasetConfig(
        model=args.model,
        train_batch_size=args.train_batch_size,
        predict_batch_size=args.predict_batch_size,
        max_input_length=args.max_input_length,
        max_output_length=args.max_output_length,
    )

    if "ultrafeedback" in args.dataset:
        dm = UltrafeedbackDPOmodule(config)
    else:
        dm = Preferencemodule(config)

    model_config = ExpertModelConfig(
        base_model=args.model,
        modifier_config=args.modifier_config,
    )

    model = model_class(
        config=model_config,
        expert_library=expert_library,
        **vars(args),
    )
    if args.rl_training == "dpo":
        # args.trainable_param_names = "^(?=.*preference_model)(?=.*prototypes).*"
        ref_model = model_class(
            config=model_config,
            expert_library=expert_library,
            **vars(args),
        )
        # eval mode
        ref_model.eval()
        module = ExpertModelDPO(
            **vars(args), preference_model=model, ref_expert_model=ref_model
        )
    elif args.rl_training == "simpo":
        # args.trainable_param_names = "^(?=.*preference_model)(?=.*prototypes).*"
        module = ExpertModelSimPO(**vars(args), preference_model=model)
    else:
        module = model
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
        save_each_epoch=args.save_each_epoch,
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
        precision=(
            int(args.precision) if args.precision in ["16", "32"] else args.precision
        ),
        val_check_interval=val_check_interval,
    )

    # initial validation only for a bunch of datasets... ?
    if args.eval_before_training:
        # validating before training fails with deepspeed
        trainer.validate(module, dm)

    if args.do_train:
        trainer.fit(module, dm)

        torch.cuda.empty_cache()

        # reload best model before pushing!
        checkpoint = (
            checkpoint_callback.best_model_path or checkpoint_callback.last_model_path
        )
        if args.compute_strategy == "deepspeed":
            from deepspeed.utils.zero_to_fp32 import (
                convert_zero_checkpoint_to_fp32_state_dict,
            )

            new_path = checkpoint.replace(".ckpt", "_fp32.ckpt")

            @rank_zero_only_and_wait(before=True, after=True)
            def convert_ckpt(path, new_path):
                convert_zero_checkpoint_to_fp32_state_dict(path, new_path)

            convert_ckpt(checkpoint, new_path)
            checkpoint = torch.load(new_path)
        else:
            checkpoint = torch.load(checkpoint)["state_dict"]

        module.load_state_dict(checkpoint)
        trainer.test(module, dm)

        @rank_zero_only_and_wait(before=False, after=True)
        def upload_library(expert_library, module):
            if expert_library is not None:
                # refresh expert library: so we dont overwrite the readme if the remote has changed.
                expert_library.refresh_from_remote()

                if isinstance(module, MoEModel):
                    with expert_library.batched_commit():
                        for expert_name in module.experts_names:
                            expert = module.get_expert_instance(expert_name)
                            expert_library.add_expert(expert, expert_name)
                elif isinstance(module, ExpertModel):
                    expert = module.as_expert()
                    expert_name = (
                        args.expert_name
                        or args.finetune_task_name
                        or generate_random_string()
                    )
                    expert_library.add_expert(expert, expert_name)
                else:
                    raise ValueError("Model class not recognized")

        # upload_library(expert_library, module)


if __name__ == "__main__":
    args = MultiExpertConfig.parse()  ## in case we only train the routing
    run_multitask(args)
