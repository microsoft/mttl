import inspect
import os
import shutil
import sys
from tempfile import TemporaryDirectory

import torch
from pytorch_lightning import Trainer, seed_everything

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from mttl.arguments import ExpertConfig
from mttl.datamodule.base import get_datamodule
from mttl.logging import logger, setup_logging
from mttl.models.expert_model import ExpertModel, MoEModel
from mttl.models.library.expert import Expert, load_expert
from mttl.models.library.expert_library import ExpertLibrary, LocalExpertLibrary
from mttl.models.lightning.callbacks import (
    DownstreamEvalCallback,
    LiveCheckpointCallback,
    NanoMMLUCallback,
    RougeCallback,
)
from mttl.models.lightning.expert_module import ExpertModule
from mttl.models.lightning.loggers import get_pl_loggers
from mttl.models.monitors import get_monitors
from mttl.utils import remote_login
from projects.modular_llm.compute_transfer_matrix import TransferMatrixConfig
from projects.modular_llm.compute_transfer_matrix import (
    run_eval as produce_transfer_matrix,
)
from mttl.models.lightning.callbacks import UpdateSparseMask


@torch.no_grad()
def add_sparse_model_to_library(checkpoint, expert_name, module, expert_library):
    from mttl.models.library.expert import load_expert

    expert_dump = load_expert(checkpoint, expert_name=expert_name)
    if expert_dump.name is None:
        raise ValueError(
            "Expert name not found in checkpoint. Need to explicitly provide one as argument."
        )
    expert_dump.expert_weights_shape = {}
    for m_name, m in dict(module.named_modules()).items():
        if "sparse_layer" in m_name:
            layer_name = ".".join(m_name.split(".")[1:])
            assert f"{layer_name}.weight" in expert_dump.expert_weights
            expert_dump.expert_weights[f"{layer_name}.weight"] = m.weight[
                m.weight_mask != 0
            ].data
            expert_dump.expert_weights_shape[f"{layer_name}.weight"] = m.weight.shape
    expert_library.add_expert(expert_dump, force=True)


def create_transfer_matrix(args, checkpoint):
    ########################
    # create transfer matrix
    config = TransferMatrixConfig()
    for k, v in vars(args).items():
        if k in vars(config):
            setattr(config, k, v)
    config.eval_base = False
    config.eval_metric = "rougeL"

    expert: Expert = load_expert(checkpoint)
    expert.expert_info.expert_name = str(args.finetune_task_name)
    expert.expert_info.expert_task_name = str(args.finetune_task_name)
    temp_dir = TemporaryDirectory()
    destination = temp_dir.name
    LocalExpertLibrary.from_expert_dict({"checkpoint": expert}, destination=destination)
    config.library_id = destination
    config.finetune_task_name = (
        args.finetune_task_name.split(",")
        if not isinstance(args.finetune_task_name, list)
        else args.finetune_task_name
    )
    if len(config.finetune_task_name) < 50:
        produce_transfer_matrix(config, debug=False)
    ########################
    temp_dir.cleanup()


def run_multitask(args: ExpertConfig):
    seed_everything(args.seed, workers=True)

    # get directory of the current file
    args.num_train_epochs = int(args.num_train_epochs)  # make sure to convert to int
    setup_logging(args.output_dir)
    logger.info("Args: {}".format(args.to_json()))

    remote_login(args.remote_token)
    expert_library = None
    if args.library_id:
        expert_library = ExpertLibrary.get_expert_library(
            repo_id=args.library_id,
            create=True,
            destination_id=args.destination_library_id,
        )

    loggers = get_pl_loggers(args)

    dm = get_datamodule(args)
    args.n_tasks = len(dm._task_names)
    args.task_names = dm._task_names

    model_class = ExpertModule
    module = model_class(**vars(args))

    # get metric monitors for models
    callbacks = get_monitors(args)
    if "mbpp" in args.dataset:
        monitor = "downstream/mbpp"
        mode = "max"
    else:
        monitor = "val/loss"
        mode = "min"

    # ----------------------------------
    # Iterative masking using Callback
    # ----------------------------------
    # NOTE: Don't move this block, it's important we call maskCallBack before others
    if hasattr(args, "use_sparse_model"):
        if args.use_sparse_model:
            assert len(args.task_names) == 1, print(
                "sparse mask does not support more than 1 task"
            )
            maskCallback = UpdateSparseMask(
                update_interval=100,
                num_train_steps=len(dm.train_dataloader()),
                save_mask_dir=args.library_id,
                task_name=args.task_names[0],
                sparse_training_type="iterative",  # default: 'iterative', options: ['iterative', 'one_shot']
                parameter_selection_procedure=args.parameter_selection_procedure,
            )  # use "max_connection_sensitivity" for default
            callbacks.append(maskCallback)

    checkpoint_callback = LiveCheckpointCallback(
        dirpath=args.output_dir,
        monitor=monitor,
        save_last=True,
        mode=mode,
        save_each_epoch=args.save_each_epoch,
    )
    callbacks.append(checkpoint_callback)

    if args.eval_rouge_flag:
        rouge = RougeCallback(
            get_datamodule(args, for_generation=True),
            every_n_epochs=3 if args.num_train_epochs > 5 else 1,
        )
        callbacks.append(rouge)
    else:
        logger.warning(
            "Deactivating rouge callback as it is not enabled in the config. Please set `eval_rouge_flag=True`."
        )

    if args.eval_mmlu_flag:
        mmlu = NanoMMLUCallback(
            get_datamodule(args, dataset_override="mmlu", for_generation=True),
            every_n_epochs=3 if args.num_train_epochs > 3 else 1,
        )
        callbacks.append(mmlu)
    else:
        logger.warning(
            "Deactivating mmlu callback as it is not enabled in the config. Please set `eval_mmlu_flag=True`."
        )

    if args.pipeline_eval_tasks:
        if args.pipeline_eval_tasks == "all":
            args.pipeline_eval_tasks = "arc-challenge,arc-easy,boolq,hellaswag,humaneval,mbpp,openbookqa,piqa,bbh-fast,winogrande"

        eval = DownstreamEvalCallback(args)
        callbacks.append(eval)
    else:
        logger.warning(
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
        log_every_n_steps=args.gradient_accumulation_steps,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        precision=(
            int(args.precision) if args.precision in ["16", "32"] else args.precision
        ),
        val_check_interval=val_check_interval,
    )

    # initial validation only for a bunch of datasets... ?
    if args.compute_strategy != "deepspeed":
        # validating before training fails with deepspeed
        trainer.validate(module, dm)

    if args.do_train:
        trainer.fit(module, dm)

        torch.cuda.empty_cache()

        # reload best model before pushing!
        checkpoint = (
            checkpoint_callback.best_model_path or checkpoint_callback.last_model_path
        )

        module.load_state_dict(torch.load(checkpoint, weights_only=False)["state_dict"])
        trainer.test(module, dm)

        if expert_library is not None:
            # refresh expert library: so we dont overwrite the readme if the remote has changed.
            expert_library.refresh_from_remote()

            if isinstance(module, ExpertModule):
                expert_name = args.expert_name or args.finetune_task_name
                if args.use_sparse_model:
                    add_sparse_model_to_library(
                        checkpoint, expert_name, module, expert_library
                    )
                else:
                    expert_library.add_expert_from_ckpt(
                        checkpoint, expert_name, force=True
                    )
            else:
                raise ValueError("Model class not recognized")

        if args.create_transfer_matrix:
            create_transfer_matrix(args, checkpoint)


if __name__ == "__main__":
    args = ExpertConfig.parse()
    run_multitask(args)
