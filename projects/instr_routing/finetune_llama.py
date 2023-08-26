import os
import sys
import json
import torch
import wandb
import pytorch_lightning as pl

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from mttl.models.poly import get_selector
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from projects.instr_routing.models.clm import CLM
from mttl.callbacks import ProgressCallback
from mttl.datamodule.alpaca_data_module import AlpacaDataModule
from mttl.datamodule.platypus_module import PlatypusModule
from mttl.config import Config
from mttl.models.monitors import get_monitors
from mttl.utils import get_mlf_logger
from mttl.models.modify_model import modify_transformer
from transformers import AutoModelForCausalLM, LlamaForCausalLM
from config import RoutingConfig

from huggingface_hub import login

# register models
import projects.instr_routing.models.routing  # noqa: F401
import projects.instr_routing.models.aux_routing  # noqa: F401


def remove_non_serializable(d):
    """
    Recursively remove non-JSON serializable values from a dictionary.
    """
    for k, v in d.items():
        if isinstance(v, (list, dict)):
            remove_non_serializable(v)
        elif not json.dumps(v, default=lambda x: None):
            del d[k]


def run_multitask(args):
    seed_everything(args.seed, workers=True)
    # get directory of the current file
    print(os.path.dirname(os.path.realpath(__file__)))

    if args.hf_token_hub:
        login(token=args.hf_token_hub)

    if args.example_to_ids_path:
        raise NotImplementedError()

    # select dataloader
    model_class = CLM
    if args.dataset == "alpaca":
        dm = AlpacaDataModule(args)
    elif args.dataset == "platypus":
        dm = PlatypusModule(args)
    else:
        raise NotImplementedError()

    args.n_tasks = len(dm.task_to_id)
    module = model_class(**vars(args), tokenizer=dm.tokenizer)

    if args.switch_to_average > 0:
        module.model.switch_selector_to_average(
            selector_to_replace=get_selector(args).__class__
        )

    # legit logging
    loggers = []
    if os.environ.get("WANDB_API_KEY") or args.wandb_project:
        project = (
            "alpaca_tuning_ncb" if args.wandb_project is None else args.wandb_project
        )
        project = os.environ.get("WANDB_PROJECT", project)
        project += f"_{args.dataset}"
        wandb_logger = pl.loggers.WandbLogger(
            project=project,
            name=os.environ.get("AMLT_JOB_NAME", args.exp_name),  # , config=args_
        )
        wandb_logger.experiment.save("*.py")
        loggers.append(wandb_logger)
    else:
        wandb_logger = None

    mlf_logger = get_mlf_logger()
    if mlf_logger:
        loggers.append(mlf_logger)

    tb_logger = pl.loggers.TensorBoardLogger(save_dir=args.output_dir)
    loggers.append(tb_logger)
    loggers.append(pl.loggers.CSVLogger(save_dir=args.output_dir, name="csv_metrics"))

    kwargs = {"val_check_interval": args.eval_every} if args.eval_every else {}

    # get metric monitors for models
    callbacks = []
    callbacks.append(ProgressCallback())

    monitor = "val/loss"
    mode = "min"

    model_name = args.model.replace("/", "_")
    # check if wandb run exists
    if wandb_logger:
        # get run id
        run_id = wandb_logger.experiment.id
        model_name += run_id

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

    trainer = Trainer(
        devices=-1,
        accelerator="gpu",
        logger=loggers,
        num_sanity_val_steps=5,
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
        fast_dev_run=args.fast_dev_run,
        **kwargs,
    )

    trainer.fit(module, dm)

    path_best_model = trainer.checkpoint_callback.best_model_path
    path_last_model = trainer.checkpoint_callback.last_model_path

    ckpt_path = "best" if not args.fast_dev_run and path_best_model else "last"
    trainer.validate(dataloaders=dm, ckpt_path=ckpt_path)

    if args.use_test_set and not args.fast_dev_run and path_best_model:
        module.model.checkpoint_tested = "best"
        trainer.test(dataloaders=dm, ckpt_path=ckpt_path)

    module.model.checkpoint_tested = "last"
    trainer.test(dataloaders=dm, ckpt_path="last")

    print(f"Best model path: {path_best_model}")
    print(f"Last model path: {path_last_model}")

    ds_limit = args.eval_ds_limit if not args.fast_debug_run else 0.05
    torch.cuda.empty_cache()

    # load best model
    if path_best_model:
        del module
        best_model = CLM.load_from_checkpoint(path_best_model, tokenizer=dm.tokenizer)
    else:
        best_model = module

    # empty memory
    del (
        dm,
        trainer,
        loggers,
        callbacks,
        checkpoint_callback,
        wandb_logger,
        mlf_logger,
    )

    if args.eval_superni:
        print("#" * 50)
        print("Evaluating on super NI")
        from projects.instr_routing.eval.ni.eval_ni import eval_ni

        rouge_L_super_ni = eval_ni(
            args,
            best_model,
            nshot=2,
            data_dir=os.environ["NI_DATA_DIR"],
            eval_batches=args.eval_batches,
        )
        if wandb.run is not None:
            wandb.log({"rouge_L_super_ni": rouge_L_super_ni})
        tb_logger.experiment.add_scalar("tasks/sni", rouge_L_super_ni, trainer.global_step)
        print("SuperNI RougeL: {:.2f}".format(rouge_L_super_ni))

    if args.eval_mmlu:
        from projects.instr_routing.eval.mmlu.eval_mmlu import eval_mmlu

        print("#" * 50)
        print("Evaluating on MMLU")
        acc = eval_mmlu(
            args,
            best_model,
            data_dir=os.environ["MMLU_DATA_DIR"],
            eval_batches=args.eval_batches,
        )
        if wandb.run is not None:
            wandb.log({"mmlu_acc": acc})
        tb_logger.experiment.add_scalar("tasks/mmlu", acc, trainer.global_step)
        print("MMLU accuracy: {:.2f}".format(acc))


if __name__ == "__main__":
    args = RoutingConfig.parse()
    run_multitask(args)
