import os
import sys
import json
import torch
import wandb
import logging
import pytorch_lightning as pl
from huggingface_hub import login
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from mttl.callbacks import MMLUCallback
from mttl.datamodule.alpaca_data_module import AlpacaDataModule
from mttl.datamodule.platypus_module import PlatypusModule
from mttl.datamodule.flan100k_module import Flan100kModule
from mttl.utils import get_mlf_logger, setup_logging, logger
from mttl.models.monitors import SelectorMetricsLog, SelectorRoutingsLog
from mttl.dist_utils import is_main_process
from mttl.models.modifiers.routing import RoutingSelector

# register models
import models.vsmear  # noqa: F401
import models.softmoe  # noqa: F401
from models.monitors import SelectorMetricsLog, SelectorRoutingsLog
from models.clm import CLM
from config import RoutingConfig

torch.set_float32_matmul_precision("high")


def remove_non_serializable(d):
    """
    Recursively remove non-JSON serializable values from a dictionary.
    """
    for k, v in d.items():
        if isinstance(v, (list, dict)):
            remove_non_serializable(v)
        elif not json.dumps(v, default=lambda x: None):
            del d[k]


def eval_superni(best_model, args, tb_logger, trainer, n_shot=0, sufix="", subsample=-1):
    from eval_ni import eval_ni

    logger.info(f"Evaluating on super NI {sufix}")
    # all_results_original -- dict of results on sni eval obtained by running the original evaluate.py
    rougel_ni_all = eval_ni(
        args,
        best_model,
        nshot=n_shot,
        subsample=subsample,
        max_input_length=-1,
        data_dir=os.environ["NI_DATA_DIR"],
    )
    rougel_ni = rougel_ni_all["all"]["mean"]
    if wandb.run is not None:
        wandb.log({f"rouge_L_super_ni_nshot_{n_shot}sht{sufix}": rougel_ni})
        # per task
        data = [
            [label, val]
            for (label, val) in rougel_ni_all["per_task"].items()
            if "rougeL" in label
        ]
        table = wandb.Table(data=data, columns=["task_sni", f"mean_rougeL{n_shot}sht{sufix}"])
        wandb.log(
            {
                f"sni_per_task_rougeL_{n_shot}sht{sufix}": wandb.plot.bar(
                    table,
                    "task_sni",
                    f"mean_rougeL{n_shot}sht{sufix}",
                    title=f"sni_per_task_rougeL_{n_shot}sht{sufix}",
                )
            }
        )
        # per category
        data = [
            [label, val]
            for (label, val) in rougel_ni_all["per_category"].items()
            if "rougeL" in label
        ]
        table2 = wandb.Table(data=data, columns=["category_sni", f"mean_rougeL{n_shot}sht{sufix}"])
        wandb.log(
            {
                f"sni_per_category_rougeL_{n_shot}sht{sufix}": wandb.plot.bar(
                    table2,
                    "category_sni",
                    f"mean_rougeL{n_shot}sht{sufix}",
                    title=f"sni_per_category_rougeL_{n_shot}sht{sufix}",
                )
            }
        )

    if args.tensorboard:
        tb_logger.experiment.add_scalar(
            f"tasks/sni_{n_shot}sht{sufix}", rougel_ni, trainer.global_step
        )
    with open(
        os.path.join(args.output_dir, f"sni_results_{n_shot}sht{sufix}.json"), "w"
    ) as f:
        json.dump(rougel_ni_all, f, indent=2)
    logger.info("SuperNI RougeL_{}sht{}: {:.2f}".format(n_shot, sufix, rougel_ni))


def eval_downstream(best_model, args, tb_logger, trainer, sufix="", subsample=-1):
    if args.eval_mmlu:
        from eval_mmlu import eval_mmlu

        logger.info("Evaluating on MMLU")
        em_mmlu_all = eval_mmlu(
            args,
            best_model,
            subsample=subsample,
            data_dir=os.environ["MMLU_DATA_DIR"],
        )
        mmlu_em = em_mmlu_all["all"]["mean"]
        if wandb.run is not None:
            wandb.log({f"mmlu_acc{sufix}": mmlu_em})
            # report per task peformance
            data = [[label, val["mean"]] for (label, val) in em_mmlu_all.items()]
            table = wandb.Table(data=data, columns=["task", "mean_acc"])
            wandb.log(
                {
                    f"mmlu_per_task_acc_{sufix}": wandb.plot.bar(
                        table, "task", "mean_acc", title="mmlu_per_task_acc"
                    )
                }
            )

        if args.tensorboard:
            tb_logger.experiment.add_scalar(
                f"tasks/mmlu{sufix}", mmlu_em, trainer.global_step
            )
        with open(os.path.join(args.output_dir, "mmlu_results.json"), "w") as f:
            json.dump(em_mmlu_all, f, indent=2)
        logger.info("MMLU accuracy{}: {:.2f}".format(sufix, mmlu_em))

    if args.eval_superni:
        eval_superni(best_model, args, tb_logger, trainer, n_shot=0, sufix=sufix, subsample=subsample)
        eval_superni(best_model, args, tb_logger, trainer, n_shot=2, sufix=sufix, subsample=subsample)


def run_multitask(args):
    seed_everything(args.seed, workers=True)

    # get directory of the current file
    setup_logging(args.output_dir)

    logger.info("Args: {}".format(args.to_json()))

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
    elif args.dataset == "flan100k":
        dm = Flan100kModule(args)
    else:
        raise NotImplementedError()

    args.n_tasks = len(dm.task_to_id)
    module = model_class(**vars(args), tokenizer=dm.tokenizer)

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

    if args.tensorboard:
        tb_logger = pl.loggers.TensorBoardLogger(save_dir=args.output_dir)
        loggers.append(tb_logger)

    loggers.append(pl.loggers.CSVLogger(save_dir=args.output_dir, name="csv_metrics"))

    kwargs = {"val_check_interval": args.eval_every} if args.eval_every else {}

    # get metric monitors for models
    callbacks = []

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
    callbacks.append(SelectorRoutingsLog(args))
    callbacks.append(SelectorMetricsLog())
    if args.mmlu_callback:
        callbacks.append(MMLUCallback(5))

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
    ckpt_path = "best" if path_best_model else "last"

    if args.load_in_8bit:
        # To prevent final spike in valid loss, we first load the model and path is to the evaluator
        # There is a bug with pl for 8bit model when calling .cuda() on it, to("cuda") works however.
        best_model = CLM.load_from_checkpoint(
            path_best_model, tokenizer=dm.tokenizer
        ).to("cuda")
        trainer.validate(dataloaders=dm, model=best_model)
    else:
        trainer.validate(dataloaders=dm, ckpt_path=ckpt_path)
    
    if is_main_process():
        if path_best_model:
            del module
            torch.cuda.empty_cache()

            best_model = CLM.load_from_checkpoint(
                path_best_model,
                tokenizer=dm.tokenizer,
                load_in_8bit=False,
            ).to("cuda")
        else:
            torch.cuda.empty_cache()
            best_model = module.to("cuda")

        tb_logger = None if not args.tensorboard else tb_logger
        eval_downstream(best_model, args, tb_logger, trainer, sufix="")
        # eval downstream with averaged model
        success = best_model.model.switch_selector_to_average(selector_to_replace=RoutingSelector, config=args) if hasattr(best_model.model, "switch_selector_to_average") else False
        if success and args.eval_avg:
            eval_downstream(best_model, args, tb_logger, trainer, sufix="_avg_exps")


if __name__ == "__main__":
    args = RoutingConfig.parse()
    run_multitask(args)
