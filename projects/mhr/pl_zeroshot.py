import copy
import os

import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything

from mttl.callbacks import ProgressCallback
from mttl.datamodule.ni_original_data_module import NIOriginalDataModule
from mttl.datamodule.t0_data_module import T0FinetuneDataModule
from mttl.models.encoder_decoder import Finetuner
from mttl.models.t0_encoder_decoder import T0EncoderDecoder
from mttl.utils import (
    get_checkpoint_path,
    get_mlf_logger,
    setup_logging,
    get_training_strategy,
    get_mhr_hf_checkpoint,
)
from mttl.config import Config
from pl_finetune import ARGS_TO_OVERWRITE

torch.set_float32_matmul_precision("high")

def zeroshot(args, use_mlf=True):
    seed_everything(args.seed, workers=True)

    # build the pretrained model
    if args.checkpoint or args.load_from_hf:
        if args.checkpoint:
            ckpt_path = get_checkpoint_path(
                args.checkpoint, use_last=args.finetune_use_last_checkpoint
            )
        else:
            ckpt_path = get_mhr_hf_checkpoint(args)

        ckpt = torch.load(ckpt_path)
        ckpt_args = ckpt["hyper_parameters"]
        ckpt_dict = ckpt["state_dict"]
        args.old_exp_name = ckpt_args["exp_name"]

        for arg_name in ARGS_TO_OVERWRITE:
            # we over-write with a new one only if the argument was a default one
            if arg_name in ckpt_args and not args.was_overridden(arg_name):
                print("Overwriting", arg_name, "=", ckpt_args[arg_name])
                setattr(args, arg_name, ckpt_args[arg_name])
    else:
        ckpt_path = None
        ckpt_args = args

    if args.dataset in ["ni"]:
        finetuner_cls = Finetuner
    else:
        finetuner_cls = T0EncoderDecoder

    # data
    if args.dataset == "ni":
        dm = NIOriginalDataModule(args)
    elif args.dataset == "t0":
        dm = T0FinetuneDataModule(args)

    kwargs = copy.deepcopy(vars(args))
    kwargs.pop("checkpoint")
    # economic checkpointing for finetuning, we don't need to save the full backbone, only parameters that we are training.
    kwargs["save_if_loaded"] = False
    module = finetuner_cls(**kwargs, tokenizer=dm.tokenizer)

    if ckpt_path is None:
        print("Skipping loading from checkpoint...")
    else:
        # breakpoint()
        module.load_state_dict(ckpt_dict, strict=False)

    # allocate new module logits for the new task
    if args.model_modifier and "poly" in args.model_modifier:
        module.model.resize_module_logits(1)
        module.model.switch_selector_to_average(**{"config": args})

    def setup_and_test():
        callbacks = [ProgressCallback()]

        # legit logging
        loggers = []
        if os.environ.get("WANDB_API_KEY"):
            wandb_logger = pl.loggers.WandbLogger(
                project=args.wandb_project,
                name=args.exp_name,
            )
            wandb_logger.experiment.save("*.py")
            loggers.append(wandb_logger)
        else:
            wandb_logger = None

        mlf_logger = get_mlf_logger()
        if mlf_logger and use_mlf:
            loggers.append(mlf_logger)

        loggers.append(
            pl.loggers.CSVLogger(save_dir=args.output_dir, name="csv_metrics")
        )

        # TODO: is there a way to avoid creating a trainer given that we don't train ?
        trainer = Trainer(
            enable_checkpointing=False,
            devices=1,
            default_root_dir=args.output_dir,
            accelerator="gpu",
            logger=loggers,
            num_sanity_val_steps=0,
            max_steps=args.total_steps,
            max_epochs=args.num_train_epochs,
            gradient_clip_val=args.max_grad_norm,
            log_every_n_steps=10,
            strategy=get_training_strategy(args.compute_strategy),
            limit_val_batches=1.0,
            limit_train_batches=1.0,
            # limit_test_batches=5,
            precision=args.precision,
            callbacks=callbacks,
            accumulate_grad_batches=args.gradient_accumulation_steps,
        )

        trainer.test(module, dm, ckpt_path=None)

        results = module.test_results
        return results

    results = setup_and_test()

    if args.dataset == "ni":
        # remove all eventual checkpoints
        os.system(f'find /tmp/sni/ -name "*.ckpt" -type f -delete')
        os.system(f'find /tmp/sni/ -name "*.pt" -type f -delete')

    return results


def zeroshot_ni(args, seeds=[13], use_mlf=True):
    all_results = []

    for seed in seeds:
        args.seed = seed

        # use mlf logger only for the first seed, otw it will complain for duplicated hps
        results = zeroshot(
            args,
            use_mlf=(seed == seeds[0] and use_mlf),
        )
        all_results.extend(results)

    for result in all_results:
        result["prefix"] = args.zeroshot_task_name

    # whatever
    print(all_results)

    df = pd.DataFrame.from_dict(all_results)
    df.to_csv(os.path.join(args.output_dir, "result.csv"))

    return df


def zeroshot_t0(args, seeds=[42], use_mlf=True):
    all_results = []

    for i, seed in enumerate(seeds):
        args.seed = seed

        # use mlf logger only for the first seed, otw it will complain for duplicated hps
        results = zeroshot(
            args,
            use_mlf=use_mlf and i == 0,
        )
        all_results.extend(results)

    for result in all_results:
        result["prefix"] = args.finetune_task_name

    # whatever
    print(all_results)

    df = pd.DataFrame.from_dict(all_results)
    df.to_csv(os.path.join(args.output_dir, "result.csv"))
    return df


if __name__ == "__main__":
    args = Config.parse()
    setup_logging(args.output_dir)

    print("arguments")
    print(vars(args))

    if args.dataset == "ni":
        zeroshot_ni(args)
    elif args.dataset == "t0":
        zeroshot_t0(args)
    else:
        raise ValueError("Dataset not recognized.")
