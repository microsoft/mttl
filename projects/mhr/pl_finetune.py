import copy
import os

import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything

from mttl.callbacks import ProgressCallback
from mttl.datamodule.t0_data_module import T0FinetuneDataModule
from mttl.models.encoder_decoder import Finetuner
from mttl.models.monitors import get_monitors
from mttl.models.t0_encoder_decoder import T0EncoderDecoder
from mttl.utils import (
    CustomModelCheckpoint,
    get_checkpoint_path,
    get_mlf_logger,
    setup_logging,
)
from mttl.config import Config

# When loading a checkpoint for evaluation, which args from old checkpoint
# should overwrite the incoming arguments ?
ARGS_TO_OVERWRITE = [
    "dataset",
    "config",
    "finegrained",
    "lora_rank",
    "max_input_length",
    "max_output_length",
    "model",
    "n_skills",
    "n_splits",
    "n_tasks",
    "use_t0_templates_as_tasks",
    "module_logits_relaxed_bernoulli",
    "module_logits_straight_through",
    "poly_use_shared_skill",
    "poly_average_correction",
    "router_selector",
    "router_granularity",
    "lora_rank",
    "modify_modules",
    "modify_layers",
    "model_modifier",
    "use_task_descriptions",
    "trainable_param_names",
    "use_precomputed_task_embeddings",
    "num_pos_examples",
    "example_to_ids_path",
]


def finetune(args, use_mlf=True, do_zs=True):
    seed_everything(args.seed, workers=True)

    # build the pretrained model
    if args.checkpoint:
        ckpt_path = get_checkpoint_path(
            args.checkpoint, use_last=args.finetune_use_last_checkpoint
        )

        if ckpt_path.startswith("az://"):
            import fsspec

            with fsspec.open(ckpt_path, "rb") as f:
                ckpt = torch.load(f)
        else:
            # local checkpoint
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

    switch_to_avg_modules = False
    skip_load_skills = False

    if args.dataset in ["ni"]:
        finetuner_cls = Finetuner
    else:
        finetuner_cls = T0EncoderDecoder

    if args.finetune_type == "F":
        # full model fine-tuning
        args.trainable_param_names = ".*"
    elif args.finetune_type == "A":
        # if we start from a full trained model, then we should load appropriately
        if ckpt_args.get(
            "trainable_param_names", ckpt_args.get("finetune_full_model", False)
        ) not in [".*", True]:
            # should be a shared model
            assert ckpt_args["model_modifier"] in ["lora", "ia3"]
            args.model_modifier = ckpt_args["model_modifier"]
    elif args.finetune_type == "Z":
        # fix skills, train only Z
        assert args.model_modifier and "poly" in args.model_modifier
        args.trainable_param_names = ".*selector.*"
    elif args.finetune_type == "MuZ":
        # don't train the module allocation, average of modules
        assert args.model_modifier and "poly" in args.model_modifier
        switch_to_avg_modules = True
    elif args.finetune_type == "PolyRand":
        # random polytropon
        skip_load_skills = True

    # data
    monitor = "val/metric_perf"
    if args.dataset == "ni":
        dm = NIOriginalDataModule(args)
    elif args.dataset == "t0":
        dm = T0FinetuneDataModule(args)

    kwargs = copy.deepcopy(vars(args))
    kwargs.pop("checkpoint")
    # economic checkpointing for finetuning, we don't need to save the full backbone, only parameters that we are training.
    kwargs["save_if_loaded"] = False
    module = finetuner_cls(**kwargs, tokenizer=dm.tokenizer)

    if skip_load_skills or ckpt_path is None:
        print("Skipping loading from checkpoint...")
    else:
        module.load_state_dict(ckpt_dict, strict=False)

    # allocate new module logits for the new task
    if args.model_modifier and "poly" in args.model_modifier:
        if switch_to_avg_modules:
            module.model.switch_selector_to_average()
        else:
            # resize to accomodate for new task
            module.model.resize_module_logits(1)

    def fit_and_test(zero_shot=False):
        callbacks = [ProgressCallback()]

        if not args.finetune_skip_es:
            ckpt_callback = CustomModelCheckpoint(
                dirpath="/tmp/sni/",
                monitor=monitor,
                filename=f"{args.model}"
                + "-{epoch:02d}-"
                + "{"
                + f"{monitor}"
                + ":.2f}",
                save_top_k=1,
                mode="max",
                save_weights_only=True,  #  try to save some HD space
            )
            callbacks.append(ckpt_callback)
        callbacks.extend(get_monitors(args))

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

        if args.finetune_skip_es:
            check_val_every_n_epoch = 10000
        else:
            check_val_every_n_epoch = 10 if args.dataset in ["ni"] else 50

        trainer = Trainer(
            enable_checkpointing=not args.finetune_skip_es,
            devices=1,
            default_root_dir=args.output_dir,
            accelerator="gpu",
            logger=loggers,
            num_sanity_val_steps=0,
            max_steps=args.total_steps,
            max_epochs=args.num_train_epochs,
            gradient_clip_val=args.max_grad_norm,
            log_every_n_steps=10,
            check_val_every_n_epoch=check_val_every_n_epoch,
            strategy="auto" if not args.compute_strategy else args.compute_strategy,
            limit_val_batches=1.0,
            limit_train_batches=1.0,
            precision=int(args.precision)
            if args.precision in ["16", "32"]
            else args.precision,
            callbacks=callbacks,
            accumulate_grad_batches=args.gradient_accumulation_steps,
        )

        if zero_shot:
            trainer.test(module, dm, ckpt_path=None)

        trainer.fit(module, dm)

        if args.finetune_skip_es:
            ckpt_path = None
            trainer.validate(module, dm, ckpt_path=ckpt_path)
        else:
            ckpt_path = "best"
        trainer.test(module, dm, ckpt_path=ckpt_path)

        results = [module.best_val_result] + module.test_results
        return results

    results = fit_and_test(zero_shot=do_zs)

    # remove all eventual checkpoints
    os.system(f'find /tmp/sni/ -name "*.ckpt" -type f -delete')
    os.system(f'find /tmp/sni/ -name "*.pt" -type f -delete')
    return results


def finetune_ni(args, seeds=[13, 42, 58], use_mlf=True, do_zs=True):
    all_results = []

    for seed in seeds:
        args.seed = seed

        # use mlf logger only for the first seed, otw it will complain for duplicated hps
        results = finetune(
            args,
            use_mlf=(seed == seeds[0] and use_mlf),
            do_zs=(seed == seeds[0] and do_zs),
        )
        all_results.extend(results)

    for result in all_results:
        result["prefix"] = args.finetune_task_name

    # whatever
    print(all_results)

    df = pd.DataFrame.from_dict(all_results)
    df.to_csv(os.path.join(args.output_dir, "result.csv"))

    return df


def finetune_t0(args, seeds=[42, 1024, 0], use_mlf=True, do_zs=True):
    all_results = []

    for i, seed in enumerate(seeds):
        args.seed = seed

        # use mlf logger only for the first seed, otw it will complain for duplicated hps
        results = finetune(
            args,
            use_mlf=use_mlf and i == 0,
            do_zs=do_zs,
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

    if args.dataset == "ni":
        finetune_ni(args)
    elif args.dataset == "t0":
        finetune_t0(args)
    else:
        raise ValueError("Dataset not recognized.")
