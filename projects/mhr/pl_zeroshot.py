import copy
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer, seed_everything

from mttl.callbacks import ProgressCallback
from mttl.config import parse_config
from mttl.datamodule.ni_data_module import NIDataModule
from mttl.datamodule.t0_data_module import T0FinetuneDataModule
from mttl.datamodule.xfit_data_module import XFitDataModule
from mttl.models.encoder_decoder import Finetuner
from mttl.models.t0_encoder_decoder import T0EncoderDecoder
from mttl.utils import get_checkpoint_path, get_mlf_logger

from pl_finetune import ARGS_TO_OVERWRITE
from mhr_config import MHRConfig


def evaluate_zeroshot(config):
    seed_everything(config.seed, workers=True)

    if config.checkpoint:
        ckpt_path = get_checkpoint_path(config.checkpoint, config.checkpoint_step)
        ckpt_beef = torch.load(ckpt_path)
        ckpt_args = ckpt_beef["hyper_parameters"]
        ckpt_dict = ckpt_beef["state_dict"]
        config.old_exp_name = ckpt_args["exp_name"]

        for arg_name in ARGS_TO_OVERWRITE:
            if arg_name in ckpt_args and not config.was_overridden(arg_name):
                print("Overwriting", arg_name, "=", ckpt_args[arg_name])
                setattr(config, arg_name, ckpt_args[arg_name])

    kwargs = copy.deepcopy(vars(config))
    kwargs.pop("checkpoint", None)

    # data
    if config.dataset == "xfit":
        dm = XFitDataModule(config)
        model_class = Finetuner
    elif config.dataset == "ni":
        dm = NIDataModule(config)
        model_class = Finetuner
    elif config.dataset == "bb":
        dm = BBDataModule(config)
        model_class = T0EncoderDecoder
    elif config.dataset == "t0":
        dm = T0FinetuneDataModule(config)
        model_class = T0EncoderDecoder
    else:
        raise NotImplementedError()

    module = model_class(**kwargs, tokenizer=dm.tokenizer)
    if config.checkpoint:
        print("Loading from checkpoint...", ckpt_path)
        module.load_state_dict(ckpt_dict, strict=False)

    # when evaluating polytropon with polytropon routing in zero-shot settings, we need to switch to average
    if args.model_modifier and "poly" in args.model_modifier and "poly" in args.poly_selector:
        module.model.switch_selector_to_average()

    # legit logging
    loggers = []
    if os.environ.get("WANDB_API_KEY"):
        wandb_logger = pl.loggers.WandbLogger(
            project=config.wandb_project,
            name=config.exp_name,
        )
        wandb_logger.experiment.save("*.py")
        loggers.append(wandb_logger)
    else:
        wandb_logger = None

    mlf_logger = get_mlf_logger()
    if mlf_logger:
        loggers.append(mlf_logger)

    trainer = Trainer(
        devices=1,
        accelerator="gpu",
        logger=loggers,
        num_sanity_val_steps=0,
        default_root_dir=config.output_dir,
        log_every_n_steps=5 if config.debug else 50,
        strategy=None if not config.compute_strategy else config.compute_strategy,
        limit_val_batches=0,
        limit_train_batches=0,
        callbacks=[ProgressCallback()],
    )
    results = trainer.test(module, dm)
    print(results)
    return results


if __name__ == "__main__":
    args = MHRConfig.parse()
    evaluate_zeroshot(args)
