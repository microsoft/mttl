import hashlib
import json
import os
import re
from collections import defaultdict, deque
from enum import Enum
from typing import Callable, Optional, Union

import prettytable
import pytorch_lightning as pl
import torch
from pytorch_lightning import LightningModule
from transformers.file_utils import PushToHubMixin
from transformers.utils import cached_file

from mttl.logging import logger
from mttl.models.get_optimizer import get_optimizer
from mttl.models.get_scheduler import get_scheduler
from mttl.utils import get_checkpoint_path

CHECKPOINT_PATH_IN_HUB = "checkpoint.ckpt"


def download_from_hub(repo_id) -> str:
    """Download checkpoint from hub."""
    from mttl.models.library.expert_library import ExpertLibrary

    return ExpertLibrary.get_expert_library(repo_id).hf_hub_download(
        repo_id=repo_id, filename=CHECKPOINT_PATH_IN_HUB
    )


def convert_hps_to_dict(hparams):
    hparams_allowed = {}
    # drop parameters which contain some strange datatypes as fsspec
    for k, v in hparams.items():
        v = v.name if isinstance(v, Enum) else v
        hparams_allowed[k] = v
    return hparams_allowed


class SimpleLogger(pl.loggers.logger.DummyLogger):
    def __init__(self, output_dir):
        self.output_file = os.path.join(output_dir, "metrics.json")
        os.makedirs(output_dir, exist_ok=True)

        if os.path.exists(self.output_file):
            os.remove(self.output_file)

    def log_metrics(self, metrics, step=None):
        lines = []
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            lines.append({"name": k, "value": v, "step": step})

        try:
            with open(self.output_file, "a+") as f:
                for l in lines:
                    f.write(json.dumps(l) + "\n")
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")


class OnLogCallback:
    """Adds `on_log` capability to callbacks."""

    def log(self, name, value, **kwargs):
        output = LightningModule.log(self, name, value, **kwargs)

        # call on log on each callback
        for callback in self.trainer.callbacks:
            if hasattr(callback, "on_log"):
                callback.on_log(self.trainer, self, name, value, **kwargs)
        return output


class EfficientCheckpointModule(OnLogCallback, PushToHubMixin, LightningModule):
    """Efficiently save and load checkpoints.

    Only saves and loads parameters that are either in the trainable parameters
    or have been loaded from a previous checkpoint.
    """

    def __init__(self, **kwargs):
        LightningModule.__init__(self)
        PushToHubMixin.__init__(self)

        self.loss_plugins = {}
        # If True, do not delete any parameters that were loaded in a
        # previous checkpoint, even if the parameters are not trainable.
        self.save_if_loaded_from_ckpt = kwargs.get("save_if_loaded_from_ckpt", True)

        if (
            self.save_if_loaded_from_ckpt
            and kwargs.get("compute_strategy", "") == "deepspeed"
        ):
            logger.warning(
                "`save_if_loaded_from_ckpt` is True. Because you are using deepspeed, you will be saving full model checkpoints."
            )

    def get_hash(self):
        model_hash = hashlib.sha256()
        model_hash.update(f"{self.hparams}".encode())
        return model_hash.hexdigest()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        **kwargs,
    ):
        # Load model
        instantiate_model = kwargs.pop("instantiate_model", True)
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", False)
        use_auth_token = kwargs.pop("use_auth_token", None)

        user_agent = {
            "file_type": "model",
            "framework": "pytorch",
            "from_auto_class": False,
        }

        if pretrained_model_name_or_path is not None:
            pretrained_model_name_or_path = str(pretrained_model_name_or_path)

            if os.path.isfile(pretrained_model_name_or_path) or os.path.isdir(
                pretrained_model_name_or_path
            ):
                resolved_archive_file = get_checkpoint_path(
                    pretrained_model_name_or_path
                )
            else:
                try:
                    # Load from URL or cache if already cached
                    resolved_archive_file = cached_file(
                        pretrained_model_name_or_path,
                        CHECKPOINT_PATH_IN_HUB,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        proxies=proxies,
                        resume_download=resume_download,
                        local_files_only=local_files_only,
                        use_auth_token=use_auth_token,
                        user_agent=user_agent,
                    )
                except EnvironmentError as err:
                    logger.error(err)
                    msg = (
                        f"Can't load weights for '{pretrained_model_name_or_path}'. Make sure that:\n\n"
                        f"- '{pretrained_model_name_or_path}' is a correct model identifier listed on 'https://huggingface.co/models'\n\n"
                        f"- or '{pretrained_model_name_or_path}' is the correct path to a directory containing a file named one of checkpoint.ckpt\n\n"
                    )
                    raise EnvironmentError(msg)

                if resolved_archive_file == pretrained_model_name_or_path:
                    logger.info(f"loading weights file {resolved_archive_file}")
                else:
                    logger.info(
                        f"loading weights file {pretrained_model_name_or_path} from cache at {resolved_archive_file}"
                    )
        else:
            resolved_archive_file = None

        if instantiate_model:
            return cls.load_from_checkpoint(resolved_archive_file, **kwargs)
        else:
            ckpt = torch.load(resolved_archive_file, map_location="cpu")

            return ckpt["state_dict"], ckpt["hyper_parameters"]

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path,
        **model_kwargs,
    ):
        from mttl.datamodule.utils import get_tokenizer_with_args

        tokenizer = model_kwargs.get("tokenizer", None)
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        ckpt["hyper_parameters"].update(**model_kwargs)

        if tokenizer is None and "model" in ckpt["hyper_parameters"]:
            tokenizer = get_tokenizer_with_args(
                model_name=ckpt["hyper_parameters"]["model"],
                model_family=ckpt["hyper_parameters"]["model_family"],
                padding_side=ckpt["hyper_parameters"]["padding_side"],
            )

        model = cls(**ckpt["hyper_parameters"])
        model.load_state_dict(ckpt["state_dict"], strict=False)
        return model

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        save_config: bool = True,
        state_dict: Optional[dict] = None,
        save_function: Callable = torch.save,
        push_to_hub: bool = False,
        save_full_model: bool = False,
        **kwargs,
    ):
        ckpt = self.state_dict()

        if not save_full_model:
            self._delete_non_trainable_params(ckpt)

        hparams_allowed = {}
        # drop parameters which contain some strange datatypes as fsspec
        for k, v in self.hparams.items():
            v = v.name if isinstance(v, Enum) else v
            hparams_allowed[k] = v

        save_package = {
            "state_dict": ckpt,
            "hyper_parameters": hparams_allowed,
        }
        if hasattr(self, "expert_info"):
            save_package["expert_info"] = self.expert_info.__dict__

        output_model_file = os.path.join(save_directory, CHECKPOINT_PATH_IN_HUB)
        torch.save(save_package, output_model_file)
        logger.info(f"Model weights saved in {output_model_file}")
        return output_model_file

    def load_state_dict(self, ckpt, **kwargs):
        # store params that might have been loaded from a previous checkpoint
        self._params_from_checkpoint = (
            set(ckpt.keys()) if self.save_if_loaded_from_ckpt else set()
        )
        for name, _ in self.state_dict().items():
            if name in ckpt:
                logger.info("Loading {} from state dict.".format(name))
        return super().load_state_dict(ckpt, strict=False)

    def _delete_non_trainable_params(self, state_dict):
        if not hasattr(self, "_params_from_checkpoint"):
            self._params_from_checkpoint = set()

        if not hasattr(self, "trainable_param_names"):
            self.trainable_param_names = [
                n for n, p in self.named_parameters() if p.requires_grad
            ]

        keys = [k for k in state_dict.keys()]

        # remove also parameters in the loss plugins, these need not be saved
        # (auxiliary parameters for the losses)
        plugin_param_keys = set()
        for _, plugin in self.loss_plugins.items():
            plugin_param_keys.update(plugin.state_dict().keys())

        deleted = []
        for key in keys:
            # we can safely avoid dumping this parameter if it is both
            # not in the trainable parameters and was not loaded from checkpoint
            if (
                not (key in self.trainable_param_names)
                and not (key in self._params_from_checkpoint)
            ) or key in plugin_param_keys:
                del state_dict[key]
                deleted.append(key)

        logger.info("Deleted from state dict: {}".format(len(deleted)))

    def on_save_checkpoint(self, ckpt):
        self._delete_non_trainable_params(ckpt["state_dict"])

    def on_load_checkpoint(self, ckpt):
        print("Loading checkpoint...")

        load_result = self.load_state_dict(ckpt["state_dict"])

        assert (
            len(load_result.unexpected_keys) == 0
        ), f"Load model failed, unexpected keys {load_result.unexpected_keys.__str__()}"

    def configure_optimizers(self):
        args = self.hparams
        self.ml_optimizer = self.ml_scheduler = None

        optimizer, self.trainable_param_names = get_optimizer(
            self, args, no_decay=["bias", "LayerNorm.weight"]
        )
        global_bs = get_global_batch_size(
            args.train_batch_size, args.gradient_accumulation_steps
        )

        if args.total_steps == -1:
            args.total_steps = (
                len(self.trainer.datamodule.train_dataset) // global_bs
            ) * self.trainer.max_epochs

        if args.warmup_steps == -1 or args.warmup_proportion > 0.0:
            logger.warning(
                "Warmup proportion is set to {}, has priority over warmup_steps".format(
                    args.warmup_proportion
                )
            )

            args.warmup_steps = int(args.warmup_proportion * args.total_steps)

        logger.info("Optimizer setup:")
        logger.info("Total steps: {}".format(args.total_steps))
        logger.info("Warmup steps: {}".format(args.warmup_steps))
        logger.info("Scheduler: {}".format(args.scheduler))

        scheduler = get_scheduler(optimizer, args)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }


def get_global_batch_size(batch_size, accumulation_steps):
    """Computes the global batch size."""
    try:
        world_size = torch.distributed.get_world_size()
    except:
        world_size = 1
    global_bs = batch_size * world_size * accumulation_steps
    return global_bs
