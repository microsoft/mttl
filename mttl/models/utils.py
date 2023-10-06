from enum import Enum
import os
from typing import Any, Callable, Optional, Union
from pytorch_lightning import LightningModule
import torch

from transformers.utils import cached_file
from transformers.file_utils import PushToHubMixin
from mttl.utils import logger, get_checkpoint_path
from mttl.models.get_optimizer import get_optimizer
from mttl.models.get_scheduler import get_scheduler


CHECKPOINT_PATH_IN_HUB = "checkpoint.ckpt"


def transfer_batch_to_device(batch, device):
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)
    return batch


def convert_and_push_to_hub(
    ckpt_path, repo_id, auto_search=True, use_last=False,
) -> None:
    """Searches into local path for the checkpoint with lowest validation loss,
    then uploads that.

    if use_last is True, then uses the last checkpoint `last.ckpt` instead
    of the one with lowest validation loss.
    """
    import huggingface_hub
    from mttl.utils import get_checkpoint_path

    if auto_search:
        ckpt_path = get_checkpoint_path(ckpt_path, use_last=use_last)

    logger.info("Uploading checkpoint at {}".format(ckpt_path))

    huggingface_hub.create_repo(repo_id, repo_type="model", exist_ok=True)
    huggingface_hub.upload_file(
        path_or_fileobj=ckpt_path, repo_id=repo_id, path_in_repo=CHECKPOINT_PATH_IN_HUB
    )


def download_from_hub(repo_id) -> str:
    """Download checkpoint from hub."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=repo_id, filename=CHECKPOINT_PATH_IN_HUB, repo_type="model"
    )


class EfficientCheckpointModule(LightningModule, PushToHubMixin):
    """Efficiently save and load checkpoints.

    Only saves and loads parameters that are either in the trainable parameters
    or have been loaded from a previous checkpoint.
    """

    def __init__(self, **kwargs):
        LightningModule.__init__(self)
        PushToHubMixin.__init__(self)

        self.loss_plugins = {}
        self.save_if_loaded = kwargs.get("save_if_loaded", True)

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

            if os.path.isdir(pretrained_model_name_or_path):
                resolved_archive_file = get_checkpoint_path(
                    pretrained_model_name_or_path
                )
            else:
                try:
                    # Load from URL or cache if already cached
                    resolved_archive_file = cached_file(
                        pretrained_model_name_or_path,
                        "checkpoint.ckpt",
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
    
        if tokenizer is None:
            tokenizer = get_tokenizer_with_args(
                model_name=ckpt["hyper_parameters"]["model"],
                model_family=ckpt["hyper_parameters"]["model_family"],
                padding_side=ckpt["hyper_parameters"]["padding_side"],
            )

        model = cls(**ckpt["hyper_parameters"], tokenizer=tokenizer)
        model.load_state_dict(ckpt["state_dict"], strict=False)
        return model

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        save_config: bool = True,
        state_dict: Optional[dict] = None,
        save_function: Callable = torch.save,
        push_to_hub: bool = False,
        **kwargs,
    ):
        ckpt = self.state_dict()

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

        output_model_file = os.path.join(save_directory, "checkpoint.ckpt")
        torch.save(save_package, output_model_file)
        logger.info(f"Model weights saved in {output_model_file}")

    def load_state_dict(self, ckpt, **kwargs):
        # store params that might have been loaded from a previous checkpoint
        self._params_from_checkpoint = (
            set(ckpt.keys()) if self.save_if_loaded else set()
        )
        for name, _ in self.state_dict().items():
            if name in ckpt:
                logger.info("Loading {} from state dict.".format(name))
        return super().load_state_dict(ckpt, strict=False)

    def _delete_non_trainable_params(self, state_dict):
        if not hasattr(self, "_params_from_checkpoint"):
            self._params_from_checkpoint = set()

        if not hasattr(self, "trainable_param_names"):
            self.trainable_param_names = [n for n, p in self.named_parameters() if p.requires_grad]

        keys = [k for k in state_dict.keys()]

        # remove also parameters in the loss plugins, these need not be saved
        # (auxiliary parameters for the losses)
        plugin_param_keys = set()
        for _, plugin in self.loss_plugins.items():
            plugin_param_keys.update(plugin.state_dict().keys())

        for key in keys:
            # we can safely avoid dumping this parameter if it is both
            # not in the trainable parameters and was not loaded from checkpoint
            if (
                not (key in self.trainable_param_names)
                and not (key in self._params_from_checkpoint)
            ) or key in plugin_param_keys:
                del state_dict[key]
                logger.info("Deleting from state dict: {}".format(key))

    def on_save_checkpoint(self, ckpt):
        self._delete_non_trainable_params(ckpt["state_dict"])

    def on_load_checkpoint(self, ckpt):
        print("Loading checkpoint...")

        load_result = self.load_state_dict(ckpt["state_dict"])

        assert (
            len(load_result.unexpected_keys) == 0
        ), f"Load model failed, unexpected keys {load_result.unexpected_keys.__str__()}"

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        return transfer_batch_to_device(batch, device)

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
            logger.info(
                "Warmup proportion is set to {}, has priority over warmup_steps".format(
                    args.warmup_proportion
                )
            )

            args.warmup_steps = int(args.warmup_proportion * args.total_steps)

        # args.scheduler = "linear_decay_with_warmup"
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
