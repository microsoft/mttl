from enum import Enum
import hashlib
import os
from typing import Any, Callable, Optional, Union
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
import torch
import json
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
    expert,
    repo_id,
) -> None:
    import huggingface_hub
    import io

    with io.BytesIO() as buffer:
        torch.save(expert.asdict(), buffer)

        huggingface_hub.create_repo(repo_id, repo_type="model", exist_ok=True)
        huggingface_hub.upload_file(
            path_or_fileobj=buffer, repo_id=repo_id, path_in_repo=CHECKPOINT_PATH_IN_HUB
        )


def download_from_hub(repo_id) -> str:
    """Download checkpoint from hub."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=repo_id, filename=CHECKPOINT_PATH_IN_HUB, repo_type="model"
    )


def convert_hps_to_dict(hparams):
    hparams_allowed = {}
    # drop parameters which contain some strange datatypes as fsspec
    for k, v in hparams.items():
        v = v.name if isinstance(v, Enum) else v
        hparams_allowed[k] = v
    return hparams_allowed


def prepare_model_for_kbit_training(model, use_gradient_checkpointing=True):
    r"""
    This method wraps the entire protocol for preparing a model before running a training. This includes:
        1- Cast the layernorm in fp32 2- making output embedding layer require grads 3- Add the upcasting of the lm
        head to fp32

    Args:
        model, (`transformers.PreTrainedModel`):
            The loaded model from `transformers`
    """
    loaded_in_kbit = getattr(model, "is_loaded_in_8bit", False) or getattr(
        model, "is_loaded_in_4bit", False
    )

    # cast all non INT8 parameters to fp32
    for param in model.parameters():
        if (param.dtype == torch.float16) or (param.dtype == torch.bfloat16):
            param.data = param.data.to(torch.float32)

    if loaded_in_kbit and use_gradient_checkpointing:
        # For backward compatibility
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        # FIX for enabling gradient of the auxiliary loss
        # enable gradient checkpointing for memory efficiency
        from functools import partial

        notfailing_checkpoint = partial(
            torch.utils.checkpoint.checkpoint, use_reentrant=False
        )
        torch.utils.checkpoint.checkpoint = notfailing_checkpoint
        model.gradient_checkpointing_enable()

    return model


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

        with open(self.output_file, "a+") as f:
            for l in lines:
                f.write(json.dumps(l) + "\n")


class OnLogCallback:
    """Adds `on_log` capability to callbacks."""

    def log(self, name, value, **kwargs):
        output = LightningModule.log(self, name, value, **kwargs)

        # call on log on each callback
        for callback in self.trainer.callbacks:
            if hasattr(callback, "on_log"):
                callback.on_log(self.trainer, self, name, value)
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
        self.save_if_loaded = kwargs.get("save_if_loaded", True)

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

        if tokenizer is None and "model" in ckpt["hyper_parameters"]:
            tokenizer = get_tokenizer_with_args(
                model_name=ckpt["hyper_parameters"]["model"],
                model_family=ckpt["hyper_parameters"]["model_family"],
                padding_side=ckpt["hyper_parameters"]["padding_side"],
            )

        expert_info = ckpt.get("expert_info", None)
        model = cls(
            **ckpt["hyper_parameters"], expert_info=expert_info, tokenizer=tokenizer
        )
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

        output_model_file = os.path.join(save_directory, "checkpoint.ckpt")
        torch.save(save_package, output_model_file)
        logger.info(f"Model weights saved in {output_model_file}")
        return output_model_file

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
            logger.warn(
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


def prepare_model_for_kbit_training(model, use_gradient_checkpointing=False):
    r"""
    This method wraps the entire protocol for preparing a model before running a training. This includes:
        1- Cast the layernorm in fp32 2- making output embedding layer require grads 3- Add the upcasting of the lm
        head to fp32

    Args:
        model, (`transformers.PreTrainedModel`):
            The loaded model from `transformers`
    """
    loaded_in_kbit = getattr(model, "is_loaded_in_8bit", False) or getattr(
        model, "is_loaded_in_4bit", False
    )

    # cast all non INT8 parameters to fp32
    for param in model.parameters():
        if (param.dtype == torch.float16) or (param.dtype == torch.bfloat16):
            param.data = param.data.to(torch.float32)

    if loaded_in_kbit and use_gradient_checkpointing:
        # For backward compatibility
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        # FIX for enabling gradient of the auxiliary loss
        # enable gradient checkpointing for memory efficiency
        from functools import partial

        notfailing_checkpoint = partial(
            torch.utils.checkpoint.checkpoint, use_reentrant=False
        )
        torch.utils.checkpoint.checkpoint = notfailing_checkpoint
        model.gradient_checkpointing_enable()
        # FIX for enabling gradient of the auxiliary loss

    return model


def model_loader_helper(model_name, device_map="auto", load_in_8bit=False):
    from transformers import PreTrainedModel, LlamaForCausalLM, AutoModelForCausalLM

    if isinstance(model_name, PreTrainedModel):
        return model_name

    if "llama" in model_name:
        model_object = LlamaForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=load_in_8bit,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
        )
    elif "phi-2" in model_name:
        model_object = AutoModelForCausalLM.from_pretrained(
            os.environ["PHI_PATH"],
            load_in_8bit=load_in_8bit,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True,
        )
    elif "stabilityai" in model_name:
        model_object = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype="auto",
        )
    else:
        model_object = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            load_in_8bit=load_in_8bit,
            trust_remote_code=True,
        )
    return model_object
