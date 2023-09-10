import os
from typing import Any
from pytorch_lightning import LightningModule
import torch
import json


def transfer_batch_to_device(batch, device):
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)
    return batch


class EfficientCheckpointModule(LightningModule):
    """Efficiently save and load checkpoints.
    
    Only saves and loads parameters that are either in the trainable parameters
    or have been loaded from a previous checkpoint.
    """

    def __init__(self, **kwargs):
        super().__init__()

        self.loss_plugins = {}
        self.save_if_loaded = kwargs.get("save_if_loaded", True)

    def load_state_dict(self, ckpt, **kwargs):
        """Always load state dict with strict = False, so that we can
        early stop successfully.
        """
        # store params that might have been loaded from a previous checkpoint
        self._params_from_checkpoint = (
            set(ckpt.keys()) if self.save_if_loaded else set()
        )
        print("Loading keys...", list(ckpt.keys()))
        return super().load_state_dict(ckpt, strict=False)

    def on_save_checkpoint(self, ckpt):
        if not hasattr(self, "_params_from_checkpoint"):
            self._params_from_checkpoint = set()

        # remove also parameters in the loss plugins, these need not be saved
        # (auxiliary parameters for the losses)
        plugin_param_keys = set()
        for _, plugin in self.loss_plugins.items():
            plugin_param_keys.update(plugin.state_dict().keys())

        keys = [k for k in ckpt["state_dict"].keys()]

        for key in keys:
            # we can safely avoid dumping this parameter if it is both
            # not in the trainable parameters and was not loaded from checkpoint
            if (
                not (key in self.trainable_param_names)
                and not (key in self._params_from_checkpoint)
            ) or key in plugin_param_keys:
                del ckpt["state_dict"][key]
                print("Deleting from state dict:", key)

    def on_load_checkpoint(self, ckpt):
        print("Loading checkpoint...")

        load_result = self.load_state_dict(ckpt["state_dict"])

        assert (
            len(load_result.unexpected_keys) == 0
        ), f"Load model failed, unexpected keys {load_result.unexpected_keys.__str__()}"

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        return transfer_batch_to_device(batch, device)


def get_global_batch_size(batch_size, accumulation_steps):
    """Computes the global batch size."""
    try:
        world_size = torch.distributed.get_world_size()
    except:
        world_size = 1
    global_bs = batch_size * world_size * accumulation_steps
    return global_bs
