from pytorch_lightning import LightningModule
import torch
from dataclasses import dataclass
from typing import List
from dataclasses import dataclass, field


class EfficientCheckpointModule(LightningModule):
    """Efficiently save and load checkpoints.
    
    Only saves and loads parameters that are either in the trainable parameters
    or have been loaded from a previous checkpoint.
    """

    def __init__(self, **kwargs):
        super().__init__()

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


@dataclass
class RoutingInfo:
    task_ids: torch.Tensor
    hashes: List[str]
    instruction_hashes: List[str] = None
    example_ids: List[int] = None
    pad_token_mask: torch.Tensor = None
    inst_token_mask: torch.Tensor = None
    labels: torch.Tensor = None

    @classmethod 
    def from_batch(cls, batch: dict):
        ri = cls(
            task_ids=batch["task_ids"],   
            hashes=batch.get("hashes", None),
            example_ids=batch.get("example_ids", None),
            instruction_hashes=batch.get("instruction_hashes", None),
            pad_token_mask = batch.get("pad_token_mask", None),
            inst_token_mask = batch.get("inst_token_mask", None),
            labels=batch.get("labels", None),
        )
        if "distances" in batch:
            # used for evaluation of soft clustering tuned models
            setattr(ri, "distances", batch["distances"])
        return ri

    def repeat_interleave(self, repeats):
        # useful for beam search
        self.task_ids = self.task_ids.repeat_interleave(repeats)
        if self.hashes:
            self.hashes = [h for h in self.hashes for _ in range(repeats)]
        if self.instruction_hashes:
            self.instruction_hashes = [h for h in self.instruction_hashes for _ in range(repeats)]
        self.example_ids = (
            self.example_ids.repeat_interleave(repeats)
            if self.example_ids is not None
            else None
        )


def get_global_batch_size(batch_size, accumulation_steps):
    """Computes the global batch size."""
    try:
        world_size = torch.distributed.get_world_size()
    except:
        world_size = 1
    global_bs = batch_size * world_size * accumulation_steps
    return global_bs
