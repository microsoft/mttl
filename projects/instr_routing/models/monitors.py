from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import math

from torch.distributions import Categorical
from pytorch_lightning import Callback

from mttl.utils import average_dicts
from mttl.models.modifiers.routing import RoutingSelector
from mttl.models.adapters import Adapter


class SelectorRoutingsLog(Callback):
    LOG_EVERY = 16

    def __init__(self):
        self.routings = {}

    def aggregate_and_maybe_log(self, trainer, pl_module, current_step, split) -> None:
        # get routing attributes of all layers
        for name, module in pl_module.named_modules():
            if isinstance(module, RoutingSelector) and hasattr(module, "routings"):
                if name not in self.routings:
                    self.routings[name] = []
                self.routings[name].append(module.routings.detach())

        if not self.routings:
            return

        if current_step % self.LOG_EVERY == 0 and trainer.global_step > 0:
            stats = []

            for name, stats in self.routings.items():
                layer_routing_dist = torch.cat(stats, dim=0)
                batch_size = layer_routing_dist.size(0)
                layer_routing_dist = layer_routing_dist.view(batch_size, -1)
                dims = layer_routing_dist.shape[1]
                layer_routing_mean = layer_routing_dist.mean(0)
                h_mean = Categorical(probs=layer_routing_mean).entropy() / math.log(
                    dims
                )
                mean_h = Categorical(
                    probs=layer_routing_dist
                ).entropy().mean() / math.log(dims)
                mi = h_mean - mean_h

                stats.append(
                    {
                        "routing_mi": mi,
                        "routing_ents": mean_h,
                    }
                )

            stats = average_dicts(stats)
            for k, v in stats.items():
                pl_module.log(
                    f"{split}/{k}", v, on_epoch=True, on_step=True, sync_dist=True, prog_bar=True
                )
            self.routings.clear()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        self.aggregate_and_maybe_log(
            trainer, pl_module, trainer.global_step, split="train"
        )

    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.aggregate_and_maybe_log(
            trainer, pl_module, batch_idx, split="val"
        )


class SelectorMetricsLog(Callback):
    LOG_EVERY = 16

    def __init__(self):
        self.metrics = {}

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        # get routing attributes of all layers
        for name, module in pl_module.named_modules():
            if isinstance(module, RoutingSelector) and hasattr(module, "metrics"):
                if name not in self.metrics:
                    self.metrics[name] = []
                self.metrics[name].append(module.metrics)

        if not self.metrics:
            return

        if trainer.global_step % self.LOG_EVERY == 0 and trainer.global_step > 0:
            stats = []

            for name, stats in self.metrics.items():
                avg_metrics = average_dicts(stats)
                stats.append(avg_metrics)

            stats = average_dicts(stats)
            for k, v in stats.items():
                pl_module.log(
                    f"train/{k}", v, on_epoch=True, on_step=True, sync_dist=True
                )
            self.metrics.clear()
