from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import math

from torch.distributions import Categorical
from pytorch_lightning import Callback

from mttl.utils import average_dicts
from mttl.models.modifiers.routing import RoutingSelector
from collections import defaultdict


class Averager:
    def __init__(self, weight: float = 1):
        self.weight = weight
        self.reset()

    def reset(self):
        self.total = defaultdict(float)
        self.counter = defaultdict(float)

    def update(self, stats):
        for key, value in stats.items():
            self.total[key] = self.total[key] * (1 - self.weight) + value * self.weight
            self.counter[key] = self.counter[key] * (1 - self.weight) + self.weight

    def average(self):
        averaged_stats = {
            key: tot / (
                self.counter[key] if self.weight else 1.
            ) for key, tot in self.total.items()
        }
        self.reset()
        return averaged_stats


class SelectorRoutingsLog(Callback):
    ACC_OVER = 10

    def __init__(self):
        self.averager = Averager(0.9)
        self.acc_routings = {}

    def aggregate_and_maybe_log(self, trainer, pl_module, current_step, split) -> None:
        # get routing attributes of all layers
        for name, module in pl_module.named_modules():
            if isinstance(module, RoutingSelector) and hasattr(module, "routings"):
                if name not in self.acc_routings:
                    self.acc_routings[name] = []
                self.acc_routings[name].append(module.routings.detach())

        if not self.acc_routings:
            return

        if current_step % self.ACC_OVER == 0 and trainer.global_step > 0:
            layer_stats = []

            for name, stats in self.acc_routings.items():
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

                layer_stats.append(
                    {
                        "routing_mi": mi,
                        "routing_ents": mean_h,
                    }
                )

            global_stats = average_dicts(layer_stats)
            self.averager.update(global_stats)
            global_stats = self.averager.average()

            for k, v in global_stats.items():
                pl_module.log(
                    f"{split}/{k}",
                    v,
                    on_epoch=True,
                    on_step=True,
                    sync_dist=True,
                    prog_bar=True,
                )

            self.acc_routings.clear()

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
        self.aggregate_and_maybe_log(trainer, pl_module, batch_idx, split="val")


class SelectorMetricsLog(Callback):
    def __init__(self):
        self.averager = Averager(weight=0.9)
        self.metrics = {}

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        # get routing attributes of all layers
        for name, module in pl_module.named_modules():
            if isinstance(module, RoutingSelector) and hasattr(module, "metrics"):
                self.metrics[name] = module.metrics

        global_stats = average_dicts(list(self.metrics.values()))
        self.averager.update(global_stats)

        average_so_far = self.averager.average()
        for k, v in average_so_far.items():
            pl_module.log(
                f"train/{k}",
                v,
                on_epoch=True,
                on_step=True,
                sync_dist=True,
                prog_bar=True,
            )
        self.metrics.clear()
