from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import math
import wandb
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from torch.distributions import Categorical
from pytorch_lightning import Callback

from mttl.utils import agg_dicts, Averager
from mttl.models.modifiers.routing import RoutingSelector
from collections import defaultdict


class SelectorRoutingsLog(Callback):
    ACC_OVER = 10

    def __init__(self, args):
        self.averager = Averager(0.5)
        self.acc_routings = {}
        self.log_per_layer = args.selector_log_per_layer

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
            layer_stats, layer_names = [], []  
            for name, stats in self.acc_routings.items():
                layer_names.append(name)
                layer_routing_dist = torch.cat(stats, dim=0)
                batch_size = layer_routing_dist.size(0)
                layer_routing_dist = layer_routing_dist.view(
                    batch_size, -1
                )  # TODO: wopuld it also work fine for multihead routing?
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

            global_stats = agg_dicts(layer_stats)
            global_stats = self.averager.update(
                {**global_stats}
            )  # average accross layers of the metrics

            for k, v in global_stats.items():
                pl_module.log(
                    f"{split}/{k}",
                    v,
                    on_step=True,
                    sync_dist=True,
                    prog_bar=True,
                )
            if (
                self.log_per_layer
                and wandb.run is not None
                and split == "val"
                and not isinstance(pl_module.loggers[0], pl.loggers.logger.DummyLogger)
            ):
                wandb_logger = [
                    lgr
                    for lgr in pl_module.loggers
                    if isinstance(lgr, pl.loggers.wandb.WandbLogger)
                ][0]
                for mertric in global_stats.keys():
                    plt.clf()
                    metric_dict = {
                        name: stat[mertric]
                        for name, stat in zip(layer_names, layer_stats)
                    }
                    values = metric_dict.values()
                    _ = plt.plot(range(len(values)), values)
                    wandb_logger.log_image(
                        f"{split}/{mertric}_per_layer",
                        [wandb.Image(plt)],
                        step=pl_module.global_step,
                    )
                    plt.clf()

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
        self.averager = Averager(weight=0.5)
        self.metrics = {}

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        # get routing attributes of all layers
        for name, module in pl_module.named_modules():
            if isinstance(module, RoutingSelector) and hasattr(module, "metrics"):
                self.metrics[name] = module.metrics

        layer_stats = list(self.metrics.values())

        global_stats = agg_dicts(layer_stats)
        global_stats = self.averager.update(global_stats)

        for k, v in global_stats.items():
            pl_module.log(
                f"train/{k}",
                v,
                on_step=True,
                sync_dist=True,
                prog_bar=True,
            )
        self.metrics.clear()
