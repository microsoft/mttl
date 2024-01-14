import torch
import numpy as np
import math
from torch.distributions import Bernoulli, Categorical
from pytorch_lightning import Callback
from mttl.utils import agg_dicts
from typing import Any

import pytorch_lightning as pl
import matplotlib.pyplot as plt
from torch.distributions import Categorical
from pytorch_lightning import Callback

from mttl.utils import agg_dicts, Averager
from mttl.models.modifiers.routing import RoutingSelector
from mttl.models.modifiers.poly import PolytroponSelector, PerTokenPolytroponSelector

try:
    import wandb
except:
    pass


def get_monitors(config):
    monitors = []
    if (
        config.model_modifier
        and "poly" in config.model_modifier
        and (
            (config.router_selector and "poly" in config.router_selector)
            or config.router_selector is None
        )
    ):
        monitors += [PolytroponLog()]

    if config.model_modifier and "llama_adapter" in config.model_modifier:
        monitors += [AlphaLog()]

    return monitors


class PolytroponLog(Callback):
    """Log polytropon metrics of interest, sparsity / efficiency / discreteness."""

    LOG_EVERY = 500

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if trainer.global_step == 0 or trainer.global_step % self.LOG_EVERY > 0:
            return

        def layer_stats(Z):
            prob = torch.sigmoid(Z)
            discreteness = (
                Bernoulli(logits=Z).entropy().sum().item()
                / np.log(2)
                / np.prod(Z.shape)
            )
            sparsity = (prob + 0.5).floor().mean()
            categorical = prob.mean(0) / prob.mean(0).sum()
            eff = (
                Categorical(probs=categorical).entropy() / math.log(Z.size(-1))
            ).item()

            return {
                "sparsity": sparsity,
                "discreteness_fixed": discreteness,
                "eff_fixed": eff,
            }

        # iterate over encoder and decoder layers
        model_family = getattr(pl_module.model.config, "model_family", "gpt")
        if model_family == "encdec":
            stats = {"encoder": [], "decoder": []}
        elif model_family == "gpt":
            stats = {"": []}

        seen = 0
        for coder in stats.keys():
            if len(coder) > 0:
                mod = getattr(pl_module.model, coder)
            else:
                mod = pl_module.model

            for module in mod.modules():
                if hasattr(module, "module_logits"):
                    stats[coder] += [layer_stats(module.module_logits)]
                    seen += 1

            # average over layers
            if len(stats[coder]):
                stats[coder] = agg_dicts(stats[coder])

                for k, v in stats[coder].items():
                    pl_module.log(
                        f"Z/{coder}.{k}", v, on_epoch=True, on_step=True, sync_dist=True
                    )

        # Finally, log seen task information
        counts = None
        if PolytroponSelector.seen_samples_per_task is not None:
            counts = PolytroponSelector.seen_samples_per_task
        elif PerTokenPolytroponSelector.seen_samples_per_token is not None:
            counts = PerTokenPolytroponSelector.seen_samples_per_token

        if counts is not None:
            is_seen = counts > 0
            n_seen, n_unseen = is_seen.sum(), (~is_seen).sum()
            seen_tasks = counts[is_seen]
            seen_min, seen_max = seen_tasks.min(), seen_tasks.max()
            to_log = {
                "n_seen": n_seen,
                "n_unseen": n_unseen,
                "seen_min": seen_min,
                "seen_max": seen_max,
            }
            for k, v in to_log.items():
                pl_module.log(
                    f"Z/{k}",
                    v.float().item(),
                    on_epoch=True,
                    on_step=True,
                    sync_dist=True,
                )


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


class AlphaLog(Callback):
    LOG_EVERY = 5

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if trainer.global_step == 0 or trainer.global_step % self.LOG_EVERY > 0:
            return

        gate_values = []
        for mod in pl_module.modules():
            if hasattr(mod, "adapter_gate"):
                gate_values += [mod.adapter_gate.mean().item()]

        to_log = {
            "train/alpha_mean": sum(gate_values) / len(gate_values),
            "train/alpha_max": max(gate_values),
            "train/alpha_min": min(gate_values),
        }

        pl_module.log_dict(
            to_log,
            on_epoch=True,
            on_step=True,
            sync_dist=True,
        )
