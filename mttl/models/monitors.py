import math
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import Callback
from torch.distributions import Bernoulli, Categorical

from mttl.models.containers.selectors.base import Selector
from mttl.utils import agg_dicts

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

    else:
        monitors += [SelectorLog(config)]

    return monitors


class PolytroponLog(Callback):
    """Log polytropon metrics of interest, sparsity / efficiency / discreteness."""

    LOG_EVERY = 500

    @torch.no_grad()
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


class SelectorLog(Callback):
    def __init__(self, args):
        self.plot_every = 10
        self.log_per_layer = True

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

    def aggregate_and_maybe_log(self, trainer, pl_module, current_step, split) -> None:
        # get routing attributes of all layers
        if current_step % self.plot_every != 0:
            return
        all_routing_gates = []
        for name, module in pl_module.named_modules():
            if isinstance(module, Selector) and hasattr(module, "routing_gates"):
                if isinstance(module.routing_gates, list):
                    gates = np.mean([torch.mean(gate) for gate in module.routing_gates])
                    all_routing_gates.append(gates.item())
                    module.routing_gates = []
                else:
                    continue
        if len(all_routing_gates) > 0:
            # log and empty routing_gates
            # we log averaged gates across samples and layers
            if wandb.run is not None:
                wandb_logger = [
                    lgr
                    for lgr in pl_module.loggers
                    if isinstance(lgr, pl.loggers.wandb.WandbLogger)
                ][0]
                prefix = (
                    f"{pl_module._log_pref}" if hasattr(pl_module, "_log_pref") else ""
                )
                wandb_logger.log_metrics(
                    {f"{prefix}{split}/routing_gates": np.mean(all_routing_gates)}
                )
                if self.log_per_layer:
                    plt.clf()
                    _ = plt.plot(range(len(all_routing_gates)), all_routing_gates)
                    wandb_logger.log_image(
                        f"{prefix}{split}/routing_gates_per_layer",
                        [wandb.Image(plt)],
                        step=pl_module.global_step,
                    )
