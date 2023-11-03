import torch
import numpy as np
import math
from torch.distributions import Bernoulli, Categorical
from pytorch_lightning import Callback
from mttl.utils import agg_dicts


def get_monitors(config):
    monitors = []
    if (
        config.model_modifier
        and "poly" in config.model_modifier
        and config.router_selector
        and "poly" in config.router_selector
    ):
        monitors += [PolytroponLog()]
    if "llama_adapter" in config.model_modifier:
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
        stats = {"encoder": [], "decoder": []}

        seen = 0
        for coder in stats.keys():
            mod = getattr(pl_module.model, coder)
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
