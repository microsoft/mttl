import torch
import numpy as np
import math
from torch.distributions import Bernoulli, Categorical
from pytorch_lightning import Callback
from mttl.utils import average_dicts


def get_monitors(config):
    monitors = []
    if config.model_modifier and "poly" in config.model_modifier and "poly" in config.poly_selector:
        monitors += [PolytroponLog()]

    return monitors


class PolytroponLog(Callback):
    """Log polytropon metrics of interest, sparsity / efficiency / discreteness."""

    LOG_EVERY = 500

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if (
            trainer.global_step == 0
            or trainer.global_step % self.LOG_EVERY > 0
        ):
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
            if hasattr(pl_module.model, coder):
                mod = getattr(pl_module.model, coder)
                for module in mod.modules():
                    if hasattr(module, "module_logits"):
                        stats[coder] += [layer_stats(module.module_logits)]
                        seen += 1

                # average over layers
                if len(stats[coder]):
                    stats[coder] = average_dicts(stats[coder])

                    for k, v in stats[coder].items():
                        pl_module.log(
                            f"Z/{coder}.{k}", v, on_epoch=True, on_step=True, sync_dist=True
                        )
