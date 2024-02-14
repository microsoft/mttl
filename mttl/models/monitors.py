import torch
import numpy as np
import math
from torch.distributions import Bernoulli, Categorical
from pytorch_lightning import Callback
from mttl.utils import agg_dicts
from mttl.models.modifiers.poly import PolytroponSelector


def get_monitors(config):
    monitors = []
    if (
        config.model_modifier
        and "poly" in config.model_modifier
        and config.router_selector
        and "poly" in config.router_selector
    ):
        monitors += [PolytroponLog(), PolytroponUniformInterpolation()]

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


class PolytroponUniformInterpolation(Callback):
    """ Bring Z towards Uniform distribution """

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if trainer.global_step == 0 or pl_module.config.uniform_interp_steps == 0: 
            return
        
        n_steps_left = pl_module.config.total_steps - trainer.global_step

        # how many steps in are we ? if this negative, we have not started
        n_steps_interp = pl_module.config.uniform_interp_steps - n_steps_left
        should_interp = n_steps_interp >= 0

        if not should_interp: 
            return

        if n_steps_interp == 0:
            # activate 
            alpha_container = {'alpha': None}
            pl_module.alpha_container = alpha_container

            for module in pl_module.modules():
                if isinstance(module, PolytroponSelector):
                    module.interpolate_to_uniform(alpha_container)

        # cosine(0) = 1
        # cosine(pi / 2) = 0
        interp_completed_frac = n_steps_interp / pl_module.config.uniform_interp_steps

        # alpha = 1 means use only Z, alpha = 0 is Uniform dist
        # to anneal from Z to U, we want alpha from 1 to 0
        alpha = np.cos(interp_completed_frac * np.pi / 2)

        pl_module.alpha_container['alpha'] = alpha
        pl_module.log('train/alpha', alpha, on_epoch=True, on_step=True, sync_dist=True)