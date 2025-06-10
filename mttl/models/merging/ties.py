"""
TIES merge algorithm for Expert objects.

This module implements the TIES (Task Interference Elimination through Sparse merging) algorithm.
"""

import copy
from typing import List

import torch

from mttl.logging import logger
from mttl.models.library.expert import Expert
from mttl.models.library.library_transforms import TiesMergeConfig


def ties_merge(experts: List[Expert], config: TiesMergeConfig) -> Expert:
    """
    Merge experts using TIES merge algorithm.

    Args:
        experts: List of Expert objects to merge
        config: TiesMergeConfig containing merge parameters

    Returns:
        Expert: Merged expert
    """
    if not experts:
        raise ValueError("Cannot merge empty list of experts")

    logger.info("Averaging {} experts".format(len(experts)))

    base_expert = copy.deepcopy(experts[0])
    base_expert.name = "ties_weighted_expert"

    state_dict_keys = list(base_expert.expert_weights.keys())

    # Build n_tasks x D experts
    # TODO: No need to build this matrix, can be done 1 expert at a time
    expert_vectors = []
    for expert in experts:
        expert_vectors += [
            torch.nn.utils.parameters_to_vector(
                list(expert.expert_weights[k] for k in state_dict_keys)
            )
        ]

    expert_vectors = torch.stack(expert_vectors, dim=0)
    per_exp_th = expert_vectors.abs().quantile(1.0 - config.top_k, dim=1)
    keep_param = expert_vectors.abs() >= per_exp_th.view(-1, 1)

    mean_valid_per_task = keep_param.float().mean(1)
    assert torch.all((mean_valid_per_task - config.top_k).abs() < 1e-4)

    used, kept, total = 0, 0, 0

    for param_name in state_dict_keys:
        # stack the expert weights
        expert_weights = torch.stack(
            [expert.expert_weights[param_name] for expert in experts], dim=0
        )

        # keep weights over the threshold
        TH = per_exp_th.view(-1, *((1,) * (expert_weights.ndim - 1)))
        keep_mask = expert_weights.abs() >= TH
        expert_weights = expert_weights * keep_mask

        if config.only_sparsify:
            final_param = expert_weights.mean(0)
            used += keep_mask.sum().item()
        else:
            # sign majority vote
            sign_per_dim = expert_weights.sign().sum(0, keepdim=True).sign()
            sign_per_dim = expert_weights.sum(0, keepdim=True).sign()

            # keep only weights whose sign agree with the majority
            use_for_avg = expert_weights.sign() == sign_per_dim

            deno = use_for_avg.sum(0).clamp(min=1.0)
            sum_param = (expert_weights * use_for_avg).sum(0)
            final_param = sum_param / deno
            used += (use_for_avg & (sign_per_dim != 0.0)).sum().item()

        kept += (expert_weights.abs() > TH).sum()
        total += expert_weights.numel()

        base_expert.expert_weights[param_name].data.copy_(final_param)

    logger.info(
        "Params not reset to 0 in TIES merge: {:.10f}%".format(100.0 * kept / total)
    )
    logger.info(
        "Params used to compute TIES mean: {:.10f}%".format(100.0 * used / total)
    )

    # manually change the config of the expert to remove the tie_params
    base_expert.expert_config.tie_params = None

    return base_expert
