"""
Standalone merging functions for Expert objects.

This module provides standalone merging routines that take a list of Expert objects
as input and return a merged Expert. These functions are decoupled from library
transforms and can be used independently.
"""

import copy
from typing import List

import torch
from tqdm.auto import tqdm

from mttl.logging import logger
from mttl.models.library.expert import Expert
from mttl.models.library.library_transforms import (
    TiesMergeConfig,
    WeightedLinearMergeConfig,
    WudiMergeConfig,
)


def wudi_merge(experts: List[Expert], config: WudiMergeConfig) -> Expert:
    """
    Merge experts using WuDi merge algorithm.
    
    Args:
        experts: List of Expert objects to merge
        config: WudiMergeConfig containing merge parameters
        
    Returns:
        Expert: Merged expert
    """
    if not experts:
        raise ValueError("Cannot merge empty list of experts")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Merging {} experts using WuDi merge".format(len(experts)))

    base_expert = copy.deepcopy(experts[0])
    base_expert.name = "wudi_merged_expert"

    # Get all parameter keys that we want to merge
    keys = [key for key in base_expert.expert_weights.keys()]

    for key in keys:
        # Stack all expert weights for this parameter
        values = torch.stack([expert.expert_weights[key] for expert in experts])

        values = values.to(device)

        # Initialize merged vector as sum of all vectors
        merging_vector = torch.nn.Parameter(torch.sum(values, dim=0))
        optimizer = torch.optim.Adam(
            [merging_vector], lr=config.lr, weight_decay=0
        )

        # Compute L2 norms
        l2_norms = torch.square(
            torch.norm(values.reshape(values.shape[0], -1), p=2, dim=-1)
        )

        # Optimize merging vector
        for _ in tqdm(range(config.iter), desc=f"Optimizing parameter {key}"):
            disturbing_vectors = merging_vector.unsqueeze(0) - values
            inner_product = torch.matmul(disturbing_vectors, values.transpose(1, 2))

            loss = torch.sum(
                torch.square(inner_product) / l2_norms.unsqueeze(-1).unsqueeze(-1)
            )
            loss = loss.requires_grad_(True)  # Ensure loss requires gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        merging_vector = merging_vector / len(experts)
        # Update base expert weights with optimized merging vector
        base_expert.expert_weights[key].data.copy_(merging_vector.data.cpu())

    return base_expert


def weighted_linear_merge(experts: List[Expert], config: WeightedLinearMergeConfig) -> Expert:
    """
    Merge experts using weighted linear averaging.
    
    Args:
        experts: List of Expert objects to merge
        config: WeightedLinearMergeConfig containing merge parameters
        
    Returns:
        Expert: Merged expert
    """
    if not experts:
        raise ValueError("Cannot merge empty list of experts")
    
    expert_names = [expert.name for expert in experts]
    logger.info("Averaging {} experts".format(len(experts)))

    base_expert = copy.deepcopy(experts[0])
    base_expert.name = "weighted_expert"

    if config.weights is not None:
        assert set(config.weights.keys()) == set(
            expert_names
        ), "Weights must have the same keys as the experts"
        if not (1 - 1e-6) <= sum(config.weights.values()) <= (1 + 1e-6):
            logger.warning(
                "Weights do not sum to 1.0, please make sure this is intended"
            )

        # scale the base expert
        for k, v in base_expert.expert_weights.items():
            base_expert.expert_weights[k] *= config.weights[expert_names[0]]

    for expert in experts[1:]:
        # Validate that the expert is compatible
        assert type(expert.expert_info.expert_config) == type(
            base_expert.expert_info.expert_config
        ), "Expert configs must be the same type"
        assert set(expert.expert_weights.keys()) == set(
            base_expert.expert_weights.keys()
        ), "Expert weights must have the same keys"

        weight = 1.0
        if config.weights is not None:
            weight = config.weights[expert.expert_info.expert_name]

        for k, v in expert.expert_weights.items():
            base_expert.expert_weights[k] += v * weight

    # Normalize the final expert
    if config.weights is None:
        for k, v in base_expert.expert_weights.items():
            base_expert.expert_weights[k] /= len(experts)

    # manually change the config of the expert to remove the tie_params
    base_expert.expert_config.tie_params = None

    return base_expert


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