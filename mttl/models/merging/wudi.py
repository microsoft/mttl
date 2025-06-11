"""
WuDi (Weight Disturbance) merge algorithm for Expert objects.

This module implements the WuDi merge algorithm from https://arxiv.org/pdf/2503.08099v1
"""

import copy
from typing import List

import torch
from tqdm.auto import tqdm

from mttl.logging import logger
from mttl.models.library.expert import Expert
from mttl.models.library.library_transforms import WudiMergeConfig


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
        merging_vector = torch.nn.Parameter(
            torch.sum(values, dim=0), requires_grad=True
        )
        optimizer = torch.optim.Adam([merging_vector], lr=config.lr, weight_decay=0)

        # Compute L2 norms
        l2_norms = torch.square(
            torch.norm(values.reshape(values.shape[0], -1), p=2, dim=-1)
        )

        # Optimize merging vector
        pbar = tqdm(range(config.iter), desc=f"Optimizing parameter {key}")
        prev_loss = float("inf")
        patience = 5  # Number of steps to wait for improvement
        no_improve_count = 0
        min_delta = 1e-4  # Minimum change in loss to be considered improvement

        for step in pbar:
            disturbing_vectors = merging_vector.unsqueeze(0) - values
            inner_product = torch.matmul(disturbing_vectors, values.transpose(1, 2))

            loss = torch.sum(
                torch.square(inner_product) / l2_norms.unsqueeze(-1).unsqueeze(-1)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Check if loss improvement is significant
            if abs(prev_loss - loss.item()) < min_delta:
                no_improve_count += 1
            else:
                no_improve_count = 0

            # Early stopping if no significant improvement for patience steps
            if no_improve_count >= patience:
                logger.info(f"Early stopping at step {step} due to minimal loss change")
                break

            prev_loss = loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        merging_vector = merging_vector / len(experts)
        # Update base expert weights with optimized merging vector
        base_expert.expert_weights[key].data.copy_(merging_vector.data.cpu())

    return base_expert
