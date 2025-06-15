"""
WuDi (Weight Disturbance) merge algorithm for Expert objects.

This module implements the WuDi merge algorithm from https://arxiv.org/pdf/2503.08099v1
"""

import copy
from typing import Dict, List

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


def wudi_merge_after(
    experts: List[Expert], config: WudiMergeConfig
) -> Dict[str, torch.Tensor]:
    """
    Merge experts using WuDi merge algorithm after computing task vectors (LoRA A @ LoRA B).

    This variant computes the outer product of LoRA A and B matrices first, then applies
    WuDi merge to the resulting task vectors for each layer.

    Args:
        experts: List of Expert objects to merge
        config: WudiMergeConfig containing merge parameters

    Returns:
        Dict[str, torch.Tensor]: Dictionary mapping layer names to merged task vectors
    """
    if not experts:
        raise ValueError("Cannot merge empty list of experts")

    logger.info("Merging {} experts using WuDi merge after".format(len(experts)))

    one_expert = experts[0]
    # Get the layer names from the model
    layer_names = [name.split(".lora_")[0] for name in one_expert.expert_weights.keys()]
    layer_names = list(set(layer_names))

    # Get the task vectors for each expert
    task_vectors_experts = {}
    for expert in experts:
        task_vectors = _get_task_vectors(expert)
        task_vectors_experts[expert.name] = task_vectors

    task_merged_vectors = {}
    # WuDi merge the task vectors
    for layer in layer_names:
        # Get the experts for this layer
        task_vectors = [task_vectors_experts[expert.name][layer] for expert in experts]

        task_vectors = torch.stack(task_vectors, dim=0)
        # Get the redundant task vector
        merged_task_vector = _get_optimized_task_vector(
            layer_name=layer,
            task_vectors=task_vectors,
            iter=config.iter,
            lr=config.lr,
        )

        # Save the merged task vector in each layer
        task_merged_vectors[layer] = merged_task_vector / len(experts)

    return task_merged_vectors


def _get_task_vectors(expert: Expert) -> Dict[str, torch.Tensor]:
    """
    Get the incremental weights for each layer, LoRA A outer product LoRA B.

    Args:
        expert: Expert object containing LoRA weights

    Returns:
        Dict[str, torch.Tensor]: Dictionary mapping layer names to task vectors
    """
    task_vectors = {}
    for key in expert.expert_weights.keys():
        base_layer_name = key.split(".lora_")[
            0
        ]  # Get base layer name by removing .lora_a or .lora_b
        if base_layer_name not in task_vectors:
            task_vectors[base_layer_name] = None

    for layer in task_vectors.keys():
        lora_a = expert.expert_weights[f"{layer}.lora_a"]
        lora_b = expert.expert_weights[f"{layer}.lora_b"]
        task_vectors[layer] = lora_a.data @ lora_b.data

    return task_vectors


def _get_optimized_task_vector(
    layer_name: str, task_vectors: torch.Tensor, iter: int, lr: float
) -> torch.Tensor:
    """
    Minimize Σᵢ (1/||τᵢ,ₗ||²F) ||(τₘ,ₗ - τᵢ,ₗ)(τᵢ,ₗ)ᵀ||²F

    Return the optimized merged task vector for each layer.

    Args:
        layer_name: Name of the layer being optimized
        task_vectors: Stacked task vectors for the layer
        iter: Number of optimization iterations
        lr: Learning rate for optimization

    Returns:
        torch.Tensor: Optimized merged task vector
    """
    task_vectors = task_vectors.cuda()
    merging_vector = torch.nn.Parameter((torch.sum(task_vectors, dim=0)))
    optimizer = torch.optim.Adam([merging_vector], lr=lr, weight_decay=0)

    l2_norms = torch.square(
        torch.norm(task_vectors.reshape(task_vectors.shape[0], -1), p=2, dim=-1)
    )

    pbar = tqdm(range(iter), desc=f"Optimizing parameter {layer_name}")
    prev_loss = float("inf")
    patience = 5  # Number of steps to wait for improvement
    no_improve_count = 0
    min_delta = 1e-4  # Minimum change in loss to be considered improvement

    for step in pbar:
        disturbing_vectors = merging_vector.unsqueeze(0) - task_vectors
        inner_product = torch.matmul(disturbing_vectors, task_vectors.transpose(1, 2))
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

    return merging_vector
