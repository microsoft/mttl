"""
Phatgoose transform algorithm for Expert objects.

This module implements Phatgoose transform that computes prototype vectors
for expert selection through training selector gates.
"""

import re
from typing import Dict, List

import torch

from mttl.logging import logger
from mttl.models.library.expert import Expert
from mttl.models.library.library_transforms import PhatgooseTransformConfig
from mttl.models.containers import ExpertContainer


def extract_phatgoose_prototypes(model) -> Dict[str, torch.Tensor]:
    """
    Extract Phatgoose prototypes from a trained model with selector gates.

    Args:
        model: Trained model with ExpertContainer modules containing selectors

    Returns:
        Dict mapping layer names to prototype vectors
    """
    prototypes = {}
    for name, module in model.model.named_modules():
        if isinstance(module, ExpertContainer) and hasattr(
            module.selector, "get_prototypes"
        ):
            # expand dict
            prototypes_module = {}
            for k, v in module.selector.get_prototypes().items():
                prototypes_module[f"{name}.selector.{k}.v"] = v
            prototypes = {**prototypes, **prototypes_module}

    return prototypes


def validate_phatgoose_training(
    model_state_before: Dict[str, torch.Tensor],
    model_state_after: Dict[str, torch.Tensor],
) -> bool:
    """
    Validate that Phatgoose training only updated selector gates and not frozen parameters.

    Args:
        model_state_before: Model state dict before training
        model_state_after: Model state dict after training

    Returns:
        True if training was valid, raises AssertionError otherwise
    """
    frozen_sum_before, unfrozen_sum_before = 0, 0
    frozen_sum_after, unfrozen_sum_after = 0, 0

    for key in model_state_before.keys():
        value_before = model_state_before[key]
        value_after = model_state_after[key]

        if re.match(".*selector.gates.*.v", key):
            unfrozen_sum_before += value_before.sum()
            unfrozen_sum_after += value_after.sum()
        else:
            frozen_sum_before += value_before.sum()
            frozen_sum_after += value_after.sum()

    assert (
        frozen_sum_before == frozen_sum_after
    ), "Frozen params changed during training"
    assert (
        unfrozen_sum_before != unfrozen_sum_after
    ), "Unfrozen params did not change during training"

    return True


def initialize_phatgoose_gates(model) -> Dict[str, torch.Tensor]:
    """
    Initialize Phatgoose selector gates to zero and return initial state.

    Args:
        model: Model with selector gates to initialize

    Returns:
        Dict of initial state for validation
    """
    initial_state = {}
    for key, value in model.state_dict().items():
        if re.match(".*selector.gates.*.v", key):
            assert torch.allclose(
                value, torch.zeros_like(value)
            ), "Gate should be 0 init"
            value.requires_grad = True
        else:
            value.requires_grad = False
        initial_state[key] = value.clone()

    return initial_state
