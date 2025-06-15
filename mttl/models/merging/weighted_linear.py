"""
Weighted linear merge algorithm for Expert objects.

This module implements weighted linear averaging of expert parameters.
"""

import copy
from typing import List

from mttl.logging import logger
from mttl.models.library.expert import Expert
from mttl.models.library.library_transforms import WeightedLinearMergeConfig


def weighted_linear_merge(
    experts: List[Expert], config: WeightedLinearMergeConfig
) -> Expert:
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
