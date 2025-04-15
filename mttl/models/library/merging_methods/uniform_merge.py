from __future__ import annotations
import copy
from dataclasses import dataclass
import torch
from mttl.models.library.merging_methods.base_merge import (
    BaseMerge,
    BaseMergeConfig,
)
from mttl.logging import logger


@dataclass
class UniformMergeConfig(BaseMergeConfig):
    merging_method: str = "uniform_merge_expert"
    alpha: float = 1.0


class UniformMerge(BaseMerge):
    """
    Computes a uniform weight mixture across experts of a given library
    """

    def __init__(self, config: UniformMergeConfig = None):
        super().__init__(config or UniformMergeConfig())

    def merge_expert(
        self,
        experts,
        expert_vectors,
        trainable_params,
        base_expert,
        base_model_state_dict,
        expert_type,
    ):
        used, total = 0, 0
        for param_name in base_model_state_dict.keys():
            if param_name in trainable_params:
                # stack the expert weights
                expert_weights = self.extract_expert_weight(
                    base_model_state_dict, experts, param_name, expert_type
                )

                # collect mask
                keep_mask = torch.ones_like(expert_weights)
                # keep weights over the threshold
                expert_weights = expert_weights * keep_mask
                # uniform
                final_param = expert_weights.mean(0)

                # -----------------------------------------------------
                # base_weight + sum of the "filtered" task-vector
                # W = W + delta_W
                # source : (a) https://openreview.net/pdf?id=6t0Kwf8-jrj (b) https://arxiv.org/pdf/2306.01708
                final_param = (
                    base_model_state_dict[param_name] + self.config.alpha * final_param
                )

                used += keep_mask.sum().item()
                total += expert_weights.numel()

                base_expert.expert_weights[param_name].data.copy_(final_param)
            else:
                base_expert.expert_weights[param_name] = copy.deepcopy(
                    base_model_state_dict[param_name]
                )

        logger.info(
            "Params used to compute Ties mean: {:.10f}%".format(100.0 * used / total)
        )
        return base_expert
