import copy
import torch
from dataclasses import dataclass
from mttl.models.library.merge_models.base_merge import BaseMerge, BaseMergeConfig
from mttl.models.library.expert_library import ExpertLibrary
from mttl.models.library.merge_models.utils import topk_multiple_experts
from mttl.models.utils import logger


@dataclass
class ModelBreadcrumbsConfig(BaseMergeConfig):
    merging_method: str = "model_breadcrumbs_expert"
    alpha: float = (
        0.4  # scaling factor # source : (a) https://openreview.net/pdf?id=6t0Kwf8-jrj (b) https://arxiv.org/pdf/2306.01708
    )
    beta: float = 0.9  # 90% beta=sparsity, keep-ratio=1-beta
    gamma: float = 0.99  # mask out top 1%


class ModelBreadcrumbs(BaseMerge):
    """
    Computes a uniform weight mixture across experts of a given library
    """

    def __init__(self, config: ModelBreadcrumbsConfig = None):
        super().__init__(config or ModelBreadcrumbsConfig())

    @torch.no_grad()
    def compute_per_task_threhold(self, expert_vectors):

        # take the absolute value:
        expert_vectors = torch.stack(expert_vectors, dim=0).abs()
        lower_topk = int(expert_vectors.size(1) * self.config.beta)
        upper_topk = int(expert_vectors.size(1) * self.config.gamma)

        per_exp_lth = topk_multiple_experts(expert_vectors, lower_topk, TH_type="lower")
        per_exp_uth = topk_multiple_experts(expert_vectors, upper_topk, TH_type="upper")

        return per_exp_lth, per_exp_uth

    @torch.no_grad()
    def merge_expert(
        self,
        experts,
        expert_vectors,
        trainable_params,
        base_expert,
        base_model_state_dict,
        expert_type,
    ):
        # Compute Threshold score, TH
        per_exp_lth, per_exp_uth = self.compute_per_task_threhold(expert_vectors)
        used, total = 0, 0
        for param_name in base_model_state_dict.keys():
            if param_name in trainable_params:
                # stack the expert weights
                expert_weights = self.extract_expert_weight(
                    base_model_state_dict, experts, param_name, expert_type
                )

                # keep weights over the threshold
                Lower_TH = per_exp_lth.view(-1, *((1,) * (expert_weights.ndim - 1)))
                Upper_TH = per_exp_uth.view(-1, *((1,) * (expert_weights.ndim - 1)))

                keep_mask = torch.logical_and(
                    expert_weights.abs() > Lower_TH, expert_weights.abs() < Upper_TH
                )
                # keep_mask = (expert_weights.abs() > Lower_TH and expert_weights.abs() < Upper_TH)
                expert_weights = expert_weights * keep_mask

                # base_weight + sum of the "filtered" task-vector
                final_param = base_model_state_dict[
                    param_name
                ] + self.config.alpha * expert_weights.sum(0)

                used += keep_mask.sum().item()
                total += expert_weights.numel()

                base_expert.expert_weights[param_name].data.copy_(final_param)
            else:
                base_expert.expert_weights[param_name] = copy.deepcopy(
                    base_model_state_dict[param_name]
                )
        logger.info(
            "Params used to compute Model-breadcrumb mean: {:.10f}%".format(
                100.0 * used / total
            )
        )

        return base_expert
