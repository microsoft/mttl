import copy
import torch
from dataclasses import dataclass
from mttl.models.library.merge_models.base_merge import BaseMerge, BaseMergeConfig
from mttl.models.library.expert_library import ExpertLibrary
from mttl.models.library.merge_models.utils import topk_multiple_experts
from mttl.models.utils import logger


@dataclass
class TiesMergeSimpleConfig(BaseMergeConfig):
    top_k: float = 0.2
    merging_method: str = "ties_merge_expert"
    alpha: float = (
        0.4  # scaling factor # source : (a) https://openreview.net/pdf?id=6t0Kwf8-jrj (b) https://arxiv.org/pdf/2306.01708
    )
    beta: float = (
        0.8  # 80% beta=sparsity, keep-ratio=1-beta, fig 3 https://arxiv.org/pdf/2306.01708 suggest to keep top20% params
    )


class TiesMergeSimple(BaseMerge):
    """
    Computes a uniform weight mixture across experts of a given library
    """

    def __init__(self, config: TiesMergeSimpleConfig = None):
        super().__init__(config or TiesMergeSimpleConfig())

        assert self.config.top_k > 0.0 and self.config.top_k <= 1.0

    def compute_per_task_threhold(self, expert_vectors):
        # take the absolute value:
        expert_vectors = torch.stack(expert_vectors, dim=0).abs()
        topk = int(expert_vectors.size(1) * self.config.beta)
        per_exp_lth = topk_multiple_experts(expert_vectors, topk, TH_type="lower")

        return per_exp_lth

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
        # ----------------------------------------------------------------------
        # Compute Threshold score, TH
        per_exp_lth = self.compute_per_task_threhold(expert_vectors)

        used, total = 0, 0
        for param_name in base_model_state_dict.keys():
            if param_name in trainable_params:
                # stack the expert weights
                expert_weights = self.extract_expert_weight(
                    base_model_state_dict, experts, param_name, expert_type
                )

                # keep weights over the threshold
                TH = per_exp_lth.view(
                    -1, *((1,) * (expert_weights.ndim - 1))
                )  # reshape
                keep_mask = expert_weights.abs() > TH
                expert_weights = expert_weights * keep_mask

                # sign majority vote
                # sign_per_dim = expert_weights.sign().sum(0, keepdim=True).sign()
                sign_per_dim = expert_weights.sum(0, keepdim=True).sign()
                # resolve zero signs: https://github.com/rezazzr/breadcrumbs/blob/main/src/task_vectors.py#L334
                majority_sign = torch.sign(sign_per_dim.sum())
                sign_per_dim[sign_per_dim == 0] = majority_sign

                # keep only weights whose sign agree with the majority
                use_for_avg = expert_weights.sign() == sign_per_dim

                deno = (use_for_avg != 0).sum(0).clamp(min=1.0)
                sum_param = (expert_weights * use_for_avg).sum(0)
                final_param = sum_param / deno
                used += (use_for_avg & (sign_per_dim != 0.0)).sum().item()

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
