import torch
import numpy as np
import copy
from dataclasses import dataclass
from huggingface_hub import hf_hub_download
from mttl.models.library.merging_methods.base_merge import (
    BaseMerge,
    BaseMergeConfig,
)
from mttl.models.expert_model import ExpertModel, ExpertModelConfig
from mttl.models.lightning.expert_module import ExpertModule
from mttl.models.utils import model_loader_helper
from mttl.models.library.merging_methods.utils import (
    load_mask,
    convert_idx_2_mask,
    dict_to_config,
)


@dataclass
class UniformSparsConfig(BaseMergeConfig):
    top_k: float = 0.2
    merging_method: str = "uniform_sparse_merge_expert"
    alpha: float = (
        1  # scaling factor # source : (a) https://openreview.net/pdf?id=6t0Kwf8-jrj (b) https://arxiv.org/pdf/2306.01708
    )


class UniformSparse(BaseMerge):
    """
    Computes a uniform weight mixture across experts of a given library
    """

    def __init__(self, config: UniformSparsConfig = None):
        super().__init__(config or UniformSparsConfig())
        assert self.config.top_k > 0.0 and self.config.top_k <= 1.0

    @torch.no_grad()
    def merge_expert(
        self,
        experts,
        trainable_params,
        base_expert,
        base_model_state_dict,
        expert_type,
    ):
        for param_name in base_model_state_dict.keys():
            if param_name in trainable_params:
                # stack the expert weights
                expert_weights = self.extract_expert_weight(
                    base_model_state_dict, experts, param_name, expert_type
                )

                sum_param = expert_weights.sum(0)
                mask_overlaps = torch.stack(
                    [(e != 0).float() for e in expert_weights], dim=0
                ).sum(0)

                mask_overlaps[mask_overlaps == 0] = 1
                final_param = sum_param / mask_overlaps
                final_param = (
                    base_model_state_dict[param_name] + self.config.alpha * final_param
                )

                base_expert.expert_weights[param_name].data.copy_(final_param)
            else:
                base_expert.expert_weights[param_name] = copy.deepcopy(
                    base_model_state_dict[param_name]
                )

        return base_expert

    @torch.no_grad()
    def transform(self, library):
        experts, expert_type, base_expert, trainable_params = self.pre_configure(
            library
        )

        train_cfg = base_expert.training_config
        if isinstance(train_cfg, dict):
            train_cfg = dict_to_config(train_cfg)
        # change the config to load the base model
        train_cfg.model_modifier = None  # load only the base model
        train_cfg.device_map = "cpu"
        train_cfg.trainable_param_names = ".*"  # change trainable param to all
        base_model = model_loader_helper(
            train_cfg.model,
            load_in_8bit=train_cfg.load_in_8bit,
            load_in_4bit=train_cfg.load_in_4bit,
            device_map=getattr(train_cfg, "device_map", "cpu"),
        )
        # wrap base-model with `ExpertModel` class
        base_model = ExpertModel(
            ExpertModelConfig(base_model=base_model), **vars(train_cfg)
        )
        base_model_state_dict = dict(base_model.state_dict())

        base_expert = self.merge_expert(
            experts, trainable_params, base_expert, base_model_state_dict, expert_type
        )
        # load state_dict into model
        assert set(base_model.state_dict().keys()) == set(
            base_expert.expert_weights.keys()
        ), "Expert weights must have the same keys"
        base_model.load_state_dict(base_expert._expert_weights)
        return base_model
