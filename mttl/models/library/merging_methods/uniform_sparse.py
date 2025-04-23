import torch
import numpy as np
from dataclasses import dataclass
from huggingface_hub import hf_hub_download
from mttl.models.library.merging_methods.base_merge import (
    BaseMerge,
    BaseMergeConfig,
)
from mttl.models.expert_model import ExpertModel, ExpertModelConfig
from mttl.models.lightning.expert_module import ExpertModule


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
        self, experts, trainable_params, base_expert, base_model_state_dict, expert_type
    ):
        param_dict = {}
        for param_name, base_w in base_expert.model.state_dict().items():
            if param_name in trainable_params:
                # ignore bias
                if "weight" in param_name:
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

                    layer_name = ".".join(param_name.split(".")[:-2])
                    updated_param_name = f"{layer_name}.weight"
                    param_dict[updated_param_name] = final_param

        return param_dict

    @torch.no_grad()
    def transform(self, library):
        experts, expert_type, base_expert, base_model, trainable_params = (
            self.pre_configure(library)
        )
        an_expert = library[next(iter(library.keys()))]
        base_model_state_dict = dict(base_model.state_dict())
        base_expert.training_config["device_map"] = "cpu"
        # base_expert = ExpertModel(ExpertModelConfig(base_model=base_model),
        #                           **base_expert.training_config)
        base_expert = ExpertModule(**base_expert.training_config)
        trainable_params = [
            n for n in an_expert.expert_weights.keys() if ("sparse_layer" in n)
        ]
        assert trainable_params != [], print("could not find sparse-layer modules")
        base_model_state_dict = base_expert.model.state_dict()

        param_dict = self.merge_expert(
            experts, trainable_params, base_expert, base_model_state_dict, expert_type
        )

        config = base_expert.training_config
        config.model_modifier = None  # load only the base model
        config.device_map = "cpu"
        config.trainable_param_names = ".*"  # allows to train all linear layers
        merged_model = ExpertModel(
            ExpertModelConfig(base_model=base_model), **vars(config)
        )

        for param_name, base_w in merged_model.state_dict().items():
            if param_name in param_dict:
                param_dict[param_name] = base_w + param_dict[param_name].to(
                    base_w.dtype
                )
            else:
                param_dict[param_name] = base_w

        assert set(merged_model.state_dict().keys()) == set(
            param_dict.keys()
        ), "Expert weights must have the same keys"
        merged_model.load_state_dict(param_dict)

        return merged_model
