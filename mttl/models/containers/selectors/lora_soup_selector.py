from mttl.models.containers.selectors.base import (
    ExpertsAndWeightsSelectorOutput,
    BatchSequenceExpertsAndWeightsSelectorOutput,
    forward_with_cache,
    SelectorConfig,
    Selector,
)
from dataclasses import dataclass
import torch.nn as nn
import torch
from mttl.models.containers.selectors.lora_merge_eign_input_analysis import (
    AdaptiveLoRAMerger,
)


@dataclass
class LoraSoupSelectorConfig(SelectorConfig):
    pass


@Selector.register("lora_soup_router", LoraSoupSelectorConfig)
class LoraSoupSelector(Selector):
    """
    LoraSoupSelector is a selector that uses a learnable routing mechanism to select experts.
    refer to the code: https://github.com/aksh555/LoRA-Soups
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.softmax = nn.Softmax(dim=-1)
        self.module_logits_dict = nn.ParameterDict()

    def _get_weights(self):
        wts_before_softmax = torch.cat(list(self.module_logits_dict.values()), dim=0)
        wts_after_softmax = self.softmax(wts_before_softmax)
        return wts_after_softmax

    def on_add_expert(self, expert_name, expert_info=None, is_default=False):
        if expert_name not in self.module_logits_dict:
            self.module_logits_dict[expert_name] = torch.nn.Parameter(
                torch.randn(1).to(self.device)
            )

    @forward_with_cache
    def forward(self, input, **kwargs) -> ExpertsAndWeightsSelectorOutput:
        # Get base weights from learned routing
        weights = self._get_weights()
        # Get expert adapters
        experts = list(self.module_logits_dict.keys())
        # print(self.layer_name, proj_coeffs)
        return ExpertsAndWeightsSelectorOutput(experts, weights)


@dataclass
class LoraSoupSelectorEignConfig(SelectorConfig):
    pass


@Selector.register("lora_soup_router_eign", LoraSoupSelectorEignConfig)
class LoraSoupSelectorEign(LoraSoupSelector):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.merger = None
        self.merger_cache = {}

    @forward_with_cache
    def forward(self, input, **kwargs) -> ExpertsAndWeightsSelectorOutput:
        if self.layer_name in self.merger_cache:
            # print(f"Using cached merger for {self.layer_name}")
            self.merger = self.merger_cache[self.layer_name]
        else:
            # print(f"Computing merger for {self.layer_name}")
            lora_a = kwargs["container"].lora_a
            lora_b = kwargs["container"].lora_b
            # Compute different coefficient methods
            math_lora_A = lora_a["zhan1993_mathqa_trained_from_lorasoup"]
            math_lora_B = lora_b["zhan1993_mathqa_trained_from_lorasoup"]
            code_lora_A = lora_a["zhan1993_code_trained_from_lorasoup"]
            code_lora_B = lora_b["zhan1993_code_trained_from_lorasoup"]
            self.merger = AdaptiveLoRAMerger(
                math_lora_A.T, math_lora_B.T, code_lora_A.T, code_lora_B.T
            )
            self.merger_cache[self.layer_name] = self.merger
        proj_coeffs = self.merger.compute_projection_based_coefficients(input, top_k=1)
        weights = torch.tensor(proj_coeffs).to(self.device)
        experts = list(self.module_logits_dict.keys())
        return ExpertsAndWeightsSelectorOutput(experts, weights)


@dataclass
class LoraSoupPriorSelectorConfig(SelectorConfig):
    pass


@Selector.register("lora_soup_prior_router", LoraSoupPriorSelectorConfig)
class LoraSoupPriorSelector(LoraSoupSelector):
    """
    LoraSoupPriorSelector extends LoraSoupSelector to incorporate prior routing information.
    It blends the learned routing weights with prior routing weights using a learnable alpha parameter.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # Learnable parameter to control blending between prior and current routing
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def _get_weights(self, prior_weights=None):
        # Get base weights from parent class
        current_weights = super()._get_weights()

        if prior_weights is not None:
            # Blend prior and current weights using learnable alpha
            weights = self.alpha * prior_weights + (1 - self.alpha) * current_weights
        else:
            weights = current_weights

        return weights

    @forward_with_cache
    def forward(
        self, input, prior_weights=None, **kwargs
    ) -> ExpertsAndWeightsSelectorOutput:
        weights = self._get_weights(prior_weights)
        experts = list(self.module_logits_dict.keys())
        return ExpertsAndWeightsSelectorOutput(experts, weights)
