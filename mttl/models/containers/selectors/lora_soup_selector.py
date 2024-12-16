from mttl.models.containers.selectors.base import (
    ExpertsAndWeightsSelectorOutput,
    forward_with_cache,
    SelectorConfig,
    Selector,
)
from dataclasses import dataclass
import torch.nn as nn
import torch


@dataclass
class LoraSoupSelectorConfig(SelectorConfig):
    pass


@Selector.register("lora_soup_selector", LoraSoupSelectorConfig)
class LoraSoupSelector(Selector):
    """
    LoraSoupSelector is a selector that uses a learnable routing mechanism to select experts.
    refer to the code: https://github.com/aksh555/LoRA-Soups
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.lora_learnable_weights = nn.Parameter(torch.randn(1), requires_grad=True)
        self.softmax = nn.Softmax(dim=-1)
        self.module_logits_dict = nn.ParameterDict()

    def _get_weights(self):
        wts_before_softmax = torch.cat(
            list(self.lora_learnable_weights.values()), dim=0
        )
        wts_after_softmax = self.softmax(wts_before_softmax)
        return wts_after_softmax

    def on_add_expert(self, expert_name, expert_info=None, is_default=False):
        if expert_name not in self.module_logits_dict:
            # to do: we did not use the logits here.
            self.module_logits_dict[expert_name] = torch.nn.Parameter(
                torch.randn(1).to(self.device)
            )

    @forward_with_cache
    def forward(self, input, **kwargs) -> ExpertsAndWeightsSelectorOutput:
        weights = self._get_weights()
        experts = list(self.module_logits_dict.keys())
        return ExpertsAndWeightsSelectorOutput(experts, weights)
