from typing import Any, Dict

import torch
from torch import nn

from mttl.models.containers import ExpertContainer
from mttl.models.containers.selectors import (
    BatchExpertsSelectorOutput,
    Selector,
    TaskNameSelector,
)
from mttl.models.library.expert import Expert
from mttl.models.modifiers.hard_prompts import HardPrompt, HardPromptConfig


class HardPromptDecoderWrapper(nn.Module):
    def __init__(self, transformer, expert_container):
        super().__init__()

        self.transformer = transformer
        self.expert_container = expert_container
        self.transformer_prepare_inputs_for_generation = (
            transformer.prepare_inputs_for_generation
        )

    def add_expert(self, expert, action="route", is_default=False):
        return self.expert_container.add_expert(expert, action, is_default)

    # make sure we have all the attributes from `transformer`
    # by overwritting `getattr`
    def __getattr__(self, name):
        if name in ["transformer", "config"]:
            return super().__getattr__(name)
        else:
            return getattr(self.transformer, name)

    def forward(self, *args, **kwargs):
        # Should be padded (**right**)
        assert len(args) <= 1, "should have at most `input_ids`"

        if len(args) == 1:
            input_ids = args[0]
        else:
            input_ids = kwargs.pop("input_ids")

        attention_mask = kwargs.get("attention_mask", None)
        labels = kwargs.get("labels", None)

        input_ids, attention_mask, labels = self.expert_container.forward(
            input_ids, attention_mask=attention_mask, labels=labels
        )

        kwargs["attention_mask"] = attention_mask
        kwargs["input_ids"] = input_ids
        kwargs["labels"] = labels

        out = self.transformer(*(), **kwargs)
        return out

    def generate(self, *args, **kwargs):
        if len(args) == 1:
            input_ids = args[0]
        else:
            input_ids = kwargs["inputs"]
        (
            kwargs["inputs"],
            kwargs["attention_mask"],
            _,
        ) = self.expert_container.forward(input_ids, kwargs["attention_mask"])
        return self.transformer.generate(*args, **kwargs)


def add_hard_prompt_to_transformer(
    transformer,
    expert: Expert,
    action="route",
    is_default=False,
):
    expert_config = expert.expert_config

    # create a shared prompt container holding the experts
    if not isinstance(transformer, HardPromptDecoderWrapper):
        expert_container = HardPromptExpertContainer(
            expert_config,
            selector=None,
        )
        # patch the decoder
        transformer = HardPromptDecoderWrapper(transformer, expert_container)

    transformer.add_expert(
        expert,
        action=action,
        is_default=is_default,
    )
    return transformer


class HardPromptExpertContainer(ExpertContainer):
    def __init__(self, config, selector=None):
        super().__init__(config, layer=None)

        self.config = config
        self.selector: Selector = selector or TaskNameSelector()

        self.default_expert_name = None
        self.merged_expert_names = []
        self.experts = nn.ModuleDict({})

    def on_add_expert(
        self,
        expert: Expert,
        action="route",
        is_default=False,
    ) -> None:
        from mttl.models.modifiers.modify_model import get_modifier_type

        if action == "merge":
            raise ValueError("Merging is not supported for hard prompts.")

        if get_modifier_type(expert.expert_config) == "hard_prompt":
            expert_module = HardPrompt(
                expert.expert_config, prompt_init=expert.expert_weights
            )
        else:
            raise NotImplementedError("Not implemented for this modifier.")

        self.experts[expert.name] = expert_module

    def route(self, input_ids, selection, attention_mask=None, labels=None):
        if isinstance(selection, BatchExpertsSelectorOutput):
            return HardPrompt.parallel_forward(
                [self[module] for module in selection.experts],
                input_ids,
                attention_mask,
                labels,
            )
        else:
            raise ValueError("Cannot process the desired selection.")

    def __setitem__(self, key, value: HardPrompt):
        if not isinstance(value, HardPrompt):
            raise ValueError("Only `HardPrompt` instances are allowed.")
        self.experts[key] = value

    def __getitem__(self, key):
        return self.experts[key]

    def __len__(self):
        return len(self.experts)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        if len(self.experts) > 0:
            selection = self.selector(
                input_ids, attention_mask=attention_mask, labels=labels
            )
            return self.route(
                input_ids, selection, attention_mask=attention_mask, labels=labels
            )
        return input_ids, attention_mask, labels
