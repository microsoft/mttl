import torch
from torch import nn
from typing import Any, Dict
from mttl.models.modifiers.base import ModifyMixin

from mttl.models.containers.selectors import (
    Selector,
    TaskNameSelector,
    BatchExpertsSelectorOutput,
)
from mttl.models.containers import ExpertContainer
from mttl.models.modifiers.hard_prompts import HardPrompt, HardPromptConfig
from mttl.models.modifiers.modify_model import register_modifier
from mttl.models.library.expert import Expert


class HardPromptDecoderWrapper(nn.Module):
    def __init__(self, transformer, expert_container):
        super().__init__()

        self.transformer = transformer
        self.expert_container = expert_container
        self.transformer_prepare_inputs_for_generation = (
            transformer.prepare_inputs_for_generation
        )

    def add_expert(self, *args, **kwargs):
        return self.expert_container.add_expert(*args, **kwargs)

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
    expert_weights = expert.expert_weights

    # create a shared prompt container holding the experts
    if not isinstance(transformer, HardPromptDecoderWrapper):
        expert_container = HardPromptExpertContainer(
            expert_config,
            transformer.info_container,
            selector=None,
        )
        # patch the decoder
        transformer = HardPromptDecoderWrapper(transformer, expert_container)

    transformer.add_expert(
        expert,
        expert_weights,
        action=action,
        is_default=is_default,
    )
    return transformer


class HardPromptExpertContainer(ExpertContainer):
    def __init__(self, config, info_container, selector=None):
        super().__init__(config, info_container, layer=None)

        self.config = config
        self.selector: Selector = selector or TaskNameSelector(info_container)

        self.default_expert_name = None
        self.merged_expert_names = []
        self.experts = nn.ModuleDict({})

    def add_expert(
        self,
        expert: Expert,
        expert_weights: Dict[str, torch.Tensor],
        action="route",
        is_default=False,
    ) -> None:
        from mttl.models.modifiers.modify_model import get_modifier_type

        if expert.name in self.experts:
            raise ValueError(
                "An expert with name {} already exists.".format(expert.name)
            )

        if action == "merge":
            raise ValueError("Merging is not supported for hard prompts.")

        if is_default:
            self.default_expert_name = expert.name

        if get_modifier_type(expert.expert_config) == "hard_prompt":
            expert_module = HardPrompt(expert.expert_config, prompt_init=expert_weights)
        else:
            raise NotImplementedError("Not implemented for this modifier.")

        self.experts[expert.name] = expert_module
        self.add_expert_to_selector(expert.name, expert_info=expert.expert_info)

    def route(self, input_ids, selection, attention_mask=None, labels=None):
        if isinstance(selection, BatchExpertsSelectorOutput):
            return HardPrompt.parallel_forward(
                [self.get(module) for module in selection.experts],
                input_ids,
                attention_mask,
                labels,
            )
        else:
            raise ValueError("Cannot process the desired selection.")

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
