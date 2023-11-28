import torch
from torch import nn
from typing import Any, Dict
from mttl.models.modifiers.base import ModifyMixin

from mttl.models.modifiers.expert_containers.selectors import Selector, TaskNameSelector
from mttl.models.modifiers.expert_containers import ExpertContainer
from mttl.models.modifiers.hard_prompts import HardPrompt, HardPromptConfig
from mttl.models.modifiers.modify_model import register_modifier


class HardPromptDecoderWrapper(nn.Module):
    def __init__(self, config, transformer, expert_container):
        super().__init__()

        self.transformer = transformer
        self.config = config
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
        if len(args) > 0:
            input_ids = args[0]
            args = args[1:]
        else:
            input_ids = kwargs["input_ids"]
        (
            kwargs["input_ids"],
            kwargs["attention_mask"],
            _,
        ) = self.expert_container.forward(input_ids, kwargs["attention_mask"])
        return self.transformer.generate(*args, **kwargs)


def add_hard_prompt_to_transformer(
    transformer,
    expert_name,
    expert_config,
    expert_weights,
    action="route",
    is_default=False,
    selectors={},
    config=None,
):
    # create a shared prompt container holding the experts
    if not isinstance(transformer, HardPromptDecoderWrapper):
        expert_container = HardPromptExpertContainer(
            expert_config,
            transformer.task_id_container,
            selector=None,
        )
        # patch the decoder
        transformer = HardPromptDecoderWrapper(config, transformer, expert_container)

    transformer.add_expert(
        expert_name,
        expert_config,
        expert_weights,
        action=action,
        is_default=is_default,
    )
    return transformer


class HardPromptExpertContainer(ExpertContainer):
    def __init__(self, config, task_id_container, selector=None):
        super().__init__()
        self.config = config
        self.selector: Selector = selector or TaskNameSelector()
        self.selector.info_container = task_id_container

        self.info_container = task_id_container
        self.default_expert_name = None
        self.merged_expert_names = []
        self.experts = nn.ModuleDict({})

    def add_expert(
        self,
        name: str,
        expert_config: Any,
        expert_weights: Dict[str, torch.Tensor],
        action="route",
        is_default=False,
    ) -> None:
        from mttl.models.modifiers.modify_model import get_modifier_type

        if name in self.experts:
            raise ValueError("An expert with name {} already exists.".format(name))

        if action == "merge":
            raise ValueError("Merging is not supported for hard prompts.")

        if is_default:
            self.default_expert_name = name

        if get_modifier_type(expert_config) == "hard_prompt":
            expert_module = HardPrompt(expert_config, prompt_init=expert_weights)
        else:
            raise NotImplementedError("Not implemented for this modifier.")

        self.experts[name] = expert_module
        self.add_expert_to_selector(name)

    def add_expert_to_selector(self, expert_name: str):
        if expert_name in self.experts:
            self.selector.add_expert(expert_name)
            self.selector.default_expert_name = self.default_expert_name

    def route(self, input_ids, routing: list, attention_mask=None, labels=None):
        load_experts = []

        for sample_weights in routing:
            if len(sample_weights) > 1:
                raise ValueError(
                    "HardPromptExpertContainer only supports one expert per task."
                )
            selected_expert = list(sample_weights.keys())[0]
            load_experts.append(self.experts[selected_expert])
        return HardPrompt.parallel_forward(
            load_experts, input_ids, attention_mask, labels
        )

    def __getitem__(self, key):
        return self.experts[key]

    def __len__(self):
        return len(self.experts)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        if len(self.experts) > 0:
            weights: list = self.selector(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            return self.route(
                input_ids, attention_mask=attention_mask, labels=labels, routing=weights
            )
        return input_ids, attention_mask, labels
