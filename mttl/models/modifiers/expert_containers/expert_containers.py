from pyparsing import abstractmethod
import torch
from torch import nn
from typing import Any, Dict
from mttl.models.modifiers.base import MergeableAdapter, ModifyMixin
from mttl.models.modifiers.hard_prompts import HardPrompt
from mttl.models.modifiers.base import Adapter, MergeableAdapter, ModifyMixin
from mttl.models.modifiers.lora import LoRA, SkilledLoRA
from mttl.models.modifiers.kv_adapter import KVAdapter
from mttl.models.modifiers.expert_containers.selectors import *
from mttl.utils import logger


class ExpertContainer:
    @abstractmethod
    def add_expert(
        self,
        name: str,
        expert_config: Any,
        expert_weights: Dict[str, torch.Tensor],
        action="merge",
        is_default=False,
    ) -> None:
        pass

    @abstractmethod
    def forward(self, input, **kwargs):
        pass

    def add_expert_to_selector(self, expert_name: str):
        if expert_name in self.experts:
            self.selector.add_expert(expert_name)
            self.selector.default_expert_name = self.default_expert_name

    def __getitem__(self, key):
        return self.experts[key]

    def __len__(self):
        return len(self.experts)


class LoRAExpertContainer(MergeableAdapter, ExpertContainer, ModifyMixin):
    def __init__(self, config, task_id_container, layer, selector=None):
        super().__init__()
        self.config = config
        self.layer = layer
        self.selector: Selector = selector or TaskNameSelector()
        self.selector.info_container = task_id_container

        if not isinstance(self.layer, nn.Linear):
            raise ValueError(
                "Expert containers for layers other than nn.Linear have not been implemented, current layer is {}".format(
                    self.layer.__class__.__name__
                )
            )

        self.info_container = task_id_container
        self.default_expert_name = None
        self.merged_expert_names = []
        self.experts = nn.ModuleDict({})

    def add_expert(
        self,
        name: str,
        expert_config: Any,
        expert_weights: Dict[str, torch.Tensor],
        action="merge",
        is_default=False,
    ) -> None:
        if name in self.experts:
            raise ValueError("An expert with name {} already exists.".format(name))

        if is_default and action == "merge":
            raise ValueError(
                "Cannot set is_default if this expert is merged, change to 'route'."
            )

        # hack this for now, but build a proper config for each module
        if expert_config.model_modifier == "lora":
            expert_module = LoRA(expert_config, self.layer)
            expert_module.load_lora_weights(expert_weights)
        else:
            raise NotImplementedError("ExpertContainer only supports LoRA experts.")

        if action == "merge":
            # weight is merged with layer so we can discard it now
            if expert_config.model_modifier == "lora":
                expert_module.merge_with_layer()
                self.merged_expert_names.append(name)
            else:
                raise NotImplementedError("Merging experts only supports LoRA experts.")
        else:
            # we keep track of the expert weights
            if name in self.experts:
                raise ValueError("An expert with name {} already exists.".format(name))
            self.experts[name] = expert_module

        if is_default:
            self.default_expert_name = name

        self.add_expert_to_selector(name)

    def merge_experts_together(self, weights=None):
        """
        Merges experts to one expert according to weights, if weights are not given, it uses the selector to get the weights.
        Does not merge the layer.
        """
        if weights is None:
            assert self.selector is not None
            weights: dict = self.selector.get_routing_weights()

        merged_weights = {}
        for name, expert in self.experts.items():
            assert name in weights, f"Weight for expert {name} is not given"
            expert_state_dict = expert.state_dict()
            weight = weights[name]
            for k, v in expert_state_dict.items():
                value = weight * v
                if k in merged_weights:
                    merged_weights[k] += value
                else:
                    merged_weights[k] = value
        self.experts = nn.ModuleDict({})
        self.add_expert("merged_expert", self.config, merged_weights, action="route")

    def merge_with_layer(self):
        if len(self.experts) > 0:
            for name, expert_module in self.experts.items():
                assert isinstance(
                    expert_module, LoRA
                ), "Only LoRA experts can be merged with the layer for now."
                expert_module.merge_with_layer()
                self.merged_expert_names.append(name)
                self.experts.pop(name)

    def route(self, input, routing: list):
        load_experts = []
        weights = []

        for sample_weights in routing:
            exps = []
            ws = []
            for expert_name, weight in sample_weights.items():
                if expert_name not in self.experts:
                    if not self.default_expert_name:
                        raise ValueError(
                            "The expert for this task {} does not exists. Consider setting a default expert!".format(
                                expert_name
                            )
                        )
                    else:
                        selected_expert = self.default_expert_name
                else:
                    selected_expert = expert_name
                exps.append(self.experts[selected_expert])
                ws.append(weight)
            assert len(exps) == len(ws)
            load_experts.append(exps)
            weights.append(torch.tensor(ws))
        return SkilledLoRA.parallel_linear_forward(input, load_experts, weights)

    def forward(self, input, **kwargs):
        if len(self.experts) > 0:
            weights: list = self.selector(input)
            output = self.route(input, weights)
            return output
        return self.layer(input)


class HardPromptExpertContainer(ExpertContainer):
    def __init__(self, config, task_id_container, layer, selector=None):
        super().__init__()
        self.config = config
        self.layer = layer
        self.selector: Selector = selector or TaskNameSelector()
        self.selector.info_container = task_id_container

        if not isinstance(self.layer, nn.Embedding):
            raise ValueError(
                "Expert containers for layers other than nn.Embedding have not been implemented, current layer is {}".format(
                    self.layer.__class__.__name__
                )
            )

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
        if name in self.experts:
            raise ValueError("An expert with name {} already exists.".format(name))

        if action == "merge":
            raise ValueError("Merging is not supported for hard prompts.")

        if is_default:
            self.default_expert_name = name

        if expert_config.model_modifier == "hard_prompt":
            expert_module = HardPrompt(expert_config, prompt_init=expert_weights)
        else:
            raise NotImplementedError("Not implemented for this modifier.")

        self.experts[name] = expert_module
        self.add_expert_to_selector(name)

    def add_expert_to_selector(self, expert_name: str):
        if expert_name in self.experts:
            self.selector.add_expert(expert_name)
            self.selector.default_expert_name = self.default_expert_name

    def route(self, inputs, routing: list):
        load_experts = []

        for sample_weights in routing:
            if len(sample_weights) > 1:
                raise ValueError(
                    "HardPromptExpertContainer only supports one expert per task."
                )
            selected_expert = list(sample_weights.keys())[0]
            load_experts.append(self.experts[selected_expert])
        return HardPrompt.parallel_forward(load_experts, **inputs)

    def __getitem__(self, key):
        return self.experts[key]

    def __len__(self):
        return len(self.experts)

    def forward(self, **kwargs):
        if len(self.experts) > 0:
            weights: list = self.selector(kwargs)
            return self.route(kwargs, routing=weights)
        return kwargs["input_ids"], kwargs.get("attention_mask"), kwargs.get("labels")


class KVExpertContainer(ExpertContainer, KVAdapter):
    def __init__(self, config, task_id_container, layer, selector=None):
        super(Adapter, self).__init__()

        self.config = config
        self.layer = layer
        self.selector: KVSelector = selector or KVTaskNameSelector()
        self.selector.info_container = task_id_container

        # Check if layer is an attention layer :
        if not hasattr(self.layer, "k_proj"):
            raise ValueError(
                "`KVExpertContainer` should wrap an attention layer. {}".format(
                    self.layer.__class__.__name__
                )
            )

        self.info_container = task_id_container
        self.default_expert_name = None
        self.experts = nn.ModuleDict({})

        # Needed to mimich behavior of `KVAdapter`
        self.an_expert = None

    def __getattr__(self, name):
        try:
            return super(Adapter, self).__getattr__(name)
        except AttributeError:
            return getattr(self.an_expert, name)

    # Delegate Routing ops to the selectors
    def route(self, query, keys, attn_layer):
        if callable(getattr(self.selector, "route", None)):
            return self.selector.route(self.experts, query, keys, attn_layer)

        return self.an_expert.route(query, keys, attn_layer)

    def get_kv_weights(self, k_proj, v_proj):
        return self.selector.get_kv_weights(self.experts, k_proj, v_proj)

    def get_gate(self, adapter_weights):
        return self.selector.get_gate(self.experts, adapter_weights)

    def add_expert(
        self,
        name: str,
        expert_config: Any,
        expert_weights: Dict[str, torch.Tensor],
        action="route",
        is_default=False,
    ) -> None:
        if name in self.experts:
            raise ValueError("An expert with name {} already exists.".format(name))

        if action == "merge":
            raise ValueError("Merging is not supported for `KVAdapters`.")

        expert_module = KVAdapter(expert_config, self.layer)
        expert_module.load_adapter_weights(expert_weights)
        self.experts[name] = expert_module
        self.an_expert = expert_module

        if is_default:
            self.default_expert_name = name

        self.add_expert_to_selector(name)

    def forward(self, *args, **kwargs):
        # Copying the forward pass of KVAdapter, for some reason super() does not work
        return self.attn_fwd(self.attn_layer, self, *args, **kwargs)
