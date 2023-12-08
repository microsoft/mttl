from pyparsing import abstractmethod
import torch
from torch import nn
from typing import Any, Dict, List
from mttl.models.modifiers.base import MergeableAdapter, ModifyMixin
from mttl.models.modifiers.base import Adapter, MergeableAdapter, ModifyMixin
from mttl.models.modifiers.lora import LoRA, SkilledLoRA, SkilledLoRAView
from mttl.models.modifiers.kv_adapter import KVAdapter
from mttl.models.modifiers.expert_containers.selectors import *
from mttl.utils import logger
from mttl.models.modifiers.expert_containers.module_graph import Expert


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

    def add_expert_to_selector(self, expert_name: str, **kwargs):
        self.selector.add_expert(expert_name, **kwargs)
        self.selector.default_expert_name = self.default_expert_name

    def get(self, key):
        if key not in self.experts:
            if self.default_expert_name is None:
                raise ValueError(
                    "Expert with name {} does not exist and no default expert is set.".format(
                        key
                    )
                )
            return self.experts[self.default_expert_name]
        return self.experts[key]

    def __getitem__(self, key):
        return self.experts[key]

    def __len__(self):
        return len(self.experts)


class LoRAExpertContainer(MergeableAdapter, ExpertContainer, ModifyMixin):
    def __init__(self, config, task_id_container, layer, selector=None):
        super().__init__()
        self.config = config
        self.layer = layer
        self.selector = selector or TaskNameSelector()
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
        name,
        expert: Expert,
        expert_weights,
        action="merge",
        is_default=False,
    ) -> None:
        expert_config = expert.expert_config
        expert_task_name = expert.expert_info.expert_task_name

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

        self.add_expert_to_selector(name, expert_task_name=expert_task_name)

    def get_merged_weights(self, with_global_names=True, **merger_kwargs):
        """
        Merges experts to one expert according to weights, if weights are not given, it uses the selector to get the weights.
        Does not merge the layer.
        """
        weights_ = {}
        for k, v in self.selector.get_merged_weights(self, **merger_kwargs).items():
            key = k if not with_global_names else self.layer_name + "." + k
            weights_[key] = v
        return self.config, weights_

    def merge_with_layer(self):
        if not len(self.experts):
            return

        for _, expert_module in self.experts.items():
            expert_module.merge_with_layer()
        self.merged_expert_names.extend(self.experts)
        self.experts.clear()

    def route(self, input, selection, **kwargs):
        """Depending on the selection output, we and merge differently."""
        from mttl.models.modifiers.lora import SkilledLoRA, SkilledLoRAView

        if isinstance(selection, BatchModulesAndWeightsSelectorOutput):
            skilled_loras = [
                SkilledLoRAView.from_loras([self.get(x_name) for x_name in b_modules])
                for b_modules in selection.modules
            ]
            weights = [torch.tensor(x_weights) for x_weights in selection.weights]
            return SkilledLoRA.parallel_linear_weighted_forward(
                input, skilled_loras, weights
            )
        elif isinstance(selection, ModulesAndWeightsSelectorOutput):
            skilled_lora = SkilledLoRAView.from_loras(
                [self.get(module) for module in selection.modules]
            )
            return SkilledLoRA.parallel_linear_weighted_forward(
                input, [skilled_lora], [selection.weights]
            )
        elif isinstance(selection, ModulesSelectorOutput):
            return LoRA.parallel_linear_forward(
                input, [self.get(module) for module in selection.modules]
            )

    def forward(self, input, **kwargs):
        if len(self.experts) > 0:
            selection = self.selector(input, **kwargs)
            return self.route(input, selection, **kwargs)
        return self.layer(input)


class KVExpertContainer(KVAdapter, ExpertContainer):
    """Expert Container for KVAdapters.
    Unlike the LoRAExpertContainer, the KVExpertContainer is a KVAdapter itself,

    See `KVSelector` for info on how the routing is done.
    See `KVAdapter` for info on the control flow of the forward pass.
    """

    def __init__(self, config, task_id_container, layer, selector=None):
        KVAdapter.__init__(self, config, layer)

        self.config = config
        self.layer = layer
        self.selector: KVSelector = selector or KVTaskNameSelector()
        self.selector.info_container = task_id_container
        self.info_container = task_id_container

        # Check if layer is an attention layer :
        if not hasattr(self.attn_layer, "k_proj") and self.config.model != "phi-2":
            raise ValueError(
                "`KVExpertContainer` should wrap an attention layer. {}".format(
                    self.attn_layer.__class__.__name__
                )
            )

        self.default_expert_name = None
        self.experts = nn.ModuleDict({})

    # skip creating the adapter weights
    def create_for_layer(self, attn_layer):
        pass

    # Delegate Routing ops to the selectors
    def route(self, query, keys, attn_layer=None):
        if callable(getattr(self.selector, "route", None)):
            return self.selector.route(self.experts, query, keys, attn_layer)

        # This behavior is problematic! you need `get_gate` to call the adapter method
        return super().route(query, keys, attn_layer)

    # Delegate Routing ops to the selectors
    def aggregate(self, adapter_weights, adapter_v):
        if callable(getattr(self.selector, "aggregate", None)):
            return self.selector.aggregate(self.experts, adapter_weights, adapter_v)

        # This behavior is problematic! you need `get_gate` to call the adapter method
        return super().aggregate(adapter_weights, adapter_v)

    def get_kv_weights(self, k_proj, v_proj):
        return self.selector.get_kv_weights(self.experts, k_proj, v_proj)

    def get_gate(self, adapter_weights):
        return self.selector.get_gate(self.experts, adapter_weights)

    def add_expert(
        self,
        name,
        expert: Expert,
        expert_weights,
        action="route",
        is_default=False,
        **kwargs,
    ) -> None:
        expert_config = expert.expert_config

        if name in self.experts:
            raise ValueError("An expert with name {} already exists.".format(name))

        if action == "merge":
            raise ValueError("Merging is not supported for `KVAdapters`.")

        expert_module = KVAdapter(expert_config, self.attn_layer)
        expert_module.load_adapter_weights(expert_weights)

        self.experts[name] = expert_module

        if is_default:
            self.default_expert_name = name

        self.add_expert_to_selector(name)
