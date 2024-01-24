from pyparsing import abstractmethod
import torch
from torch import nn
from typing import List
from mttl.config import Config
from mttl.models.modifiers.base import (
    ModifierConfig,
    MergeableAdapter,
    ModifierConfig,
    ModifyMixin,
)

from mttl.utils import logger
from mttl.models.modifiers.lora import LoRA, LoRAConfig, SkilledLoRA, SkilledLoRAConfig
from mttl.models.modifiers.kv_adapter import KVAdapter, KVAdapterConfig
from mttl.models.modifiers.expert_containers.selectors import *
from mttl.models.modifiers.expert_containers.module_graph import Expert


class ExpertContainer:
    __supports_configs__ = []

    def __init__(self, config, task_id_container, layer, selector=None):
        self.config = config
        self.layer = layer
        self.task_id_container = task_id_container
        self.selector = selector or TaskNameSelector(task_id_container)

        self.experts: dict = None
        self.expert_infos = {}
        self.expert_names = []
        self.default_expert_name = None

    def _add_expert(self, expert_name, expert_info, expert_module):
        self.expert_infos[expert_name] = expert_info
        self.expert_names.append(expert_name)
        self.experts[expert_name] = expert_module
        self.add_expert_to_selector(expert_name, expert_info=expert_info)

    def _check_config(self, expert_config: Union[Config, ModifierConfig]):
        """Checks if the config is supported and converts it to the supported config type if needed."""
        if isinstance(expert_config, Config):
            # patches the config to be a LoRAConfig for the future
            from mttl.models.modifiers.base import ModifierConfig

            expert_config = ModifierConfig.from_training_config(expert_config)

        if type(expert_config) not in self.__supports_configs__:
            raise ValueError(
                "Unsupported expert config type {} for this type of expert container.".format(
                    type(expert_config)
                )
            )

    def export_experts(self) -> List[Expert]:
        experts = []
        for name, expert_module in self.experts.items():
            expert = Expert(
                expert_info=self.expert_infos[name],
                expert_weights={
                    self.layer_name + "." + n: v for n, v in expert_module.state_dict()
                },
            )
            experts.append(expert)
        return experts

    @abstractmethod
    def add_expert(
        self,
        expert: Expert,
        action="merge",
        is_default=False,
    ) -> None:
        pass

    @property
    def layer_name(self):
        return self.__layer_name__

    @abstractmethod
    def forward(self, input, **kwargs):
        pass

    def add_expert_to_selector(self, expert_name: str, **kwargs):
        self.selector.add_expert(expert_name, **kwargs)
        self.selector.default_expert_name = self.default_expert_name

    def get(self, key: Union[int, str]):
        if type(key) == int:
            key = self.expert_names[key]

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
    __supports_configs__ = [LoRAConfig]

    def __init__(self, config, task_id_container, layer, selector=None):
        MergeableAdapter.__init__(self)
        super().__init__(config, task_id_container, layer, selector)

        if not isinstance(self.layer, nn.Linear):
            raise ValueError(
                "Expert containers for layers other than nn.Linear have not been implemented, current layer is {}".format(
                    self.layer.__class__.__name__
                )
            )

        self.merged_expert_names = []
        self.experts = nn.ModuleDict({})

    def add_expert(
        self,
        expert: Expert,
        action="merge",
        is_default=False,
    ) -> None:
        from mttl.models.modifiers.expert_containers import filter_expert_weights

        if expert.expert_weights is not None:
            expert_weights = filter_expert_weights(
                self.__layer_name__, expert.expert_weights
            )
        else:
            expert_weights = None

        if expert.name in self.expert_infos:
            raise ValueError(
                "An expert with name {} already exists.".format(expert.name)
            )

        if is_default and action == "merge":
            raise ValueError(
                "Cannot set is_default if this expert is merged, change to 'route'."
            )

        # back-compatibility, in previous versions, the expert config was a training config
        self._check_config(expert.expert_config)

        expert_module = LoRA(expert.expert_config, self.layer)

        if expert_weights is not None:
            expert_module.load_lora_weights(expert_weights)

        if action == "merge":
            # weight is merged with layer so we can discard it now
            expert_module.merge_with_layer()
            self.merged_expert_names.append(expert.name)

        else:
            if is_default:
                self.default_expert_name = expert.name

            self._add_expert(expert.name, expert.expert_info, expert_module)

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
        elif isinstance(selection, BatchSequenceModulesAndWeightsSelectorOutput):
            if selection.modules is not None:
                assert isinstance(
                    selection.modules, torch.Tensor
                ), "Tensor expected, return indices of selected experts!"

                indices = selection.modules.reshape(-1, selection.modules.shape[-1])
                weights = selection.weights.reshape(-1, selection.weights.shape[-1])

                # set of active indices
                unique_indices, inverse_indices = torch.unique(
                    indices, return_inverse=True
                )

                # form a skilled lora for each unique index, we could potentially skip this stack step
                # to save some memory space, but let's leave it for now
                skilled_loras = [
                    SkilledLoRAView.from_loras(
                        [self.get(int(expert_index)) for expert_index in unique_indices]
                    )
                ]

                # express weights in the new basis of unique indices
                # i.e.
                # indices          = [[10, 20], [15, 5]]
                # weights          = [[0.2, 0.8], [0.9, 0.1]]
                # unique indices   = [5, 10, 15, 20]
                # inverse_indices  = [[1, 3], [2, 0]]
                # inverse_weights  = [[0, 0.2, 0, 0.8], [0.1, 0, 0.9, 0.]]
                inverse_weights = torch.zeros(
                    weights.shape[0], len(unique_indices), device=weights.device
                )
                inverse_weights = torch.scatter_add(
                    inverse_weights, 1, inverse_indices, weights
                )

                module_output = SkilledLoRA.parallel_linear_weighted_forward(
                    input.view(-1, input.size(-1)),
                    skilled_loras,
                    inverse_weights,
                    merge_after=False,
                )
            else:
                # we have no indices, so we just use a linear combination of all the experts
                # for each position and batch example
                if hasattr(self, "_skilled_loras"):
                    skilled_loras = self._skilled_loras
                else:
                    logger.warn("Storing skilled loras for reuse.")

                    # store skilled lora view for reuse locally
                    skilled_loras = [
                        SkilledLoRAView.from_loras(
                            [
                                self.get(int(expert_index))
                                for expert_index in range(len(self))
                            ]
                        )
                    ]
                    self._skilled_loras = skilled_loras

                weights = selection.weights.reshape(-1, selection.weights.shape[-1])
                module_output = SkilledLoRA.parallel_linear_weighted_forward(
                    input.view(-1, input.size(-1)), skilled_loras, [weights]
                )
            return module_output.view(input.shape[0], input.shape[1], -1)

    def forward(self, input, **kwargs):
        if len(self.experts) > 0:
            selection = self.selector(input, **kwargs)
            return self.route(input, selection, **kwargs)
        return self.layer(input)


class CoalescedLoRAExpertContainer(LoRAExpertContainer):
    """A coalesced version of the LoRA expert container, where the experts are kept
    in memory in a single parameter.
    """

    __supports_configs__ = [LoRAConfig]

    def __init__(self, config, task_id_container, layer, selector=None):
        MergeableAdapter.__init__(self)
        super().__init__(config, task_id_container, layer, selector)

        if not isinstance(self.layer, nn.Linear):
            raise ValueError(
                "Expert containers for layers other than nn.Linear have not been implemented, current layer is {}".format(
                    self.layer.__class__.__name__
                )
            )

        self.merged_expert_names = []

        # create a skilled lora config with 0 skills
        dummy_config = SkilledLoRAConfig(
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            lora_init_b_random=config.lora_init_b_random,
            lora_rank=config.lora_rank,
            n_skills=0,
        )
        self.experts = SkilledLoRA(dummy_config, layer)

    def _add_expert(self, expert_name, expert_info, expert_module):
        self.expert_infos[expert_name] = expert_info
        self.expert_names.append(expert_name)
        self.experts.add_skill(expert_module)
        self.add_expert_to_selector(expert_name, expert_info=expert_info)

    def get_merged_weights(self, with_global_names=True, **merger_kwargs):
        """
        Merges experts to one expert according to weights, if weights are not given, it uses the selector to get the weights.
        Does not merge the layer.
        """
        raise ValueError(
            "Get merged weights is impossible for coalesced expert container."
        )

    def merge_with_layer(self):
        raise ValueError("Cannot merge with layer for coalesced expert container.")

    def route(self, input, selection, **kwargs):
        if isinstance(selection, ModulesAndWeightsSelectorOutput):
            module_output = SkilledLoRA.parallel_linear_weighted_forward(
                input, [self.experts], [selection.weights]
            )
            return module_output
        elif isinstance(selection, ModulesSelectorOutput):
            indices = torch.LongTensor(
                [
                    self.expert_names.index(module)
                    if module in self.expert_names
                    else self.expert_names.index(self.default_expert_name)
                    for module in selection.modules
                ]
            ).unsqueeze(1)
            weights = (
                torch.zeros(
                    (len(selection.modules), self.experts.n_skills),
                )
                .scatter_add(1, indices, torch.ones((len(selection.modules), 1)))
                .to(device=self.experts.lora_a.device, dtype=torch.float32)
            )
            module_output = SkilledLoRA.parallel_linear_weighted_forward(
                input, [self.experts], [weights]
            )
            return module_output
        elif isinstance(
            selection, BatchSequenceModulesAndWeightsSelectorOutput
        ) or isinstance(selection, BatchModulesAndWeightsSelectorOutput):
            # selection.weights can be 2 dim if sentence routing or 3 dim if per-token routing
            # selection.modules can be 2 dim ... or 3 dim if ... or None if no top-k
            if selection.modules is not None:
                assert isinstance(
                    selection.modules, torch.Tensor
                ), "Tensor expected, return indices of selected experts!"
                weights = torch.zeros(
                    (
                        selection.weights.shape[0],
                        selection.weights.shape[1],
                        self.experts.n_skills,
                    )
                    if selection.weights.ndim == 3
                    else (selection.weights.shape[0], self.experts.n_skills),
                    device=selection.weights.device,
                ).scatter_add(
                    selection.weights.ndim - 1, selection.modules, selection.weights
                )
            else:
                weights = selection.weights

            if weights.ndim == 2:
                # only weights for each example
                module_output = SkilledLoRA.parallel_linear_weighted_forward(
                    input, [self.experts], [weights]
                )
                return module_output
            else:
                # weights for examples and sequence length
                weights = weights.view(-1, weights.shape[-1])
                module_output = SkilledLoRA.parallel_linear_weighted_forward(
                    input.view(-1, input.shape[-1]), [self.experts], [weights]
                )
                return module_output.view(input.shape[0], input.shape[1], -1)

    def forward(self, input, **kwargs):
        if len(self.experts) > 0:
            selection = self.selector(input, container=self, **kwargs)
            return self.route(input, selection, **kwargs)
        else:
            return self.layer(input)


class KVExpertContainer(KVAdapter, ExpertContainer):
    """Expert Container for KVAdapters.
    Unlike the LoRAExpertContainer, the KVExpertContainer is a KVAdapter itself,

    See `KVSelector` for info on how the routing is done.
    See `KVAdapter` for info on the control flow of the forward pass.
    """

    __supports_configs__ = [KVAdapterConfig]

    def __init__(self, config, task_id_container, layer, selector=None):
        super().__init__(
            config,
            task_id_container,
            layer,
            selector or KVTaskNameSelector(task_id_container),
        )

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
        expert: Expert,
        action="route",
        is_default=False,
        **kwargs,
    ) -> None:
        from mttl.models.modifiers.expert_containers import filter_expert_weights

        expert_weights = filter_expert_weights(
            self.__layer_name__, expert.expert_weights
        )

        if expert.name in self.experts:
            raise ValueError(
                "An expert with name {} already exists.".format(expert.name)
            )

        if action == "merge":
            raise ValueError("Merging is not supported for `KVAdapters`.")

        expert_config = ModifierConfig.from_training_config(expert.expert_config)
        self._check_config(expert_config)

        expert_module = KVAdapter(expert_config, self.attn_layer)
        expert_module.load_adapter_weights(expert_weights)

        if is_default:
            self.default_expert_name = expert.name

        self._add_expert(expert.name, expert.expert_info, expert_module)
