from pyparsing import abstractmethod
import torch
from torch import nn
from typing import Dict, List, Union
from mttl.config import Config
from mttl.models.modifiers.base import (
    ModifierConfig,
    MergeableAdapter,
    ModifierConfig,
    ModifyMixin,
)
from mttl.models.modifiers.expert_containers.selectors import (
    BatchExpertsAndWeightsSelectorOutput,
    BatchExpertsSelectorOutput,
    BatchSequenceExpertsAndWeightsSelectorOutput,
    ExpertsAndWeightsSelectorOutput,
    KVTaskNameSelector,
    SelectorOutput,
)

from mttl.utils import logger, warn_once
from mttl.models.modifiers.lora import LoRA, LoRAConfig, SkilledLoRA, SkilledLoRAConfig
from mttl.models.modifiers.kv_adapter import KVAdapter, KVAdapterConfig
from mttl.models.modifiers.expert_containers.expert import Expert
from mttl.models.modifiers.modify_model import get_modifier_type


class ExpertContainer:
    __supports_configs__ = []

    def __init__(self, config, info_container, layer, selector=None):
        from mttl.models.modifiers.expert_containers.selectors import TaskNameSelector

        self.config = config
        self.layer = layer
        self.info_container = info_container
        self.selector = selector or TaskNameSelector(info_container)

        self.experts: dict = None
        self.expert_infos = {}
        self.expert_names = []
        self.default_expert_name = None

    def assign_selector(self, selector):
        del self.selector
        self._modules.pop("selector", None)
        # propagate experts to the selector
        self.selector = selector
        for expert_name, expert_info in self.expert_infos.items():
            self.add_expert_to_selector(expert_name, expert_info=expert_info)

    def add_expert_to_selector(self, expert_name: str, **kwargs):
        self.selector.add_expert(expert_name, **kwargs)
        self.selector.default_expert_name = self.default_expert_name

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
        if not hasattr(self, "__layer_name__"):
            raise ValueError("Dependency injection for layer name has not been done.")
        return self.__layer_name__

    def _get_expert_weights(self, expert_name):
        return self.experts[expert_name].state_dict()

    @abstractmethod
    def forward(self, input, **kwargs):
        pass

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

    def get_merged_params(self, with_global_names=True, **merger_kwargs):
        """
        Merges experts to one expert according to selector weights.
        """
        merged_params = {}
        merging_weights = self.selector.get_merging_weights(
            **merger_kwargs
        )  # expert_name: weight
        for exp_name, merging_weight in merging_weights.items():
            for k, parameter in self._get_expert_weights(exp_name).items():
                key = k if not with_global_names else self.layer_name + "." + k
                if k not in merged_params:
                    merged_params[key] = parameter * merging_weight
                else:
                    merged_params[key] += parameter * merging_weight

        return merged_params

    def __getitem__(self, key):
        return self.experts[key]

    def __len__(self):
        return len(self.experts)


class LoRAExpertContainer(MergeableAdapter, ExpertContainer, ModifyMixin):
    __supports_configs__ = [LoRAConfig]

    def __init__(
        self,
        config: LoRAConfig,
        info_container,
        layer,
        selector=None,
        lora_merge_after=False,
    ):
        MergeableAdapter.__init__(self)
        super().__init__(config, info_container, layer, selector)
        self.lora_merge_after = lora_merge_after

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

        # We may want to add a SkilledLoRA directly, if we are loading an MHR model for example
        lora_type = get_modifier_type(expert.expert_config)
        LoRA_cls = {"lora": LoRA, "skilled_lora": SkilledLoRA}[lora_type]
        modifier_module = LoRA_cls(
            expert.expert_config, self.layer, layer_name=self.__layer_name__
        )

        if expert.expert_weights:
            expert_weights = filter_expert_weights(
                self.__layer_name__, expert.expert_weights
            )
            modifier_module.load_lora_weights(expert_weights)

        if action == "merge":
            # weight is merged with layer so we can discard it now
            modifier_module.merge_with_layer()
            self.merged_expert_names.append(expert.name)
        else:
            if is_default:
                self.default_expert_name = expert.name

            self._add_expert(expert.name, expert.expert_info, modifier_module)

    def merge_with_layer(self):
        if not len(self.experts):
            return

        for _, expert_module in self.experts.items():
            expert_module.merge_with_layer()

        self.merged_expert_names.extend(self.experts)
        self.experts.clear()

    def _get_expert_weights(self, expert_name):
        return {
            n: w
            for n, w in self.experts[expert_name].state_dict().items()
            if "lora" in n
        }

    def _convert_expert_names_to_indices(
        self, expert_names, use_default_expert=True
    ) -> torch.Tensor:
        indices = []

        for expert_name in expert_names:
            if type(expert_name) in [list, tuple]:
                indices.append(self._convert_expert_names_to_indices(expert_name))
            else:
                if expert_name in self.expert_names:
                    index = self.expert_names.index(expert_name)
                elif use_default_expert:
                    index = self.expert_names.index(self.default_expert_name)
                else:
                    raise ValueError(
                        "Expert name not found in the list of experts: {}".format(
                            expert_name
                        )
                    )
                indices.append(index)
        return indices

    def route(self, input, selection, **kwargs):
        """Depending on the selection output, we and merge differently."""
        from mttl.models.modifiers.lora import SkilledLoRA, SkilledLoRAView

        if isinstance(selection, ExpertsAndWeightsSelectorOutput):
            # In this case, we have a list of experts and their weights
            # and these are shared across all the batch examples
            skilled_lora = SkilledLoRAView.from_loras(
                [self.get(module) for module in selection.experts]
            )
            return SkilledLoRA.parallel_linear_weighted_forward(
                input,
                [skilled_lora],
                selection.weights,
                dim_names=selection.dim_names,
                merge_after=self.lora_merge_after,
            )
        elif isinstance(selection, BatchExpertsSelectorOutput):
            # In this case, we have exactly one expert per example in the batch with no weights
            return LoRA.parallel_linear_forward(
                input, [self.get(module) for module in selection.experts]
            )
        elif isinstance(
            selection,
            (
                BatchExpertsAndWeightsSelectorOutput,
                BatchSequenceExpertsAndWeightsSelectorOutput,
            ),
        ):
            # In this case, we have exactly multiple experts per example (and possible per token) in the batch with weights
            # The selectors might return a list of expert names, in this case we need to convert them to indices
            # If expert names are not returned, it means that we are scoring all the experts
            if selection.experts is not SelectorOutput.ALL_EXPERTS:
                if not isinstance(selection.experts, torch.Tensor):
                    # convert expert names to indices
                    selection.experts = torch.LongTensor(
                        self._convert_expert_names_to_indices(
                            selection.experts,
                            use_default_expert=self.default_expert_name is not None,
                        )
                    ).to(selection.weights.device)

                # set of active indices
                unique_indices, inverse_indices = torch.unique(
                    selection.experts, return_inverse=True
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
                    *(selection.weights.shape[:-1] + (len(unique_indices),)),
                    device=selection.weights.device,
                    dtype=selection.weights.dtype,
                )
                inverse_weights = torch.scatter_add(
                    inverse_weights,
                    selection.weights.ndim - 1,
                    inverse_indices,
                    selection.weights,
                )
                module_output = SkilledLoRA.parallel_linear_weighted_forward(
                    input,
                    skilled_loras,
                    inverse_weights,
                    dim_names=selection.dim_names,
                    merge_after=self.lora_merge_after,
                )
            else:
                # we have no indices, so we assume that we have weights for all the experts
                assert selection.weights.shape[-1] == len(self.experts)

                warn_once(
                    "Creating skilled loras for all experts, you might want to use CoalescedLoRAContainer instead, set USE_COALESCED_LORA=True in your environment variables."
                )

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

                module_output = SkilledLoRA.parallel_linear_weighted_forward(
                    input,
                    skilled_loras,
                    selection.weights,
                    dim_names=selection.dim_names,
                    merge_after=self.lora_merge_after,
                )
            return module_output.view(input.shape[0], input.shape[1], -1)

    def forward(self, input, **kwargs):
        if len(self.experts) > 0:
            selection = self.selector(input, container=self, **kwargs)
            return self.route(input, selection, **kwargs)
        return self.layer(input)


class CoalescedLoRAExpertContainer(LoRAExpertContainer):
    """A coalesced version of the LoRA expert container, where the experts are kept
    in memory in a single parameter.
    """

    __supports_configs__ = [SkilledLoRAConfig, LoRAConfig]

    def __init__(self, config, info_container, layer, selector=None, **kwargs):
        MergeableAdapter.__init__(self)
        super().__init__(config, info_container, layer, selector)

        if not isinstance(self.layer, nn.Linear):
            raise ValueError(
                "Expert containers for layers other than nn.Linear have not been implemented, current layer is {}".format(
                    self.layer.__class__.__name__
                )
            )

        self.merged_expert_names = []

        # create a skilled lora config with 0 skills
        self.dummy_config = SkilledLoRAConfig(
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            lora_init_b_random=config.lora_init_b_random,
            lora_rank=config.lora_rank,
            n_splits=config.n_splits if isinstance(config, SkilledLoRAConfig) else 1,
            n_skills=0,
            phi_2_align_heads=(
                config.phi_2_align_heads
                if isinstance(config, SkilledLoRAConfig)
                else False
            ),
        )
        self.experts = SkilledLoRA(self.dummy_config, layer)

    def _get_expert_weights(self, expert_name) -> Dict:
        from mttl.models.modifiers.modify_model import get_modifier_type

        index_of = self.expert_names.index(expert_name)
        weights = self.experts.get_skill_weights(index_of)

        modifier_type = get_modifier_type(self.expert_infos[expert_name].expert_config)

        if modifier_type == "lora":
            assert self.dummy_config.n_splits == 1
            # squeeze the first dimension and the n_splits dimension
            return {n: w.squeeze() for n, w in weights.items()}
        else:
            # should be skilled lora
            return weights

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
        if isinstance(selection, BatchExpertsSelectorOutput):
            # in order to use this container, we need to create one-hot weights for the experts
            batch_size = len(selection.experts)

            indices = torch.LongTensor(
                self._convert_expert_names_to_indices(
                    selection.experts,
                    use_default_expert=self.default_expert_name is not None,
                )
            ).to(selection.weights.device)

            # one-hot encode the indices
            weights = (
                torch.zeros(
                    (batch_size, self.experts.n_skills),
                )
                .scatter_add(
                    1, indices.unsqueeze(1), torch.ones((len(selection.experts), 1))
                )
                .to(device=self.experts.lora_a.device, dtype=torch.float32)
            )

            module_output = SkilledLoRA.parallel_linear_weighted_forward(
                input, [self.experts], weights, dim_names=["batch", "experts"]
            )
            return module_output
        elif isinstance(
            selection, BatchSequenceExpertsAndWeightsSelectorOutput
        ) or isinstance(selection, BatchExpertsAndWeightsSelectorOutput):
            if selection.experts is not SelectorOutput.ALL_EXPERTS:
                # we are in top-k or sparse selection mode
                if not isinstance(selection.experts, torch.Tensor):
                    selection.experts = torch.LongTensor(
                        self._convert_expert_names_to_indices(
                            selection.experts,
                            use_default_expert=self.default_expert_name is not None,
                        )
                    ).to(selection.weights.device)

                # we need to expand the weights to the full size of the expert set
                weights = torch.zeros(
                    (selection.weights.shape[:-1] + (self.experts.n_skills,)),
                    device=selection.weights.device,
                ).scatter_add(
                    selection.weights.ndim - 1, selection.experts, selection.weights
                )
            else:
                # we select all experts, weight have already the right shape
                weights = selection.weights
                assert weights.shape[-1] == self.experts.n_skills

            module_output = SkilledLoRA.parallel_linear_weighted_forward(
                input, [self.experts], weights, dim_names=selection.dim_names
            )
            return module_output

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

    def __init__(self, config, info_container, layer, selector=None, **kwargs):
        super().__init__(
            config,
            info_container,
            layer,
            selector or KVTaskNameSelector(info_container),
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
