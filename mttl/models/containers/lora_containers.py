import torch
from pyparsing import Union
from torch import Tensor, nn

from mttl.logging import warn_once
from mttl.models.containers.base import ExpertContainer
from mttl.models.containers.selectors.selector_output import (
    BatchExpertsAndWeightsSelectorOutput,
    BatchExpertsSelectorOutput,
    BatchSequenceExpertsAndWeightsSelectorOutput,
    ExpertsAndWeightsSelectorOutput,
    SelectorOutput,
)
from mttl.models.library.expert import Expert
from mttl.models.modifiers.lora import LoRA, LoRAConfig, SkilledLoRA, SkilledLoRAConfig
from mttl.models.modifiers.modify_model import get_modifier_name


class LoRAExpertContainer(ExpertContainer):
    __supports_configs__ = [LoRAConfig]

    def __init__(
        self,
        config: LoRAConfig,
        layer,
        selector=None,
        lora_merge_after=False,
    ):
        super().__init__(config, layer, selector)

        self.lora_merge_after = lora_merge_after

        if not isinstance(self.layer, nn.Linear):
            raise ValueError(
                "Expert containers for layers other than nn.Linear have not been implemented, current layer is {}".format(
                    self.layer.__class__.__name__
                )
            )

        self.merged_expert_names = []
        self.experts = nn.ModuleDict({})

    def on_add_expert(
        self,
        expert: Expert,
        action="route",
        is_default=False,
    ) -> None:
        from mttl.models.containers import filter_expert_weights

        # back-compatibility, in previous versions, the expert config was a training config
        self._check_config(expert.expert_config)

        # We may want to add a SkilledLoRA directly, if we are loading an MHR model for example
        LoRA_cls = {"lora": LoRA, "skilled_lora": SkilledLoRA}[
            get_modifier_name(expert.expert_config)
        ]

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
            self.experts[expert.name] = modifier_module

    def merge_with_layer(self):
        if not len(self.experts):
            return

        for expert_name, expert_module in self.experts.items():
            expert_module.merge_with_layer()

        self.merged_expert_names.extend(self.experts)
        self.expert_infos.clear()
        self.experts.clear()

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

    def __getitem__(self, name):
        return self.experts[name]


class CoalescedLoRAExpertContainer(LoRAExpertContainer):
    """A coalesced version of the LoRA expert container, where the experts are kept
    in memory in a single parameter.
    """

    __supports_configs__ = [LoRAConfig, SkilledLoRAConfig]

    def __init__(
        self,
        config,
        layer,
        selector=None,
        lora_merge_after=False,
        **kwargs,
    ):
        super().__init__(config, layer, selector, lora_merge_after)

        if not isinstance(self.layer, nn.Linear):
            raise ValueError(
                "Expert containers for layers other than nn.Linear have not been implemented, current layer is {}".format(
                    self.layer.__class__.__name__
                )
            )

        # create a skilled lora config with 0 skills
        self.dummy_config = SkilledLoRAConfig(
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            lora_init_b_random=config.lora_init_b_random,
            lora_rank=config.lora_rank,
            n_splits=config.n_splits if isinstance(config, SkilledLoRAConfig) else 1,
            n_skills=0,
        )
        self.experts = SkilledLoRA(self.dummy_config, layer)

    def __getitem__(self, name) -> Union[LoRA, SkilledLoRA]:
        """Returns either a LoRA or a SkilledLoRA module.

        Arrow adds lora modules to the container, while MHR adds
        skilled lora modules to the container.
        """
        index_of: int = self.expert_names.index(name)
        weights: dict[str, Tensor] = self.experts.get_skill_weights(index_of)

        config = self.expert_infos[name].expert_config
        modifier_name = get_modifier_name(config)

        if modifier_name == "lora":
            assert self.dummy_config.n_splits == 1
            # squeeze the first dimension and the n_splits dimension
            lora = LoRA(config, self.layer)
            lora.load_lora_weights({n: w.squeeze() for n, w in weights.items()})
            return lora
        elif modifier_name == "skilled_lora":
            # should be skilled lora
            skilled_lora = SkilledLoRA(config, self.layer)
            skilled_lora.load_lora_weights(weights)
            return skilled_lora
        else:
            raise ValueError("Unknown modifier type, expected LoRA or SkilledLoRA.")

    def on_add_expert(self, expert: Expert, action="route", is_default=False) -> None:
        from mttl.models.containers import filter_expert_weights

        if action == "merge":
            raise ValueError(
                "Merging is not supported for `CoalescedLoRAExpertContainer`."
            )

        # back-compatibility, in previous versions, the expert config was a training config
        self._check_config(expert.expert_config)

        # We may want to add a SkilledLoRA directly, if we are loading an MHR model for example
        lora_type = get_modifier_name(expert.expert_config)
        LoRA_cls = {"lora": LoRA, "skilled_lora": SkilledLoRA}[lora_type]

        modifier_module = LoRA_cls(
            expert.expert_config, self.layer, layer_name=self.__layer_name__
        )

        if expert.expert_weights:
            expert_weights = filter_expert_weights(
                self.__layer_name__, expert.expert_weights
            )
            modifier_module.load_lora_weights(expert_weights)

        self.experts.add_skill(modifier_module)

    def route(self, input, selection, **kwargs):
        if isinstance(selection, BatchExpertsSelectorOutput):
            # in order to use this container, we need to create one-hot weights for the experts
            batch_size = len(selection.experts)

            indices = torch.LongTensor(
                self._convert_expert_names_to_indices(
                    selection.experts,
                    use_default_expert=self.default_expert_name is not None,
                )
            )

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
                input,
                [self.experts],
                weights,
                dim_names=["batch", "experts"],
                merge_after=self.lora_merge_after,
            )
            return module_output
        elif (
            isinstance(selection, BatchSequenceExpertsAndWeightsSelectorOutput)
            or isinstance(selection, BatchExpertsAndWeightsSelectorOutput)
            or isinstance(selection, ExpertsAndWeightsSelectorOutput)
        ):
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
                    dtype=selection.weights.dtype,
                ).scatter_add(
                    selection.weights.ndim - 1, selection.experts, selection.weights
                )
            else:
                # we select all experts, weight have already the right shape
                weights = selection.weights
                assert weights.shape[-1] == self.experts.n_skills

            module_output = SkilledLoRA.parallel_linear_weighted_forward(
                input,
                [self.experts],
                weights,
                dim_names=selection.dim_names,
                merge_after=self.lora_merge_after,
            )
            return module_output
        else:
            raise ValueError("Unknown selection type.")

    def forward(self, input, **kwargs):
        if len(self.experts) > 0:
            selection = self.selector(input, container=self, **kwargs)
            return self.route(input, selection, **kwargs)
        else:
            return self.layer(input)
