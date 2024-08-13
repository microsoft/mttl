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
    """A coalesced version of the LoRA expert container, where the experts are kept
    in memory in a single parameter.
    """

    __supports_configs__ = [LoRAConfig, SkilledLoRAConfig]

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

    def __init__(
        self,
        config,
        layer,
        selector=None,
        lora_merge_after=False,
        **kwargs,
    ):
        super().__init__(config, layer, selector)

        if not isinstance(self.layer, nn.Linear):
            raise ValueError(
                "Expert containers for layers other than nn.Linear have not been implemented, current layer is {}".format(
                    self.layer.__class__.__name__
                )
            )

        self.lora_merge_after = lora_merge_after
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

                # set of active indices
                unique_indices, inverse_indices = torch.unique(
                    selection.experts, return_inverse=True
                )

                experts = SkilledLoRAView(
                    self.experts.config,
                    self.experts.layers,
                    self.experts.lora_a[unique_indices],
                    self.experts.lora_b[unique_indices],
                )

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
                weights = torch.scatter_add(
                    inverse_weights,
                    selection.weights.ndim - 1,
                    inverse_indices,
                    selection.weights,
                )
            else:
                # we select all experts, weight have already the right shape
                weights = selection.weights
                experts = [self.experts]
                assert weights.shape[-1] == self.experts.n_skills

            module_output = SkilledLoRA.parallel_linear_weighted_forward(
                input,
                experts,
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
