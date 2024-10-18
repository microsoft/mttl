from dataclasses import dataclass

import torch
from torch import nn

from mttl.models.containers.base import ContainerFullException, ExpertContainer
from mttl.models.containers.selectors.product_key import PKSelectorConfig, PKSSelector
from mttl.models.containers.selectors.selector_output import (
    MultiheadBatchSequenceExpertsAndWeightsSelectorOutput,
)
from mttl.models.library.expert import Expert
from mttl.models.modifiers.base import Modifier, ModifierConfig

# from mttl.models.modifiers.mlp import PEERConfig, PEERModifier

# diff architectures name those layers differently
DOWN_NAMES = ["fc1", "c_fc"]
UP_NAMES = ["fc2", "c_proj"]


@dataclass
class PEERConfig(ModifierConfig):
    n_heads: int = 8
    moe__num_experts: int = 100
    emb_dim: int = 128
    down_proj_layer: str = "fc1"
    up_proj_layer: str = "fc2"


@Modifier.register("peer", config_cls=PEERConfig)
class PEERMLPContainer(ExpertContainer, Modifier):
    """
    PEER layer from Mixture of A Million Experts (https://arxiv.org/pdf/2407.04153)

    Right now it assumes that it receives a module -- an MLP block, that has attributes fc1 and fc2.
    It upcycles the base model. Yet, for now the experts are innitialized randomly.

    """

    __supports_configs__ = [PEERConfig]

    def __init__(
        self,
        config: PEERConfig,
        module,
        selector_config: PKSelectorConfig = None,
        **kwargs,
    ):
        super().__init__(config, module)
        self._num_experts = 0
        down_names = DOWN_NAMES + [
            config.down_proj_layer
        ]  # names of the up and down projection layers in the MLP block
        up_names = UP_NAMES + [
            config.up_proj_layer
        ]  # needed to infer the dimentions of the MLP block

        assert any(
            hasattr(module, name) for name in down_names + up_names
        ), "Module must have fc1 and fc2 attributes, this is only applicable to MLP block for"
        n_idx = [i for i, name in enumerate(down_names) if hasattr(module, name)][0]

        self.activation = module.act
        self.input_dim = getattr(module, down_names[n_idx]).in_features
        self.output_dim = getattr(module, up_names[n_idx]).out_features
        if selector_config:
            self.selector = PKSSelector(selector_config, in_d=self.input_dim)
        # to enable selector instantiation without having to cary the original module's weights
        self.dtype = next(self.layer.parameters()).dtype

        self.layer = nn.Identity()
        self.expert_name = None
        self.layer.in_features = self.input_dim

    @property
    def num_experts(self):
        return self._num_experts

    def initialize_experts(self, expert_config: PEERConfig) -> None:
        self._num_experts = expert_config.moe__num_experts
        assert (
            self._num_experts**0.5
        ).is_integer(), "Number of experts must be a square number"
        self.peer_weight_down_embed = nn.Embedding(
            num_embeddings=self._num_experts,
            embedding_dim=self.input_dim,
            dtype=self.dtype,
        )
        self.peer_weight_up_embed = nn.Embedding(
            num_embeddings=self._num_experts,
            embedding_dim=self.output_dim,
            dtype=self.dtype,
        )

    def forward(self, input, **kwargs):
        routing: MultiheadBatchSequenceExpertsAndWeightsSelectorOutput = self.selector(
            input
        )
        indices, scores = (
            routing.experts,
            routing.weights,
        )  # both shape b, s, heads, experts

        w_down = self.peer_weight_down_embed(indices)  # b, s, heads, experts, input_dim
        w_up = self.peer_weight_up_embed(indices)  # b, s, heads, experts, output_dim

        x = torch.einsum("bsd,bshed->bshe", input, w_down)  # b, s, heads, experts
        x = self.activation(x)
        x *= scores
        x = torch.einsum("bshe,bshed->bsd", x, w_up)
        return x

    def add_expert(self, expert: Expert, **kwargs) -> None:
        return self.on_add_expert(expert, **kwargs)

    def on_add_expert(self, expert: Expert, **kwargs) -> None:
        """
        'initialize_experts' is called from here instead of __init__ to allow for laoding expert weights from expert object that is passed here
        """
        expert_config: PEERConfig = expert.expert_config
        if self._num_experts == expert_config.moe__num_experts:
            raise ContainerFullException()
        self.initialize_experts(expert_config)
        self.expert_infos[expert.name] = expert.expert_info
        if expert.expert_weights:
            self.load_state_dict(expert.expert_weights)
        self.expert_name = expert.name

    def __getitem__(self, name):
        if name != self.expert_name:
            raise ValueError(
                f"Expert with name {name} does not exist in this container."
            )
        return self

    def __len__(self):
        return self._num_experts
