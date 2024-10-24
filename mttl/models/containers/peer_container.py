import torch
from torch import nn

from mttl.models.containers.base import ContainerFullException, ExpertContainer
from mttl.models.containers.selectors.product_key import PKSelectorConfig, PKSSelector
from mttl.models.containers.selectors.selector_output import (
    MultiheadBatchSequenceExpertsAndWeightsSelectorOutput,
)
from mttl.models.library.expert import Expert
from mttl.models.modifiers.mlp import PEERConfig, PEERModifier

# diff architectures name those layers differently
DOWN_NAMES = ["fc1", "c_fc"]
UP_NAMES = ["fc2", "c_proj"]


class PEERMLPContainer(ExpertContainer):
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
        self.num_experts = 0
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
        self.layer.in_features = self.input_dim
        self.experts = PEERModifier(config)

    def initialize_experts(self, expert_config: PEERConfig) -> None:
        self.num_experts = expert_config.moe_num_experts
        assert (
            self.num_experts**0.5
        ).is_integer(), "Number of experts must be a square number"
        self.peer_weight_down_embed = nn.Embedding(
            num_embeddings=self.num_experts,
            embedding_dim=self.input_dim,
            dtype=self.dtype,
        )
        self.peer_weight_up_embed = nn.Embedding(
            num_embeddings=self.num_experts,
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
        expert_config: PEERConfig = expert.expert_config
        if self.num_experts == expert_config.moe_num_experts:
            raise ContainerFullException()
        self.initialize_experts(expert_config)

    def __getitem__(self, key):
        pass
