# from https://github.com/huggingface/peft/blob/main/src/peft/tuners/oft/layer.py
from __future__ import annotations

import math
import warnings
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from mttl.models.modifiers.base import Modifier, ModifierConfig, ModifyMixin


class MultiplicativeDropoutLayer(nn.Module):
    """
    Implements the multiplicative dropout layer for OFT.
    """

    def __init__(self, p=0.0):
        """
        Initializes the multiplicative dropout layer.

        Parameters:
        p (float): The probability of dropping out a block. Defaults to 0.0.
        """
        super().__init__()
        self.p = p

    def forward(self, x):
        """
        Applies multiplicative dropout to the input tensor.

        Parameters:
        x (Tensor): The input tensor of shape (D, H, H), where `D` represents
                    the number of OFT blocks, and `H` is the size of the square blocks along the last two dimensions,
                    the block size in OFT.
        """
        if self.training:
            # Ensure the last two dimensions are the same
            if x.shape[-1] != x.shape[-2]:
                raise ValueError("The last two dimensions of input should be the same!")

            D, H, _ = x.shape

            # If block share, skip the multiplicative dropout
            if D == 1:
                return x

            num_to_replace = int(self.p * D)
            num_zeros = D - num_to_replace
            mask = torch.cat(
                [
                    torch.ones(num_to_replace, device=x.device),
                    torch.zeros(num_zeros, device=x.device),
                ]
            )
            mask = mask[torch.randperm(D)].view(D, 1, 1)
            eye_matrix = torch.eye(H, device=x.device).repeat(D, 1, 1)
            x = (1 - mask) * x + mask * eye_matrix
        return x


class OFTConfig(ModifierConfig):
    r: int = 8
    oft_block_size: int = 0
    oft_dropout: float = 0.0
    coft: bool = False
    eps: float = 6e-5
    block_share: bool = False
    init_weights: Union[bool, str] = True
    bias: bool = (
        "none"  # metadata={"help": "Bias type for OFT. Can be 'none', 'all' or 'oft_only'"}
    )

    def __post_init__(self):
        if self.r == 0 and self.oft_block_size == 0:
            raise ValueError(
                f"Either `r` or `oft_block_size` must be non-zero. Currently, r = {self.r} and oft_block_size = {self.oft_block_size}."
            )
        if not (self.r != 0) ^ (self.oft_block_size != 0):
            raise ValueError(
                f"You can only specify either r ({self.r}) or oft_block_size ({self.oft_block_size}), but not both simultaneously, because r x oft_block_size == in_features."
            )


@Modifier.register("oft", config_cls=OFTConfig)
class OFTLayer(Modifier, ModifyMixin):
    """
    Implements the OFT layer from https://arxiv.org/pdf/2306.07280.
    """

    # All names of layers that may contain adapter weights
    adapter_layer_names = ("oft_r", "oft_s")
    # other_param_names is defined on parent class
    other_param_names = ("r", "oft_block_size", "oft_dropout")

    def __init__(self, config, layer: nn.Module, **kwargs) -> None:
        """
        Initializes the OFT layer.

        Note, currently only support linear layer and convolutional layer, with further support for other layers to be
        added soon.

        Parameters:
        base_layer: the pretrained model layer
        """
        super().__init__()

        self.base_layer = layer
        # OFT info
        self.config = config
        self.oft_r = nn.Parameter()  # nn.ParameterDict({})
        self.oft_s = nn.Parameter()  # nn.ParameterDict({})
        self.r = {}
        self.oft_block_size = {}
        self.oft_dropout = None
        self.coft = {}
        self.eps = {}
        self.block_share = {}
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        self.kwargs = kwargs

        if isinstance(self.base_layer, nn.Linear):
            in_features, out_features = (
                self.base_layer.in_features,
                self.base_layer.out_features,
            )
        elif isinstance(self.base_layer, nn.Conv2d):
            in_features, out_features = (
                self.base_layer.in_channels,
                self.base_layer.out_channels,
            )
        else:
            raise ValueError(f"Unsupported layer type {type(self.base_layer)}")

        self.in_features = in_features
        self.out_features = out_features

        self.update_layer(
            r=config.r,
            oft_block_size=config.oft_block_size,
            module_dropout=config.oft_dropout,
            coft=config.coft,
            eps=config.eps,
            block_share=config.block_share,
            init_weights=config.init_weights,
        )

    @property
    def _available_adapters(self) -> set[str]:
        return {*self.oft_r}

    def set_scale(self, adapter, scale):
        if adapter not in self.scaling:
            # Ignore the case where the adapter is not in the layer
            return

        warnings.warn(
            "Scaling operation for OFT not supported! Automatically set scale to 1."
        )

    def scale_layer(self, scale: float) -> None:
        if scale == 1:
            return

        for active_adapter in self.active_adapters:
            if active_adapter not in self.oft_r.keys():
                continue

            warnings.warn(
                "Scaling operation for OFT not supported! Automatically set scale to 1."
            )

    def unscale_layer(self, scale=None) -> None:
        for active_adapter in self.active_adapters:
            if active_adapter not in self.oft_r.keys():
                continue

            warnings.warn(
                "Unscaling operation for OFT not supported! Keeping scale to 1."
            )

    def update_layer(
        self,
        r,
        oft_block_size,
        module_dropout,
        coft,
        eps,
        block_share,
        init_weights,
    ):
        """
        Update the linear layer with trainable OFT weights. Override for other layer types.
        """
        """Internal function to create oft adapter

        Args:
            adapter_name (`str`): Name for the adapter to add.
            r (`int`): Rank for the added adapter.
            oft_block_size (`int`): The block size for added adapter.
            module_dropout (`float`):
                The multiplicative dropout probability for disabling adapter blocks during training.
            coft (`bool`): Whether to use the constrained variant of OFT or not.
            eps (`float`):
                The control strength of COFT. The freedom of rotation. Only has an effect if `coft` is set to True.
            block_share (`bool`): Whether to share the OFT parameters between blocks or not.
            init_weights (`bool`): Whether to initialize weights.
        """
        # Initialize the MultiplicativeDropoutLayer for module_dropout > 0.0.
        if module_dropout > 0.0:
            oft_dropout_layer = MultiplicativeDropoutLayer(p=module_dropout)
        else:
            oft_dropout_layer = nn.Identity()
        self.oft_dropout = oft_dropout_layer  # update(nn.ModuleDict({adapter_name: oft_dropout_layer}))

        if r == 0 and oft_block_size != 0:
            if (
                self.in_features % oft_block_size != 0
                or oft_block_size > self.in_features
            ):
                old_oft_block_size = oft_block_size
                oft_block_size = self.adjust_oft_parameters(
                    self.in_features, oft_block_size
                )
                warnings.warn(
                    f"Invalid `oft_block_size` ({old_oft_block_size})! Adjusted `oft_block_size` to ({oft_block_size})."
                )
            r = int(self.in_features // oft_block_size)
        elif r != 0 and oft_block_size == 0:
            if self.in_features % r != 0 or r > self.in_features:
                old_r = r
                r = self.adjust_oft_parameters(self.in_features, r)
                warnings.warn(f"Invalid `r` ({old_r})! Adjusted `r` to ({r}).")
            oft_block_size = int(self.in_features // r)
        else:
            raise ValueError(
                "Something went wrong, please report this error: https://github.com/huggingface/peft/issues"
            )

        self.coft = coft
        self.block_share = block_share
        self.eps = (
            eps * math.ceil(self.out_features / r) * math.ceil(self.out_features / r)
        )

        # Create weights with provided shape
        if block_share:
            self.oft_r = nn.Parameter(
                torch.empty(
                    1, math.ceil(self.in_features / r), math.ceil(self.in_features / r)
                )
            )
        else:
            self.oft_r = nn.Parameter(
                torch.empty(
                    r, math.ceil(self.in_features / r), math.ceil(self.in_features / r)
                )
            )
        self.oft_s = nn.Parameter(torch.empty(int(self.out_features), 1))

        # Initialize weights
        self.reset_oft_parameters(init_weights)

        # set oft r and block size
        self.r = r
        self.oft_block_size = oft_block_size

        # Move new weights to device
        # self._move_adapter_to_device_of_base_layer(adapter_name)
        # self.set_adapter(self.active_adapters)

    def reset_oft_parameters(self, init_weights):
        """
        Reset the OFT parameters.
        """
        if init_weights is False:
            nn.init.normal_(self.oft_r, mean=0.0, std=0.1)
            nn.init.normal_(self.oft_s, mean=1.0, std=0.1)
            return

        # if adapter_name in self.oft_r.keys():
        if init_weights is True:
            # initialize oft_r to zero
            nn.init.zeros_(self.oft_r)
            nn.init.ones_(self.oft_s)
        else:
            raise ValueError(f"Unknown initialization {init_weights=}")

    def _cayley_batch(self, data: torch.Tensor) -> torch.Tensor:
        """
        Perform the Cayley parametrization on a batch of skew-symmetric matrices.

        Args:
            data: A batch of skew-symmetric matrices of shape (b, r, c).
        """
        b, r, c = data.shape
        # Ensure the input matrix is skew-symmetric
        skew_mat = 0.5 * (data - data.transpose(1, 2))
        id_mat = (
            torch.eye(r, device=data.device).unsqueeze(0).expand(b, r, c)
        )  # noqa: E741

        # Perform the Cayley parametrization
        Q = torch.linalg.solve(id_mat + skew_mat, id_mat - skew_mat, left=False)

        return Q

    # Copied from https://github.com/Zeju1997/oft/blob/84cebb965df69781e3d9c3c875f5980b421eaf24/oft-control/oft.py#L155
    def _block_diagonal(self, oft_r: torch.Tensor, rank: int) -> torch.Tensor:
        if oft_r.shape[0] == 1:
            # block share
            blocks = [oft_r[0, ...] for i in range(rank)]
        else:
            blocks = [oft_r[i, ...] for i in range(rank)]

        # Use torch.block_diag to create the block diagonal matrix
        A = torch.block_diag(*blocks)

        return A

    # Copied from https://github.com/Zeju1997/oft/blob/84cebb965df69781e3d9c3c875f5980b421eaf24/oft-control/oft.py#L52
    def _project_batch(self, oft_r, eps=1e-5):
        # scaling factor for each of the smaller block matrix
        eps = eps * 1 / torch.sqrt(torch.tensor(oft_r.shape[0]))
        I = (  # noqa: E741
            torch.zeros(
                (oft_r.size(1), oft_r.size(1)), device=oft_r.device, dtype=oft_r.dtype
            )
            .unsqueeze(0)
            .expand_as(oft_r)
        )
        diff = oft_r - I
        norm_diff = torch.norm(oft_r - I, dim=(1, 2), keepdim=True)
        mask = (norm_diff <= eps).bool()
        out = torch.where(mask, oft_r, I + eps * (diff / norm_diff))
        return out

    def adjust_oft_parameters(self, in_features, params):
        """
        Adjust the OFT parameters to be divisible by the in_features dimension.
        """
        if params < in_features:
            higher_params = params
            while higher_params <= in_features and in_features % higher_params != 0:
                higher_params += 1
        else:
            return in_features

        lower_params = params
        while lower_params > 1 and in_features % lower_params != 0:
            lower_params -= 1

        if (params - lower_params) <= (higher_params - params):
            return lower_params
        else:
            return higher_params

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        previous_dtype = x.dtype
        oft_rotation = torch.eye(
            self.in_features, device=x.device, dtype=previous_dtype
        )
        oft_scale = torch.ones(
            (int(self.out_features), 1), device=x.device, dtype=previous_dtype
        )

        oft_r = self.oft_r
        oft_s = self.oft_s
        dropout = self.oft_dropout

        rank = self.r
        coft = self.coft
        eps = self.eps

        if coft:
            with torch.no_grad():
                oft_r.copy_(self._project_batch(oft_r, eps=eps))

        orth_rotate = self._cayley_batch(oft_r)
        orth_rotate = dropout(orth_rotate)
        oft_mat = self._block_diagonal(orth_rotate, rank)

        oft_rotation = oft_mat @ oft_rotation
        oft_scale = oft_s * oft_scale

        x = x.to(self.base_layer.weight.data.dtype)

        orig_weight = self.base_layer.weight.data
        orig_weight = torch.transpose(orig_weight, 0, 1)
        oft_rotation = oft_rotation.to(previous_dtype)
        orig_weight = orig_weight.to(previous_dtype)
        rotated_weight = torch.mm(oft_rotation, orig_weight)
        rotated_weight = torch.transpose(rotated_weight, 0, 1)

        scaled_rotated_weight = rotated_weight * oft_scale

        scaled_rotated_weight = scaled_rotated_weight.to(previous_dtype)
        bias = (
            self.base_layer.bias.to(previous_dtype)
            if self.base_layer.bias is not None
            else None
        )
        result = F.linear(input=x, weight=scaled_rotated_weight, bias=bias)

        result = result.to(previous_dtype)
        return result
