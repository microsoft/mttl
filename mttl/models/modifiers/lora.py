from dataclasses import dataclass
import os
import re
from typing import List
import numpy as np
import torch
from torch import nn
import math
from torch import nn
import torch
import math
import bitsandbytes as bnb

from mttl.utils import logger
from mttl.models.modifiers import register_modifier
from mttl.models.modifiers.base import (
    MergeableAdapter,
    ModifyMixin,
    ModifierConfig,
    Adapter,
)


@dataclass
class LoRAConfig(ModifierConfig):
    lora_rank: int = 4
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    lora_init_b_random: bool = False


@register_modifier("lora", config_cls=LoRAConfig)
class LoRA(MergeableAdapter, ModifyMixin):
    def __init__(
        self,
        config: LoRAConfig,
        layer: nn.Module,
        **kwargs,
    ):
        super().__init__()

        # assign self variables
        self.config = config
        self.rank = config.lora_rank
        self.alpha = config.lora_alpha
        self.dropout = config.lora_dropout
        self.in_features = layer.in_features
        self.out_features = layer.out_features
        self.init_b_random = config.lora_init_b_random
        self.training_steps = 0.0
        self.scaling = self.alpha / self.rank
        self.forward_fn = None
        self.layer = layer

        if self.dropout > 0.0:
            self.dropout_layer = nn.Dropout(self.dropout)
        else:
            self.dropout_layer = lambda x: x

        if hasattr(layer, "weight"):
            self.weight = layer.weight

        if hasattr(layer, "bias"):
            self.bias = layer.bias

        self.create_for_layer(layer)
        self.reset_parameters()
        self.merged_with_layer = False

    def load_lora_weights(self, state_dict):
        self.lora_a.data.copy_(state_dict["lora_a"])
        self.lora_b.data.copy_(state_dict["lora_b"])

    def merge_with_layer(self):
        """Merge this adapter with the layer!"""
        if isinstance(self.layer, nn.Linear):
            self.merged_with_layer = True
            # for back-compatibility, try the two sides:
            if self.lora_a.data.shape[0] == self.layer.weight.shape[0]:
                to_merge = self.lora_a.data @ self.lora_b.data
            else:
                to_merge = (self.lora_a.data @ self.lora_b.data).T
            to_merge = to_merge * self.scaling

            if isinstance(self.layer, bnb.nn.Linear8bitLt):
                if self.layer.state.SCB is None:
                    self.layer.state.SCB = self.layer.weight.SCB

                # Dequantize the result of identity matrix and int8 weight because bitsandbytes does not support int8
                # dequantization directly
                im = (
                    torch.eye(self.layer.weight.data.shape[-1])
                    .contiguous()
                    .half()
                    .to(self.weight.device)
                )
                im, imt, SCim, SCimt, coo_tensorim = bnb.functional.double_quant(im)
                im, Sim = bnb.functional.transform(im, "col32")

                if self.layer.state.CxB is None:
                    (
                        self.layer.state.CxB,
                        self.layer.state.SB,
                    ) = bnb.functional.transform(
                        self.layer.weight.data, to_order=self.layer.state.formatB
                    )

                out32, Sout32 = bnb.functional.igemmlt(
                    im, self.layer.state.CxB, Sim, self.layer.state.SB
                )
                output = bnb.functional.mm_dequant(
                    out32, Sout32, SCim, self.layer.state.SCB, bias=None
                ).t()
                w_data = output.to(to_merge.dtype).to(to_merge.device) + to_merge

                self.layer.weight = bnb.nn.Int8Params(
                    w_data.to("cpu"),
                    requires_grad=False,
                    has_fp16_weights=self.layer.weight.has_fp16_weights,
                ).to(self.layer.weight.device)
                self.layer.state.reset_grads()
            else:
                self.layer.weight.data.add_(to_merge.to(self.layer.weight.device))
        else:
            raise NotImplementedError("LoRA only supports nn.Linear layers.")

    def create_for_layer(self, layer):
        if isinstance(layer, nn.Linear):
            self.lora_a = nn.Parameter(
                torch.empty(layer.in_features, self.rank).to(
                    device=layer.weight.device
                ),
            )
            self.lora_b = nn.Parameter(
                torch.empty(self.rank, layer.out_features).to(
                    device=layer.weight.device
                ),
            )
            self.forward_fn = self.forward_linear_
        else:
            raise NotImplementedError("LoRA only supports nn.Linear layers.")

    def forward_linear_(self, input, **kwargs):
        output = self.layer(input)
        if self.merged_with_layer:
            return output
        else:
            input_lora = input.to(self.lora_a.dtype)
            input_lora = self.dropout_layer(input_lora)
            adapter_out = (
                torch.matmul(torch.matmul(input_lora, self.lora_a), self.lora_b)
                * self.scaling
            )
            return output + adapter_out.to(input.dtype)

    @classmethod
    def parallel_linear_forward(cls, input, loras):
        if any([lora.merged_with_layer for lora in loras]):
            raise ValueError("Cannot parallelize merged loras.")
        if len(set([lora.layer for lora in loras])) > 1:
            raise ValueError("Cannot parallelize loras applied to different layers.")

        # (batch, in_features, rank)
        lora_a = torch.stack([lora.lora_a for lora in loras], dim=0)
        # (batch, rank, out_features)
        lora_b = torch.stack([lora.lora_b for lora in loras], dim=0)

        # (batch,)
        scaling = torch.cat(
            [torch.FloatTensor([lora.scaling]) for lora in loras], dim=0
        ).to(device=lora_a.device, dtype=lora_a.dtype)

        # (n_examples, seq_len, out_features)
        layer_out = loras[0].layer(input)
        input_lora = input.to(loras[0].lora_a.dtype)
        input_lora = loras[0].dropout_layer(input_lora)

        adapter_out = (
            torch.bmm(torch.bmm(input_lora, lora_a), lora_b) * scaling[:, None, None]
        )
        return layer_out + adapter_out.to(dtype=input.dtype)

    def reset_parameters(self):
        gain = nn.init.calculate_gain(nonlinearity="leaky_relu", param=math.sqrt(5))
        std = gain / math.sqrt(self.in_features)
        with torch.no_grad():
            self.lora_a.uniform_(-std, std)

        # ensure that initially, adding the adapter does not change the output
        if self.init_b_random:
            with torch.no_grad():
                self.lora_b.uniform_(-std, std)
        else:
            torch.nn.init.zeros_(self.lora_b)

    def forward(self, *args, **kwargs):
        return self.forward_fn(*args, **kwargs)


@dataclass
class SkilledLoRAConfig(LoRAConfig):
    n_skills: int = 1
    n_splits: int = 1
    phi_2_align_heads: bool = False


@register_modifier("skilled_lora", config_cls=SkilledLoRAConfig)
class SkilledLoRA(LoRA):
    def __init__(
        self,
        config: SkilledLoRAConfig,
        layer: nn.Module,
        **kwargs,
    ):
        self.n_splits = config.n_splits
        self.n_skills = config.n_skills
        super().__init__(config, layer)

    def __len__(self):
        return self.n_skills

    def add_skill(self, lora: LoRA):
        self.n_skills += 1

        self.lora_a.data = torch.cat(
            [
                self.lora_a.data,
                lora.lora_a.data.reshape(
                    1, self.n_splits, self.in_features // self.n_splits, self.rank
                ).to(device=self.lora_a.device, dtype=self.lora_a.dtype),
            ],
            dim=0,
        )
        self.lora_b.data = torch.cat(
            [
                self.lora_b.data,
                lora.lora_b.data.reshape(
                    1, self.rank, self.n_splits, self.out_features // self.n_splits
                ).to(device=self.lora_a.device, dtype=self.lora_a.dtype),
            ],
            dim=0,
        )

    def create_for_layer(self, layer):
        if isinstance(layer, nn.Linear):
            self.lora_a = nn.Parameter(
                torch.empty(
                    self.n_skills,
                    self.n_splits,
                    layer.in_features // self.n_splits,
                    self.rank,
                ).to(device=self.weight.device)
            )
            self.lora_b = nn.Parameter(
                torch.empty(
                    self.n_skills,
                    self.rank,
                    self.n_splits,
                    layer.out_features // self.n_splits,
                ).to(device=self.weight.device)
            )
            self.forward_fn = self.forward_linear_
        else:
            raise NotImplementedError("SkilledLoRA only supports nn.Linear layers.")

    def forward_linear_(self, input, weights):
        layer_out = self.layer(input)
        input_lora = input.to(self.lora_a.dtype)
        input_lora = self.dropout_layer(input_lora)

        bs = input.size(0)

        # Standard polytropon routing : (batch_size, dim_in, dim_out)
        if weights.ndim < 4:
            # these are task ids
            if weights.ndim == 1:
                # use indexing!
                wrm_steps = 0
                if self.training_steps < wrm_steps:
                    A = self.lora_a[torch.zeros_like(weights).long()]
                    B = self.lora_b[torch.zeros_like(weights).long()]
                else:
                    if self.training_steps == wrm_steps:
                        self.lora_a.data.copy_(
                            self.lora_a.data[:1].repeat(self.n_skills, 1, 1, 1)
                        )
                        self.lora_b.data.copy_(
                            self.lora_b.data[:1].repeat(self.n_skills, 1, 1, 1)
                        )
                    A = self.lora_a[weights.long(), :, :, :]
                    B = self.lora_b[weights.long(), :, :, :]
            elif weights.ndim == 3:
                weights = weights.to(self.lora_a.dtype)
                A = torch.einsum("bqs,sqdr->bqdr", (weights, self.lora_a))
                B = torch.einsum("bqs,srqd->brqd", (weights, self.lora_b))

            A = A.reshape(bs, self.in_features, self.rank)
            B = B.reshape(bs, self.rank, self.out_features)
            adapter_out = input_lora.bmm(A).bmm(B) * self.scaling

        # Per Token Routing : (batch_size, seq_len, dim_in, dim_out)
        elif weights.ndim == 4:
            weights = weights.to(self.lora_a.dtype)
            # b: batch, l: seq_len, d: d_in/d_out, r: rank
            A = torch.einsum("blqs,sqdr->blqdr", (weights, self.lora_a))
            B = torch.einsum("blqs,srqd->blqrd", (weights, self.lora_b))
            A = A.reshape(bs, -1, self.in_features, self.rank)
            B = B.transpose(2, 3).reshape(bs, -1, self.rank, self.out_features)
            adapter_out = torch.einsum("bld,bldr->blr", (input_lora, A))
            adapter_out = torch.einsum("blr,blrd->bld", (adapter_out, B)) * self.scaling

        return layer_out + adapter_out.to(input.dtype)

    @classmethod
    def parallel_linear_forward(cls, input, skilled_loras, weights):
        return cls.parallel_linear_weighted_forward(input, skilled_loras, weights)

    @classmethod
    def parallel_linear_weighted_forward(
        cls,
        input: torch.Tensor,
        skilled_loras: List["SkilledLoRAView"],
        weights: List[torch.Tensor],
        merge_after: bool = False,
    ):
        """
        Executes multiple skilled loras in parallel.

        I.e. this is useful for the situations in which each example in the batch
        need to be processed by a different combination of skills,
              --> skills     --> weights
        ex1 : [[a, d, f]     [[0.1, 0.2, 0.7]
        ex2 :  [c, g, h]]     [0.3, 0.4, 0.3]]

        This also handles the case in which the same skilled lora is applied to multiple examples,
        in this case, we broadcast the same combination to all the examples in the batch,
              --> skills      --> weights
        *   : [[a, d, f]]     [[0.1, 0.2, 0.7]]

        It also handles another case, in which we have a single shared skilled lora applied with different weights
        depending on the example,
              --> skills      --> weights
        *   : [[a, d, f]]     [[0.1, 0.2, 0.7],
                               [0.3, 0.4, 0.3]]
        """

        if len(set([lora.layer for lora in skilled_loras])) > 1:
            raise ValueError("Cannot parallelize loras applied to different layers.")

        device = skilled_loras[0].lora_a.device
        n_skills = skilled_loras[0].lora_a.shape[0]
        assert np.all(skl.n_skills == n_skills for skl in skilled_loras)

        num_skilled_loras = len(skilled_loras)

        if num_skilled_loras == 1:
            skilled_loras_a = skilled_loras[0].lora_a.unsqueeze(0)
            skilled_loras_b = skilled_loras[0].lora_b.unsqueeze(0)
        else:
            skilled_loras_a = torch.stack(
                [lora.lora_a for lora in skilled_loras], dim=0
            )
            skilled_loras_b = torch.stack(
                [lora.lora_b for lora in skilled_loras], dim=0
            )

        if type(weights) == list:
            weights = torch.stack(weights, dim=0).to(device)

        # assert skilled_loras_a.shape[2] == 1, "Only 1 split is supported for now."
        # assert skilled_loras_b.shape[3] == 1, "Only 1 split is supported for now."
        # skilled_loras_a = skilled_loras_a.squeeze(2)
        # skilled_loras_b = skilled_loras_b.squeeze(3)

        # (n_examples, seq_len, out_features)
        layer_out = skilled_loras[0].layer(input)
        phi_2_align_heads = skilled_loras[0].config.phi_2_align_heads

        input_lora = input.to(skilled_loras[0].lora_a.dtype)
        input_lora = skilled_loras[0].dropout_layer(input_lora)
        weights = weights.to(dtype=skilled_loras[0].lora_a.dtype)

        # (n_examples,)
        scaling = torch.cat(
            [torch.FloatTensor([lora.scaling]) for lora in skilled_loras], dim=0
        ).to(device=device, dtype=skilled_loras[0].lora_a.dtype)

        # make all ops in float32
        if num_skilled_loras == 1:
            # no batch, skilled lora is shared across all examples, remove batch dimension
            skilled_loras_a = skilled_loras_a.squeeze(0)
            skilled_loras_b = skilled_loras_b.squeeze(0)
            weights = weights.squeeze(0)
            scaling = scaling.squeeze(0)

            if weights.ndim == 1:
                assert not phi_2_align_heads
                # skilled_loras_a is skills x split x d x r
                # skilled_loras_b is skills x r x split x d
                if merge_after:
                    A, B = skilled_loras_a.flatten(1, 2), skilled_loras_b.flatten(2, 3)
                    # A skills x d x r
                    # B skills x r x d
                    # partial_out = torch.einsum("bd,sdr->bsr", (input_lora, A))
                    # adapter_out = torch.einsum("bsr,srd->sbd", (partial_out, B))
                    # adapter_out = torch.einsum("s,sbo->bo", (weights, adapter_out)) * scaling
                    # should be the same as:
                    adapter_out = torch.matmul(torch.matmul(input_lora, A), B)
                    adapter_out = (
                        torch.einsum("s,sbo->bo", (weights, adapter_out)) * scaling
                    )
                else:
                    A = torch.einsum("s,sqdr->qdr", (weights, skilled_loras_a))
                    B = torch.einsum("s,srqd->rqd", (weights, skilled_loras_b))

                    # combine n_splits, and d_split into a single dimension
                    A, B = A.flatten(0, 1), B.flatten(1, 2)

                    # scaling is a float (only 1 skilled lora)
                    adapter_out = torch.matmul(torch.matmul(input_lora, A), B) * scaling
            elif weights.ndim == 2:
                assert not phi_2_align_heads
                # we are in the case in which we have a single skilled lora applied with different weights
                if merge_after:
                    # skilled_loras_a is skills x split x d x r
                    # skilled_loras_b is skills x r x split x d
                    A, B = skilled_loras_a.flatten(1, 2), skilled_loras_b.flatten(2, 3)
                    # A skills x d x r
                    # B skills x r x d
                    if input_lora.ndim == 2:
                        adapter_out = torch.matmul(torch.matmul(input_lora, A), B)
                        adapter_out = (
                            torch.einsum("bs,sbo->bo", (weights, adapter_out)) * scaling
                        )
                    elif input_lora.ndim == 3:
                        partial_out = torch.einsum("bkd,sdr->sbkr", (input_lora, A))
                        adapter_out = torch.einsum("sbkr,srd->sbkd", (partial_out, B))
                        adapter_out = (
                            torch.einsum("bs,sbkd->bkd", (weights, adapter_out))
                            * scaling
                        )

                else:
                    A = torch.einsum("bs,sqdr->bqdr", (weights, skilled_loras_a))
                    B = torch.einsum("bs,srqd->brqd", (weights, skilled_loras_b))

                    # combine n_splits, and d_split into a single dimension
                    A, B = A.flatten(1, 2), B.flatten(2, 3)

                    if input_lora.ndim == 2:
                        partial_out = torch.einsum("bd,bdr->br", (input_lora, A))
                        adapter_out = torch.einsum("br,brd->bd", (partial_out, B))
                        adapter_out = adapter_out * scaling
                    elif input_lora.ndim == 3:
                        adapter_out = torch.bmm(torch.bmm(input_lora, A), B) * scaling
                    else:
                        raise NotImplementedError(
                            "Only 2D and 3D inputs are supported."
                        )
            elif weights.ndim == 3:
                if merge_after:
                    raise ValueError("Merge after is not supported for 3D weights.")
                else:
                    # we are in the case in which we have a single skilled lora applied with different weights
                    A = torch.einsum("bqs,sqdr->bqdr", (weights, skilled_loras_a))
                    B = torch.einsum("bqs,srqd->brqd", (weights, skilled_loras_b))

                    if (
                        phi_2_align_heads and B.size(-1) // A.size(-2) == 3
                    ):  # last only true for Wqkv weight
                        # phi_2 formats the B as  "... (three h d) -> ... three h d"
                        # We want to make sure that the `h` here aligns with n_splits, or `q` index
                        bs, rank, n_splits, d_split = B.shape
                        # (h, 3 * d) -> (h, 3, d)
                        B = B.view(bs, rank, n_splits, 3, d_split // 3)
                        # (bs, r, h, 3, d) -> (bs, r, 3, h, d) -> ... (bs, r, 3 * h * d)
                        B = B.transpose(2, 3).reshape(bs, rank, n_splits, d_split)

                    # combine n_splits, and d_split into a single dimension
                    A, B = A.flatten(1, 2), B.flatten(2, 3)

                    if input_lora.ndim == 2:
                        partial_out = torch.einsum("bd,bdr->br", (input_lora, A))
                        adapter_out = torch.einsum("br,brd->bd", (partial_out, B))
                        adapter_out = adapter_out * scaling
                    elif input_lora.ndim == 3:
                        adapter_out = torch.bmm(torch.bmm(input_lora, A), B) * scaling
                    else:
                        raise NotImplementedError(
                            "Only 2D and 3D inputs are supported."
                        )
        elif n_skills == 1:
            # this is basically standard lora forward, we are here by accident
            # !!!warning!!!! this ignores the weights
            return LoRA.parallel_linear_forward(
                input, [sk_lora.to_loras()[0] for sk_lora in skilled_loras]
            )
        else:
            assert skilled_loras_a.shape[2] == 1, "Only 1 split is supported for now."
            assert skilled_loras_b.shape[3] == 1, "Only 1 split is supported for now."
            skilled_loras_a = skilled_loras_a.squeeze(2)
            skilled_loras_b = skilled_loras_b.squeeze(3)
            # skilled_loras_a is batch x skills x d x r
            # skilled_loras_b is batch x skills x r x d
            if merge_after:
                if input_lora.ndim == 2:
                    partial_out = torch.einsum(
                        "bd,bsdr->sbr", (input_lora, skilled_loras_a)
                    )
                    adapter_out = torch.einsum(
                        "sbr,bsro->sbo", (partial_out, skilled_loras_b)
                    )
                    adapter_out = torch.einsum("bs,sbo->bo", (weights, adapter_out))
                    adapter_out = adapter_out * scaling[:, None]

                elif input_lora.ndim == 3:
                    partial_out = torch.einsum(
                        "bkd,bsdr->sbkr", (input_lora, skilled_loras_a)
                    )
                    adapter_out = torch.einsum(
                        "sbkr,bsrd->sbkd", (partial_out, skilled_loras_b)
                    )
                    adapter_out = torch.einsum("bs,sbko->bko", (weights, adapter_out))
                    adapter_out = adapter_out * scaling[:, None, None]
                else:
                    raise NotImplementedError("Only 2D and 3D inputs are supported.")

            else:
                A = torch.einsum("bs,bsdr->bdr", (weights, skilled_loras_a))
                B = torch.einsum("bs,bsrd->brd", (weights, skilled_loras_b))

                # (n_examples, out_features)
                if input_lora.ndim == 2:
                    partial_out = torch.einsum("bd,bdr->br", (input_lora, A))
                    adapter_out = torch.einsum("br,brd->bd", (partial_out, B))
                    adapter_out = adapter_out * scaling[:, None]
                # (n_examples, seq_len, out_features)
                else:
                    partial_out = torch.einsum("bsd,bdr->bsr", (input_lora, A))
                    adapter_out = torch.einsum("bsr,brd->bsd", (partial_out, B))
                    adapter_out = adapter_out * scaling[:, None, None]

        # adapter out is float32
        return layer_out + adapter_out.to(dtype=input.dtype)


class LoRAView(LoRA):
    """
    Avoid initializing parameters, the parameters are just a view
    on a bunch of other LoRAs parameters stacked together.
    """

    def __init__(self, config, layer, lora_a, lora_b, **kwargs):
        super().__init__(config, layer)
        self.lora_a = lora_a
        self.lora_b = lora_b

        if isinstance(layer, nn.Linear):
            self.forward_fn = self.forward_linear_
        else:
            raise NotImplementedError("LoRAView only supports nn.Linear layers.")

    def create_for_layer(self, layer):
        pass

    def reset_parameters(self):
        pass


class SkilledLoRAView(SkilledLoRA):
    """
    Avoid initializing parameters, the parameters are just a view
    on a bunch of other LoRAs parameters stacked together.
    """

    def __init__(self, config, layer, lora_a, lora_b):
        super().__init__(config, layer)
        self.lora_a = lora_a
        self.lora_b = lora_b

    def create_for_layer(self, layer):
        pass

    def reset_parameters(self):
        pass

    def to_loras(self):
        """
        Create a list of loras from a skilled lora
        """
        if self.n_splits > 1:
            raise ValueError("Cannot convert a skilled lora with n_splits > 1.")

        loras = []
        for i in range(self.n_skills):
            # squeeze n_splits if any
            lora = LoRAView(
                self.config,
                self.layer,
                self.lora_a[i].squeeze(),
                self.lora_b[i].squeeze(),
            )
            loras.append(lora)
        return loras

    @classmethod
    def from_loras(cls, loras):
        """
        Create a skilled lora from a list of loras
        """
        if len(set([lora.layer for lora in loras])) > 1:
            raise ValueError("Cannot create a SkilledLora from different loras.")

        config = SkilledLoRAConfig(
            lora_rank=loras[0].config.lora_rank,
            lora_alpha=loras[0].config.lora_alpha,
            lora_dropout=loras[0].config.lora_dropout,
            lora_init_b_random=loras[0].config.lora_init_b_random,
            n_skills=len(loras),
            n_splits=1,
        )
        skilled_lora = cls(
            config,
            loras[0].layer,
            lora_a=torch.stack([lora.lora_a for lora in loras], dim=0).unsqueeze(1),
            lora_b=torch.stack([lora.lora_b for lora in loras], dim=0).unsqueeze(2),
        )
        return skilled_lora


class TiedLoRAConfig(LoRAConfig):
    pass


@register_modifier("tied_lora", config_cls=TiedLoRAConfig)
class TiedLoRAForKQV(Adapter):
    """
    Makes sure that lora_a (d_in x r) is shared across layers.
    """

    def __init__(self, config, layer, layer_name):
        super().__init__()
        self.config = config
        self.rank = config.lora_rank

        self.layers = [layer]
        self.layer_names = [layer_name]
        self.lora_bs = nn.ParameterList([self.create_lora_b()])
        self.create_lora_a()  # shared across layers

    def add_layer(self, layer, layer_name):
        if len(self.layers) > 0:
            assert layer.out_features == self.layers[-1].out_features
            assert layer.in_features == self.layers[-1].in_features

        self.layers.append(layer)
        self.layer_names.append(layer_name)

        self.lora_bs.append(self.create_lora_b())

    def create_lora_a(self):
        self.lora_a = nn.Parameter(
            torch.empty(self.layers[0].in_features, self.rank).to(
                device=self.layers[0].weight.device
            ),
        )
        gain = nn.init.calculate_gain(nonlinearity="leaky_relu", param=math.sqrt(5))
        std = gain / math.sqrt(self.layers[0].in_features)
        with torch.no_grad():
            self.lora_a.uniform_(-std, std)

    def create_lora_b(self):
        """
        Create A and B s.t. A is d_in x r , B is r x (d_out x len(layers))
        """
        lora_b = nn.Parameter(
            torch.empty(self.rank, self.layers[0].out_features).to(
                device=self.layers[0].weight.device
            ),
        )
        torch.nn.init.zeros_(lora_b)
        return lora_b

    def get_view_for_layer(self, i):
        return LoRAView(
            self.config,
            self.layers[i],
            self.lora_a,  # shared
            self.lora_bs[i],  # specific
        )

    @classmethod
    def modify_transformer(cls, transformer, config):
        for m_name, module in dict(transformer.named_modules()).items():
            if re.fullmatch(config.modify_modules, m_name):
                # tie within the parent module
                tied_adapter = None
                for c_name, layer in dict(module.named_children()).items():
                    if re.fullmatch(config.modify_layers, c_name):
                        logger.info(f"Patching {m_name}.{c_name}...")

                        if re.fullmatch(config.tie_layers, c_name):
                            if tied_adapter is None:
                                tied_adapter = cls(config, layer, c_name)
                            else:
                                tied_adapter.add_layer(layer, c_name)
                            continue

                        setattr(
                            module,
                            c_name,
                            LoRA(config, layer),
                        )
                if tied_adapter is not None:
                    for i, (c_name, layer) in enumerate(
                        zip(tied_adapter.layer_names, tied_adapter.layers)
                    ):
                        setattr(
                            module,
                            c_name,
                            tied_adapter.get_view_for_layer(i),
                        )

        return transformer
