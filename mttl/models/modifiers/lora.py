import math
from dataclasses import dataclass
from typing import List, Union

import bitsandbytes as bnb
import numpy as np
import torch
from torch import nn

from mttl.logging import warn_once
from mttl.models.modifiers.base import MergeableModifierMixin, Modifier, ModifierConfig


@dataclass
class LoRAConfig(ModifierConfig):
    lora_rank: int = 4
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    lora_init_b_random: bool = False


@Modifier.register("lora", config_cls=LoRAConfig)
class LoRA(Modifier, MergeableModifierMixin):
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

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        """Override state dict for this adapter to avoid saving layer weights."""
        state_dict = super().state_dict(
            *args, destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        return {n: v for n, v in state_dict.items() if "lora" in n}

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

        if self.merged_with_layer or not self.enabled:
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

        if len(loras) not in [1, input.shape[0]]:
            raise ValueError("Needed either 1 lora or as many batch examples.")

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

        if lora_a.size(0) == 1:
            lora_a, lora_b = lora_a.squeeze(0), lora_b.squeeze(0)
            adapter_out = torch.einsum("bsi,ir->bsr", (input_lora, lora_a))
            adapter_out = (
                torch.einsum("bsr,ro->bso", (adapter_out, lora_b))
                * scaling[:, None, None]
            )
        else:
            adapter_out = (
                torch.bmm(torch.bmm(input_lora, lora_a), lora_b)
                * scaling[:, None, None]
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


@Modifier.register("skilled_lora", config_cls=SkilledLoRAConfig)
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

    def get_skill_weights(self, skill_index):
        if skill_index >= self.n_skills:
            raise ValueError(f"Skill index {skill_index} out of bounds.")

        return {
            "lora_a": self.lora_a[skill_index].unsqueeze(0).clone().detach().cpu(),
            "lora_b": self.lora_b[skill_index].unsqueeze(0).clone().detach().cpu(),
        }

    def set_skill(self, lora: Union[LoRA, "SkilledLoRA"], skill_index):
        """Copy the weights of the given lora to the given skill index."""
        if skill_index >= self.lora_a.data.shape[0]:
            raise ValueError(f"Skill index {skill_index} out of bounds.")

        self.lora_a.data[skill_index] = lora.lora_a.data.reshape(
            1, self.n_splits, self.in_features // self.n_splits, self.rank
        ).to(device=self.lora_a.device, dtype=self.lora_a.dtype)

        self.lora_b.data[skill_index] = lora.lora_b.data.reshape(
            1, self.rank, self.n_splits, self.out_features // self.n_splits
        ).to(device=self.lora_a.device, dtype=self.lora_a.dtype)

    def add_skill(self, lora: Union[LoRA, "SkilledLoRA"]) -> None:
        """Adds a skill to the skilled lora by copying the weights of the given lora."""
        self.lora_a.data = torch.cat(
            [
                self.lora_a.data,
                torch.zeros(1, *self.lora_a.data.shape[1:]).to(
                    device=self.lora_a.device, dtype=self.lora_a.dtype
                ),
            ]
        )
        self.lora_b.data = torch.cat(
            [
                self.lora_b.data,
                torch.zeros(1, *self.lora_b.data.shape[1:]).to(
                    device=self.lora_b.device, dtype=self.lora_b.dtype
                ),
            ]
        )

        self.set_skill(lora, self.n_skills)
        self.n_skills += 1

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

        if not self.enabled:
            return layer_out

        input_lora = input.to(self.lora_a.dtype)
        input_lora = self.dropout_layer(input_lora)

        bs = input.size(0)
        if weights.ndim == 1:
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

        # Standard polytropon routing : (batch_size, dim_in, dim_out)
        elif weights.ndim == 3:
            weights = weights.to(self.lora_a.dtype)
            A = torch.einsum("bqs,sqdr->bqdr", (weights, self.lora_a))
            B = torch.einsum("bqs,srqd->brqd", (weights, self.lora_b))

        A = A.reshape(bs, self.in_features, self.rank)
        B = B.reshape(bs, self.rank, self.out_features)
        adapter_out = input_lora.bmm(A).bmm(B) * self.scaling

        return layer_out + adapter_out.to(input.dtype)

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
    def parallel_linear_weighted_forward(
        cls,
        input: torch.Tensor,
        skilled_loras: List["SkilledLoRAView"],
        weights: torch.Tensor,
        dim_names: List[str],
        merge_after: bool = False,
    ):
        """
        Executes multiple skilled loras in parallel, weights are stored in `weights`.

        We handle different scenarios here, situations in which each example in the batch
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

        We handle all these scenarios at once, by creating a weights tensor of size ["batch", "skills", "splits", "experts"]

        dim_names specifies the names of the dimensions currently in the weights tensor, e.g. ["batch", "experts"],
        we unsqueeze the remaining dimensions.
        """
        if len(set([lora.layer for lora in skilled_loras])) > 1:
            raise ValueError("Cannot parallelize loras applied to different layers.")

        if len(dim_names) != weights.ndim:
            raise ValueError("Not all dimensions are present in the weights tensor.")

        device = skilled_loras[0].lora_a.device
        n_skills = skilled_loras[0].lora_a.shape[0]
        assert np.all(skl.n_skills == n_skills for skl in skilled_loras)

        if n_skills == 1:
            # For Phatgoose, we have a single skill, but we still need a selector
            warn_once(
                f"You are using Skilled LoRA with only one skill. Make sure this is needed"
            )

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

        expected_dims = ["batch", "sequence", "splits", "experts"]
        for i, dim in enumerate(expected_dims):
            if dim not in dim_names:
                weights = weights.unsqueeze(i)
        weights = weights.to(dtype=skilled_loras[0].lora_a.dtype)

        # (n_examples, seq_len, out_features)
        layer_out = skilled_loras[0].layer(input)

        input_lora = input.to(skilled_loras[0].lora_a.dtype)
        input_lora = skilled_loras[0].dropout_layer(input_lora)

        # (n_examples,)
        scaling = torch.cat(
            [torch.FloatTensor([lora.scaling]) for lora in skilled_loras], dim=0
        ).to(device=device, dtype=skilled_loras[0].lora_a.dtype)

        # (batch, dimension)
        if input_lora.ndim == 2:
            input_lora = input_lora.unsqueeze(1)

        # b = batch
        # l = sequence
        # q = splits
        # e = experts
        if merge_after:
            partial_out = torch.einsum(
                "bld,beqdr->bleqr", (input_lora, skilled_loras_a)
            )
            adapter_out = torch.einsum(
                "bleqr,berqd->bleqd", (partial_out, skilled_loras_b)
            )
            adapter_out = torch.einsum("blqe,bleqd->blqd", (weights, adapter_out))
            adapter_out = adapter_out.flatten(2, 3)
        else:
            A = torch.einsum("blqe,beqdr->blqdr", (weights, skilled_loras_a))
            B = torch.einsum("blqe,berqd->blrqd", (weights, skilled_loras_b))
            batch_size, sequence_length, rank, n_splits, d_split = B.shape

            # flatten the "splits" (q) dimension
            A, B = A.flatten(2, 3), B.flatten(3, 4)

            partial_out = torch.einsum("bld,bldr->blr", (input_lora, A))
            adapter_out = torch.einsum("blr,blrd->bld", (partial_out, B))

        adapter_out = adapter_out * scaling

        # squeeze again sequence dimension ("l") if needed
        if layer_out.ndim == 2:
            adapter_out = adapter_out.squeeze(1)

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
