from dataclasses import dataclass
from typing import List
import torch
from torch import nn
import math
from torch import nn
import torch
import math
import bitsandbytes as bnb

from mttl.models.modifiers import register_modifier
from mttl.models.modifiers.base import MergeableAdapter, ModifyMixin, ModifierConfig


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

        # (n_examples, in_features, rank)
        lora_a = torch.stack([lora.lora_a for lora in loras], dim=0)
        # (n_examples, rank, out_features)
        lora_b = torch.stack([lora.lora_b for lora in loras], dim=0)
        # (n_examples,)
        scaling = torch.cat(
            [torch.FloatTensor([lora.scaling]) for lora in loras], dim=0
        ).to(device=lora_a.device)

        # (n_examples, seq_len, out_features)
        layer_out = loras[0].layer(input)
        input_lora = input.to(loras[0].lora_a.dtype)

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


class SkilledLoRA(LoRA):
    def __init__(
        self,
        config: SkilledLoRAConfig,
        layer: nn.Module,
    ):
        self.n_splits = config.n_splits
        self.n_skills = config.n_skills
        super().__init__(config, layer)

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
        bs = input.size(0)

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
        else:
            weights = weights.to(self.lora_a.dtype)
            A = torch.einsum("bqs,sqdr->bqdr", (weights, self.lora_a))
            B = torch.einsum("bqs,srqd->brqd", (weights, self.lora_b))

        A = A.view(bs, self.in_features, self.rank)
        B = B.view(bs, self.rank, self.out_features)
        adapter_out = input_lora.bmm(A).bmm(B) * self.scaling
        return layer_out + adapter_out.to(input.dtype)

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
        need to be processed by a different combination of skills.
              --> skills     --> weights
        ex1 : [[a, d, f]     [[0.1, 0.2, 0.7]
        ex2 :  [c, g, h]]     [0.3, 0.4, 0.3]]

        This also handles the case in which the same skilled lora is applied to multiple example
        i.e.:

              --> skills      --> weights
        *   : [[a, d, f]]     [[0.1, 0.2, 0.7]]

        in this case, we broadcast the same combination to all the examples in the batch.
        """
        if merge_after:
            raise NotImplementedError("`merge_after` is not implemented for now.")

        if len(set([lora.layer for lora in skilled_loras])) > 1:
            raise ValueError("Cannot parallelize loras applied to different layers.")

        device = skilled_loras[0].lora_a.device
        skilled_loras_a = torch.stack([lora.lora_a for lora in skilled_loras], dim=0)
        skilled_loras_b = torch.stack([lora.lora_b for lora in skilled_loras], dim=0)
        weights = torch.stack(weights, dim=0).to(device)

        assert skilled_loras_a.shape[3] == 1, "Only 1 split is supported for now."
        assert skilled_loras_b.shape[4] == 1, "Only 1 split is supported for now."
        skilled_loras_a = skilled_loras_a.squeeze(3)
        skilled_loras_b = skilled_loras_b.squeeze(4)

        # (n_examples,)
        scaling = torch.cat(
            [torch.FloatTensor([lora.scaling]) for lora in skilled_loras], dim=0
        ).to(device=device)

        layer_out = skilled_loras[0].layer(input)
        input = input.to(dtype=skilled_loras[0].lora_a.dtype)

        # no batch!
        if skilled_loras_a.shape[0] == 1:
            # skilled lora is shared across all examples, remove batch dimension
            skilled_loras_a = skilled_loras_a.squeeze(0)
            skilled_loras_b = skilled_loras_b.squeeze(0)
            weights = weights.squeeze(0)
            scaling = scaling.squeeze(0)

            A = torch.einsum("s,sdr->dr", (weights, skilled_loras_a))
            B = torch.einsum("s,srd->rd", (weights, skilled_loras_b))

            # (n_examples, seq_len, out_features)
            adapter_out = torch.matmul(torch.matmul(input, A), B) * scaling
        else:
            A = torch.einsum("bs,bsdr->bdr", (weights, skilled_loras_a))
            B = torch.einsum("bs,bsrd->brd", (weights, skilled_loras_b))

            # (n_examples, seq_len, out_features)
            adapter_out = torch.bmm(torch.bmm(input, A), B) * scaling
        return layer_out + adapter_out.to(dtype=layer_out.dtype)


class SkilledLoRAView(SkilledLoRA):
    """
    Avoid initializing parameters, the parameters are just a view
    on a bunch of other LoRAs parameters stacked together.
    """

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
        layer = loras[0].layer
        skilled_lora = cls(config, layer)
        skilled_lora.lora_a = torch.stack(
            [lora.lora_a.unsqueeze(1) for lora in loras], dim=0
        )
        skilled_lora.lora_b = torch.stack(
            [lora.lora_b.unsqueeze(2) for lora in loras], dim=0
        )
        return skilled_lora


class SkilledLoRA_MergeLoraAfterOP(SkilledLoRA):
    def __init__(
        self,
        config,
        layer,
    ):
        super().__init__(config, layer)
        self.merge_after_op = config.merge_after_op

    def forward_linear_(self, input, weights):
        if not self.merge_after_op:
            return super().forward_linear_(input, weights)

        layer_out = self.layer(input)
        input_lora = input.to(self.lora_a.dtype)

        bs, _, _ = weights.size()
        adapter_out = torch.einsum(
            "bsd,qkdr->bsqkr", (input_lora, self.lora_a)
        )  # bs seq x n_splits x n_skills x rank
        adapter_out = torch.einsum(
            "bsqkr,qkrd->bsqkd", (adapter_out, self.lora_b)
        )  # bs x seq x n_splits x n_skills x D
        adapter_out = torch.einsum(
            "bsqkd,bqk->bsd", (adapter_out, weights)
        )  # bs x seq x n_splits x D
        adapter_out *= self.scaling

        return layer_out + adapter_out.to(input.dtype)
