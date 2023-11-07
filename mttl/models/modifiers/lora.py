from dataclasses import dataclass
import torch
from torch import nn
import math
from mttl.models.modifiers import register_modifier
from mttl.models.modifiers.base import Adapter, ModifyMixin, ModifierConfig


@dataclass
class LoRAConfig(ModifierConfig):
    lora_rank: int = 4
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    lora_warmup: bool = False
    lora_init_b_random: bool = False


@register_modifier("lora", config_cls=LoRAConfig)
class LoRA(Adapter, ModifyMixin):
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
        self.use_warmup = config.lora_warmup
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

    def create_for_layer(self, layer):
        if isinstance(layer, nn.Linear):
            self.lora_a = nn.Parameter(torch.empty(layer.in_features, self.rank))
            self.lora_b = nn.Parameter(torch.empty(self.rank, layer.out_features))
            self.forward_fn = self.forward_linear_
        else:
            raise NotImplementedError("LoRA only supports nn.Linear layers.")

    def forward_linear_(self, input, **kwargs):
        layer_out = self.layer(input)
        input_lora = input.to(self.lora_a.dtype)
        if self.training:
            self.training_steps += 1
        adapter_out = (
            torch.matmul(torch.matmul(input_lora, self.lora_a), self.lora_b)
            * self.scaling
        )
        warmup = min(self.training_steps / 10_000, 1)
        if self.use_warmup:
            adapter_out = adapter_out * warmup
        return layer_out + adapter_out.to(input.dtype)

    def reset_parameters(self):
        gain = nn.init.calculate_gain(nonlinearity="leaky_relu", param=math.sqrt(5))
        std = gain / math.sqrt(self.in_features)
        with torch.no_grad():
            self.lora_a.uniform_(-std, std)

        # ensure that initially, adding the adapter does not change the output
        if self.use_warmup or self.init_b_random:
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
                    self.n_splits,
                    self.n_skills,
                    layer.in_features // self.n_splits,
                    self.rank,
                )
            )
            self.lora_b = nn.Parameter(
                torch.empty(
                    self.n_splits,
                    self.n_skills,
                    self.rank,
                    layer.out_features // self.n_splits,
                )
            )
            self.forward_fn = self.forward_linear_
        else:
            raise NotImplementedError("SkilledLoRA only supports nn.Linear layers.")

    def forward_linear_(self, input, weights):
        layer_out = self.layer(input)
        input_lora = input.to(self.lora_a.dtype)
        if self.training:
            self.training_steps += 1

        bs = input.size(0)
        if weights.ndim == 1:
            # use indexing!
            A = self.lora_a[:, weights.long(), :, :]
            B = self.lora_b[:, weights.long(), :, :]
        else:
            A = torch.einsum("bqs,qsdr->bqdr", (weights, self.lora_a))
            B = torch.einsum("bqs,qsrd->bqrd", (weights, self.lora_b))

        A = A.reshape(bs, self.in_features, self.rank)
        B = B.transpose(1, 2).reshape(bs, self.rank, self.out_features)
        adapter_out = input_lora.bmm(A).bmm(B) * self.scaling

        warmup = min(self.training_steps / 10_000, 1)
        if self.use_warmup:
            adapter_out = adapter_out * warmup

        return layer_out + adapter_out.to(input.dtype)


class SkilledLoRA_MergeLoraAfterOP(SkilledLoRA):
    def __init__(
        self,
        config: SkilledLoRAConfig,
        layer: nn.Module,
    ):
        super().__init__(config, layer)
        self.merge_after_op = config.merge_after_op

    def forward_linear_(self, input, weights):
        if not self.merge_after_op:
            return super().forward_linear_(input, weights)

        layer_out = self.layer(input)
        input_lora = input.to(self.lora_a.dtype)
        if self.training:
            self.training_steps += 1

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
        warmup = min(self.training_steps / 10_000, 1)
        if self.use_warmup:
            adapter_out = adapter_out * warmup

        return layer_out + adapter_out.to(input.dtype)
