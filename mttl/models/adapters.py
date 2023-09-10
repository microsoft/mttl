from typing import Optional
from torch import nn
import torch
import math
from torch.autograd import Function
from torch.nn.modules.module import Module


class Adapter(nn.Module):
    @property
    def layer_name(self):
        if not hasattr(self, '__layer_name__'):
            raise ValueError("Layer name not set, dependency injection not done properly?")

        return self.__layer_name__


class LoRA(Adapter):
    def __init__(
        self,
        config,
        layer,
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
        if self.training:
            self.training_steps += 1
        adapter_out = torch.matmul(torch.matmul(input, self.lora_a), self.lora_b) * self.scaling
        warmup = min(self.training_steps / 10_000, 1)
        if self.use_warmup:
            adapter_out = adapter_out * warmup
        return self.layer(input) + adapter_out

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


class IA3(Adapter):
    def __init__(self, config, layer):
        super().__init__()

        assert isinstance(
            layer, nn.Linear
        ), f"IA3 can only be applied to torch.nn.Linear, but {layer} is {type(layer)}."

        self.layer = layer
        self.multi_lora_b = nn.Parameter(torch.ones(layer.out_features))

    def forward(self, input):
        return self.layer(input) * self.multi_lora_b


class LN(Adapter):
    def __init__(self, config, layer):
        super().__init__()
        
        self.out_features = layer.weight.size(0)
        self.weight = layer.weight
        self.variance_epsilon = layer.variance_epsilon

        assert self.out_features % config.n_splits == 0

        self.lora_b = nn.Parameter(self.weight.data)

    def forward(self, input):
        # layer norm should always be calculated in float32
        variance = input.to(torch.float32).pow(2).mean(-1, keepdim=True)
        input = input / torch.sqrt(variance + self.variance_epsilon)

        if self.weight.dtype == torch.float16:
            input = input.to(torch.float16)
        return self.lora_b.unsqueeze(0) * input


class SkilledLoRA(LoRA):
    def __init__(
        self,
        config,
        layer,
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
                )
            )
            self.lora_b = nn.Parameter(
                torch.empty(
                    self.n_skills,
                    self.rank,
                    self.n_splits,
                    layer.out_features // self.n_splits,
                )
            )
            self.forward_fn = self.forward_linear_
        else:
            raise NotImplementedError("SkilledLoRA only supports nn.Linear layers.")

    def forward_linear_(self, input, weights):
        if self.training:
            self.training_steps += 1

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
                    self.lora_a.data.copy_(self.lora_a.data[:1].repeat(self.n_skills, 1, 1, 1))
                    self.lora_b.data.copy_(self.lora_b.data[:1].repeat(self.n_skills, 1, 1, 1))
                A = self.lora_a[weights.long(), :, :, :]
                B = self.lora_b[weights.long(), :, :, :]
        else:
            A = torch.einsum("bqs,sqdr->bqdr", (weights, self.lora_a))
            B = torch.einsum("bqs,srqd->brqd", (weights, self.lora_b))

        A = A.view(bs, self.in_features, self.rank)
        B = B.view(bs, self.rank, self.out_features)
        adapter_out = torch.bmm(torch.bmm(input, A), B) * self.scaling
        return self.layer(input) + adapter_out


class LoRAKnoweldgeContainer(LoRA):
    def __init__(
        self,
        config,
        layer,
    ):
        super().__init__(config, layer)

        self.km_names = set()
        self.km_to_id = {}
        self.n_skills = 0

    def add_module(self, name: str, weights: torch.Tensor) -> None:
        if name in self.km_names:
            raise ValueError("A knowledge module with this name already exists.")

        self.km_names.add(name)
        self.km_to_id[name] = len(self.km_to_id)
        self.add_module_weights(self.km_to_id[name], weights)

    def add_module_weights(self, km_id: int, weights: torch.Tensor) -> None:
        self.lora_b.data.expand(self.lora_b.size(0) + 1, -1, -1)

        self.lora_a[km_id].copy_(weights["lora_a"].data.unsqueeze(0))
        self.lora_b[km_id].copy_(weights["lora_a"].data.unsqueeze(0))

    def create_for_layer(self, layer):
        if isinstance(layer, nn.Linear):
            self.lora_a = nn.Parameter(
                torch.empty(
                    0,
                    layer.in_features,
                    self.rank,
                )
            )
            self.lora_b = nn.Parameter(
                torch.empty(
                    0,
                    self.rank,
                    layer.out_features,
                )
            )
            self.forward_fn = self.forward_linear_
        else:
            raise NotImplementedError("LoRAKnoweldgeContainer only supports nn.Linear layers.")

    def forward_linear_(self, input, weights):
        if self.training:
            self.training_steps += 1

        bs = input.size(0)
        if weights.ndim == 1:
            # use indexing!
            wrm_steps = 1_000
            if self.training_steps < wrm_steps:
                A = self.lora_a[torch.zeros_like(weights).long()]
                B = self.lora_b[torch.zeros_like(weights).long()]
            else:
                if self.training_steps == wrm_steps:
                    self.lora_a.data.copy_(self.lora_a.data[:1].repeat(self.n_skills, 1, 1, 1))
                    self.lora_b.data.copy_(self.lora_b.data[:1].repeat(self.n_skills, 1, 1, 1))
                A = self.lora_a[weights.long(), :, :, :]
                B = self.lora_b[weights.long(), :, :, :]
        else:
            A = torch.einsum("bqs,sqdr->bqdr", (weights, self.lora_a))
            B = torch.einsum("bqs,srqd->brqd", (weights, self.lora_b))

        A = A.view(bs, self.in_features, self.rank)
        B = B.view(bs, self.rank, self.out_features)
        adapter_out = torch.bmm(torch.bmm(input, A), B) * self.scaling
        return self.layer(input) + adapter_out
