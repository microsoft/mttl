import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import math

from transformers.models.t5.modeling_t5 import T5LayerNorm
from mttl.models.modifiers import register_modifier


class LoRALinear(nn.Module):
    def __init__(self, config, linear_layer):
        super().__init__()

        assert isinstance(
            linear_layer, nn.Linear
        ), f"LoRA can only be applied to torch.nn.Linear, but {linear_layer} is {type(linear_layer)}."

        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.rank = config.lora_rank
        self.weight = linear_layer.weight
        self.bias = linear_layer.bias
        self.linear_layer = linear_layer
        self.lora_alpha = getattr(config, "lora_alpha", 1.0)
        self.scaling = self.lora_alpha / self.rank
        self.lora_a = nn.Parameter(torch.randn(config.lora_rank, linear_layer.in_features))
        self.lora_b = nn.Parameter(torch.zeros(linear_layer.out_features, config.lora_rank))
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain(nonlinearity="leaky_relu", param=math.sqrt(5))
        std = gain / math.sqrt(self.in_features)
        with torch.no_grad():
            self.lora_a.uniform_(-std, std)

        torch.nn.init.zeros_(self.lora_b)

    def forward(self, input):
        # general implementation for lora (adding and scaling)
        adapter_out = (input @ self.lora_a.T)
        adapter_out = (adapter_out @ self.lora_b.T)
        return self.linear_layer(input) + adapter_out * self.scaling


class GatorLinear(nn.Module):
    def __init__(self, config, linear_layer):
        super().__init__()

        assert isinstance(
            linear_layer, nn.Linear
        ), f"GatorLinear can only be applied to torch.nn.Linear, but {linear_layer} is {type(linear_layer)}."

        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.weight = linear_layer.weight
        self.bias = linear_layer.bias

        self.multi_lora_b_i = nn.Parameter(torch.zeros(linear_layer.in_features, 1))
        self.multi_lora_b_o = nn.Parameter(torch.zeros(1, linear_layer.out_features))
        self.multi_lora_b = nn.Parameter(torch.zeros(linear_layer.out_features,))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.multi_lora_b_i, a=math.sqrt(5))
        torch.nn.init.ones_(self.multi_lora_b)

    def forward(self, input):
        # general implementation for lora (adding and scaling)
        hidden = F.linear(input, self.weight, self.bias)
        hidden = hidden * self.multi_lora_b + torch.matmul(torch.matmul(input, self.multi_lora_b_i), self.multi_lora_b_o)
        return hidden


class LNAdapter(nn.Module):
    def __init__(self, config, layer):
        super().__init__()

        self.n_splits = config.n_splits
        self.n_tasks = config.n_tasks
        self.n_skills = config.n_skills
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

    def extra_repr(self):
        return "n_splits={}, n_skills={}, out_features={}".format(
            self.n_splits, self.n_skills, self.out_features,
        )


class IA3Linear(nn.Module):
    def __init__(self, config, linear_layer):
        super().__init__()

        assert isinstance(
            linear_layer, nn.Linear
        ), f"IA3Linear can only be applied to torch.nn.Linear, but {linear_layer} is {type(linear_layer)}."

        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.linear_layer = linear_layer
        self.weight = linear_layer.weight
        self.bias = linear_layer.bias
        self.multi_lora_b = nn.Parameter(torch.ones(linear_layer.out_features))

    def forward(self, input):
        return self.linear_layer(input) * self.multi_lora_b


def modify_with_adapter(transformer, config, adapter_klass):
    for m_name, module in dict(transformer.named_modules()).items():
        if re.fullmatch(config.modify_modules, m_name):
            for c_name, layer in dict(module.named_children()).items():
                if re.fullmatch(config.modify_layers, c_name):
                    setattr(
                        module,
                        c_name,
                        adapter_klass(config, layer),
                    )
    return transformer


@register_modifier("ia3")
def modify_with_ia3(transformer, config):
    return modify_with_adapter(transformer, config, IA3Linear)


@register_modifier("lora")
def modify_with_lora(transformer, config):
    return modify_with_adapter(transformer, config, LoRALinear)


@register_modifier("gator")
def modify_with_gator(transformer, config):
    return modify_with_adapter(transformer, config, GatorLinear)


@register_modifier("ln")
def modify_with_ln(transformer, config):
    return modify_with_adapter(transformer, config, LNAdapter)