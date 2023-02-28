import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import math


class LoRALinear(nn.Module):
    def __init__(self, linear_layer, rank, init_b_random=False):
        super().__init__()

        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.rank = rank
        self.init_b_random = init_b_random
        self.weight = linear_layer.weight
        self.bias = linear_layer.bias
        self.lora_a = nn.Parameter(torch.randn(rank, linear_layer.in_features))
        self.lora_b = nn.Parameter(torch.zeros(linear_layer.out_features, rank))
        self.reset_parameters()

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

    def forward(self, input):
        # general implementation for lora (adding and scaling)
        weight = self.weight
        weight = weight + torch.matmul(self.lora_b, self.lora_a) / self.rank
        return F.linear(input, weight, self.bias)


class IA3Linear(nn.Module):
    def __init__(self, config, linear_layer):
        super().__init__()

        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.weight = linear_layer.weight
        self.bias = linear_layer.bias
        self.multi_lora_b = nn.Parameter(torch.ones(linear_layer.out_features))

    def forward(self, input):
        hidden = F.linear(input, self.weight, self.bias)
        hidden = hidden * self.multi_lora_b
        return hidden


def modify_with_lora(transformer, config):
    for m_name, module in dict(transformer.named_modules()).items():
        if re.fullmatch(config.lora_modules, m_name):
            for c_name, layer in dict(module.named_children()).items():
                if re.fullmatch(config.lora_layers, c_name):
                    assert isinstance(
                        layer, nn.Linear
                    ), f"LoRA can only be applied to torch.nn.Linear, but {layer} is {type(layer)}."
                    setattr(
                        module,
                        c_name,
                        LoRALinear(layer, config.lora_rank),
                    )
    return transformer


def modify_with_ia3(transformer, config):
    for m_name, module in dict(transformer.named_modules()).items():
        if re.fullmatch(config.lora_modules, m_name):
            for c_name, layer in dict(module.named_children()).items():
                if re.fullmatch(config.lora_layers, c_name):
                    assert isinstance(
                        layer, nn.Linear
                    ), f"IA3 can only be applied to torch.nn.Linear, but {layer} is {type(layer)}."
                    setattr(
                        module,
                        c_name,
                        IA3Linear(config, layer),
                    )
    return transformer
