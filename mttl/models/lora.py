import torch        
import torch.nn as nn
import torch.nn.functional as F
import re
import math

from transformers.models.t5.modeling_t5 import T5LayerNorm



class LoRALinear(nn.Module):
    def __init__(self, config, linear_layer):
        super().__init__()

        assert isinstance(
            linear_layer, nn.Linear
        ), f"LoRA can only be applied to torch.nn.Linear, but {linear_layer} is {type(linear_layer)}."

        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.rank = config.rank
        self.init_b_random = config.init_b_random
        self.weight = linear_layer.weight
        self.bias = linear_layer.bias
        self.lora_a = nn.Parameter(torch.randn(config.rank, linear_layer.in_features))
        self.lora_b = nn.Parameter(torch.zeros(linear_layer.out_features, config.rank))
        self.merged = False
        self.lora_alpha: int = config.lora_alpha if hasattr(config, "lora_alpha") else 1
        self.scaling = self.lora_alpha / self.rank
        lora_dropout = config.lora_dropout if hasattr(config, "lora_dropout") else 0.
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        self.fan_in_fan_out: bool = False
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
    
    # def merge(self, lora_a=None, lora_b=None):              
    #     lora_a = lora_a if lora_a is not None else self.lora_a
    #     lora_b = lora_b if lora_b is not None else self.lora_b
    #     if not self.merged:
    #         self.weight.data = self.weight.data + torch.matmul(self.lora_b, self.lora_a) / self.rank
    #         self.merged = True
    #     return self.weight

    # def forward(self, input):
    #     # general implementation for lora (adding and scaling)
    #     weight = self.weight
    #     if not self.merged:
    #         weight = weight + torch.matmul(self.lora_b, self.lora_a) / self.rank
    #     return F.linear(input, weight, self.bias)

    
    def forward(self, x: torch.Tensor): # implementation from https://github.com/microsoft/LoRA/blob/main/loralib/layers.py#L64
        def T(w):
            return w.T if self.fan_in_fan_out else w
        if self.rank > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.rank > 0:
                result += (self.lora_dropout(x) @ self.lora_a.T @ self.lora_b.T) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)


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
        self.lora_b = nn.Parameter(torch.zeros(linear_layer.out_features,))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.zeros_(self.lora_b)

    def forward(self, input):
        # general implementation for lora (adding and scaling)
        output = F.linear(input, self.weight, self.bias)
        gator = (self.lora_b.unsqueeze(0).unsqueeze(1) * input).sum(-1, keepdim=True) * self.lora_b.unsqueeze(0).unsqueeze(1)
        return output + gator

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
        self.weight = linear_layer.weight
        self.bias = linear_layer.bias
        self.multi_lora_b = nn.Parameter(torch.ones(linear_layer.out_features))

    def forward(self, input):
        hidden = F.linear(input, self.weight, self.bias)
        hidden = hidden * self.multi_lora_b
        return hidden


def modify_with_adapter(transformer, config, adapter_klass):    
    for m_name, module in dict(transformer.named_modules()).items():
        if re.fullmatch(config.lora_modules, m_name):
            for c_name, layer in dict(module.named_children()).items():
                if re.fullmatch(config.lora_layers, c_name) and config.model_modifier not in c_name :
                    setattr(
                        module,
                        c_name,
                        adapter_klass(config, layer),
                    )
    return transformer

adapter_class = {
    # `TODO`: add your adapter here
    'lora': LoRALinear
}

def modify_with_ia3(transformer, config):
    return modify_with_adapter(transformer, config, IA3Linear)


def modify_with_lora(transformer, config):
    return modify_with_adapter(transformer, config, LoRALinear)


def modify_with_gator(transformer, config):
    return modify_with_adapter(transformer, config, GatorLinear)


def modify_with_ln(transformer, config):
    return modify_with_adapter(transformer, config, LNAdapter)