import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import math


class DoRALinear(nn.Module):
    """_summary_

    Returns:
        _type_: This is a implement of DoRA
    """

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
        # m = Magnitude column-wise across output dimension
        self.weight_m_wdecomp = nn.Parameter(1, linear_layer.out_features)
        self.lora_dropout = nn.Dropout(0.05)

        self.reset_parameters()

    def forward(self, x):
        adapted = self.weight + torch.matmul(self.lora_b, self.lora_a) / self.rank
        norm_scale = (
            self.weight_m_wdecomp.view(-1)
            / (torch.linalg.norm(adapted, dim=1)).detach()
        )

        org_result = F.linear(x, self.weight, self.bias)

        dropout_x = self.lora_dropout(x)
        result = org_result + (norm_scale - 1) * F.linear(
            dropout_x, self.weight, self.bias
        )
        result += norm_scale * (
            self.lora_b(self.lora_a(dropout_x.to(self.lora_a.weight.dtype)))
        )
        return result

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


class KronA(nn.Module):
    def __init__(self, config, linear_layer):
        super().__init__()

        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.weight = linear_layer.weight
        self.bias = linear_layer.bias
        self.lora_a = nn.Parameter(torch.randn(32, 64))
        self.lora_b = nn.Parameter(torch.randn(64, 32))
        self.reset_parameters()
        self.norm = nn.LayerNorm([64, 32, 32, 64])

    def reset_parameters(self):
        gain = nn.init.calculate_gain(nonlinearity="leaky_relu", param=math.sqrt(5))
        std = gain / math.sqrt(self.in_features)
        with torch.no_grad():
            self.lora_a.uniform_(-std, std)

        # ensure that initially, adding the adapter does not change the output
        with torch.no_grad():
            self.lora_b.uniform_(-std, std)

    def kronecker_product(self, a, b):
        """
        Kronecker product of matrices a and b with leading batch dimensions.
        Batch dimensions are broadcast. The number of them mush
        :type a: torch.Tensor
        :type b: torch.Tensor
        :rtype: torch.Tensor
        """
        # return torch.stack([torch.kron(ai, bi) for ai, bi in zip(a,b)], dim=0)
        siz1 = torch.Size(torch.tensor(a.shape[-2:]) * torch.tensor(b.shape[-2:]))
        res = a.unsqueeze(-1).unsqueeze(-3) * b.unsqueeze(-2).unsqueeze(-4)
        # res = self.norm(res)
        siz0 = res.shape[:-4]
        out = res.reshape(siz0 + siz1)
        return out

    def kronecker_product_einsum(self, A: torch.Tensor, B: torch.Tensor):
        """
        Batched Version of Kronecker Products
        :param A: has shape (a, c)
        :param B: has shape (k, p)
        :return: (ak, cp)
        """
        res = torch.einsum("ac,kp->akcp", A, B)
        res = self.norm(res)

        return res.reshape(A.size(0) * B.size(0), A.size(1) * B.size(1))

    def forward(self, input):
        weight = self.weight
        weight = weight + self.kronecker_product_einsum(self.lora_b, self.lora_a)
        return F.linear(input, weight, self.bias)

    def forward(self, input):
        hidden = F.linear(input, self.weight, self.bias)
        self.multi_lora_b = self.quantum_layer(self.qubits, self.quantum_weights)
        # hidden = self.multi_lora_b
        hidden = hidden * self.multi_lora_b.float()
        return hidden


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


def modify_with_krona(transformer, config):
    for m_name, module in dict(transformer.named_modules()).items():
        if re.fullmatch(config.lora_modules, m_name):
            for c_name, layer in dict(module.named_children()).items():
                if re.fullmatch(config.lora_layers, c_name):
                    assert isinstance(
                        layer, nn.Linear
                    ), f"kron can only be applied to torch.nn.Linear, but {layer} is {type(layer)}."
                    setattr(
                        module,
                        c_name,
                        KronA(config, layer),
                    )
    return transformer


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


def modify_with_dora(transformer, config):
    for m_name, module in dict(transformer.named_modules()).items():
        if re.fullmatch(config.lora_modules, m_name):
            for c_name, layer in dict(module.named_children()).items():
                if re.fullmatch(config.lora_layers, c_name):
                    assert isinstance(
                        layer, nn.Linear
                    ), f"DoRA can only be applied to torch.nn.Linear, but {layer} is {type(layer)}."
                    setattr(
                        module,
                        c_name,
                        DoRALinear(layer, config.lora_rank),
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
