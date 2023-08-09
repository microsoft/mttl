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
        self.rank = config.lora_rank
        self.weight = linear_layer.weight
        self.bias = linear_layer.bias
        self.lora_a = nn.Parameter(
            torch.randn(config.lora_rank, linear_layer.in_features)
        )
        self.lora_b = nn.Parameter(
            torch.zeros(linear_layer.out_features, config.lora_rank)
        )
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain(nonlinearity="leaky_relu", param=math.sqrt(5))
        std = gain / math.sqrt(self.in_features)
        with torch.no_grad():
            self.lora_a.uniform_(-std, std)

        torch.nn.init.zeros_(self.lora_b)

    def forward(self, input):
        # general implementation for lora (adding and scaling)
        weight = self.weight
        adapter_out = torch.matmul(input, self.lora_a.T)
        adapter_out = torch.matmul(adapter_out, self.lora_b.T) / self.rank
        return F.linear(input, self.weight, self.bias) + adapter_out


class LoRATensor(nn.Module):
    def __init__(self, config, tensor_layer):
        super().__init__()

        assert isinstance(
            tensor_layer, nn.Linear
        ), f"LoRA can only be applied to torch.nn.Linear, but {tensor_layer} is {type(tensor_layer)}."

        self.in_features = tensor_layer.in_features
        self.out_features = tensor_layer.out_features
        self.lora_rank = config.lora_rank
        self.order = config.order
        self.tensor_rank = config.tensor_rank
        self.weight = tensor_layer.weight
        self.bias = tensor_layer.bias

        self.embedding_dim_leaf_a = math.ceil((self.in_features) ** (1 / self.order))
        self.embedding_dim_leaf_b = math.ceil((self.out_features) ** (1 / self.order))

        self.weight_leafs_a = nn.Parameter(
            self.weight.new_empty(
                self.order,
                self.tensor_rank,
                self.lora_rank,
                self.embedding_dim_leaf_a,
            )
        )

        self.weight_leafs_b = nn.Parameter(
            self.weight.new_empty(
                self.order,
                self.tensor_rank,
                self.lora_rank,
                self.embedding_dim_leaf_b,
            )
        )

        # What if I just use one layer normalization
        self.layerone_normalization_a = nn.LayerNorm(
            normalized_shape=[self.lora_rank, self.embedding_dim_leaf_a**2]
        )
        # self.layertwo_normalization_a = nn.LayerNorm(
        #     normalized_shape=[self.lora_rank, self.embedding_dim_leaf_a**2])
        self.layerone_normalization_b = nn.LayerNorm(
            normalized_shape=[self.lora_rank, self.embedding_dim_leaf_b**2]
        )
        # self.layertwo_normalization_b = nn.LayerNorm(
        #     normalized_shape=[self.lora_rank, self.embedding_dim_leaf_b**2])
        self.reset_parameters()

    def tensor_product_construct(self, weight_leafs, flag="up"):
        w = weight_leafs
        if self.order == 2:
            w01 = w[0, :, :, :, None] * w[1, :, :, None, :]
            # print(w[:,:,:,:].size())
            w01 = w01.view(self.tensor_rank, self.lora_rank, -1)
            if flag == "up":
                w01 = self.layerone_normalization_a(w01)
            elif flag == "down":
                w01 = self.layerone_normalization_b(w01)
            # print(w01.size())
            w01 = w01.view(self.tensor_rank, self.lora_rank, -1)
            weight = w01.sum(dim=0)
        elif self.order == 4:
            w01 = w[0, :, :, :, None] * w[1, :, :, None, :]
            # print(w[:,:,:,:].size())
            w01 = w01.view(self.tensor_rank, self.lora_rank, -1)
            if flag == "up":
                w01 = self.layerone_normalization_a(w01)
            elif flag == "down":
                w01 = self.layerone_normalization_b(w01)
            # print(w01.size())
            w23 = w[2, :, :, :, None] * w[3, :, :, None, :]
            w23 = w23.view(self.tensor_rank, self.lora_rank, -1)
            if flag == "up":
                w23 = self.layerone_normalization_a(w23)
            elif flag == "down":
                w23 = self.layerone_normalization_b(w23)

            w0123 = w01[:, :, :, None] * w23[:, :, None, :]
            w0123 = w0123.view(self.tensor_rank, self.lora_rank, -1)
            weight = w0123.sum(dim=0)
        elif self.order == 8:
            w01 = w[0, :, :, :, None] * w[1, :, :, None, :]
            w01 = w01.view(self.tensor_rank, self.lora_rank, -1)
            w23 = w[2, :, :, :, None] * w[3, :, :, None, :]
            w23 = w23.view(self.tensor_rank, self.lora_rank, -1)
            w45 = w[4, :, :, :, None] * w[5, :, :, None, :]
            w45 = w45.view(self.tensor_rank, self.lora_rank, -1)
            w67 = w[6, :, :, :, None] * w[7, :, :, None, :]
            w67 = w67.view(self.tensor_rank, self.lora_rank, -1)
            w0123 = w01[:, :, :, None] * w23[:, :, None, :]
            w0123 = w0123.view(self.tensor_rank, self.lora_rank, -1)
            w4567 = w45[:, :, :, None] * w67[:, :, None, :]
            w4567 = w4567.view(self.tensor_rank, self.lora_rank, -1)
            w01234567 = w0123[:, :, :, None] * w4567[:, :, None, :]
            w01234567 = w01234567.view(self.tensor_rank, self.lora_rank, -1)
            weight = w01234567.sum(0)
        if flag == "down":
            tpr = weight[:, : self.out_features]
        elif flag == "up":
            tpr = weight[:, : self.in_features]
        else:
            raise ValueError("signal must be i)input or ii)output")
        return tpr

    def reset_parameters(self):
        gain = nn.init.calculate_gain(nonlinearity="leaky_relu", param=math.sqrt(5))
        std = gain / math.sqrt(self.in_features)
        with torch.no_grad():
            self.weight_leafs_a.uniform_(-std, std)

        torch.nn.init.zeros_(self.weight_leafs_b)

    def forward(self, input):
        # general implementation for lora (adding and scaling)
        weight = self.weight

        self.lora_a = self.tensor_product_construct(
            self.weight_leafs_a, flag="up"
        )  # [tensor rank, rank, D]
        self.lora_b = self.tensor_product_construct(self.weight_leafs_b, flag="down")
        self.lora_a = self.lora_a.transpose(1, 0)
        weight = weight + torch.matmul(self.lora_a, self.lora_b) / self.lora_rank
        # adapter_out = torch.matmul(input, self.lora_a.T)
        # adapter_out = torch.matmul(adapter_out, self.lora_b.T) / self.rank
        return F.linear(input, weight, self.bias)


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
        self.multi_lora_b = nn.Parameter(
            torch.zeros(
                linear_layer.out_features,
            )
        )
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.multi_lora_b_i, a=math.sqrt(5))
        torch.nn.init.ones_(self.multi_lora_b)

    def forward(self, input):
        # general implementation for lora (adding and scaling)
        hidden = F.linear(input, self.weight, self.bias)
        hidden = hidden * self.multi_lora_b + torch.matmul(
            torch.matmul(input, self.multi_lora_b_i), self.multi_lora_b_o
        )
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
            self.n_splits,
            self.n_skills,
            self.out_features,
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
                if re.fullmatch(config.lora_layers, c_name):
                    setattr(
                        module,
                        c_name,
                        adapter_klass(config, layer),
                    )
    return transformer


adapter_class = {
    # `TODO`: add your adapter here
    "lora": LoRALinear
}


def modify_with_ia3(transformer, config):
    return modify_with_adapter(transformer, config, IA3Linear)


def modify_with_lora(transformer, config):
    return modify_with_adapter(transformer, config, LoRALinear)


def modify_with_tlora(transformer, config):
    return modify_with_adapter(transformer, config, LoRATensor)


def modify_with_gator(transformer, config):
    return modify_with_adapter(transformer, config, GatorLinear)


def modify_with_ln(transformer, config):
    return modify_with_adapter(transformer, config, LNAdapter)
