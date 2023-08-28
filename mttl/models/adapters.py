from torch import nn
import torch
import math


class Adapter(nn.Module):
    pass


class LoRA(Adapter):
    def __init__(
        self,
        in_features,
        out_features,
        lora_rank,
        lora_alpha,
        lora_warmup=0,
        lora_dropout=0.0,
    ):
        super().__init__()

        # assign self variables
        self.in_features = in_features
        self.out_features = out_features
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.use_warmup = lora_warmup
        self.training_steps = 0.0
        self.scaling = self.lora_alpha / self.lora_rank

        # define lora parameters
        self.lora_a = nn.Parameter(torch.empty(self.in_features, self.lora_rank))
        self.lora_b = nn.Parameter(torch.empty(self.lora_rank, self.out_features))

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain(nonlinearity="leaky_relu", param=math.sqrt(5))
        std = gain / math.sqrt(self.in_features)
        with torch.no_grad():
            self.lora_a.uniform_(-std, std)

        # ensure that initially, adding the adapter does not change the output
        if self.use_warmup:
            with torch.no_grad():
                self.lora_b.uniform_(-std, std)
        else:
            torch.nn.init.zeros_(self.lora_b)

    def forward(self, input):
        return input.mm(self.lora_a).mm(self.lora_b) * self.scaling


class SkilledLoRA(LoRA):
    def __init__(
        self,
        in_features,
        out_features,
        n_skills,
        lora_rank,
        lora_alpha,
        lora_warmup=0,
        lora_dropout=0.0,
        n_splits=1,
    ):
        super().__init__(in_features, out_features, lora_rank, lora_alpha, lora_warmup, lora_dropout)

        self.n_splits = n_splits
        self.n_skills = n_skills
        
        self.lora_a = nn.Parameter(
            torch.empty(
                self.n_splits,
                self.n_skills,
                in_features // self.n_splits,
                self.rank,
            )
        )
        self.lora_b = nn.Parameter(
            torch.empty(
                self.n_splits,
                self.n_skills,
                self.rank,
                out_features // self.n_splits,
            )
        )
        self.training_steps += 1
        self.reset_parameters()

    def forward(self, input, weights):
        if self.training:
            self.training_steps += 1

        bs, n_splits, n_skills = weights.size()
        A = torch.einsum("bqs,qsdr->bqdr", (weights, self.lora_a))
        B = torch.einsum("bqs,qsrd->bqrd", (weights, self.lora_b))
        A = A.reshape(bs, self.in_features, self.rank)
        B = B.transpose(1, 2).reshape(bs, self.rank, self.out_features)
        adapter_out = input.bmm(A).bmm(B) * self.scaling

        warmup = min(self.training_steps / 10_000, 1)
        if self.use_warmup:
            adapter_out = adapter_out * warmup

        return adapter_out
