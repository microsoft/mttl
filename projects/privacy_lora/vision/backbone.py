"""Minimal ViT + LoRA factor module for medical-image superposition tests.

Pure PyTorch (no torchvision / timm / HF-ViT). Uses a custom attention block so
LoRA on the output projection is invoked via Module.forward (nn.MultiheadAttention
bypasses forward and reads .weight directly).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Frozen dense linear + trainable LoRA factors (A: in×r, B: r×out), matching mttl."""

    def __init__(self, linear: nn.Linear, rank: int = 16, alpha: Optional[float] = None, freeze_a: bool = False):
        super().__init__()
        self.linear = linear
        for p in self.linear.parameters():
            p.requires_grad = False
        self.rank = rank
        self.scaling = float(alpha if alpha is not None else rank) / rank
        device, dtype = linear.weight.device, linear.weight.dtype
        self.lora_a = nn.Parameter(torch.empty(linear.in_features, rank, device=device, dtype=dtype))
        self.lora_b = nn.Parameter(torch.zeros(rank, linear.out_features, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.lora_a.requires_grad = not freeze_a
        self.lora_b.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + (x @ self.lora_a @ self.lora_b) * self.scaling


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, n, d)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        hid = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hid)
        self.fc2 = nn.Linear(hid, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.fc2(F.gelu(self.fc1(self.norm2(x))))
        return x


class MiniViT(nn.Module):
    """Compact ViT for MedMNIST (resized to 64×64). LoRA on one block's attn out-proj."""

    def __init__(
        self,
        num_classes: int,
        img_size: int = 64,
        patch: int = 8,
        dim: int = 256,
        depth: int = 6,
        heads: int = 4,
        lora_layer: int = 3,
        rank: int = 16,
        freeze_a: bool = False,
        in_ch: int = 3,
    ):
        super().__init__()
        assert img_size % patch == 0
        self.img_size = img_size
        self.patch = patch
        n_patches = (img_size // patch) ** 2
        self.lora_layer = lora_layer
        self.patch_embed = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, n_patches + 1, dim))
        self.blocks = nn.ModuleList([TransformerBlock(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        for p in self.parameters():
            p.requires_grad = False
        block = self.blocks[lora_layer]
        block.attn.proj = LoRALinear(block.attn.proj, rank=rank, alpha=rank, freeze_a=freeze_a)

    def lora_module(self) -> LoRALinear:
        return self.blocks[self.lora_layer].attn.proj  # type: ignore

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None):
        b = x.shape[0]
        tok = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls.expand(b, -1, -1)
        h = torch.cat([cls, tok], dim=1) + self.pos
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h)
        logits = self.head(h[:, 0])
        out = {"logits": logits}
        if labels is not None:
            out["loss"] = F.cross_entropy(logits, labels)
        return out
