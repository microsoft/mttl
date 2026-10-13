"""Model loading and fine-tuning methods (LoRA variants, full FT, IA3, head-only)."""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from mttl.models.modifiers.lora import LoRA, LoRAConfig
from mttl.models.modifiers.modify_model import modify_transformer

ATTN = "q_proj|k_proj|v_proj|o_proj"
MLP = "gate_proj|up_proj|down_proj"
ALL = ATTN + "|" + MLP


def load_base_model(model_name: str, device: str, dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    on_cuda = str(device).startswith("cuda")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map={"": device} if on_cuda else None,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    if getattr(model, "hf_device_map", None) is None:
        model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, tokenizer


# --------------------------------------------------------------------------- #
# IA3
# --------------------------------------------------------------------------- #
class IA3Linear(nn.Module):
    def __init__(self, layer: nn.Linear):
        super().__init__()
        self.layer = layer
        self.ia3 = nn.Parameter(torch.ones(layer.out_features, device=layer.weight.device, dtype=torch.float32))

    def forward(self, x):
        return self.layer(x) * self.ia3.to(x.dtype)


def _wrap_linears(model, layer_regex: str, module_regex: str, wrapper):
    n = 0
    for m_name, module in list(model.named_modules()):
        if not re.fullmatch(module_regex, m_name):
            continue
        for c_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and re.fullmatch(layer_regex, c_name):
                setattr(module, c_name, wrapper(child))
                n += 1
    return n


# --------------------------------------------------------------------------- #
# Attach a fine-tuning method
# --------------------------------------------------------------------------- #
def attach_method(model, method: str, rank: int = 16, alpha: Optional[float] = None, targets: str = "attn",
                  modules_regex: str = ".*", freeze_a: bool = False, lora_dropout: float = 0.05,
                  variant: str = "lora") -> Dict:
    """method in {lora, full, ia3, head}; variant in {lora, rslora}."""
    layer_regex = {"attn": ATTN, "mlp": MLP, "all": ALL}.get(targets, targets)
    info = {"method": method, "targets": layer_regex}
    if method == "lora":
        alpha = rank if alpha is None else alpha
        cfg = LoRAConfig(modify_modules=modules_regex, modify_layers=layer_regex, lora_rank=rank,
                         lora_alpha=alpha, lora_dropout=lora_dropout)
        modify_transformer(model, cfg)
        for _, m in lora_modules(model):
            if variant == "rslora":
                m.scaling = alpha / (rank**0.5)
            # keep the random initialisation of A for the A - A(0) attack variant
            m.register_buffer("lora_a_init", m.lora_a.detach().clone(), persistent=False)
        for name, p in model.named_parameters():
            if ".lora_" in name:
                p.requires_grad = not (freeze_a and name.endswith("lora_a"))
        model.train()
    elif method == "ia3":
        n = _wrap_linears(model, "k_proj|v_proj|down_proj", modules_regex, IA3Linear)
        info["n_ia3"] = n
        for name, p in model.named_parameters():
            p.requires_grad = name.endswith(".ia3")
    elif method == "head":
        head = model.get_output_embeddings()
        if head.weight is model.get_input_embeddings().weight:
            # untie
            new = nn.Linear(head.in_features, head.out_features, bias=False, device=head.weight.device, dtype=head.weight.dtype)
            new.weight.data.copy_(head.weight.data)
            model.set_output_embeddings(new)
            model.config.tie_word_embeddings = False
        for p in model.get_output_embeddings().parameters():
            p.requires_grad = True
    elif method == "full":
        for p in model.parameters():
            p.requires_grad = True
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    else:
        raise ValueError(method)
    info["n_trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return info


def lora_modules(model) -> List[Tuple[str, LoRA]]:
    return [(n, m) for n, m in model.named_modules() if isinstance(m, LoRA)]


def set_lora_enabled(model, enabled: bool):
    for _, m in lora_modules(model):
        m._enabled = enabled


def trainable_state(model) -> Dict[str, torch.Tensor]:
    return {n: p.detach().to("cpu", copy=True) for n, p in model.named_parameters() if p.requires_grad}


def load_trainable_state(model, state: Dict[str, torch.Tensor]):
    params = dict(model.named_parameters())
    for n, t in state.items():
        params[n].data.copy_(t.to(params[n].device, dtype=params[n].dtype))


def lora_state(model) -> Dict[str, torch.Tensor]:
    out = {}
    for n, m in lora_modules(model):
        out[n + ".lora_a"] = m.lora_a.detach().cpu().clone()
        out[n + ".lora_b"] = m.lora_b.detach().cpu().clone()
    return out


def load_lora_state(model, state: Dict[str, torch.Tensor]):
    mods = dict(lora_modules(model))
    for k, v in state.items():
        n, f = k.rsplit(".", 1)
        p = getattr(mods[n], f)
        if p.shape != v.shape:  # rank changed (truncation)
            setattr(mods[n], f, nn.Parameter(v.to(p.device, dtype=p.dtype)))
        else:
            p.data.copy_(v.to(p.device, dtype=p.dtype))


@torch.no_grad()
def truncate_lora_rank(model, keep_rank: int) -> Dict:
    """Defense D2, scaling-preserving.

    ΔW_eff = scaling * A B  ->  SVD  ->  A' = U_r sqrt(S_r), B' = sqrt(S_r) V_r^T,
    and *scaling is set to 1* so the retained components are not re-amplified
    (the previous version reset scaling to alpha / r', inflating ΔW by r/r').
    Returns the mean retained spectral energy.
    """
    retained = []
    for _, m in lora_modules(model):
        a = m.lora_a.data.float()
        b = m.lora_b.data.float()
        delta = (a @ b) * float(m.scaling)
        u, s, vh = torch.linalg.svd(delta, full_matrices=False)
        r = min(keep_rank, int(s.numel()))
        retained.append(float((s[:r] ** 2).sum() / (s**2).sum().clamp_min(1e-12)))
        new_a = (u[:, :r] * s[:r].sqrt().unsqueeze(0)).to(m.lora_a.dtype)
        new_b = (s[:r].sqrt().unsqueeze(1) * vh[:r, :]).to(m.lora_b.dtype)
        m.lora_a = nn.Parameter(new_a.to(m.lora_a.device))
        m.lora_b = nn.Parameter(new_b.to(m.lora_b.device))
        m.rank = r
        m.scaling = 1.0
    return {"keep_rank": keep_rank, "retained_energy": sum(retained) / max(len(retained), 1)}


@torch.no_grad()
def merge_lora_states(states: List[Dict[str, torch.Tensor]], method: str = "ties", densify: float = 0.3) -> Dict[str, torch.Tensor]:
    """D4: merge k shard adapters (uniform or TIES-style) in the *product* space.

    LoRA factors of independently trained adapters are not aligned, so we merge
    ΔW_k = A_k B_k and re-factorize to rank r via SVD.
    """
    keys = sorted({k.rsplit(".", 1)[0] for k in states[0]})
    merged = {}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for key in keys:
        deltas = torch.stack([sd[key + ".lora_a"].to(dev).float() @ sd[key + ".lora_b"].to(dev).float() for sd in states])
        if method == "uniform":
            d = deltas.mean(0)
        else:
            flat = deltas.reshape(deltas.size(0), -1)
            k = max(1, int(densify * flat.size(1)))
            idx = torch.topk(flat.abs(), k=k, dim=1).indices
            mask = torch.zeros_like(flat).scatter_(1, idx, 1.0)
            trimmed = flat * mask
            sign = torch.sign(trimmed.sum(0, keepdim=True))
            sign[sign == 0] = 1
            agreed = trimmed * (torch.sign(trimmed) == sign).float()
            denom = (agreed != 0).float().sum(0).clamp_min(1.0)
            d = (agreed.sum(0) / denom).reshape_as(deltas[0])
        r = states[0][key + ".lora_a"].shape[1]
        u, s, vh = torch.linalg.svd(d, full_matrices=False)
        merged[key + ".lora_a"] = (u[:, :r] * s[:r].sqrt()).to(states[0][key + ".lora_a"].dtype).cpu()
        merged[key + ".lora_b"] = (s[:r].sqrt().unsqueeze(1) * vh[:r]).to(states[0][key + ".lora_b"].dtype).cpu()
    return merged


def set_lora_scaling(model, value: float):
    for _, m in lora_modules(model):
        m.scaling = value
