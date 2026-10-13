"""Train / load LoRA adapters for causal LM on clinical notes."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from mttl.logging import logger
from mttl.models.modifiers.lora import LoRA, LoRAConfig
from mttl.models.modifiers.modify_model import modify_transformer


@dataclass
class TrainConfig:
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    lora_rank: int = 16
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05
    modify_layers: str = "q_proj|k_proj|v_proj|o_proj"
    modify_modules: str = ".*"
    learning_rate: float = 1e-4
    num_train_epochs: int = 3
    batch_size: int = 1
    grad_accum: int = 8
    max_length: int = 512
    seed: int = 42
    optimizer: str = "adamw"  # adamw | sgd
    freeze_a: bool = False
    dp_epsilon: Optional[float] = None
    dp_max_grad_norm: float = 1.0
    dp_delta: float = 1e-5


class NoteDataset(Dataset):
    def __init__(self, texts: Sequence[str], tokenizer, max_length: int):
        self.texts = list(texts)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def load_base_model(model_name: str, device: str, dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    return model, tokenizer


def attach_lora(model, cfg: TrainConfig) -> LoRAConfig:
    lora_cfg = LoRAConfig(
        modify_modules=cfg.modify_modules,
        modify_layers=cfg.modify_layers,
        lora_rank=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
    )
    modify_transformer(model, lora_cfg)
    for name, param in model.named_parameters():
        if ".lora_" not in name:
            param.requires_grad = False
        elif cfg.freeze_a and ".lora_a" in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Trainable LoRA params: %s", f"{n_train:,}")
    return lora_cfg


def collect_lora_modules(model) -> List[Tuple[str, LoRA]]:
    out = []
    for name, module in model.named_modules():
        if isinstance(module, LoRA):
            out.append((name, module))
    return out


def lora_state_dict(model) -> Dict[str, torch.Tensor]:
    state = {}
    for name, module in collect_lora_modules(model):
        state[f"{name}.lora_a"] = module.lora_a.detach().cpu().contiguous().clone()
        state[f"{name}.lora_b"] = module.lora_b.detach().cpu().contiguous().clone()
    return state


def load_lora_state(model, state: Dict[str, torch.Tensor]) -> None:
    modules = dict(collect_lora_modules(model))
    for key, tensor in state.items():
        name, field = key.rsplit(".", 1)
        module = modules[name]
        getattr(module, field).data.copy_(tensor.to(getattr(module, field).device, dtype=getattr(module, field).dtype))


def count_lora_params(model) -> int:
    return sum(
        module.lora_a.numel() + module.lora_b.numel()
        for _, module in collect_lora_modules(model)
    )


def merge_lora_into_base(model) -> None:
    for _, module in collect_lora_modules(model):
        module.merge_with_layer()


def disable_lora(model) -> None:
    for _, module in collect_lora_modules(model):
        module.disable()


def enable_lora(model) -> None:
    for _, module in collect_lora_modules(model):
        module.enable()


def truncate_lora_rank(model, keep_rank: int) -> None:
    """Post-hoc SVD truncation of each adapter (defense D2)."""
    for _, module in collect_lora_modules(model):
        a = module.lora_a.data.float()
        b = module.lora_b.data.float()
        # delta = a @ b  (in_features, out_features)
        delta = a @ b
        u, s, vh = torch.linalg.svd(delta, full_matrices=False)
        r = min(keep_rank, int(s.numel()))
        u_r = u[:, :r]
        s_r = s[:r]
        vh_r = vh[:r, :]
        dtype = module.lora_a.dtype
        device = module.lora_a.device
        new_a = (u_r * s_r.sqrt().unsqueeze(0)).to(device=device, dtype=dtype)
        new_b = (s_r.sqrt().unsqueeze(1) * vh_r).to(device=device, dtype=dtype)
        module.lora_a = nn.Parameter(new_a)
        module.lora_b = nn.Parameter(new_b)
        module.rank = r
        module.scaling = module.alpha / max(r, 1)


def _gaussian_sigma(epsilon: float, delta: float, steps: int, noise_multiplier_guess: float = 1.0) -> float:
    # Conservative analytic approx: sigma ~= sqrt(2 log(1.25/delta)) / epsilon * sqrt(steps)
    # We use a simpler fixed mapping for the paper sweep.
    if epsilon is None or epsilon <= 0:
        return 0.0
    return max(0.5, (math.sqrt(2.0 * math.log(1.25 / max(delta, 1e-12))) / epsilon))


def train_lora_on_notes(
    texts: Sequence[str],
    cfg: TrainConfig,
    output_dir: str,
    device: Optional[str] = None,
) -> Dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)
    torch.manual_seed(cfg.seed)

    model, tokenizer = load_base_model(cfg.model_name, device)
    attach_lora(model, cfg)
    model.train()

    ds = NoteDataset(texts, tokenizer, cfg.max_length)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

    params = [p for p in model.parameters() if p.requires_grad]
    if cfg.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, lr=cfg.learning_rate, momentum=0.9)
    else:
        optimizer = torch.optim.AdamW(params, lr=cfg.learning_rate)

    steps_per_epoch = max(1, math.ceil(len(loader) / cfg.grad_accum))
    total_steps = steps_per_epoch * cfg.num_train_epochs
    sigma = _gaussian_sigma(cfg.dp_epsilon, cfg.dp_delta, total_steps) if cfg.dp_epsilon else 0.0

    history = []
    global_step = 0
    t0 = time.time()
    for epoch in range(cfg.num_train_epochs):
        running = 0.0
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(loader, desc=f"train epoch {epoch}")
        for step, batch in enumerate(pbar):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / cfg.grad_accum
            loss.backward()
            running += out.loss.item()

            if (step + 1) % cfg.grad_accum == 0 or (step + 1) == len(loader):
                if sigma > 0:
                    # Per-step DP-LoRA approximation: clip then Gaussian noise on LoRA grads.
                    torch.nn.utils.clip_grad_norm_(params, cfg.dp_max_grad_norm)
                    for p in params:
                        if p.grad is None:
                            continue
                        noise = torch.randn_like(p.grad) * (sigma * cfg.dp_max_grad_norm)
                        p.grad.add_(noise)
                else:
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            pbar.set_postfix({"loss": f"{running / (step + 1):.4f}"})
        history.append({"epoch": epoch, "train_loss": running / max(step + 1, 1)})
        logger.info("epoch %s loss %.4f", epoch, history[-1]["train_loss"])

    state = lora_state_dict(model)
    ckpt_path = os.path.join(output_dir, "adapter.pt")
    meta = {
        "train_config": asdict(cfg),
        "n_texts": len(texts),
        "lora_params": count_lora_params(model),
        "train_seconds": time.time() - t0,
        "history": history,
        "dp_sigma": sigma,
    }
    torch.save({"lora_state": state, "meta": meta}, ckpt_path)
    with open(os.path.join(output_dir, "metrics.json"), "w") as handle:
        json.dump(meta, handle, indent=2)
    logger.info("Saved adapter to %s (%s params)", ckpt_path, meta["lora_params"])

    # Free GPU before returning paths.
    del model
    torch.cuda.empty_cache()
    return {"adapter_path": ckpt_path, "meta": meta, "tokenizer_name": cfg.model_name}


def load_adapted_model(
    model_name: str,
    adapter_path: str,
    device: Optional[str] = None,
    merge: bool = False,
    cfg_override: Optional[TrainConfig] = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    blob = torch.load(adapter_path, map_location="cpu")
    meta = blob["meta"]
    train_cfg = TrainConfig(**meta["train_config"])
    if cfg_override is not None:
        train_cfg = cfg_override
    model, tokenizer = load_base_model(model_name or train_cfg.model_name, device)
    attach_lora(model, train_cfg)
    load_lora_state(model, blob["lora_state"])
    if merge:
        merge_lora_into_base(model)
    model.eval()
    return model, tokenizer, train_cfg, meta
