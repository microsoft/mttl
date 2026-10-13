"""MedMNIST loaders for vision LoRA experiments (public, non-credentialed)."""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

DATA_ROOT = os.environ.get("PRIVACY_LORA_MEDMNIST", "privacy_lora_v2/medmnist")

# flag -> (class_name in medmnist, num_classes, channels, clinical note)
DATASETS = {
    "pathmnist": ("PathMNIST", 9, 3, "colorectal histology (pathology)"),
    "dermamnist": ("DermaMNIST", 7, 3, "dermatoscopy of pigmented skin lesions"),
    "pneumoniamnist": ("PneumoniaMNIST", 2, 1, "pediatric chest X-ray pneumonia"),
}


def _ensure_medmnist():
    try:
        import medmnist  # noqa: F401
    except ImportError:
        import subprocess
        import sys

        subprocess.check_call([sys.executable, "-m", "pip", "install", "medmnist", "-q"])


def load_split(name: str, split: str = "train") -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Return (images uint8 HxWxC or HxW, labels int64, n_classes, in_ch)."""
    _ensure_medmnist()
    import medmnist
    from medmnist import INFO

    if name not in DATASETS:
        raise KeyError(name)
    cls_name, n_classes, in_ch, _ = DATASETS[name]
    info = INFO[name]
    DataClass = getattr(medmnist, info["python_class"])
    os.makedirs(DATA_ROOT, exist_ok=True)
    ds = DataClass(split=split, download=True, root=DATA_ROOT)
    # medmnist stores .imgs / .labels
    imgs = np.asarray(ds.imgs)
    labels = np.asarray(ds.labels).reshape(-1).astype(np.int64)
    return imgs, labels, n_classes, in_ch


class MedImages(Dataset):
    def __init__(self, imgs: np.ndarray, labels: np.ndarray, img_size: int = 64, in_ch: int = 3):
        self.imgs = imgs
        self.labels = labels
        self.img_size = img_size
        self.in_ch = in_ch

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        im = self.imgs[idx]
        if im.ndim == 2:
            im = np.stack([im, im, im], axis=-1) if self.in_ch == 3 else im[:, :, None]
        if im.shape[-1] == 1 and self.in_ch == 3:
            im = np.repeat(im, 3, axis=-1)
        if im.shape[-1] == 3 and self.in_ch == 1:
            im = im.mean(axis=-1, keepdims=True)
        pil = Image.fromarray(im.astype(np.uint8) if im.shape[-1] == 3 else im.squeeze(-1).astype(np.uint8))
        if self.in_ch == 1:
            pil = pil.convert("L")
        else:
            pil = pil.convert("RGB")
        pil = pil.resize((self.img_size, self.img_size), Image.BILINEAR)
        arr = np.asarray(pil).astype(np.float32) / 255.0
        if arr.ndim == 2:
            arr = arr[None, :, :]
        else:
            arr = arr.transpose(2, 0, 1)
        # ImageNet-ish normalization for RGB; identity-ish for gray
        if arr.shape[0] == 3:
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
            arr = (arr - mean) / std
        else:
            arr = (arr - 0.5) / 0.5
        x = torch.from_numpy(arr)
        y = int(self.labels[idx])
        return x, y


def sample_pool(name: str, m: int, seed: int, img_size: int = 64) -> Dict:
    imgs, labels, n_classes, in_ch = load_split(name, "train")
    rng = np.random.RandomState(seed)
    # Prefer diverse labels when possible
    idx = rng.choice(len(labels), size=min(m * 8, len(labels)), replace=False)
    chosen = []
    seen = set()
    for i in idx:
        if labels[i] not in seen or len(chosen) < m:
            chosen.append(int(i))
            seen.add(int(labels[i]))
        if len(chosen) >= m:
            break
    while len(chosen) < m:
        j = int(rng.randint(0, len(labels)))
        if j not in chosen:
            chosen.append(j)
    chosen = chosen[:m]
    sub_imgs = imgs[chosen]
    sub_labs = labels[chosen]
    ds = MedImages(sub_imgs, sub_labs, img_size=img_size, in_ch=3 if in_ch == 3 else 3)
    # force 3-channel for MiniViT simplicity (gray duplicated)
    xs, ys = [], []
    for k in range(m):
        x, y = ds[k]
        if x.shape[0] == 1:
            x = x.repeat(3, 1, 1)
        xs.append(x)
        ys.append(y)
    return {
        "name": name,
        "n_classes": n_classes,
        "in_ch": 3,
        "indices": chosen,
        "images": torch.stack(xs, 0),
        "labels": torch.tensor(ys, dtype=torch.long),
        "clinical": DATASETS[name][3],
    }
