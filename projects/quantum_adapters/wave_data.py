"""Waveform dataloaders for WavLM / HuBERT / wav2vec 2.0 (raw 16 kHz audio).

Reuses the task registry and decoders from ``projects/audio_merging/audio_data.py``.
"""

from __future__ import annotations

import os
import sys

import torch

_AUDIO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "audio_merging"))
if _AUDIO_DIR not in sys.path:
    sys.path.insert(0, _AUDIO_DIR)

from audio_data import (  # noqa: E402
    ALL_TASKS,
    AUDIO_TASKS,
    TARGET_SR,
    _dataloader_kwargs,
    _load_full,
    _normalize_label,
    _train_test_split,
    load_mono_wav,
)
from datasets import Audio  # noqa: E402


class WaveformClassificationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        hf_dataset,
        label_col,
        label2id,
        audio_col="audio",
        wf_sr=None,
        max_samples=None,
    ):
        self.ds = hf_dataset
        self.label_col = label_col
        self.label2id = label2id
        self.audio_col = audio_col
        self.wf_sr = wf_sr
        self.max_samples = max_samples

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        example = self.ds[idx]
        wav = load_mono_wav(
            example[self.audio_col],
            target_sr=TARGET_SR,
            default_sr=self.wf_sr,
        )
        wav = torch.from_numpy(wav).float()
        if self.max_samples is not None and wav.numel() > self.max_samples:
            wav = wav[: self.max_samples]
        label = self.label2id[_normalize_label(example[self.label_col])]
        return {"input_values": wav, "labels": label}


def collate_waveforms(batch):
    wavs = [b["input_values"] for b in batch]
    labels = torch.tensor([b["labels"] for b in batch], dtype=torch.long)
    lengths = [int(w.numel()) for w in wavs]
    max_len = max(lengths)
    padded = wavs[0].new_zeros(len(wavs), max_len)
    mask = torch.zeros(len(wavs), max_len, dtype=torch.long)
    for i, wav in enumerate(wavs):
        padded[i, : wav.numel()] = wav
        mask[i, : wav.numel()] = 1
    return {
        "input_values": padded,
        "attention_mask": mask,
        "labels": labels,
    }


def get_waveform_dataloaders(
    task_name: str,
    train_batch_size: int = 8,
    eval_batch_size: int = 16,
    num_workers: int = 0,
    subsample_train: int = -1,
    subsample_test: int = -1,
    seed: int = 42,
    max_seconds: float = 10.0,
):
    """Returns (train_loader, test_loader, num_classes, label2id)."""
    if task_name not in AUDIO_TASKS:
        raise ValueError(f"Unknown task {task_name}. Available: {ALL_TASKS}")
    spec = AUDIO_TASKS[task_name]
    dataset = _load_full(spec)
    audio_col = spec.get("audio_col", "audio")
    wf_sr = spec.get("wf_sr")
    for split in list(dataset.keys()):
        feats = dataset[split].features
        if audio_col in feats and getattr(feats[audio_col], "dtype", None) != "list":
            try:
                dataset[split] = dataset[split].cast_column(
                    audio_col, Audio(sampling_rate=TARGET_SR, decode=False)
                )
            except Exception:
                pass

    train_hf, test_hf = _train_test_split(spec, dataset, seed)
    label_col = spec["label_col"]
    labels = sorted(
        {_normalize_label(v) for v in train_hf[label_col]}
        | {_normalize_label(v) for v in test_hf[label_col]},
        key=lambda x: (isinstance(x, str), x),
    )
    label2id = {lab: i for i, lab in enumerate(labels)}
    num_classes = len(label2id)

    if subsample_train and subsample_train > 0:
        n = min(subsample_train, len(train_hf))
        train_hf = train_hf.shuffle(seed=seed).select(range(n))
    if subsample_test and subsample_test > 0:
        n = min(subsample_test, len(test_hf))
        test_hf = test_hf.shuffle(seed=seed).select(range(n))

    max_samples = int(max_seconds * TARGET_SR) if max_seconds and max_seconds > 0 else None
    train_loader = torch.utils.data.DataLoader(
        WaveformClassificationDataset(
            train_hf, label_col, label2id, audio_col, wf_sr, max_samples
        ),
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=collate_waveforms,
        drop_last=True,
        **_dataloader_kwargs(num_workers),
    )
    test_loader = torch.utils.data.DataLoader(
        WaveformClassificationDataset(
            test_hf, label_col, label2id, audio_col, wf_sr, max_samples
        ),
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=collate_waveforms,
        **_dataloader_kwargs(num_workers),
    )
    return train_loader, test_loader, num_classes, label2id
