"""Data utilities for audio-classification model merging with AST (ViT for audio).

We evaluate on six diverse audio classification tasks spanning environmental
sound, urban sound, music genre, speech emotion and keyword spotting:
  * ESC-50          - environmental sound (50 classes)
  * UrbanSound8K     - urban sound (10 classes)
  * GTZAN            - music genre (10 classes)
  * RAVDESS          - speech emotion (8 classes)
  * CREMA-D          - speech emotion (6 classes)
  * Speech Commands  - keyword spotting (~35 classes)

Audio is decoded and resampled to 16 kHz and turned into log-mel spectrograms by
the AST feature extractor (``input_values`` of shape (1024, 128)).
"""

import numpy as np
import torch
from datasets import Audio, load_dataset
from transformers import AutoFeatureExtractor

# task_name -> dict(repo, config, label_col, split)
# split strategy:
#   "native"  -> dataset already provides train/test
#   "fold:N"  -> column 'fold' present; fold == N is the test set
#   "ratio"   -> single split; deterministic 80/20 split
AUDIO_TASKS = {
    "esc50": dict(repo="ashraq/esc50", config=None, label_col="target", split="fold:5"),
    "urbansound8k": dict(
        repo="danavery/urbansound8K", config=None, label_col="classID", split="fold:10"
    ),
    "gtzan": dict(
        repo="sanchit-gandhi/gtzan", config=None, label_col="genre", split="ratio"
    ),
    "ravdess": dict(
        repo="xbgoose/ravdess", config=None, label_col="emotion", split="ratio"
    ),
    "cremad": dict(
        repo="AbstractTTS/CREMA-D", config=None, label_col="major_emotion", split="ratio"
    ),
    "speech_commands": dict(
        repo="google/speech_commands", config="v0.02", label_col="label", split="native"
    ),
}

ALL_TASKS = list(AUDIO_TASKS.keys())

TARGET_SR = 16000


def _load_full(spec):
    repo, config = spec["repo"], spec["config"]
    kwargs = dict(trust_remote_code=True)
    if config:
        return load_dataset(repo, config, **kwargs)
    return load_dataset(repo, **kwargs)


def _normalize_label(value):
    """Return a hashable canonical label (str for text labels, int for ints)."""
    if isinstance(value, (int, np.integer)):
        return int(value)
    return str(value)


def _train_test_split(spec, dataset, seed):
    """Resolve the (train_hf, test_hf) HF datasets given the split strategy."""
    strategy = spec["split"]
    if strategy == "native":
        test_key = "test" if "test" in dataset else "validation"
        return dataset["train"], dataset[test_key]

    # single split datasets are stored under "train"
    base = dataset["train"] if "train" in dataset else dataset[list(dataset.keys())[0]]

    if strategy.startswith("fold:"):
        test_fold = int(strategy.split(":")[1])
        folds = np.array(base["fold"])
        test_idx = np.where(folds == test_fold)[0].tolist()
        train_idx = np.where(folds != test_fold)[0].tolist()
        return base.select(train_idx), base.select(test_idx)

    if strategy == "ratio":
        splits = base.train_test_split(test_size=0.2, seed=seed)
        return splits["train"], splits["test"]

    raise ValueError(f"Unknown split strategy {strategy}")


class AudioClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, feature_extractor, label_col, label2id):
        self.ds = hf_dataset
        self.fe = feature_extractor
        self.label_col = label_col
        self.label2id = label2id

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        example = self.ds[idx]
        audio = example["audio"]
        wav = np.asarray(audio["array"], dtype=np.float32)
        if wav.ndim > 1:  # stereo -> mono
            wav = wav.mean(axis=-1)
        features = self.fe(
            wav, sampling_rate=TARGET_SR, return_tensors="pt"
        )["input_values"][0]
        label = self.label2id[_normalize_label(example[self.label_col])]
        return {"input_values": features, "labels": label}


def collate_fn(batch):
    return {
        "input_values": torch.stack([b["input_values"] for b in batch]),
        "labels": torch.tensor([b["labels"] for b in batch], dtype=torch.long),
    }


def get_audio_dataloaders(
    task_name: str,
    model_name: str,
    train_batch_size: int = 16,
    eval_batch_size: int = 32,
    num_workers: int = 4,
    subsample_train: int = -1,
    subsample_test: int = -1,
    seed: int = 42,
):
    """Returns (train_loader, test_loader, num_classes, label2id)."""
    if task_name not in AUDIO_TASKS:
        raise ValueError(f"Unknown task {task_name}. Available: {ALL_TASKS}")
    spec = AUDIO_TASKS[task_name]

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

    dataset = _load_full(spec)
    # resample all audio to 16 kHz on decode
    for split in list(dataset.keys()):
        dataset[split] = dataset[split].cast_column(
            "audio", Audio(sampling_rate=TARGET_SR)
        )

    train_hf, test_hf = _train_test_split(spec, dataset, seed)

    # build a label -> id map that is consistent across train and test
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

    train_loader = torch.utils.data.DataLoader(
        AudioClassificationDataset(train_hf, feature_extractor, label_col, label2id),
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        AudioClassificationDataset(test_hf, feature_extractor, label_col, label2id),
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    return train_loader, test_loader, num_classes, label2id
