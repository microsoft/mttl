"""Data utilities for image-classification model merging with ViT-B/16.

We evaluate on six diverse image classification datasets: DTD, EuroSAT, GTSRB,
RESISC45, SUN397 and SVHN. All datasets are pulled from the Hugging Face Hub
(the `tanganke/*` mirrors used by the task-arithmetic / model-merging
literature) and exposed through a small torch ``Dataset`` that applies the
ViT image processor transforms.
"""

import random

import torch
from datasets import load_dataset
from PIL import Image, ImageOps
from transformers import AutoImageProcessor

# task_name -> (hugging face dataset repo, config_name or None)
VISION_TASKS = {
    "dtd": ("tanganke/dtd", None),
    "eurosat": ("tanganke/eurosat", None),
    "gtsrb": ("tanganke/gtsrb", None),
    "resisc45": ("tanganke/resisc45", None),
    "sun397": ("tanganke/sun397", None),
    "svhn": ("ufldl-stanford/svhn", "cropped_digits"),
}

ALL_TASKS = list(VISION_TASKS.keys())


class VisionClassificationDataset(torch.utils.data.Dataset):
    """Applies the ViT image processor (resize + normalize). For training we add a
    light random horizontal flip as augmentation. This avoids a torchvision
    dependency by relying purely on the HF processor + PIL.
    """

    def __init__(self, hf_dataset, processor, train=False):
        self.ds = hf_dataset
        self.processor = processor
        self.train = train

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        example = self.ds[idx]
        image = example["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        if self.train and random.random() < 0.5:
            image = ImageOps.mirror(image)
        pixel_values = self.processor(images=image, return_tensors="pt")[
            "pixel_values"
        ][0]
        return {
            "pixel_values": pixel_values,
            "labels": int(example["label"]),
        }


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "labels": torch.tensor([b["labels"] for b in batch], dtype=torch.long),
    }


def get_vision_dataloaders(
    task_name: str,
    model_name: str,
    train_batch_size: int = 32,
    eval_batch_size: int = 64,
    num_workers: int = 4,
    subsample_train: int = -1,
    subsample_test: int = -1,
    seed: int = 42,
):
    """Returns (train_loader, test_loader, num_classes, class_names)."""
    if task_name not in VISION_TASKS:
        raise ValueError(
            f"Unknown task {task_name}. Available: {list(VISION_TASKS.keys())}"
        )

    processor = AutoImageProcessor.from_pretrained(model_name)
    repo, config_name = VISION_TASKS[task_name]
    dataset = load_dataset(repo, config_name) if config_name else load_dataset(repo)

    train_split = dataset["train"]
    test_split = dataset["test"]

    if subsample_train and subsample_train > 0:
        n = min(subsample_train, len(train_split))
        train_split = train_split.shuffle(seed=seed).select(range(n))
    if subsample_test and subsample_test > 0:
        n = min(subsample_test, len(test_split))
        test_split = test_split.shuffle(seed=seed).select(range(n))

    label_feature = dataset["train"].features["label"]
    num_classes = label_feature.num_classes
    class_names = label_feature.names

    train_loader = torch.utils.data.DataLoader(
        VisionClassificationDataset(train_split, processor, train=True),
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        VisionClassificationDataset(test_split, processor, train=False),
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    return train_loader, test_loader, num_classes, class_names
