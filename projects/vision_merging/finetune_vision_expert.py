"""Finetune a LoRA adapter on top of ViT-B/16 for a single image-classification task.

The trained LoRA adapter is stored as an MTTL ``Expert`` inside a (local)
``ExpertLibrary`` so that the *exact* same merging machinery used for the NLP
experiments (``projects/modular_llm/eval_library.py``) can be reused.

Because each image-classification task has a different label space, the linear
classification head is task specific. We therefore train the LoRA adapter
(attention query/value projections) *and* the classification head, then save:
  * the LoRA weights  -> expert library (these are what get merged)
  * the classifier head -> ``<heads_dir>/<task>.pt`` (used at eval time)

We set ``lora_alpha == lora_rank`` so the LoRA scaling factor is 1.0; this makes
``lora_a @ lora_b`` exactly equal to the trained weight delta, which is what the
MTTL merge transforms operate on.
"""

import argparse
import os

import torch
from torch import nn
from tqdm.auto import tqdm
from transformers import ViTForImageClassification

from mttl.logging import logger, setup_logging
from mttl.models.library.expert import Expert, ExpertInfo
from mttl.models.library.expert_library import ExpertLibrary
from mttl.models.modifiers.lora import LoRAConfig
from mttl.models.modifiers.modify_model import modify_transformer

from vision_data import get_vision_dataloaders

# patch LoRA onto the attention query/value projections of every ViT block
ATTENTION_MODULES = r"vit\.encoder\.layer\.\d+\.attention\.attention"
ATTENTION_LAYERS = "query|value"


def build_lora_config(args) -> LoRAConfig:
    return LoRAConfig(
        modify_modules=ATTENTION_MODULES,
        modify_layers=ATTENTION_LAYERS,
        lora_rank=args.lora_rank,
        # alpha == rank => scaling factor of 1.0 so the saved delta is exact
        lora_alpha=float(args.lora_rank),
        lora_dropout=args.lora_dropout,
    )


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for batch in tqdm(loader, desc="eval", leave=False):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        logits = model(pixel_values=pixel_values).logits
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.numel()
    return correct / max(total, 1)


def extract_lora_weights(model):
    """Return only the LoRA parameters, keyed by `<layer>.lora_a` / `<layer>.lora_b`."""
    return {
        name: param.detach().cpu().clone()
        for name, param in model.state_dict().items()
        if ".lora_" in name
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--model_name", default="google/vit-base-patch16-224-in21k")
    parser.add_argument("--library_id", default="local://vision_library")
    parser.add_argument("--heads_dir", default="vision_library_heads")
    parser.add_argument("--output_dir", default="vision_output")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--head_learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--subsample_train", type=int, default=-1)
    parser.add_argument("--subsample_test", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logging(args.output_dir)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.heads_dir, exist_ok=True)

    logger.info(f"==== Finetuning expert for task `{args.task}` ====")
    train_loader, test_loader, num_classes, _ = get_vision_dataloaders(
        task_name=args.task,
        model_name=args.model_name,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        subsample_train=args.subsample_train,
        subsample_test=args.subsample_test,
        seed=args.seed,
    )
    logger.info(f"num_classes={num_classes}, train_batches={len(train_loader)}")

    model = ViTForImageClassification.from_pretrained(
        args.model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )

    lora_config = build_lora_config(args)
    modify_transformer(model, lora_config)

    # the classification head is task specific and trained alongside the adapter
    for name, param in model.named_parameters():
        if name.startswith("classifier."):
            param.requires_grad = True

    lora_params = [p for n, p in model.named_parameters() if ".lora_" in n]
    head_params = [p for n, p in model.named_parameters() if n.startswith("classifier.")]
    n_lora = sum(p.numel() for p in lora_params)
    n_head = sum(p.numel() for p in head_params)
    logger.info(f"Trainable params: lora={n_lora:,} | head={n_head:,}")

    model.to(device)

    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": args.learning_rate},
            {"params": head_params, "lr": args.head_learning_rate},
        ],
        weight_decay=args.weight_decay,
    )
    total_steps = len(train_loader) * args.num_train_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: (
            step / max(warmup_steps, 1)
            if step < warmup_steps
            else max(0.0, (total_steps - step) / max(total_steps - warmup_steps, 1))
        ),
    )
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(args.num_train_epochs):
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"train epoch {epoch}")
        for step, batch in enumerate(pbar):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            logits = model(pixel_values=pixel_values).logits
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params + head_params, 1.0)
            optimizer.step()
            scheduler.step()

            running += loss.item()
            pbar.set_postfix({"loss": f"{running / (step + 1):.4f}"})

    acc = evaluate(model, test_loader, device)
    logger.info(f"[{args.task}] individual top-1 test accuracy: {acc:.4f}")

    # ------------------------------------------------------------------ save
    expert_weights = extract_lora_weights(model)
    expert_info = ExpertInfo(
        expert_name=args.task,
        expert_task_name=args.task,
        expert_config=lora_config,
        expert_model=args.model_name,
        training_config={
            "model": args.model_name,
            "num_classes": num_classes,
            "individual_accuracy": acc,
            "lora_rank": args.lora_rank,
        },
    )
    expert = Expert(expert_info=expert_info, expert_weights=expert_weights)

    library = ExpertLibrary.get_expert_library(repo_id=args.library_id, create=True)
    library.add_expert(expert, expert_name=args.task, force=True)
    logger.info(f"Added expert `{args.task}` to library `{args.library_id}`")

    head_state = {
        "classifier.weight": model.classifier.weight.detach().cpu().clone(),
        "classifier.bias": model.classifier.bias.detach().cpu().clone(),
        "num_classes": num_classes,
        "model_name": args.model_name,
        "accuracy": acc,
    }
    head_path = os.path.join(args.heads_dir, f"{args.task}.pt")
    torch.save(head_state, head_path)
    logger.info(f"Saved classification head to {head_path}")


if __name__ == "__main__":
    main()
