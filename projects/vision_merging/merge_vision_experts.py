"""Merge LoRA experts trained on the six image-classification tasks and report
top-1 accuracy of the merged ViT-B/16 backbone on every task.

This mirrors ``projects/modular_llm/eval_library.py`` (the NLP example referenced
in ``test_merging.sh``) but for image classification: the *same* MTTL merge
transforms are reused (uniform / ties / wudi / wudi_merge_after / ...). Each task
keeps its own trained classification head; only the attention backbone is merged.
"""

import argparse
import os

import torch
from tqdm.auto import tqdm
from transformers import ViTForImageClassification

from mttl.logging import logger, setup_logging
from mttl.models.library.expert import Expert
from mttl.models.library.expert_library import ExpertLibrary
from mttl.models.library.library_transforms import (
    ISOMerge,
    ISOMergeConfig,
    KnotMerge,
    KnotMergeConfig,
    TiesMerge,
    TiesMergeConfig,
    TiesMergeAfter,
    TiesMergeAfterConfig,
    TSVMerge,
    TSVMergeConfig,
    UniformMergeAfter,
    UniformMergeAfterConfig,
    WeightedLinearMerge,
    WeightedLinearMergeConfig,
    WudiMerge,
    WudiMergeAfter,
    WudiMergeConfig,
)

from vision_data import get_vision_dataloaders


def expert_to_layer_deltas(expert: Expert) -> dict:
    """Convert a LoRA expert into {base_layer_name: delta_W} (delta = lora_a @ lora_b).

    Because we trained with lora_alpha == lora_rank, the LoRA scaling is 1.0, so
    this product is exactly the trained weight delta.
    """
    layers = sorted(
        {k.split(".lora_")[0] for k in expert.expert_weights.keys()}
    )
    deltas = {}
    for layer in layers:
        lora_a = expert.expert_weights[f"{layer}.lora_a"]
        lora_b = expert.expert_weights[f"{layer}.lora_b"]
        deltas[layer] = lora_a.data @ lora_b.data
    return deltas


def compute_merged_deltas(
    method: str, library: ExpertLibrary, wudi_iter: int = 300, wudi_lr: float = 1e-5
) -> dict:
    """Run a merge transform and always return {base_layer_name: delta_W}."""
    if method == "uniform":
        expert = WeightedLinearMerge(WeightedLinearMergeConfig()).transform(library)
        return expert_to_layer_deltas(expert)
    if method == "ties":
        expert = TiesMerge(TiesMergeConfig()).transform(library)
        return expert_to_layer_deltas(expert)
    if method == "wudi":
        # WudiMerge returns task vectors keyed by `<layer>.lora_{a,b}`; reduce to deltas
        merged = WudiMerge(WudiMergeConfig(iter=wudi_iter, lr=wudi_lr)).transform(library)
        fake = Expert(expert_info=None, expert_weights=merged)
        return expert_to_layer_deltas(fake)
    if method == "uniform_merge_after":
        return UniformMergeAfter(UniformMergeAfterConfig()).transform(library)
    if method == "ties_merge_after":
        return TiesMergeAfter(TiesMergeAfterConfig(mask_rate=0.8)).transform(library)
    if method == "wudi_merge_after":
        return WudiMergeAfter(
            WudiMergeConfig(iter=wudi_iter, lr=wudi_lr)
        ).transform(library)
    if method == "iso":
        # ISOMerge returns {layer: delta_W} in (in, out) orientation, like the *_after methods
        return ISOMerge(ISOMergeConfig()).transform(library)
    if method == "tsv":
        # TSVMerge returns {layer: delta_W} in (in, out) orientation; recompute=True avoids
        # reusing a stale `tsv_ingredients.pt` cache from a different library.
        return TSVMerge(TSVMergeConfig()).transform(library, recompute=True)
    if method == "knots":
        # KnotMerge returns delta_W in (out, in); transpose to (in, out) so it matches the
        # convention apply_deltas_to_backbone expects (which transposes once more to (out, in)).
        merged = KnotMerge(KnotMergeConfig()).transform(library, recompute=True)
        return {layer: d.T for layer, d in merged.items()}
    raise ValueError(f"Unknown merge method {method}")


def apply_deltas_to_backbone(state_dict, deltas, scaling_coefficient=1.0):
    """Return a copy of `state_dict` with merged deltas added to `<layer>.weight`."""
    new_state = {k: v.clone() for k, v in state_dict.items()}
    for layer, delta in deltas.items():
        weight_key = f"{layer}.weight"
        if weight_key not in new_state:
            logger.warning(f"{weight_key} not found in backbone; skipping")
            continue
        w = new_state[weight_key]
        d = delta.to(w.dtype).to(w.device)
        # Deltas are built as `lora_a @ lora_b` with lora_a:(in,rank), lora_b:(rank,out),
        # i.e. shape (in, out). The forward pass is `input @ (lora_a @ lora_b)`, which for
        # nn.Linear (`input @ weight.T`) corresponds to a weight delta of (lora_a @ lora_b).T,
        # shape (out, in). The attention q/v projections are square (768x768), so a
        # shape-based guard never transposes and would silently apply the wrong orientation.
        d = d.T
        if d.shape != w.shape:
            raise ValueError(
                f"delta shape {tuple(d.shape)} != weight shape {tuple(w.shape)} for {weight_key}"
            )
        new_state[weight_key] = w + scaling_coefficient * d
    return new_state


@torch.no_grad()
def evaluate_task(task, model_name, backbone_state, heads_dir, device, args):
    head_path = os.path.join(heads_dir, f"{task}.pt")
    head = torch.load(head_path, map_location="cpu", weights_only=False)
    num_classes = head["num_classes"]

    model = ViTForImageClassification.from_pretrained(
        model_name, num_labels=num_classes, ignore_mismatched_sizes=True
    )
    # load merged backbone (vit.*) then the task-specific head
    missing, unexpected = model.load_state_dict(backbone_state, strict=False)
    model.classifier.weight.data.copy_(head["classifier.weight"])
    model.classifier.bias.data.copy_(head["classifier.bias"])
    model.to(device).eval()

    _, test_loader, _, _ = get_vision_dataloaders(
        task_name=task,
        model_name=model_name,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        subsample_test=args.subsample_test,
        seed=args.seed,
    )

    correct, total = 0, 0
    for batch in tqdm(test_loader, desc=f"eval[{task}]", leave=False):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        preds = model(pixel_values=pixel_values).logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.numel()
    return correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library_id", default="local://vision_library")
    parser.add_argument("--heads_dir", default="vision_library_heads")
    parser.add_argument("--model_name", default="google/vit-base-patch16-224-in21k")
    parser.add_argument(
        "--merge_method",
        default="wudi_merge_after",
        help="base | individual | uniform | ties | wudi | uniform_merge_after | ties_merge_after | wudi_merge_after | iso | tsv | knots",
    )
    parser.add_argument("--scaling_coefficient", type=float, default=1.0)
    parser.add_argument(
        "--wudi_iter", type=int, default=300, help="optimization steps for wudi / wudi_merge_after"
    )
    parser.add_argument(
        "--wudi_lr", type=float, default=1e-5, help="learning rate for wudi / wudi_merge_after"
    )
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--subsample_test", type=int, default=-1)
    parser.add_argument("--output_dir", default="vision_output")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logging(args.output_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    library = ExpertLibrary.get_expert_library(repo_id=args.library_id)
    tasks = list(library.keys())
    logger.info(f"Loaded {len(tasks)} experts: {tasks}")

    # reference backbone state (clean ViT, no head)
    base = ViTForImageClassification.from_pretrained(
        args.model_name, num_labels=2, ignore_mismatched_sizes=True
    )
    base_state = base.state_dict()

    results = {}

    if args.merge_method == "base":
        # lower-bound baseline: pretrained backbone with NO deltas, each task's own head
        backbone_state = {k: v for k, v in base_state.items() if k.startswith("vit.")}
        for task in tasks:
            acc = evaluate_task(
                task, args.model_name, backbone_state, args.heads_dir, device, args
            )
            results[task] = acc
            logger.info(f"[base] {task}: {acc:.4f}")
    elif args.merge_method == "individual":
        # upper-bound baseline: every task uses its own adapter + own head
        for task in tasks:
            deltas = expert_to_layer_deltas(library[task])
            merged_state = apply_deltas_to_backbone(
                base_state, deltas, args.scaling_coefficient
            )
            backbone_state = {
                k: v for k, v in merged_state.items() if k.startswith("vit.")
            }
            acc = evaluate_task(
                task, args.model_name, backbone_state, args.heads_dir, device, args
            )
            results[task] = acc
            logger.info(f"[individual] {task}: {acc:.4f}")
    else:
        logger.info(f"Merging experts with method `{args.merge_method}`")
        deltas = compute_merged_deltas(
            args.merge_method, library, wudi_iter=args.wudi_iter, wudi_lr=args.wudi_lr
        )
        merged_state = apply_deltas_to_backbone(
            base_state, deltas, args.scaling_coefficient
        )
        backbone_state = {
            k: v for k, v in merged_state.items() if k.startswith("vit.")
        }
        for task in tasks:
            acc = evaluate_task(
                task, args.model_name, backbone_state, args.heads_dir, device, args
            )
            results[task] = acc
            logger.info(f"[{args.merge_method}] {task}: {acc:.4f}")

    mean_acc = sum(results.values()) / max(len(results), 1)
    hparams = f"scaling={args.scaling_coefficient} iter={args.wudi_iter} lr={args.wudi_lr}"
    logger.info("==================== RESULTS ====================")
    logger.info(f"method = {args.merge_method} | {hparams}")
    for task, acc in results.items():
        logger.info(f"  {task:<10s}: {acc * 100:.2f}")
    logger.info(f"  {'mean':<10s}: {mean_acc * 100:.2f}")
    logger.info("=================================================")

    print(f"\nmethod = {args.merge_method} | {hparams}")
    for task, acc in results.items():
        print(f"  {task:<10s}: {acc * 100:.2f}")
    print(f"  {'mean':<10s}: {mean_acc * 100:.2f}")


if __name__ == "__main__":
    main()
