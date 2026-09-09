"""Merge LoRA experts trained on audio-classification tasks and report
top-1 accuracy of the merged AST backbone on every task.

Audio counterpart of ``projects/vision_merging/merge_vision_experts.py`` and of
``projects/modular_llm/eval_library.py``: the same MTTL merge transforms are
reused. Each task keeps its own trained classification head; only the AST
attention backbone is merged.

SOATA in the ICASSP draft is exposed via dedicated merge methods:
  * ``soata`` / ``soata_ties``  -> shared-basis SVD + TIES on coordinates
  * ``soata_linear``            -> shared-basis SVD + linear coordinate blend
Legacy aliases (``knots*``) remain for backward compatibility.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
from tqdm.auto import tqdm
from transformers import ASTForAudioClassification

from mttl.logging import logger, setup_logging
from mttl.models.library.expert import Expert
from mttl.models.library.expert_library import ExpertLibrary
from mttl.models.library.library_transforms import (
    DareMerge,
    DareMergeConfig,
    ISOMerge,
    ISOMergeConfig,
    KnotMerge,
    KnotMergeConfig,
    SoataMerge,
    SoataMergeConfig,
    TaskArithmeticMerge,
    TaskArithmeticConfig,
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

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from audio_data import get_audio_dataloaders

BACKBONE_PREFIX = "audio_spectrogram_transformer."

MERGE_METHODS = [
    "base",
    "individual",
    "uniform",
    "task_arithmetic",
    "ties",
    "dare_ties",
    "dare_task_arithmetic",
    "uniform_merge_after",
    "ties_merge_after",
    "wudi",
    "wudi_merge_after",
    "iso",
    "tsv",
    "knots",
    "knots_ties",
    "knots_linear",
    "soata",
    "soata_ties",
    "soata_linear",
    "delta_linear",
]


def library_fs_path(library_id: str) -> str:
    for prefix in ("local://", "hf://"):
        if library_id.startswith(prefix):
            return library_id[len(prefix) :]
    return library_id


def expert_to_layer_deltas(expert: Expert) -> dict:
    """Convert a LoRA expert into {base_layer_name: delta_W} (delta = lora_a @ lora_b)."""
    layers = sorted({k.split(".lora_")[0] for k in expert.expert_weights.keys()})
    deltas = {}
    for layer in layers:
        lora_a = expert.expert_weights[f"{layer}.lora_a"]
        lora_b = expert.expert_weights[f"{layer}.lora_b"]
        deltas[layer] = lora_a.data @ lora_b.data
    return deltas


def _knot_path(library_id: str) -> str:
    return os.path.join(library_fs_path(library_id), "knot_ingredients.pt")


def compute_merged_deltas(
    method: str,
    library: ExpertLibrary,
    library_id: str = "local://audio_library",
    wudi_iter: int = 300,
    wudi_lr: float = 1e-5,
    ta_scaling: float = 1.0,
    dare_drop_rate: float = 0.7,
    ties_top_k: float = 0.2,
    retained_rank: int = -1,
    recompute: bool = True,
    weights: dict | None = None,
    knot_path: str | None = None,
) -> dict:
    knot_path = knot_path or _knot_path(library_id)
    if method == "uniform":
        expert = WeightedLinearMerge(
            WeightedLinearMergeConfig(weights=weights)
        ).transform(library)
        return expert_to_layer_deltas(expert)
    if method in ("delta_linear", "product_linear"):
        # Average LoRA *products* (the theorem's ΔW* = Σ λ_k ΔW_k), not LoRA factors.
        names = list(library.keys())
        if weights is None:
            coeff = {n: 1.0 / max(len(names), 1) for n in names}
        else:
            coeff = {n: float(weights[n]) for n in names}
        acc = None
        for name in names:
            deltas = expert_to_layer_deltas(library[name])
            if acc is None:
                acc = {layer: coeff[name] * delta for layer, delta in deltas.items()}
            else:
                for layer, delta in deltas.items():
                    acc[layer] = acc[layer] + coeff[name] * delta
        return acc
    if method in ("task_arithmetic", "ta"):
        expert = TaskArithmeticMerge(
            TaskArithmeticConfig(ta_scaling=float(ta_scaling))
        ).transform(library)
        return expert_to_layer_deltas(expert)
    if method == "ties":
        expert = TiesMerge(TiesMergeConfig(top_k=float(ties_top_k))).transform(library)
        return expert_to_layer_deltas(expert)
    if method in ("dare_ties", "dare"):
        expert = DareMerge(
            DareMergeConfig(
                dare_drop_rate=float(dare_drop_rate),
                dare_merge_method="ties",
                ties_top_k=float(ties_top_k),
            )
        ).transform(library)
        return expert_to_layer_deltas(expert)
    if method in ("dare_task_arithmetic", "dare_ta"):
        expert = DareMerge(
            DareMergeConfig(
                dare_drop_rate=float(dare_drop_rate),
                dare_merge_method="task_arithmetic",
                ta_scaling=float(ta_scaling),
            )
        ).transform(library)
        return expert_to_layer_deltas(expert)
    if method == "wudi":
        merged = WudiMerge(WudiMergeConfig(iter=wudi_iter, lr=wudi_lr)).transform(library)
        return expert_to_layer_deltas(Expert(expert_info=None, expert_weights=merged))
    if method == "uniform_merge_after":
        return UniformMergeAfter(UniformMergeAfterConfig()).transform(library)
    if method == "ties_merge_after":
        return TiesMergeAfter(TiesMergeAfterConfig(mask_rate=0.8)).transform(library)
    if method == "wudi_merge_after":
        return WudiMergeAfter(
            WudiMergeConfig(iter=wudi_iter, lr=wudi_lr)
        ).transform(library)
    if method == "iso":
        return ISOMerge(ISOMergeConfig()).transform(library)
    if method == "tsv":
        return TSVMerge(TSVMergeConfig()).transform(library, recompute=True)
    if method in ("knots", "knots_ties", "knots_linear"):
        # KnotMerge returns delta_W in (out, in); transpose to (in, out) so it matches
        # apply_deltas_to_backbone (which transposes once more to (out, in)).
        knot_method = "linear" if method == "knots_linear" else "ties"
        merged = KnotMerge(
            KnotMergeConfig(
                path=knot_path,
                merge_method=knot_method,
                retained_rank=int(retained_rank),
                weights=weights,
            )
        ).transform(library, recompute=recompute)
        return {layer: d.T for layer, d in merged.items()}
    if method in ("soata", "soata_ties", "soata_linear"):
        # SOATA has its own class (built on KnotMerge + energy-preserving rescale).
        # Return orientation matches KnotMerge path: (in, out) for application helper.
        soata_method = "linear" if method == "soata_linear" else "ties"
        merged = SoataMerge(
            SoataMergeConfig(
                path=knot_path,
                merge_method=soata_method,
                retained_rank=int(retained_rank),
                weights=weights,
                preserve_energy=True,
            )
        ).transform(library, recompute=recompute)
        return {layer: d.T for layer, d in merged.items()}
    raise ValueError(f"Unknown merge method {method}")


def apply_deltas_to_backbone(state_dict, deltas, scaling_coefficient=1.0):
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
    head = torch.load(
        os.path.join(heads_dir, f"{task}.pt"), map_location="cpu", weights_only=False
    )
    num_classes = head["num_classes"]

    model = ASTForAudioClassification.from_pretrained(
        model_name, num_labels=num_classes, ignore_mismatched_sizes=True
    )
    model.load_state_dict(backbone_state, strict=False)
    model.classifier.layernorm.weight.data.copy_(head["classifier.layernorm.weight"])
    model.classifier.layernorm.bias.data.copy_(head["classifier.layernorm.bias"])
    model.classifier.dense.weight.data.copy_(head["classifier.dense.weight"])
    model.classifier.dense.bias.data.copy_(head["classifier.dense.bias"])
    model.to(device).eval()

    _, test_loader, _, _ = get_audio_dataloaders(
        task_name=task,
        model_name=model_name,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        subsample_test=args.subsample_test,
        seed=args.seed,
    )

    correct, total = 0, 0
    for batch in tqdm(test_loader, desc=f"eval[{task}]", leave=False):
        input_values = batch["input_values"].to(device)
        labels = batch["labels"].to(device)
        preds = model(input_values=input_values).logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.numel()
    del model
    torch.cuda.empty_cache()
    return correct / max(total, 1)


def result_tag(args) -> str:
    """Filename stem used for JSON dumps (unique per method / rank)."""
    tag = args.merge_method
    if args.merge_method in (
        "knots",
        "knots_ties",
        "knots_linear",
        "soata",
        "soata_ties",
        "soata_linear",
    ) and args.retained_rank > 0:
        tag = f"{args.merge_method}_R{args.retained_rank}"
    return tag


def write_latex_row(method: str, scores: dict, tasks) -> str:
    cells = [f"{scores.get(t, 0.0) * 100:.1f}" for t in tasks]
    avg = scores.get("avg", 0.0) * 100
    ret = scores.get("retention")
    ret_s = f"{ret:.1f}" if isinstance(ret, (int, float)) else "--"
    return method + " & " + " & ".join(cells) + f" & {avg:.1f} & {ret_s} \\\\"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library_id", default="local://audio_library")
    parser.add_argument("--heads_dir", default="audio_library_heads")
    parser.add_argument(
        "--model_name", default="MIT/ast-finetuned-audioset-10-10-0.4593"
    )
    parser.add_argument(
        "--merge_method",
        default="knots",
        choices=MERGE_METHODS,
        help="base | individual | uniform | task_arithmetic | ties | dare_ties | "
        "dare_task_arithmetic | wudi | wudi_merge_after | iso | tsv | "
        "knots | knots_ties | knots_linear | soata | soata_ties | soata_linear",
    )
    parser.add_argument("--scaling_coefficient", type=float, default=1.0)
    parser.add_argument("--ta_scaling", type=float, default=1.0)
    parser.add_argument("--dare_drop_rate", type=float, default=0.7)
    parser.add_argument("--ties_top_k", type=float, default=0.2)
    parser.add_argument(
        "--wudi_iter", type=int, default=300, help="optimization steps for wudi / wudi_merge_after"
    )
    parser.add_argument(
        "--wudi_lr", type=float, default=1e-5, help="learning rate for wudi / wudi_merge_after"
    )
    parser.add_argument(
        "--retained_rank",
        type=int,
        default=-1,
        help="KnoT/SOATA truncated SVD rank R; -1 keeps the numerical rank.",
    )
    parser.add_argument(
        "--recompute_prototypes",
        action="store_true",
        help="Recompute the joint SVD cache (knot_ingredients.pt).",
    )
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--subsample_test", type=int, default=-1)
    parser.add_argument("--output_dir", default="audio_output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--expert_scores_json",
        default=None,
        help="JSON of single-task expert scores used to compute retention.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip if ${output_dir}/${method}.json already exists.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(args.output_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tag = result_tag(args)
    out_path = os.path.join(args.output_dir, f"{tag}.json")
    if args.skip_existing and os.path.isfile(out_path):
        logger.info("Skipping %s (already have %s)", tag, out_path)
        print(open(out_path).read())
        return

    library = ExpertLibrary.get_expert_library(repo_id=args.library_id)
    tasks = list(library.keys())
    logger.info(f"Loaded {len(tasks)} experts: {tasks}")

    base = ASTForAudioClassification.from_pretrained(
        args.model_name, num_labels=2, ignore_mismatched_sizes=True
    )
    base_state = base.state_dict()

    def backbone_only(state):
        return {k: v for k, v in state.items() if k.startswith(BACKBONE_PREFIX)}

    t0 = time.time()
    results = {}
    if args.merge_method == "base":
        backbone_state = backbone_only(base_state)
        for task in tasks:
            acc = evaluate_task(
                task, args.model_name, backbone_state, args.heads_dir, device, args
            )
            results[task] = acc
            logger.info(f"[base] {task}: {acc:.4f}")
    elif args.merge_method == "individual":
        for task in tasks:
            deltas = expert_to_layer_deltas(library[task])
            merged = apply_deltas_to_backbone(base_state, deltas, args.scaling_coefficient)
            acc = evaluate_task(
                task, args.model_name, backbone_only(merged), args.heads_dir, device, args
            )
            results[task] = acc
            logger.info(f"[individual] {task}: {acc:.4f}")
    else:
        logger.info(f"Merging experts with method `{args.merge_method}`")
        recompute = args.recompute_prototypes or not os.path.isfile(
            _knot_path(args.library_id)
        )
        deltas = compute_merged_deltas(
            args.merge_method,
            library,
            library_id=args.library_id,
            wudi_iter=args.wudi_iter,
            wudi_lr=args.wudi_lr,
            ta_scaling=args.ta_scaling,
            dare_drop_rate=args.dare_drop_rate,
            ties_top_k=args.ties_top_k,
            retained_rank=args.retained_rank,
            recompute=recompute,
        )
        merged = apply_deltas_to_backbone(base_state, deltas, args.scaling_coefficient)
        backbone_state = backbone_only(merged)
        for task in tasks:
            acc = evaluate_task(
                task, args.model_name, backbone_state, args.heads_dir, device, args
            )
            results[task] = acc
            logger.info(f"[{args.merge_method}] {task}: {acc:.4f}")

    merge_seconds = time.time() - t0
    mean_acc = sum(results.values()) / max(len(results), 1)
    results["avg"] = mean_acc

    retention = None
    if args.expert_scores_json and os.path.isfile(args.expert_scores_json):
        with open(args.expert_scores_json) as handle:
            payload = json.load(handle)
        expert_scores = payload.get("scores", payload)
        expert_avg = expert_scores.get("avg")
        if expert_avg:
            retention = 100.0 * mean_acc / expert_avg
    elif args.merge_method == "individual":
        retention = 100.0
    results["retention"] = retention

    hparams = (
        f"scaling={args.scaling_coefficient} iter={args.wudi_iter} lr={args.wudi_lr} "
        f"R={args.retained_rank}"
    )
    logger.info("==================== RESULTS ====================")
    logger.info(f"method = {args.merge_method} | {hparams}")
    for task in tasks:
        logger.info(f"  {task:<16s}: {results[task] * 100:.2f}")
    logger.info(f"  {'mean':<16s}: {mean_acc * 100:.2f}")
    if retention is not None:
        logger.info(f"  {'retention':<16s}: {retention:.1f}%")
    logger.info("=================================================")

    payload = {
        "merge_method": args.merge_method,
        "tag": tag,
        "library_id": args.library_id,
        "tasks": tasks,
        "scores": results,
        "merge_seconds": merge_seconds,
        "retained_rank": args.retained_rank,
        "scaling_coefficient": args.scaling_coefficient,
        "latex_row": write_latex_row(tag, results, tasks),
    }
    with open(out_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    logger.info("Saved %s", out_path)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
