"""Fine-tune linear / LoRA / QuAda adapters on AST or WavLM.

Examples
--------
# Parameter table (no GPU, no data):
python projects/quantum_adapters/finetune_quada.py --count_params_only

# AST + QuAda on ESC-50:
python projects/quantum_adapters/finetune_quada.py \\
    --task esc50 --backbone ast --method quada --n_mlp 1024

# WavLM SUPERB-style weighted-sum probing + generated layer weights:
python projects/quantum_adapters/finetune_quada.py \\
    --task speech_commands --backbone wavlm --method quada --gen_layer_weights
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch
from torch import nn
from tqdm.auto import tqdm

from mttl.logging import logger, setup_logging
from mttl.models.library.expert import Expert, ExpertInfo
from mttl.models.library.expert_library import ExpertLibrary
from mttl.models.modifiers.dora import DoRAConfig
from mttl.models.modifiers.lora import LoRAConfig
from mttl.models.modifiers.modify_model import modify_transformer
from mttl.models.modifiers.quada import (
    QuAdaConfig,
    QuAdaGenerator,
    collect_lora_modules,
    detach_lora_parameters,
    materialize_lora_state_dict,
    quada_resource_table,
)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_AUDIO_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "audio_merging"))
if _AUDIO_DIR not in sys.path:
    sys.path.insert(0, _AUDIO_DIR)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from audio_data import (  # noqa: E402
    _dataloader_kwargs,
    collate_fn,
    get_audio_dataloaders,
)
from wave_data import get_waveform_dataloaders  # noqa: E402

AST_DEFAULT = "MIT/ast-finetuned-audioset-10-10-0.4593"
WAVLM_DEFAULT = "microsoft/wavlm-base-plus"

BACKBONE_SPECS = {
    "ast": dict(
        default_model=AST_DEFAULT,
        modify_modules=r"audio_spectrogram_transformer\.encoder\.layer\.\d+\.attention\.attention",
        modify_layers="query|value",
        input_kind="spectrogram",
    ),
    "wavlm": dict(
        default_model=WAVLM_DEFAULT,
        modify_modules=r"wavlm\.encoder\.layers\.\d+\.attention",
        modify_layers="q_proj|v_proj",
        input_kind="waveform",
    ),
    "hubert": dict(
        default_model="facebook/hubert-base-ls960",
        modify_modules=r"hubert\.encoder\.layers\.\d+\.attention",
        modify_layers="q_proj|v_proj",
        input_kind="waveform",
    ),
}


class GeneratedAdapterModel(nn.Module):
    """Assign generated LoRA (and optional layer weights) then run the backbone."""

    def __init__(self, model, generator=None, lora_modules=None):
        super().__init__()
        self.model = model
        self.generator = generator
        self._lora_modules = list(lora_modules or [])

    def forward(self, **kwargs):
        if self.generator is not None:
            layer_logits = self.generator.assign_to_loras(self._lora_modules)
            if layer_logits is not None and hasattr(self.model, "layer_weights"):
                self.model.layer_weights = layer_logits
        return self.model(**kwargs)


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="esc50")
    parser.add_argument("--backbone", default="ast", choices=list(BACKBONE_SPECS))
    parser.add_argument("--model_name", default=None)
    parser.add_argument(
        "--method",
        default="quada",
        choices=["linear", "lora", "dora", "quada"],
    )
    parser.add_argument("--library_id", default="local://quada_library")
    parser.add_argument("--heads_dir", default="quada_heads")
    parser.add_argument("--output_dir", default="quada_output")
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--n_mlp", type=int, default=1024)
    parser.add_argument("--n_circuit_layers", type=int, default=8)
    parser.add_argument(
        "--ansatz", default="ry_cnot", choices=["ry_cnot", "rx_cnot", "ry_rz_cnot"]
    )
    parser.add_argument(
        "--index_mode", default="structured", choices=["structured", "flat"]
    )
    parser.add_argument("--gen_layer_weights", action="store_true")
    parser.add_argument("--learn_layer_weights", action="store_true")
    parser.add_argument("--n_shots", type=int, default=0, help="0 = exact statevector")
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument(
        "--freeze_circuit",
        action="store_true",
        help="Train only the decoder (PQC angles frozen).",
    )
    parser.add_argument(
        "--freeze_decoder",
        action="store_true",
        help="Train only the PQC (decoder weights frozen).",
    )
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--head_learning_rate", type=float, default=1e-3)
    parser.add_argument("--circuit_learning_rate", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--subsample_train", type=int, default=-1)
    parser.add_argument("--subsample_test", type=int, default=-1)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--max_seconds", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--count_params_only", action="store_true")
    parser.add_argument("--max_train_steps", type=int, default=-1)
    return parser


def theoretical_lora_m(n_layers=12, n_mod=2, rank=4, dim=768):
    return n_layers * n_mod * rank * (dim + dim)


def print_resource_table():
    m = theoretical_lora_m()
    rows = quada_resource_table(m, n_circuit_layers=8)
    print(f"Target LoRA entries m = {m} (Base encoder, r=4, Q/V)")
    print(
        f"{'n_mlp':>8} {'N':>4} {'circuit':>10} {'decoder':>10} {'total':>10} {'vs LoRA':>8}"
    )
    for row in rows:
        pct = 100.0 * row["total"] / m
        print(
            f"{row['n_mlp']:8d} {row['n_qubits']:4d} {row['circuit_params']:10d} "
            f"{row['decoder_params']:10d} {row['total']:10d} {pct:7.1f}%"
        )
    return rows


def split_val(train_loader, args, collate):
    train_ds = train_loader.dataset
    n_total = len(train_ds)
    n_val = max(1, int(n_total * args.val_ratio)) if args.val_ratio > 0 else 0
    n_train = max(1, n_total - n_val)
    if n_val <= 0 or n_train + n_val != n_total:
        return train_loader, None
    generator = torch.Generator().manual_seed(args.seed)
    train_sub, val_sub = torch.utils.data.random_split(
        train_ds, [n_train, n_val], generator=generator
    )
    train_loader = torch.utils.data.DataLoader(
        train_sub,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate,
        drop_last=True,
        **_dataloader_kwargs(args.num_workers),
    )
    val_loader = torch.utils.data.DataLoader(
        val_sub,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate,
        **_dataloader_kwargs(args.num_workers),
    )
    return train_loader, val_loader


def move_batch(batch, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


@torch.no_grad()
def evaluate(wrapper, loader, device):
    wrapper.eval()
    correct, total = 0, 0
    for batch in tqdm(loader, desc="eval", leave=False):
        batch = move_batch(batch, device)
        labels = batch.pop("labels")
        logits = wrapper(**batch).logits
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.numel()
    return correct / max(total, 1)


def load_backbone(args, spec, num_classes):
    model_name = args.model_name or spec["default_model"]
    if args.backbone == "ast":
        from transformers import ASTForAudioClassification

        model = ASTForAudioClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
    else:
        from transformers import AutoModelForAudioClassification

        model = AutoModelForAudioClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
            use_weighted_layer_sum=True,
        )
    return model, model_name


def head_param_names(backbone):
    if backbone == "ast":
        return ("classifier.",)
    return ("classifier.", "projector.")


def is_head_param(name, backbone):
    return name.startswith(head_param_names(backbone))


def maybe_detach_layer_weights(model, generate: bool):
    if not generate or not hasattr(model, "layer_weights"):
        return 0
    weight = model.layer_weights.detach()
    model._parameters.pop("layer_weights", None)
    model.layer_weights = weight
    return int(weight.numel())


def main():
    args = build_argparser().parse_args()
    if args.count_params_only:
        rows = print_resource_table()
        os.makedirs(args.output_dir, exist_ok=True)
        path = os.path.join(args.output_dir, "param_table.json")
        with open(path, "w") as handle:
            json.dump({"m": theoretical_lora_m(), "rows": rows}, handle, indent=2)
        print(f"Wrote {path}")
        return

    spec = BACKBONE_SPECS[args.backbone]
    if args.backbone in {"wavlm", "hubert"}:
        if args.method == "quada":
            args.gen_layer_weights = True
        else:
            args.learn_layer_weights = True
    setup_logging(args.output_dir)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.heads_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    run_name = f"{args.backbone}_{args.method}_{args.task}"
    if args.method == "quada":
        run_name += f"_mlp{args.n_mlp}_L{args.n_circuit_layers}_{args.index_mode}"
        if args.gen_layer_weights:
            run_name += "_genw"
    elif args.method in {"lora", "dora"}:
        run_name += f"_r{args.lora_rank}"

    library = ExpertLibrary.get_expert_library(repo_id=args.library_id, create=True)
    if args.skip_existing and run_name in library.keys():
        logger.info("==== Skipping `%s` (already in %s) ====", run_name, args.library_id)
        return

    logger.info("==== %s / %s / %s ====", args.method, args.backbone, args.task)
    if spec["input_kind"] == "spectrogram":
        model_name = args.model_name or spec["default_model"]
        train_loader, test_loader, num_classes, _ = get_audio_dataloaders(
            task_name=args.task,
            model_name=model_name,
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            subsample_train=args.subsample_train,
            subsample_test=args.subsample_test,
            seed=args.seed,
        )
        collate = collate_fn
    else:
        train_loader, test_loader, num_classes, _ = get_waveform_dataloaders(
            task_name=args.task,
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            subsample_train=args.subsample_train,
            subsample_test=args.subsample_test,
            seed=args.seed,
            max_seconds=args.max_seconds,
        )
        from wave_data import collate_waveforms

        collate = collate_waveforms

    train_loader, val_loader = split_val(train_loader, args, collate)
    logger.info("num_classes=%s, train_batches=%s", num_classes, len(train_loader))

    model, model_name = load_backbone(args, spec, num_classes)
    lora_config = None
    generator = None
    lora_items = []
    n_layer_weights = 0

    if args.method in {"lora", "dora", "quada"}:
        config_cls = DoRAConfig if args.method == "dora" else LoRAConfig
        lora_config = config_cls(
            modify_modules=spec["modify_modules"],
            modify_layers=spec["modify_layers"],
            lora_rank=args.lora_rank,
            lora_alpha=float(args.lora_rank),
            lora_dropout=args.lora_dropout,
        )
        modify_transformer(model, lora_config)
        lora_items = collect_lora_modules(model)
        if not lora_items:
            raise RuntimeError(
                f"No adapter modules matched {spec['modify_modules']} / "
                f"{spec['modify_layers']} on {model_name}. "
                f"First linear names: "
                + ", ".join(
                    n
                    for n, m in model.named_modules()
                    if isinstance(m, nn.Linear)
                )[:800]
            )
        logger.info("Patched %s adapter modules", len(lora_items))

    if args.learn_layer_weights or args.gen_layer_weights:
        if not hasattr(model, "layer_weights"):
            logger.warning(
                "Backbone has no layer_weights; weighted-sum generation disabled."
            )
            args.gen_layer_weights = False
            args.learn_layer_weights = False
        elif args.learn_layer_weights:
            model.layer_weights.requires_grad = True

    if args.method == "linear":
        for name, param in model.named_parameters():
            param.requires_grad = is_head_param(name, args.backbone) or (
                args.learn_layer_weights and name.endswith("layer_weights")
            )
    elif args.method == "lora":
        for name, param in model.named_parameters():
            param.requires_grad = (
                ".lora_" in name
                or is_head_param(name, args.backbone)
                or (args.learn_layer_weights and name.endswith("layer_weights"))
            )
    elif args.method == "dora":
        for name, param in model.named_parameters():
            param.requires_grad = (
                ".lora_" in name
                or name.endswith("magnitude")
                or is_head_param(name, args.backbone)
                or (args.learn_layer_weights and name.endswith("layer_weights"))
            )
    else:
        detach_lora_parameters(model)
        n_layer_weights = maybe_detach_layer_weights(model, args.gen_layer_weights)
        if args.gen_layer_weights and n_layer_weights == 0:
            n_layer_weights = 0
        generator = QuAdaGenerator(
            lora_items,
            QuAdaConfig(
                n_mlp=args.n_mlp,
                n_circuit_layers=args.n_circuit_layers,
                ansatz=args.ansatz,
                index_mode=args.index_mode,
                n_layer_weights=n_layer_weights,
                n_shots=args.n_shots or None,
                noise=args.noise,
            ),
        )
        for name, param in model.named_parameters():
            param.requires_grad = is_head_param(name, args.backbone) or (
                args.learn_layer_weights and name.endswith("layer_weights")
            )

    loras = [item[-1] for item in lora_items]
    wrapper = GeneratedAdapterModel(model, generator, loras)
    wrapper.to(device)

    head_params = [
        p for n, p in model.named_parameters() if p.requires_grad and is_head_param(n, args.backbone)
    ]
    extra_params = [
        p
        for n, p in model.named_parameters()
        if p.requires_grad and not is_head_param(n, args.backbone)
    ]
    param_groups = [{"params": head_params, "lr": args.head_learning_rate}]
    if extra_params:
        param_groups.append({"params": extra_params, "lr": args.learning_rate})
    if generator is not None:
        dec_params = list(generator.decoder.parameters())
        circ_params = list(generator.circuit.parameters())
        if args.freeze_decoder:
            for param in dec_params:
                param.requires_grad = False
        else:
            param_groups.append({"params": dec_params, "lr": args.learning_rate})
        if args.freeze_circuit:
            for param in circ_params:
                param.requires_grad = False
        else:
            param_groups.append(
                {"params": circ_params, "lr": args.circuit_learning_rate}
            )

    n_head = sum(p.numel() for p in head_params)
    n_extra = sum(p.numel() for p in extra_params)
    n_gen = (
        sum(p.numel() for p in generator.parameters() if p.requires_grad)
        if generator is not None
        else 0
    )
    adapter_params = n_extra + n_gen
    logger.info(
        "Trainable: adapter=%s head=%s (generator=%s)",
        f"{adapter_params:,}",
        f"{n_head:,}",
        f"{n_gen:,}",
    )
    if generator is not None:
        logger.info("QuAda resources: %s", generator.trainable_parameter_count())

    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    total_steps = max(len(train_loader) * args.num_train_epochs, 1)
    if args.max_train_steps > 0:
        total_steps = min(total_steps, args.max_train_steps)
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    loss_fn = nn.CrossEntropyLoss()

    best_val = -1.0
    best_state = None
    history = []
    global_step = 0
    t0 = time.time()
    stop = False
    for epoch in range(args.num_train_epochs):
        if stop:
            break
        wrapper.train()
        running = 0.0
        epoch_t0 = time.time()
        pbar = tqdm(train_loader, desc=f"train epoch {epoch}")
        for step, batch in enumerate(pbar):
            batch = move_batch(batch, device)
            labels = batch.pop("labels")
            logits = wrapper(**batch).logits
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            trainable = [p for p in wrapper.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            running += loss.item()
            global_step += 1
            pbar.set_postfix({"loss": f"{running / (step + 1):.4f}"})
            if args.max_train_steps > 0 and global_step >= args.max_train_steps:
                stop = True
                break
        val_acc = evaluate(wrapper, val_loader, device) if val_loader is not None else None
        epoch_rec = {
            "epoch": epoch,
            "train_loss": running / max(step + 1, 1),
            "val_acc": val_acc,
            "epoch_seconds": time.time() - epoch_t0,
        }
        history.append(epoch_rec)
        logger.info("[%s] epoch %s %s", args.task, epoch, epoch_rec)
        score = val_acc if val_acc is not None else -epoch_rec["train_loss"]
        if score >= best_val:
            best_val = score
            best_state = {
                k: v.detach().cpu().clone() for k, v in wrapper.state_dict().items()
            }

    train_seconds = time.time() - t0
    if best_state is not None:
        wrapper.load_state_dict(best_state, strict=False)
        wrapper.to(device)

    if generator is not None:
        generator.assign_to_loras(loras)

    acc = evaluate(wrapper, test_loader, device)
    logger.info("[%s] test accuracy: %.4f (train %.1fs)", args.task, acc, train_seconds)

    if args.method in {"lora", "dora", "quada"}:
        if generator is not None:
            generator.assign_to_loras(loras)
        expert_weights = materialize_lora_state_dict(model)
        expert_info = ExpertInfo(
            expert_name=run_name,
            expert_task_name=args.task,
            expert_config=lora_config,
            expert_model=model_name,
            training_config={
                "model": model_name,
                "method": args.method,
                "num_classes": num_classes,
                "individual_accuracy": acc,
                "lora_rank": args.lora_rank,
                "adapter_params": adapter_params,
            },
        )
        library.add_expert(
            Expert(expert_info=expert_info, expert_weights=expert_weights),
            expert_name=run_name,
            force=True,
        )

    metrics = {
        "task": args.task,
        "backbone": args.backbone,
        "method": args.method,
        "model_name": model_name,
        "test_accuracy": acc,
        "best_val": best_val if best_val >= 0 else None,
        "adapter_params": adapter_params,
        "head_params": n_head,
        "lora_rank": args.lora_rank,
        "n_mlp": args.n_mlp if args.method == "quada" else None,
        "n_circuit_layers": args.n_circuit_layers if args.method == "quada" else None,
        "index_mode": args.index_mode if args.method == "quada" else None,
        "gen_layer_weights": bool(args.gen_layer_weights),
        "freeze_circuit": bool(args.freeze_circuit),
        "freeze_decoder": bool(args.freeze_decoder),
        "ansatz": args.ansatz if args.method == "quada" else None,
        "n_shots": args.n_shots if args.method == "quada" else None,
        "noise": args.noise if args.method == "quada" else None,
        "quada": generator.trainable_parameter_count() if generator is not None else None,
        "train_seconds": train_seconds,
        "history": history,
        "num_classes": num_classes,
        "seed": args.seed,
        "run_name": run_name,
    }
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as handle:
        json.dump(metrics, handle, indent=2)
    logger.info("Wrote %s", metrics_path)


if __name__ == "__main__":
    main()
