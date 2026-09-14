"""QuAda: quantum-generated LoRA adapters.

A parameterized quantum circuit (hardware-efficient RY + CNOT ansatz) prepares
an N-qubit state. Its 2^N measurement probabilities, together with the
computational-basis bitstrings, are decoded by a shared MLP into chunks of
LoRA entries. Only the circuit angles and the decoder are trained; after
training the generated A, B matrices are materialized and inference is
entirely classical.

The data never enter the circuit. Exact state-vector simulation is the
default; finite-shot and depolarizing-noise models are available for the
hardware-robustness ablations in the paper.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

DECODER_HIDDEN = (32, 64, 128, 128, 64, 32)
MODULE_ORDER = {
    "query": 0,
    "q_proj": 0,
    "value": 1,
    "v_proj": 1,
    "key": 2,
    "k_proj": 2,
    "out_proj": 3,
    "dense": 3,
}

def _lora_cls():
    from mttl.models.modifiers.lora import LoRA

    return LoRA


LoraItem = Tuple[int, int, str, "LoRA"]


def n_qubits_for(m: int, n_mlp: int) -> int:
    """Qubit count of Eq. (qubits) in the paper, with a floor of 1."""
    n_ch = max(1, math.ceil(m / n_mlp))
    return max(1, math.ceil(math.log2(n_ch)))


def decoder_num_params(
    n_qubits: int,
    n_mlp: int,
    hidden: Sequence[int] = DECODER_HIDDEN,
    bias: bool = True,
) -> int:
    dims = [n_qubits + 1, *hidden, n_mlp]
    total = 0
    for in_dim, out_dim in zip(dims[:-1], dims[1:]):
        total += in_dim * out_dim
        if bias:
            total += out_dim
    return total


def circuit_num_params(n_qubits: int, n_layers: int, ansatz: str = "ry_cnot") -> int:
    if ansatz == "ry_rz_cnot":
        return 2 * n_qubits * n_layers
    return n_qubits * n_layers


def quada_resource_table(
    m: int,
    n_mlp_values: Sequence[int] = (256, 512, 1024, 2048, 4096),
    n_circuit_layers: int = 8,
    hidden: Sequence[int] = DECODER_HIDDEN,
    ansatz: str = "ry_cnot",
) -> List[Dict[str, int]]:
    rows = []
    for n_mlp in n_mlp_values:
        n_qubits = n_qubits_for(m, n_mlp)
        n_circ = circuit_num_params(n_qubits, n_circuit_layers, ansatz)
        n_dec = decoder_num_params(n_qubits, n_mlp, hidden)
        rows.append(
            {
                "n_mlp": n_mlp,
                "n_qubits": n_qubits,
                "n_chunks": math.ceil(m / n_mlp),
                "circuit_params": n_circ,
                "decoder_params": n_dec,
                "total": n_circ + n_dec,
            }
        )
    return rows


def parse_layer_index(name: str) -> int:
    match = re.search(r"layers?\.(\d+)", name)
    return int(match.group(1)) if match else 0


def parse_module_index(name: str) -> int:
    last = name.rsplit(".", 1)[-1]
    return MODULE_ORDER.get(last, 99)


def collect_lora_modules(model: nn.Module) -> List[LoraItem]:
    items: List[LoraItem] = []
    lora_cls = _lora_cls()
    for name, module in model.named_modules():
        if isinstance(module, lora_cls):
            items.append(
                (parse_layer_index(name), parse_module_index(name), name, module)
            )
    items.sort(key=lambda x: (x[0], x[1], x[2]))
    return items


def detach_lora_parameters(model: nn.Module) -> None:
    """Stop treating LoRA A/B as trainable Parameters.

    Generated tensors are assigned onto ``lora_a`` / ``lora_b`` each forward
    so gradients flow into the quantum generator rather than into stored
    adapter entries.
    """
    lora_cls = _lora_cls()
    for module in model.modules():
        if not isinstance(module, lora_cls):
            continue
        a = module.lora_a.detach()
        b = module.lora_b.detach()
        module._parameters.pop("lora_a", None)
        module._parameters.pop("lora_b", None)
        module.lora_a = a
        module.lora_b = b


def _binary_matrix(n_rows: int, n_bits: int, device, dtype) -> torch.Tensor:
    idx = torch.arange(n_rows, device=device)
    bits = torch.zeros(n_rows, n_bits, device=device, dtype=dtype)
    for bit in range(n_bits):
        shift = n_bits - 1 - bit
        bits[:, bit] = ((idx >> shift) & 1).to(dtype)
    return bits


def _apply_1q(state: torch.Tensor, matrix: torch.Tensor, qubit: int, n_qubits: int):
    shaped = state.reshape((2,) * n_qubits)
    out = torch.tensordot(matrix, shaped, dims=([1], [qubit]))
    perm = list(range(1, qubit + 1)) + [0] + list(range(qubit + 1, n_qubits))
    return out.permute(*perm).reshape(-1)


def apply_ry(state: torch.Tensor, theta: torch.Tensor, qubit: int, n_qubits: int):
    half = theta * 0.5
    c = torch.cos(half)
    s = torch.sin(half)
    z = torch.zeros([], dtype=state.dtype, device=state.device)
    # RY is real; embed into the state's complex dtype.
    c_c = c.to(state.dtype) + z
    s_c = s.to(state.dtype) + z
    row0 = torch.stack([c_c, -s_c])
    row1 = torch.stack([s_c, c_c])
    return _apply_1q(state, torch.stack([row0, row1]), qubit, n_qubits)


def apply_rx(state: torch.Tensor, theta: torch.Tensor, qubit: int, n_qubits: int):
    half = theta * 0.5
    c = torch.cos(half).to(state.dtype)
    s = torch.sin(half).to(state.dtype)
    minus_i_s = s * (-1j)
    row0 = torch.stack([c, minus_i_s])
    row1 = torch.stack([minus_i_s, c])
    return _apply_1q(state, torch.stack([row0, row1]), qubit, n_qubits)


def apply_rz(state: torch.Tensor, theta: torch.Tensor, qubit: int, n_qubits: int):
    half = theta * 0.5
    e_m = torch.exp(-1j * half)
    e_p = torch.exp(1j * half)
    z = torch.zeros([], dtype=state.dtype, device=state.device)
    row0 = torch.stack([e_m.to(state.dtype), z])
    row1 = torch.stack([z, e_p.to(state.dtype)])
    return _apply_1q(state, torch.stack([row0, row1]), qubit, n_qubits)


def cnot_permutation(n_qubits: int, control: int, target: int, device) -> torch.Tensor:
    dim = 1 << n_qubits
    ctrl_shift = n_qubits - 1 - control
    tgt_shift = n_qubits - 1 - target
    idx = torch.arange(dim, device=device)
    ctrl = (idx >> ctrl_shift) & 1
    return idx ^ (ctrl * (1 << tgt_shift))


class HardwareEfficientAnsatz(nn.Module):
    """L repetitions of single-qubit rotations followed by a linear CNOT chain."""

    def __init__(
        self,
        n_qubits: int,
        n_layers: int,
        ansatz: str = "ry_cnot",
    ):
        super().__init__()
        if n_qubits < 1:
            raise ValueError("n_qubits must be >= 1")
        if ansatz not in {"ry_cnot", "rx_cnot", "ry_rz_cnot"}:
            raise ValueError(f"Unknown ansatz '{ansatz}'")
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.ansatz = ansatz
        n_angles = circuit_num_params(n_qubits, n_layers, ansatz)
        self.theta = nn.Parameter(torch.empty(n_angles).uniform_(0.0, 2.0 * math.pi))
        cnot_idx = torch.stack(
            [
                cnot_permutation(n_qubits, q, q + 1, device="cpu")
                for q in range(max(n_qubits - 1, 0))
            ]
        ) if n_qubits > 1 else torch.zeros(0, 1 << n_qubits, dtype=torch.long)
        self.register_buffer("cnot_idx", cnot_idx, persistent=False)

    def forward(
        self,
        n_shots: Optional[int] = None,
        noise: float = 0.0,
    ) -> torch.Tensor:
        n = self.n_qubits
        dim = 1 << n
        state = torch.zeros(dim, dtype=torch.cfloat, device=self.theta.device)
        state = state.clone()
        state[0] = 1 + 0j
        cursor = 0
        for _ in range(self.n_layers):
            if self.ansatz == "ry_cnot":
                for q in range(n):
                    state = apply_ry(state, self.theta[cursor], q, n)
                    cursor += 1
            elif self.ansatz == "rx_cnot":
                for q in range(n):
                    state = apply_rx(state, self.theta[cursor], q, n)
                    cursor += 1
            else:
                for q in range(n):
                    state = apply_ry(state, self.theta[cursor], q, n)
                    cursor += 1
                for q in range(n):
                    state = apply_rz(state, self.theta[cursor], q, n)
                    cursor += 1
            for q in range(n - 1):
                state = state[self.cnot_idx[q]]
        probs = state.real.square() + state.imag.square()
        probs = probs / probs.sum().clamp_min(1e-12)
        if noise > 0:
            probs = (1.0 - noise) * probs + noise / dim
        if n_shots is not None and n_shots > 0:
            counts = torch.multinomial(probs, n_shots, replacement=True)
            freq = torch.bincount(counts, minlength=dim).to(probs.dtype) / float(n_shots)
            # Straight-through estimator: sampled frequencies in the forward
            # pass, exact probabilities in the backward pass.
            probs = freq + (probs - probs.detach())
        return probs


class MappingDecoder(nn.Module):
    """Shared MLP: (basis bits, probability) -> n_mlp adapter entries."""

    def __init__(
        self,
        n_qubits: int,
        n_mlp: int,
        hidden: Sequence[int] = DECODER_HIDDEN,
    ):
        super().__init__()
        dims = [n_qubits + 1, *hidden, n_mlp]
        layers: List[nn.Module] = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            linear = nn.Linear(in_dim, out_dim)
            is_last = i == len(dims) - 2
            if is_last:
                # Tiny nonzero init: full zeros block gradients (same as
                # LoRA with both A and B at 0). Keep the initial adapter
                # small so the frozen backbone is only weakly perturbed.
                nn.init.normal_(linear.weight, std=1e-3)
                nn.init.zeros_(linear.bias)
            layers.append(linear)
            if not is_last:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, bits: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([bits, probs], dim=-1))


@dataclass
class QuAdaConfig:
    n_mlp: int = 1024
    n_circuit_layers: int = 8
    ansatz: str = "ry_cnot"
    decoder_hidden: Tuple[int, ...] = DECODER_HIDDEN
    index_mode: str = "structured"  # structured | flat
    n_layer_weights: int = 0
    n_shots: Optional[int] = None
    noise: float = 0.0


class QuAdaGenerator(nn.Module):
    """Generate all LoRA A/B entries (and optional layer-weight logits)."""

    def __init__(
        self,
        lora_items: Sequence[LoraItem],
        config: Optional[QuAdaConfig] = None,
    ):
        super().__init__()
        self.config = config or QuAdaConfig()
        if not lora_items:
            raise ValueError("QuAdaGenerator requires at least one LoRA module.")
        self.lora_names = [name for _, _, name, _ in lora_items]
        self.layer_ids = [layer for layer, _, _, _ in lora_items]
        self.module_ids = [mod for _, mod, _, _ in lora_items]
        self.shapes: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
        self.module_sizes: List[int] = []
        for _, _, _, lora in lora_items:
            a_shape = (lora.in_features, lora.rank)
            b_shape = (lora.rank, lora.out_features)
            self.shapes.append((a_shape, b_shape))
            self.module_sizes.append(a_shape[0] * a_shape[1] + b_shape[0] * b_shape[1])

        self.n_modules = len(lora_items)
        self.n_layers = len(set(self.layer_ids))
        self.n_mod_types = len(set(self.module_ids))
        self.uniform_size = len(set(self.module_sizes)) == 1
        self.module_size = max(self.module_sizes)
        self.n_c = math.ceil(self.module_size / self.config.n_mlp)

        lora_m = sum(self.module_sizes)
        self.n_layer_weights = int(self.config.n_layer_weights)
        self.m = lora_m + self.n_layer_weights
        self.n_qubits = n_qubits_for(self.m, self.config.n_mlp)
        self.n_ch = math.ceil(self.m / self.config.n_mlp)
        self.n_basis = 1 << self.n_qubits

        # Structured: concat in (layer, module) order so layer identity sits
        # in the high bits of the basis index (Eq. indexmap). Flat: reverse
        # that order, destroying the layer/module bit fields.
        order = list(range(self.n_modules))
        if self.config.index_mode == "flat":
            order = list(reversed(order))
        self.concat_order = order

        self.circuit = HardwareEfficientAnsatz(
            self.n_qubits, self.config.n_circuit_layers, self.config.ansatz
        )
        self.decoder = MappingDecoder(
            self.n_qubits, self.config.n_mlp, self.config.decoder_hidden
        )
        bits = _binary_matrix(self.n_ch, self.n_qubits, device="cpu", dtype=torch.float32)
        self.register_buffer("basis_bits", bits, persistent=False)

    def trainable_parameter_count(self) -> Dict[str, int]:
        n_circ = sum(p.numel() for p in self.circuit.parameters())
        n_dec = sum(p.numel() for p in self.decoder.parameters())
        return {
            "n_qubits": self.n_qubits,
            "n_chunks": self.n_ch,
            "m": self.m,
            "circuit_params": n_circ,
            "decoder_params": n_dec,
            "total": n_circ + n_dec,
        }

    def generate_flat(self) -> torch.Tensor:
        probs = self.circuit(
            n_shots=self.config.n_shots, noise=self.config.noise
        )
        used = probs[: self.n_ch].unsqueeze(-1)
        chunks = self.decoder(self.basis_bits.to(dtype=used.dtype), used)
        return chunks.reshape(-1)

    def assign_to_loras(
        self, lora_modules: Sequence[nn.Module]
    ) -> Optional[torch.Tensor]:
        if len(lora_modules) != self.n_modules:
            raise ValueError(
                f"Expected {self.n_modules} LoRA modules, got {len(lora_modules)}"
            )
        flat = self.generate_flat()
        offset = 0
        for mod_i in self.concat_order:
            module = lora_modules[mod_i]
            a_shape, b_shape = self.shapes[mod_i]
            a_num = a_shape[0] * a_shape[1]
            b_num = b_shape[0] * b_shape[1]
            dtype = module.layer.weight.dtype
            module.lora_a = (
                flat[offset : offset + a_num].view(*a_shape).to(dtype)
            )
            module.lora_b = (
                flat[offset + a_num : offset + a_num + b_num]
                .view(*b_shape)
                .to(dtype)
            )
            offset += a_num + b_num
        if self.n_layer_weights <= 0:
            return None
        return flat[offset : offset + self.n_layer_weights]


def materialize_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Snapshot generated A/B (and DoRA magnitudes) as detached CPU tensors."""
    out = {}
    lora_cls = _lora_cls()
    for name, module in model.named_modules():
        if isinstance(module, lora_cls):
            out[f"{name}.lora_a"] = module.lora_a.detach().cpu().contiguous().clone()
            out[f"{name}.lora_b"] = module.lora_b.detach().cpu().contiguous().clone()
            mag = getattr(module, "magnitude", None)
            if mag is not None:
                out[f"{name}.magnitude"] = mag.detach().cpu().contiguous().clone()
    return out
