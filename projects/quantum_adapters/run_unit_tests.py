"""Run QuAda unit tests without pytest (cluster env may not have it)."""

from __future__ import annotations

import math
import sys
import traceback

import torch
from torch import nn

from mttl.models.modifiers.lora import LoRAConfig
from mttl.models.modifiers.quada import (
    DECODER_HIDDEN,
    HardwareEfficientAnsatz,
    QuAdaConfig,
    QuAdaGenerator,
    collect_lora_modules,
    decoder_num_params,
    detach_lora_parameters,
    materialize_lora_state_dict,
    n_qubits_for,
    quada_resource_table,
)


def _approx(a, b, abs_tol=1e-5):
    assert math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=abs_tol), (a, b)


def test_n_qubits_matches_paper_formula():
    m = 147456
    assert n_qubits_for(m, 256) == 10
    assert n_qubits_for(m, 512) == 9
    assert n_qubits_for(m, 1024) == 8
    assert n_qubits_for(m, 2048) == 7
    assert n_qubits_for(m, 4096) == 6


def test_circuit_probabilities_normalized_and_real_for_ry():
    torch.manual_seed(0)
    circuit = HardwareEfficientAnsatz(n_qubits=4, n_layers=3, ansatz="ry_cnot")
    probs = circuit()
    assert probs.shape == (16,)
    assert torch.isfinite(probs).all()
    _approx(probs.sum().item(), 1.0)
    assert (probs >= -1e-6).all()


def test_cnot_flips_target_when_control_is_one():
    circuit = HardwareEfficientAnsatz(n_qubits=2, n_layers=1, ansatz="ry_cnot")
    with torch.no_grad():
        circuit.theta.zero_()
    probs = circuit().detach()
    assert probs.argmax().item() == 0

    with torch.no_grad():
        circuit.theta.zero_()
        circuit.theta[0] = math.pi
        circuit.theta[1] = 0.0
    probs = circuit().detach()
    assert probs.argmax().item() == 3
    _approx(probs[3].item(), 1.0, abs_tol=1e-4)


def test_circuit_gradients_flow():
    torch.manual_seed(1)
    circuit = HardwareEfficientAnsatz(n_qubits=3, n_layers=2, ansatz="ry_cnot")
    probs = circuit()
    loss = (probs * torch.arange(8, dtype=probs.dtype)).sum()
    loss.backward()
    assert circuit.theta.grad is not None
    assert torch.isfinite(circuit.theta.grad).all()
    assert circuit.theta.grad.abs().sum().item() > 0


def test_resource_table_decoder_dominates():
    rows = quada_resource_table(m=147456, n_circuit_layers=8)
    by_mlp = {row["n_mlp"]: row for row in rows}
    assert by_mlp[256]["n_qubits"] == 10
    assert by_mlp[1024]["n_qubits"] == 8
    for row in rows:
        assert row["decoder_params"] > row["circuit_params"]
        assert row["total"] == row["circuit_params"] + row["decoder_params"]
        assert row["circuit_params"] == row["n_qubits"] * 8


def test_decoder_param_count_matches_linear_stack():
    n_qubits, n_mlp = 8, 1024
    expected = decoder_num_params(n_qubits, n_mlp, DECODER_HIDDEN)
    dims = [n_qubits + 1, *DECODER_HIDDEN, n_mlp]
    counted = 0
    for a, b in zip(dims[:-1], dims[1:]):
        counted += a * b + b
    assert expected == counted


def _patched_mlp(n_layers=2, hidden=16, rank=2):
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = nn.Module()
            self.attention.query = nn.Linear(hidden, hidden, bias=False)
            self.attention.value = nn.Linear(hidden, hidden, bias=False)

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Module()
            self.encoder.layer = nn.ModuleList([Block() for _ in range(n_layers)])

    model = Tiny()
    config = LoRAConfig(
        modify_modules=r"encoder\.layer\.\d+\.attention",
        modify_layers="query|value",
        lora_rank=rank,
        lora_alpha=float(rank),
    )
    from mttl.models.modifiers.modify_model import modify_transformer

    modify_transformer(model, config)
    return model


def test_generator_assigns_matching_shapes_and_trains():
    torch.manual_seed(0)
    model = _patched_mlp()
    items = collect_lora_modules(model)
    assert len(items) == 4
    detach_lora_parameters(model)
    generator = QuAdaGenerator(
        items,
        QuAdaConfig(n_mlp=32, n_circuit_layers=2, n_layer_weights=2),
    )
    loras = [item[-1] for item in items]
    layer_logits = generator.assign_to_loras(loras)
    assert layer_logits is not None
    assert tuple(layer_logits.shape) == (2,)
    for lora in loras:
        assert tuple(lora.lora_a.shape) == (16, 2)
        assert tuple(lora.lora_b.shape) == (2, 16)
        assert lora.lora_a.requires_grad

    x = torch.randn(3, 16)
    out = model.encoder.layer[0].attention.query(x)
    loss = out.square().mean() + layer_logits.square().mean()
    loss.backward()
    grads = [p.grad for p in generator.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)
    assert sum(g.abs().sum().item() for g in grads) > 0


def test_materialized_adapters_are_classical_and_finite():
    model = _patched_mlp(n_layers=1, hidden=8, rank=1)
    items = collect_lora_modules(model)
    detach_lora_parameters(model)
    generator = QuAdaGenerator(items, QuAdaConfig(n_mlp=16, n_circuit_layers=1))
    generator.assign_to_loras([item[-1] for item in items])
    state = materialize_lora_state_dict(model)
    assert state
    for tensor in state.values():
        assert tensor.device.type == "cpu"
        assert torch.isfinite(tensor).all()


def test_finite_shots_still_normalized():
    torch.manual_seed(0)
    circuit = HardwareEfficientAnsatz(n_qubits=3, n_layers=2)
    probs = circuit(n_shots=200)
    _approx(probs.sum().item(), 1.0)


def main():
    tests = [
        test_n_qubits_matches_paper_formula,
        test_circuit_probabilities_normalized_and_real_for_ry,
        test_cnot_flips_target_when_control_is_one,
        test_circuit_gradients_flow,
        test_resource_table_decoder_dominates,
        test_decoder_param_count_matches_linear_stack,
        test_generator_assigns_matching_shapes_and_trains,
        test_materialized_adapters_are_classical_and_finite,
        test_finite_shots_still_normalized,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL  {fn.__name__}")
            traceback.print_exc()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
