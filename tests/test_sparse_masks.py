import math
import os

import pytest
from pytorch_lightning import seed_everything

from mttl.models.modifiers import modify_transformer
from mttl.models.modifiers.sparse_mask import (
    MaskedLinear,
    ScatteredSparseLinearModule,
    SNIPMaskUpdateWrapper,
    SparseLinearModule,
    SparseMaskConfig,
)


def test_sm_adapter_spops():
    os.environ["CONFIG_PATH"] = "./"

    seed_everything(0)
    from transformers.models.llama.configuration_llama import LlamaConfig

    adapter_config = SparseMaskConfig(
        modify_layers="gate_proj|down_proj|up_proj",
        sps_impl="sp_add+sp_mm",
        sps_type="regular_sparse",
        keep_ratio=0.05,
        mask_updater=None,
    )

    small_config = LlamaConfig(
        vocab_size=400,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=5,
        num_attention_heads=8,
        max_position_embeddings=512,
    )
    from transformers.models.llama.modeling_llama import LlamaForCausalLM

    model = LlamaForCausalLM(small_config)
    modify_transformer(model, adapter_config)

    modules = dict(model.named_modules())
    for n, p in model.named_parameters():
        if p.requires_grad:
            assert n.endswith(".sparse_weights")
            parent_name = ".".join(n.split(".")[:-1])
            parent_module = modules[parent_name]
            assert isinstance(parent_module, SparseLinearModule)
            assert p.numel() == int(0.05 * parent_module.base_weight.numel())

    # not test for forward as it requires GPU.


def test_block_sparse():
    os.environ["CONFIG_PATH"] = "./"

    seed_everything(0)
    from transformers.models.llama.configuration_llama import LlamaConfig

    adapter_config = SparseMaskConfig(
        modify_layers="gate_proj|down_proj|up_proj",
        sps_impl="sp_add+sp_mm",
        sps_type="block_sparse",
        keep_ratio=0.05,
        mask_updater=None,
        block_size=16,
    )

    small_config = LlamaConfig(
        vocab_size=400,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=5,
        num_attention_heads=8,
        max_position_embeddings=512,
    )
    from transformers.models.llama.modeling_llama import LlamaForCausalLM

    model = LlamaForCausalLM(small_config)
    modify_transformer(model, adapter_config)

    modules = dict(model.named_modules())
    for n, p in model.named_parameters():
        if p.requires_grad:
            assert n.endswith(".sparse_weights")
            parent_name = ".".join(n.split(".")[:-1])
            parent_module = modules[parent_name]
            assert isinstance(parent_module, SparseLinearModule)
            assert (
                p.numel()
                == math.ceil(int(0.05 * parent_module.base_weight.numel()) / 16**2)
                * 16**2
            )


def test_sm_adapter_scattered(dummy_batch):
    os.environ["CONFIG_PATH"] = "./"

    seed_everything(0)
    from transformers.models.llama.configuration_llama import LlamaConfig

    adapter_config = SparseMaskConfig(
        modify_layers="gate_proj|down_proj|up_proj",
        sps_impl="scattered",
        sps_type="regular_sparse",
        keep_ratio=0.05,
        mask_updater=None,
    )

    small_config = LlamaConfig(
        vocab_size=400,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=5,
        num_attention_heads=8,
        max_position_embeddings=512,
    )
    from transformers.models.llama.modeling_llama import LlamaForCausalLM

    model = LlamaForCausalLM(small_config)
    modify_transformer(model, adapter_config)

    modules = dict(model.named_modules())
    for n, p in model.named_parameters():
        if p.requires_grad:
            assert n.endswith(".sparse_weights")
            parent_name = ".".join(n.split(".")[:-1])
            parent_module = modules[parent_name]
            assert isinstance(parent_module, ScatteredSparseLinearModule)
            assert p.numel() == int(0.05 * parent_module.base_weight.numel())

    loss = model(**dummy_batch).loss
    assert pytest.approx(loss.item(), 0.1) == 5.6253


def test_sm_adapter_masked_linear(dummy_batch):
    os.environ["CONFIG_PATH"] = "./"

    seed_everything(0)
    from transformers.models.llama.configuration_llama import LlamaConfig

    adapter_config = SparseMaskConfig(
        modify_layers="gate_proj|down_proj|up_proj",
        sps_impl="masked_linear",
        sps_type="regular_sparse",
        keep_ratio=0.05,
        mask_updater=None,
    )

    small_config = LlamaConfig(
        vocab_size=400,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=5,
        num_attention_heads=8,
        max_position_embeddings=512,
    )
    from transformers.models.llama.modeling_llama import LlamaForCausalLM

    model = LlamaForCausalLM(small_config)
    modify_transformer(model, adapter_config)

    modules = dict(model.named_modules())
    for n, p in model.named_parameters():
        if p.requires_grad:
            assert n.endswith(".sparse_weights")
            parent_name = ".".join(n.split(".")[:-1])
            parent_module = modules[parent_name]
            assert isinstance(parent_module, MaskedLinear)
            assert p.numel() == parent_module.base_weight.numel()
            assert parent_module.binary_mask.sum() == int(
                0.05 * parent_module.base_weight.numel()
            )

    loss = model(**dummy_batch).loss
    assert (
        pytest.approx(loss.item(), 0.1) == 5.6253
    )  # same for all mask types, since sparse weights are innitialized to 0.0


def test_snip_updater(dummy_batch):
    os.environ["CONFIG_PATH"] = "./"

    seed_everything(0)
    from transformers.models.llama.configuration_llama import LlamaConfig

    adapter_config = SparseMaskConfig(
        modify_layers="gate_proj|down_proj|up_proj",
        sps_impl="scattered",
        sps_type="regular_sparse",
        keep_ratio=0.05,
        mask_updater="snip",
    )

    small_config = LlamaConfig(
        vocab_size=400,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=5,
        num_attention_heads=8,
        max_position_embeddings=512,
    )
    from transformers.models.llama.modeling_llama import LlamaForCausalLM

    model = LlamaForCausalLM(small_config)
    modify_transformer(model, adapter_config)

    loss = model(**dummy_batch).loss
    assert pytest.approx(loss.item(), 0.1) == 5.6253

    modules = dict(model.named_modules())
    for n, p in model.named_parameters():
        if p.requires_grad:
            assert n.endswith(".sparse_weights")
            parent_name = ".".join(n.split(".")[:-2])
            parent_module = modules[parent_name]
            assert isinstance(parent_module, SNIPMaskUpdateWrapper)
            assert parent_module._mask_update_steps == 1


if __name__ == "__main__":
    pytest.main([__file__])
