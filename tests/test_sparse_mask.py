import math
import os
import sys

import pytest
import torch
import torch.nn as nn
from pytorch_lightning import seed_everything

from mttl.models.modifiers import modify_transformer
from mttl.models.modifiers.sparse_mask import SparseMaskAdapter, SparseMaskConfig


def test_sm_adapter():
    os.environ["CONFIG_PATH"] = "./"

    seed_everything(0)
    from transformers.models.llama.configuration_llama import LlamaConfig

    adapter_config = SparseMaskConfig(
        modify_layers="gate_proj|down_proj|up_proj",
        sparse_cat="regular_sparse",
        keep_ratio=0.05,
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

    sparse_module_count = 0
    modules = dict(model.named_modules())
    for n, p in model.named_parameters():
        if p.requires_grad:
            sparse_module_count += 1
            print("in")
            print(n)
            assert n.endswith(".sparse_layer.weight") or n.endswith(
                ".sparse_layer.bias"
            )
            parent_name = ".".join(n.split(".")[:-1])
            parent_module = modules[parent_name]

    assert sparse_module_count == 30
