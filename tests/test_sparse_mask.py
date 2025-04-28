import itertools
import math
import os
import sys

import pytest
import torch
import torch.nn as nn
from pytorch_lightning import seed_everything

from mttl.models.modifiers import modify_transformer
from mttl.models.modifiers.sparse_mask import (
    SparseMaskAdapter,
    SparseMaskConfig,
    make_sparse_model_during_training,
)


def test_sm_adapter():
    os.environ["CONFIG_PATH"] = "./"

    seed_everything(0)
    from transformers.models.llama.configuration_llama import LlamaConfig

    keep_ratio_list = [0.05, 0.1]
    sparse_cat_list = ["regular_sparse", "block_sparse"]
    mask_cat_list = ["scatter", None]
    parameter_selection_procedure_list = ["max_connection_sensitivity", "model"]

    for (
        sparse_cat,
        mask_cat,
        keep_ratio,
        parameter_selection_procedure,
    ) in itertools.product(
        sparse_cat_list,
        mask_cat_list,
        keep_ratio_list,
        parameter_selection_procedure_list,
    ):

        if parameter_selection_procedure == "model" and sparse_cat == "block_sparse":
            continue  # skip this combination

        adapter_config = SparseMaskConfig(
            modify_layers="gate_proj|down_proj|up_proj",
            sparse_cat=sparse_cat,
            keep_ratio=keep_ratio,
            mask_cat=mask_cat,
        )
        print(
            f"Testing with sparse_cat={sparse_cat}, mask_cat={mask_cat},"
            f"keep_ratio={keep_ratio}, parameter_selection_procedure={parameter_selection_procedure}"
        )

        adapter_config = SparseMaskConfig(
            modify_layers="gate_proj|down_proj|up_proj",
            sparse_cat=sparse_cat,
            keep_ratio=keep_ratio,
            mask_cat=mask_cat,
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

        # check that the correct modules have been modified
        sparse_module_count = 0
        weight_mask_count = 0
        modules = dict(model.named_modules())
        for n, p in model.named_parameters():
            if p.requires_grad:
                sparse_module_count += 1
                assert n.endswith(".sparse_layer.weight") or n.endswith(
                    ".sparse_layer.bias"
                )
                parent_name = ".".join(n.split(".")[:-1])
                parent_module = modules[parent_name]

            if n.endswith(".sparse_layer.weight_mask"):
                weight_mask_count += 1
        assert sparse_module_count == 30
        assert weight_mask_count == 15

        bs, max_seq_len = 10, 100
        batch = {
            "input_ids": torch.randint(10, 400, (bs, max_seq_len)),
            "labels": torch.randint(10, 400, (bs, max_seq_len)),
            "attention_mask": torch.ones(bs, max_seq_len, dtype=torch.int32),
        }

        # Perform a few optimizer steps, to initialize the sparse_layer weights
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        for _ in range(3):
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

        # checks that sparse mask has not been updated yet
        for m in model.modules():
            if isinstance(m, SparseMaskAdapter):
                assert torch.equal(
                    m.sparse_layer.weight_mask,
                    torch.ones_like(m.sparse_layer.weight_mask),
                )

        # calculates loss. calcs gradients for weight_mask and updates mask.
        make_sparse_model_during_training(
            model, batch, parameter_selection_procedure=parameter_selection_procedure
        )

        new_outputs = model(**batch)

        # assert mask is being applied
        assert outputs != new_outputs

        # assert sparsification works as intended
        if parameter_selection_procedure == "max_connection_sensitivity":
            # assert each layer is properly sparse
            for m in model.modules():
                if isinstance(m, SparseMaskAdapter):
                    expected_non_zero_in_mask = int(m.param_num * keep_ratio)
                    # if block sparse then need to slightly adjust expected non zero in mask
                    if sparse_cat == "block_sparse":
                        expected_blocks = int(
                            expected_non_zero_in_mask / m.BLOCK_SIZE**2
                        )
                        expected_non_zero_in_mask = expected_blocks * m.BLOCK_SIZE**2

                    actual_non_zero_in_mask = m.sparse_layer.weight_mask.sum()
                    assert actual_non_zero_in_mask == expected_non_zero_in_mask

        elif parameter_selection_procedure == "model":
            # assert model is approppriately sparse in aggregate
            expected_non_zero_in_mask = 0
            actual_non_zero_in_mask = 0
            for m in model.modules():
                if isinstance(m, SparseMaskAdapter):
                    expected_non_zero_in_mask += int(m.param_num * keep_ratio)
                    actual_non_zero_in_mask += m.sparse_layer.weight_mask.sum()

            assert expected_non_zero_in_mask == actual_non_zero_in_mask


def test_load_expert_from_checkpoint():
    from tempfile import TemporaryDirectory

    from mttl.models.expert_model import ExpertModel, ExpertModelConfig
    from mttl.models.library.expert_library import LocalExpertLibrary

    temp_dir = TemporaryDirectory()
    destination = temp_dir.name

    adapter_config = SparseMaskConfig(
        modify_layers="k_proj",
        sparse_cat="regular_sparse",
        keep_ratio=0.05,
        mask_cat="max_connection_sensitivity",
    )

    model = ExpertModel(
        ExpertModelConfig(
            "EleutherAI/gpt-neo-125m",
            expert_name="a",
            modifier_config=adapter_config,
        )
    )

    bs, max_seq_len = 10, 100
    batch = {
        "input_ids": torch.randint(10, 400, (bs, max_seq_len)),
        "labels": torch.randint(10, 400, (bs, max_seq_len)),
        "attention_mask": torch.ones(bs, max_seq_len, dtype=torch.int32),
    }

    # Perform a few optimizer steps, to initialize the sparse_layer weights
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(1):
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    outputs = model(**batch)

    model.save_pretrained(destination)
    library = LocalExpertLibrary(destination)

    reloaded = ExpertModel.from_pretrained(destination)
    reloaded_output = reloaded(**batch)

    # assert reloaded model is the same as original
    assert reloaded_output.loss.item() == outputs.loss.item() and torch.equal(
        reloaded_output.logits, outputs.logits
    )

    # calculates loss. calcs gradients for weight_mask and updates mask.
    make_sparse_model_during_training(
        model, batch, parameter_selection_procedure="max_connection_sensitivity"
    )

    new_outputs = model(**batch)

    model.save_pretrained(destination)
    reloaded = ExpertModel.from_pretrained(destination)

    reloaded_output = reloaded(**batch)
    # assert reloaded model is the same as original, even after sparse_mask update
    assert reloaded_output.loss.item() == new_outputs.loss.item() and torch.equal(
        reloaded_output.logits, new_outputs.logits
    )

    # assert mask is being applied
    assert outputs != new_outputs

    temp_dir.cleanup()
