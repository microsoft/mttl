import os
import torch
import pytest
import torch.nn.functional as F
from pytorch_lightning import seed_everything
from mttl.models.modifiers.routing import RoutingInfo
from mttl.models.modifiers import modify_transformer
from mttl.models.modifiers.lora import LoRAConfig, LoRA


def test_lora_adapter():
    os.environ["CONFIG_PATH"] = "./"

    seed_everything(0)

    # create a small Llama-like instance (not pretrained)
    # build llama config
    from transformers.models.llama.configuration_llama import LlamaConfig

    adapter_config = LoRAConfig(modify_layers="gate_proj|down_proj|up_proj")
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

    bs, max_seq_len = 10, 100

    seed_everything(0)
    batch = {
        "input_ids": torch.randint(10, 400, (bs, max_seq_len)),
        "labels": torch.randint(10, 400, (bs, max_seq_len)),
        "attention_mask": torch.ones(bs, max_seq_len, dtype=torch.int32),
    }
    new_model = modify_transformer(model, adapter_config)

    # count
    lora_adapter_count = 0
    for n, p in new_model.named_parameters():
        lora_adapter_count += "lora_a" in n or "lora_b" in n
        if "lora_a" in n:
            assert p.shape[-1] == adapter_config.lora_rank
        if "lora_b" in n:
            assert p.shape[0] == adapter_config.lora_rank

    assert lora_adapter_count == 30
    loss = model(**batch).loss

    assert pytest.approx(loss.item(), 0.0001) == 6.0908
