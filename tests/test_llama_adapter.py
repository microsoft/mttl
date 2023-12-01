import os
import torch
import pytest
from pytorch_lightning import seed_everything
from mttl.models.modifiers.routing import RoutingInfo
from mttl.models.modifiers import modify_transformer
from mttl.models.modifiers.llama_adapter import (
    LlamaAdapter,
    LLamaAdapterConfig,
    PolyLLamaAdapterConfig,
)


@pytest.mark.parametrize("adapter_type", ["llama_adapter", "poly_llama_adapter"])
def test_llama_adapter(adapter_type):
    os.environ["CONFIG_PATH"] = "./"

    if adapter_type == "llama_adapter":
        adapter_config = LLamaAdapterConfig(n_tasks=768, soft_prompt_learn_kv=True)
    else:
        adapter_config = PolyLLamaAdapterConfig(
            n_tasks=768, n_splits=1, soft_prompt_learn_kv=True
        )

    seed_everything(0)

    # create a small Llama-like instance (not pretrained)
    # build llama config
    from transformers.models.llama.configuration_llama import LlamaConfig

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
    model.task_id_container = {}
    seed_everything(0)
    batch = {
        "input_ids": torch.randint(10, 400, (bs, max_seq_len)),
        "labels": torch.randint(10, 400, (bs, max_seq_len)),
    }
    seq_len = torch.randint(0, max_seq_len, (bs,))
    attn_mask = torch.zeros(bs, max_seq_len, dtype=torch.int32)
    attn_mask[torch.arange(bs), seq_len] = 1
    attn_mask = 1 - attn_mask.cumsum(dim=-1)
    task_ids = torch.randint(0, adapter_config.n_tasks, (bs,))
    batch["attention_mask"] = attn_mask

    model.task_id_container["routing_infos"] = RoutingInfo(task_ids=task_ids)

    # Test Base Llama model
    output = model(**batch)
    assert round(output.loss.item(), 4) == 6.0915

    # Test with llama adapter
    new_model = modify_transformer(model, adapter_config)

    new_model.task_id_container["routing_infos"] = RoutingInfo(task_ids=task_ids)

    if adapter_type == "llama_adapter":
        output = new_model(**batch)
        assert round(output.loss.item(), 4) == 6.0915

        # Manually set the gates to have non-zero values
        for module in new_model.modules():
            if isinstance(module, LlamaAdapter):
                module.adapter_gate.data.fill_(10.0)

        # Test Base Llama model
        output = new_model(**batch)
        assert round(output.loss.item(), 4) == 6.0815