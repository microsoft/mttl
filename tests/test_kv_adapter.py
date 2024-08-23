import os

import pytest
import torch
from pytorch_lightning import seed_everything

from mttl.models.modifiers import modify_transformer
from mttl.models.modifiers.kv_adapter import KVAdapter, KVAdapterConfig
from mttl.models.modifiers.routing import RoutingInfo


@pytest.mark.parametrize("adapter_type", ["kv_adapter"])
@pytest.mark.parametrize("model_arg", ["llama", "gpt-neo"])
def test_llama_adapter(adapter_type, model_arg):
    os.environ["CONFIG_PATH"] = "./"

    if adapter_type == "kv_adapter":
        adapter_config = KVAdapterConfig(n_tasks=768, soft_prompt_learn_kv=True)

    adapter_config.model = model_arg
    seed_everything(0)

    # create a small Llama-like instance (not pretrained)
    # build llama config
    adapter_config.modify_modules = ".*"
    if model_arg == "llama":
        from transformers.models.llama.configuration_llama import LlamaConfig
        from transformers.models.llama.modeling_llama import LlamaForCausalLM

        adapter_config.modify_layers = ".*attn.*"

        small_config = LlamaConfig(
            vocab_size=400,
            hidden_size=512,
            intermediate_size=1024,
            num_hidden_layers=5,
            num_attention_heads=8,
            max_position_embeddings=512,
        )

        model = LlamaForCausalLM(small_config)
    elif model_arg == "gpt-neo":
        from transformers.models.gpt_neo.configuration_gpt_neo import GPTNeoConfig
        from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoForCausalLM

        adapter_config.modify_layers = ".*attention.*"

        small_config = GPTNeoConfig(
            vocab_size=400,
            hidden_size=512,
            intermediate_size=1024,
            num_layers=6,
            max_position_embeddings=512,
            num_heads=2,
            attention_types=[[["global", "local"], 3]],
        )

        model = GPTNeoForCausalLM(small_config)

    bs, max_seq_len = 10, 100

    seed_everything(0)
    batch = {
        "input_ids": torch.randint(10, 400, (bs, max_seq_len)),
        "labels": torch.randint(10, 400, (bs, max_seq_len)),
    }
    seq_len = torch.randint(0, max_seq_len, (bs,))
    attn_mask = torch.zeros(bs, max_seq_len, dtype=torch.int32)
    attn_mask[torch.arange(bs), seq_len] = 1
    attn_mask = 1 - attn_mask.cumsum(dim=-1)

    # TODO: rewrite to use taks names instead
    task_ids = torch.randint(0, adapter_config.n_tasks, (bs,))
    batch["attention_mask"] = attn_mask

    # Test with llama adapter
    new_model = modify_transformer(model, adapter_config)

    print("model : ", model_arg)
    if model_arg == "llama":
        # Test Base Llama model
        output = model(**batch)
        assert round(output.loss.item(), 4) == 6.0915

        # Make sure zero-init gate gives the same result
        output = new_model(**batch)
        assert round(output.loss.item(), 4) == 6.0915

        # Manually set the gates to have non-zero values
        n_modules = 0
        for module in new_model.modules():
            if isinstance(module, KVAdapter):
                module.adapter_gate.data.fill_(10.0)
                n_modules += 1

        # Check if the right amount of layers have been patched
        assert n_modules == small_config.num_hidden_layers

        # Test Modified Llama model
        output = new_model(**batch)
        assert round(output.loss.item(), 4) == 6.08

    if model_arg == "gpt-neo":
        # Test Base GPT neo model
        output = model(**batch)
        assert round(output.loss.item(), 4) == 6.12

        # Make sure zero-init gate gives the same result
        output = new_model(**batch)
        assert round(output.loss.item(), 4) == 6.12

        # Manually set the gates to have non-zero values
        n_modules = 0
        for module in new_model.modules():
            if isinstance(module, KVAdapter):
                module.adapter_gate.data.fill_(10.0)
                n_modules += 1

        # Check if the right amount of layers have been patched
        assert n_modules == len(small_config.attention_layers)

        # Test Modified Neo model
        output = new_model(**batch)
        assert round(output.loss.item(), 4) == 6.10
