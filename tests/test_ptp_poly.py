import os
import torch
from pytorch_lightning import seed_everything
from transformers import AutoTokenizer
from mttl.config import Config
from pathlib import Path
from mttl.models.modifiers.poly import (
    PerTokenPolyLoRAConfig,
    PerTokenPolyLoRA,
    PerTokenPolytroponSelector,
)
from mttl.models.modifiers import modify_transformer
from mttl.models.modifiers.routing import RoutingInfo


def test_ptp_poly():
    seed_everything(0)

    # dummy batch
    bs, max_seq_len = 10, 100
    batch = {
        "input_ids": torch.randint(10, 400, (bs, max_seq_len)),
        "labels": torch.randint(10, 400, (bs, max_seq_len)),
    }
    seq_len = torch.randint(10, max_seq_len, (bs,))
    label_len = torch.randint(0, 10, (bs,))
    attn_mask = torch.zeros(bs, max_seq_len, dtype=torch.int32)
    attn_mask[torch.arange(bs), seq_len] = 1
    attn_mask = 1 - attn_mask.cumsum(dim=-1)
    batch["attention_mask"] = attn_mask

    # create a small Llama-like instance (not pretrained)
    # build llama config
    from transformers.models.llama.configuration_llama import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaForCausalLM

    small_config = LlamaConfig(
        vocab_size=400,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=5,
        num_attention_heads=8,
        max_position_embeddings=512,
    )
    model = LlamaForCausalLM(small_config).eval()
    model.info_container = {}

    # non perturbed input
    output = model(**batch)
    assert round(output.loss.item(), 4) == 6.0956

    # ** Per Token Poly adapter
    adapter_config = PerTokenPolyLoRAConfig(
        modify_modules=".*self_attn",
        modify_layers="q_proj|v_proj|k_proj|o_proj",
        # vocab_size=400,
        skip_unseen_tokens=True,
        model_family="gpt",
    )

    modify_transformer(model, adapter_config)
    model.info_container["routing_infos"] = RoutingInfo.from_batch(batch)
    model.eval()

    # loss at init. same, as (i) B is init to 0 and (ii) skipping unseen tokens
    output = model(**batch)
    assert round(output.loss.item(), 4) == 6.0956

    # Set Bs to nonzero values
    PerTokenPolytroponSelector.seen_samples_per_token.fill_(0.0)
    for name, module in model.named_modules():
        if isinstance(module, PerTokenPolyLoRA):
            module.lora_b.data.uniform_(0, 1)

    # loss at init. same, as (ii) skipping unseen tokens
    output = model(**batch)
    assert round(output.loss.item(), 4) == 6.0956

    PerTokenPolytroponSelector.seen_samples_per_token.fill_(0.0)
    model.train()
    # in training mode, we don't skip, loss should be different
    output = model(**batch)
    assert round(output.loss.item(), 4) == 6.1095
    model.eval()

    # Now, loss should be different
    PerTokenPolytroponSelector.seen_samples_per_token.fill_(1.0)
    output = model(**batch)
    assert round(output.loss.item(), 4) == 6.1095

    # Set Bs to nonzero values
    for _, module in model.named_modules():
        if isinstance(module, PerTokenPolyLoRA):
            module.lora_b.data.fill_(0.0)

    # Even if tokens are seen, still same loss.

    PerTokenPolytroponSelector.seen_samples_per_token.fill_(1.0)
    # loss at init. same, as (ii) Bs are 0s
    output = model(**batch)
    assert round(output.loss.item(), 4) == 6.0956
