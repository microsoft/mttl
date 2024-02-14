import os
import torch
import pytest
import torch.nn.functional as F
from pytorch_lightning import seed_everything
from mttl.models.modifiers.routing import RoutingInfo
from mttl.models.modifiers import modify_transformer
from mttl.models.modifiers.lora import (
    LoRAConfig,
    LoRA,
    LoRAView,
    SkilledLoRAConfig,
    SkilledLoRAView,
    SkilledLoRA,
)


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


def test_skilled_lora_parallel_merge_with_weights():
    layer = torch.nn.Linear(1, 2, bias=False)
    layer.weight.requires_grad = False
    layer.weight.fill_(0.0)

    config = SkilledLoRAConfig(n_skills=2, n_splits=1, lora_alpha=1, lora_rank=3)

    lora_a = torch.randn(2, 1, 1, config.lora_rank)  # skills x splits x in_d x rank
    lora_b = torch.ones(2, config.lora_rank, 1, 2)  # skills x rank x splits x d_out
    ada1 = SkilledLoRAView(config, layer, lora_a, lora_b)

    lora_a = torch.randn(2, 1, 1, config.lora_rank)
    lora_b = torch.ones(2, config.lora_rank, 1, 2)
    ada2 = SkilledLoRAView(config, layer, lora_a, lora_b)

    # fill some dummy values
    ada1.lora_a[0, :].fill_(1.0)
    ada1.lora_a[1, :].fill_(2.0)
    ada2.lora_a[0, :].fill_(3.0)
    ada2.lora_a[1, :].fill_(4.0)

    ada1.lora_b[0, :].fill_(1.0)
    ada1.lora_b[1, :].fill_(2.0)

    input = torch.ones(2, 1)
    # the weights are interpolating the skills in ada1
    output = SkilledLoRA.parallel_linear_forward(
        input, [ada1], [torch.tensor([0.5, 0.5])]
    )
    assert output[0, 0].item() == 2.25
    assert output.shape == (2, 2)

    output = SkilledLoRA.parallel_linear_weighted_forward(
        input, [ada1], [torch.tensor([0.5, 0.5])], merge_after=True
    )
    assert output[0, 0].item() == 2.5
    assert output.shape == (2, 2)

    output = SkilledLoRA.parallel_linear_forward(
        input, [ada1, ada2], [torch.tensor([0.5, 0.5]), torch.tensor([0.0, 1.0])]
    )
    assert output[0, 0].item() == 2.25
    assert output[1, 0].item() == 4.0
    assert output.shape == (2, 2)

    output = SkilledLoRA.parallel_linear_weighted_forward(
        input,
        [ada1, ada2],
        [torch.tensor([0.5, 0.5]), torch.tensor([0.0, 1.0])],
        merge_after=True,
    )
    assert output[0, 0].item() == 2.5
    assert output[1, 0].item() == 4.0
    assert output.shape == (2, 2)

    input = torch.ones(2, 3, 1)
    # the weights are interpolating the skills in ada1
    output = SkilledLoRA.parallel_linear_forward(
        input, [ada1], [torch.tensor([0.5, 0.5])]
    )
    assert output[0, 0, 0].item() == 2.25
    assert output.shape == (2, 3, 2)

    output = SkilledLoRA.parallel_linear_weighted_forward(
        input, [ada1], [torch.tensor([0.5, 0.5])], merge_after=True
    )
    assert output[0, 0, 0].item() == 2.5
    assert output.shape == (2, 3, 2)

    output = SkilledLoRA.parallel_linear_forward(
        input, [ada1, ada2], [torch.tensor([0.5, 0.5]), torch.tensor([0.0, 1.0])]
    )
    assert output[0, 0, 0].item() == 2.25
    assert output[1, 0, 0].item() == 4.0
    assert output.shape == (2, 3, 2)

    output = SkilledLoRA.parallel_linear_weighted_forward(
        input,
        [ada1, ada2],
        [torch.tensor([0.5, 0.5]), torch.tensor([0.0, 1.0])],
        merge_after=True,
    )
    assert output[0, 0, 0].item() == 2.5
    assert output[1, 0, 0].item() == 4.0
    assert output.shape == (2, 3, 2)

    # one skilled lora with weights tied across batch
    input = torch.ones(2, 1)
    output = SkilledLoRA.parallel_linear_forward(
        input, [ada1], [torch.tensor([0.5, 0.5]), torch.tensor([0.0, 1.0])]
    )
    assert output[0, 0].item() == 2.25
    assert output[1, 0].item() == 4.0
    assert output.shape == (2, 2)

    input = torch.ones(2, 1)
    output = SkilledLoRA.parallel_linear_weighted_forward(
        input,
        [ada1],
        [torch.tensor([0.5, 0.5]), torch.tensor([0.0, 1.0])],
        merge_after=True,
    )
    assert output[0, 0].item() == 2.5
    assert output[1, 0].item() == 4.0
    assert output.shape == (2, 2)

    lora_a = torch.randn(1, config.lora_rank)
    lora_b = torch.ones(config.lora_rank, 2)
    l1 = LoRAView(config, layer, lora_a, lora_b)

    lora_a = torch.randn(1, config.lora_rank)
    lora_b = torch.ones(config.lora_rank, 2)
    l2 = LoRAView(config, layer, lora_a, lora_b)

    input = torch.ones(2, 3, 1)
    output = LoRA.parallel_linear_forward(input, [l1, l2])

    weights = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    output_sk = SkilledLoRAView.parallel_linear_weighted_forward(
        input, [SkilledLoRAView.from_loras([l1, l2])], weights
    )
    assert torch.allclose(output, output_sk)


def test_skilled_lora_view():
    layer = torch.nn.Linear(1, 2, bias=False)
    adapter_config = LoRAConfig(lora_rank=5)
    lora1 = LoRA(adapter_config, layer)
    lora2 = LoRA(adapter_config, layer)
    lora3 = LoRA(adapter_config, layer)
    skilled_lora = SkilledLoRAView.from_loras([lora1, lora2, lora3])
    assert skilled_lora.lora_a.shape == torch.Size([3, 1, 1, 5])
    assert skilled_lora.lora_b.shape == torch.Size([3, 5, 1, 2])
    assert skilled_lora.rank == 5
    assert skilled_lora.alpha == adapter_config.lora_alpha


if __name__ == "__main__":
    pytest.main([__file__])
