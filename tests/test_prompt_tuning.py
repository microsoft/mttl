import os
import sys
import torch
import torch.nn.functional as F
from pytorch_lightning import seed_everything
from transformers import AutoTokenizer
from projects.instr_routing.config import RoutingConfig
from mttl.models.modifiers.routing import RoutingInfo
from mttl.models.modifiers import modify_transformer
from mttl.models.modifiers.prompt_tuning import PromptTuning

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_llama_adapter():
    os.environ["CONFIG_PATH"] = "./"

    _args = RoutingConfig(
        "projects/instr_routing/configs/platypus/llama2_7b.json"
        + "+projects/instr_routing/configs/t0_/llama_adapter.json",
    )
    _args.n_tasks = 768
    _args.warmup_steps = 0
    _args.learning_rate = 1e-3

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
    task_ids = torch.randint(0, _args.n_tasks, (bs,))
    batch["attention_mask"] = attn_mask

    model.task_id_container["routing_infos"] = RoutingInfo(task_ids=task_ids)

    # Test Base Llama model
    output = model(**batch)
    assert round(output.loss.item(), 4) == 6.0915

    # Test with llama adapter
    _args.model_modifier = "prompt_tuning"
    new_model = modify_transformer(model, _args)

    new_model.task_id_container["routing_infos"] = RoutingInfo(task_ids=task_ids)

    # Test Base Llama model
    labels = batch.pop("labels")
    output = new_model(**batch)

    def masked_cross_entropy(logits, labels, mask):
        bs, ds, vocab_size = logits.size()
        ce = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction="none")
        ce = ce.view_as(labels)
        bs_loss = (ce * attn_mask).sum(1) / attn_mask.sum(1)
        return bs_loss.mean()

    # trim the logits to remove the soft prompt
    loss = masked_cross_entropy(output.logits, labels, batch["attention_mask"])
    assert round(loss.item(), 4) == 6.0812

    # Manually set the soft prompt embeddings to high values to induce a change
    for module in new_model.modules():
        if isinstance(module, PromptTuning):
            module.prompt_embedding.weight.data.fill_(0)

    # Test Base Llama model
    output = new_model(**batch)
    loss = masked_cross_entropy(output.logits, labels, batch["attention_mask"])
    assert round(loss.item(), 4) == 6.0662


if __name__ == "__main__":
    test_llama_adapter()
