import torch
from transformers import AutoModelForCausalLM

from mttl.datamodule.mt_seq_to_seq_module import (
    FlatMultiTaskConfig,
    FlatMultiTaskModule,
)
from mttl.models.expert_context import InfoContainer
from mttl.models.modifiers.routing import RoutingInfo


def test_packing_and_attn(tiny_flan_id):

    # Important to pick a model with only relative positional embeddings
    model_name = "EleutherAI/pythia-31m"
    common_kwargs = {
        "model": model_name,  # "EleutherAI/gpt-neo-125m",
        "train_batch_size": 4,
        "predict_batch_size": 4,
        "model_family": "gpt",
        "truncation_side": "left",
        "finetune_task_name": "cot_ecqa,stream_qed",
        "dataset": tiny_flan_id,
        "max_seq_per_pack": 4,
        "pack_sequences": False,
        "max_input_length": 1024,
    }
    config = FlatMultiTaskConfig(**common_kwargs)
    dm = FlatMultiTaskModule(config)

    ds = dm.train_dataset.select(range(100))
    collator = dm.collate_fn

    # manually do the packing steps
    tok_ds = dm.tokenize_dataset(ds)
    packed_ds = dm.pack_sequences(
        tok_ds, shuffle=False, max_sequences=config.max_seq_per_pack
    )

    assert len(packed_ds) < len(tok_ds)
    assert max([max(x) for x in packed_ds["seq_lens"]]) <= config.max_input_length
    assert max([len(x) for x in packed_ds["seq_lens"]]) <= config.max_seq_per_pack

    # extract the first packed sequence
    first_seq_len = len(packed_ds["seq_lens"][0])
    packed_ids = packed_ds["input_ids"][0]
    input_ids = tok_ds["input_ids"][:first_seq_len]

    assert len(packed_ids) == sum([len(x) for x in input_ids])

    # Check if the data is the one we expect (this can change if you change the model / tokenizer)
    assert sum([sum(x) for x in input_ids]) == sum(packed_ids) == 3348702

    packed_batch = collator([packed_ds[0]])
    input_batch = collator([ds[idx] for idx in range(first_seq_len)])

    # Check if the collated data is the one we expect
    flat_input_batch = input_batch["input_ids"].view(-1)
    flat_input_batch = flat_input_batch[flat_input_batch != dm.tokenizer.pad_token_id]

    if dm.tokenizer.pad_token_id == dm.tokenizer.eos_token_id:
        # remove the eos_token_id from packed_batch before doing a comparison
        packed_input_batch = packed_batch["input_ids"].view(-1)
        packed_input_batch = packed_input_batch[
            packed_input_batch != dm.tokenizer.eos_token_id
        ]

    # Check if collator is working correctly
    assert (flat_input_batch == packed_input_batch).all()

    # Check that sequence lengths are properly computed
    assert (
        packed_batch["seq_lens"].flatten()
        == input_batch["attention_mask"].sum(1).flatten()
    ).all()
    model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="sdpa")

    def _strip_model(model):
        model.eval()
        model.gpt_neox.layers = model.gpt_neox.layers[:1]
        model.gpt_neox.layers[0].mlp = torch.nn.Identity()
        model.embed_out = torch.nn.Identity()

    def _flatten(logits, attn_mask):
        logits = logits.flatten(0, 1)
        return logits[attn_mask.flatten() == 1]

    _strip_model(model)
    with InfoContainer(model, RoutingInfo.from_batch(packed_batch)) as context:
        packed_out = model(
            input_ids=packed_batch["input_ids"],
            attention_mask=packed_batch["attention_mask"],
        ).logits[0]

    with InfoContainer(model, RoutingInfo.from_batch(input_batch)):
        reg_out = model(
            input_ids=input_batch["input_ids"],
            attention_mask=input_batch["attention_mask"],
        ).logits
        reg_out = _flatten(reg_out, input_batch["attention_mask"])

    # remove monkey patching
    with InfoContainer(model, None):
        torch.nn.functional.scaled_dot_product_attention = (
            torch.nn.functional._default_scaled_dot_product_attention
        )
        rm_packed_out = model(
            input_ids=packed_batch["input_ids"],
            attention_mask=packed_batch["attention_mask"],
        ).logits[0]
        rm_reg_out = model(
            input_ids=input_batch["input_ids"],
            attention_mask=input_batch["attention_mask"],
        ).logits
        rm_reg_out = _flatten(rm_reg_out, input_batch["attention_mask"])

    # TEST 1 : With or without monkey patching, non packed sequences should give the same result
    assert torch.allclose(reg_out, rm_reg_out, atol=1e-5)

    # TEST 2 : With monkey patching, packed sequences should give the same result as without packing
    assert torch.allclose(reg_out, packed_out, atol=1e-4)  # Note : 1e-5 fails

    # TEST 3 : Without monkey patching, packed sequences should give different results than without packing
    assert not torch.allclose(reg_out, rm_packed_out, atol=1)
