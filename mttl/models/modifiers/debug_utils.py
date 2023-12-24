import torch
import torch.nn.functional as F
from copy import deepcopy as dc


def check_if_align(transformer, old_kw, new_kw, max_len, up_to=None):
    old_attn_mask, old_input_ids = old_kw["attention_mask"], old_kw["input_ids"]
    new_attn_mask, new_input_ids = new_kw["attention_mask"], new_kw["input_ids"]

    old_attn_mask = old_attn_mask[:, :max_len]
    old_input_ids = old_input_ids[:, :max_len]
    new_attn_mask = new_attn_mask[:, :max_len]
    new_input_ids = new_input_ids[:, :max_len]

    if up_to is not None:
        assert torch.all(old_attn_mask[:, :up_to] == new_attn_mask[:, :up_to])
        assert torch.all(old_input_ids[:, :up_to] == new_input_ids[:, :up_to])

    old_kw = {"attention_mask": old_attn_mask, "input_ids": old_input_ids}
    new_kw = {"attention_mask": new_attn_mask, "input_ids": new_input_ids}
    add_kw = {
        "attention_mask": torch.cat((old_attn_mask, old_attn_mask[:, [-1]]), dim=-1),
        "input_ids": torch.cat((old_input_ids, old_input_ids[:, [-1]]), dim=-1),
    }

    assert not transformer.training
    out_old = transformer(*(), **old_kw)
    out_new = transformer(*(), **new_kw)

    if up_to is not None:
        old_logits = out_old.logits[:, :up_to]
        new_logits = out_new.logits[:, :up_to]
    else:
        old_logits = out_old.logits
        new_logits = out_new.logits

    print(old_logits[:, :10].sum(-1))
    print(new_logits[:, :10].sum(-1))
    result = torch.allclose(old_logits, new_logits)
    return result


# Set up hooks to see at which point do the hidden reps start to diverge.
# What do I need ?
# 1) Ability to check at until a specific index if things align.
# 2) Ability to do this at multiple modules.
def build_tracking_fwd_hook(module, name, check_up_to, dump_every=2):
    if "resid_dropout" in name:
        return

    module.check_up_to = check_up_to
    module.name = name
    module.dump_every = dump_every
    module.previous_output = []
    module.previous_input = []

    def tracking_fwd_hook(module, input, output):
        # check if output is a tensor
        print("verifying module ", module.name)
        if len(module.previous_output) == 0:
            module.previous_output.append(dc(output))
            module.previous_input.append(dc(input))
        else:
            last_output = module.previous_output[-1]
            if not isinstance(last_output, torch.Tensor):
                # Most likely a tupple; let's look at the first instance.
                last_output, output = last_output[0], output[0]
                if not isinstance(last_output, torch.Tensor):
                    print("skipping module ", module.name)
                    return
            if last_output.ndim in [3, 5]:
                if torch.allclose(
                    last_output[:, : module.check_up_to],
                    output[:, : module.check_up_to],
                ):
                    print(f"mod {module.name} aligns")
                else:
                    print(f"mod {module.name} does not align")
                    breakpoint()
                    prev_input = module.previous_input[-1][0].float()
                    input = input[0].float()
                    assert torch.allclose(input[:, :10], prev_input[:, :10])
                    prev_out = F.linear(
                        prev_input, module.weight.float(), module.bias.float()
                    )
                    out = F.linear(input, module.weight.float(), module.bias.float())
                    xx = 1
            elif module.name.endswith("_id") or module.name.endswith("inner_attn"):
                if torch.allclose(
                    last_output[:, : module.check_up_to],
                    output[:, : module.check_up_to],
                ):
                    print(f"mod {module.name} aligns")
                else:
                    print("module does NOT align")
                    breakpoint()
                    xx = 1
            elif module.name.endswith("inner_attn.drop"):
                # check input
                prev_input = module.previous_input[-1][0].float()
                input = input[0].float()
                if torch.allclose(
                    prev_input[:, :, : module.check_up_to, : module.check_up_to],
                    input[:, :, : module.check_up_to, : module.check_up_to],
                ):
                    print(f"mod {module.name} aligns")
                else:
                    print("module does NOT align")
                    breakpoint()
                    xx = 1
                    breakpoint()
            else:
                print("skipping module ", module.name)
                breakpoint()
                xx = 1

        if len(module.previous_output) % module.dump_every == 0:
            module.previous_output = []
            module.previous_input = []

    module.register_forward_hook(tracking_fwd_hook)


def monitor_transformer(transformer):
    for name, module in transformer.named_modules():
        build_tracking_fwd_hook(module, name, check_up_to=10, dump_every=2)

    transformer.wrapped = True
