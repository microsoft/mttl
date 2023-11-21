import torch
from dataclasses import dataclass
from mttl.models.modifiers.base import Adapter, ModifyMixin
from mttl.models.modifiers.modify_model import register_modifier


@dataclass
class HardPromptConfig:
    max_input_length: int = 1024


@register_modifier("hard_prompt", config_cls=HardPromptConfig)
class HardPrompt(Adapter, ModifyMixin):
    def __init__(self, config, prompt_init=None, tokenizer=None):
        self.config = config
        self.prompt = prompt_init
        # dependency injection in the prompt
        self.tokenizer = tokenizer

    @classmethod
    def parallel_forward(cls, prompts, **kwargs):
        # assume left padding
        def roll_along(arr, shifts, dim):
            assert arr.ndim - 1 == shifts.ndim
            dim %= arr.ndim
            shape = (1,) * dim + (-1,) + (1,) * (arr.ndim - dim - 1)
            dim_indices = torch.arange(arr.shape[dim]).reshape(shape).to(arr.device)
            indices = (dim_indices - shifts.unsqueeze(dim)) % arr.shape[dim]
            return torch.gather(arr, dim, indices)

        padding_side = prompts[0].tokenizer.padding_side
        input_ids, attn_mask = (
            kwargs["input_ids"],
            kwargs["attention_mask"],
        )
        shifts = attn_mask.sum(1)

        # add a \n separator to be safe :-), doesn't mess w. gpt2 tokenizer!
        prompt_weights = [p.prompt + "\n" for p in prompts]
        eps = prompts[0].tokenizer(
            prompt_weights,
            return_tensors="pt",
            padding=True,
        )
        prompt_shifts = eps["attention_mask"].sum(1)

        if padding_side == "left":
            # if padding side is left then we move the padding to the right here, so that we can prepend the prompt safely
            input_ids = roll_along(input_ids, shifts, 1)
            attn_mask = roll_along(attn_mask, shifts, 1)

        if padding_side == "right":
            # if padding side of tokenizer is right, then we move the padding to the left here
            eps["input_ids"] = roll_along(eps["input_ids"], prompt_shifts, 1)
            eps["attention_mask"] = roll_along(eps["attention_mask"], prompt_shifts, 1)

        input_ids_with_prompts = torch.cat((eps["input_ids"], input_ids), dim=1)
        attn_mask_with_prompts = torch.cat((eps["attention_mask"], attn_mask), dim=1)

        # roll back
        reset_shifts = eps["attention_mask"].shape[1] - prompt_shifts
        if padding_side == "left":
            input_ids_with_prompts = roll_along(
                input_ids_with_prompts, -(attn_mask.shape[1] - shifts) - reset_shifts, 1
            )
            attn_mask_with_prompts = roll_along(
                attn_mask_with_prompts, -(attn_mask.shape[1] - shifts) - reset_shifts, 1
            )

            # now we know that all the padding is on the left, cut to max input length
            input_ids_with_prompts = input_ids_with_prompts[
                :, -prompts[0].config.max_input_length :
            ]
            attn_mask_with_prompts = attn_mask_with_prompts[
                :, -prompts[0].config.max_input_length :
            ]
        else:
            input_ids_with_prompts = roll_along(
                input_ids_with_prompts,
                -reset_shifts,
                1,
            )
            attn_mask_with_prompts = roll_along(
                attn_mask_with_prompts,
                -reset_shifts,
                1,
            )

            input_ids_with_prompts = input_ids_with_prompts[
                :, : prompts[0].config.max_input_length
            ]
            attn_mask_with_prompts = attn_mask_with_prompts[
                :, : prompts[0].config.max_input_length
            ]

        return input_ids_with_prompts, attn_mask_with_prompts

    def forward(self, batch, **kwargs):
        raise NotImplementedError("Use parallel_forward instead.")
