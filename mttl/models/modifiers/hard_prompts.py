from dataclasses import dataclass

import torch

from mttl.models.modifiers.base import Modifier


@dataclass
class HardPromptConfig:
    max_input_length: int = None
    tokenizer: str = None
    model_family: str = None


@Modifier.register("hard_prompt", config_cls=HardPromptConfig)
class HardPrompt(Modifier):
    def __init__(self, config, prompt_init=None):
        if config.model_family is None or config.tokenizer is None:
            raise ValueError(
                "Please provide a model_family and tokenizer to HardPromptConfig."
            )

        self.config = config
        self.prompt = prompt_init
        self.model_family = config.model_family
        self.tokenizer = config.tokenizer

    @classmethod
    def parallel_forward(
        cls, prompts, input_ids, attention_mask, labels=None, **kwargs
    ):
        # assume left padding
        def roll_along(arr, shifts, dim):
            assert arr.ndim - 1 == shifts.ndim
            dim %= arr.ndim
            shape = (1,) * dim + (-1,) + (1,) * (arr.ndim - dim - 1)
            dim_indices = torch.arange(arr.shape[dim]).reshape(shape).to(arr.device)
            indices = (dim_indices - shifts.unsqueeze(dim)) % arr.shape[dim]
            return torch.gather(arr, dim, indices)

        padding_side = prompts[0].tokenizer.padding_side
        shifts = attention_mask.sum(1)

        # add a \n separator to be safe :-), doesn't mess w. gpt2 tokenizer!
        prompt_weights = [p.prompt + "\n" for p in prompts]
        eps = prompts[0].tokenizer(
            prompt_weights,
            return_tensors="pt",
            padding=True,
        )
        # move it to GPU
        eps["input_ids"] = eps["input_ids"].to(input_ids.device)
        eps["attention_mask"] = eps["attention_mask"].to(input_ids.device)
        #
        prompt_shifts = eps["attention_mask"].sum(1)
        modify_labels = labels is not None and prompts[0].model_family == "gpt"

        if padding_side == "left":
            # if padding side is left then we move the padding to the right here, so that we can prepend the prompt safely
            input_ids = roll_along(input_ids, shifts, 1)
            attention_mask = roll_along(attention_mask, shifts, 1)
            if modify_labels:
                labels = roll_along(labels, shifts, 1)

        if padding_side == "right":
            # if padding side of tokenizer is right, then we move the padding to the left here
            eps["input_ids"] = roll_along(eps["input_ids"], prompt_shifts, 1)
            eps["attention_mask"] = roll_along(eps["attention_mask"], prompt_shifts, 1)

        input_ids_with_prompts = torch.cat((eps["input_ids"], input_ids), dim=1)
        attn_mask_with_prompts = torch.cat(
            (eps["attention_mask"], attention_mask), dim=1
        )
        if modify_labels:
            labels_empty = torch.zeros_like(eps["input_ids"]).fill_(-100)
            labels_with_prompts = torch.cat((labels_empty, labels), dim=1)

        # roll back
        if padding_side == "left":
            reset_shifts = attention_mask.shape[1] - shifts
            input_ids_with_prompts = roll_along(input_ids_with_prompts, reset_shifts, 1)
            attn_mask_with_prompts = roll_along(attn_mask_with_prompts, reset_shifts, 1)

            # now we know that all the padding is on the left, cut to max input length
            input_ids_with_prompts = input_ids_with_prompts[
                :, -prompts[0].config.max_input_length :
            ]
            attn_mask_with_prompts = attn_mask_with_prompts[
                :, -prompts[0].config.max_input_length :
            ]
            if modify_labels:
                labels_with_prompts = roll_along(labels_with_prompts, reset_shifts, 1)
                labels_with_prompts = labels_with_prompts[
                    :, -prompts[0].config.max_input_length :
                ]
        else:
            reset_shifts = eps["attention_mask"].shape[1] - prompt_shifts
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
            if modify_labels:
                labels_with_prompts = roll_along(labels_with_prompts, -reset_shifts, 1)
                labels_with_prompts = labels_with_prompts[
                    :, : prompts[0].config.max_input_length
                ]

        if labels is None:
            labels_with_prompts = None

        return input_ids_with_prompts, attn_mask_with_prompts, labels_with_prompts

    def forward(self, batch, **kwargs):
        raise NotImplementedError("Use parallel_forward instead.")

    @classmethod
    def modify_transformer(cls, transformer, config):
        raise NotImplementedError(
            "Use parallel_forward instead or HardPromptExpertContainer."
        )
