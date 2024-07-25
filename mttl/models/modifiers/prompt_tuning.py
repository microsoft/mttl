import re
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_utils import PreTrainedModel

from mttl.models.expert_context import InfoContainer
from mttl.models.modifiers import register_modifier
from mttl.models.modifiers.base import Adapter, ModifierConfig, ModifyMixin
from mttl.models.modifiers.debug_utils import check_if_align, monitor_transformer
from mttl.models.modifiers.kv_adapter import KVAdapterConfig

# from mttl.models.containers.selectors import PolySelectorConfig


PromptTuningRouting = None


def roll_along(arr, shifts, dim):
    """Roll each row/col of a specific dimension according to the specified shifts"""

    assert arr.ndim - 1 == shifts.ndim
    dim %= arr.ndim
    shape = (1,) * dim + (-1,) + (1,) * (arr.ndim - dim - 1)
    dim_indices = torch.arange(arr.shape[dim]).reshape(shape).to(arr.device)
    indices = (dim_indices - shifts.unsqueeze(dim)) % arr.shape[dim]
    return torch.gather(arr, dim, indices)


def split_soft_prompts(logits, label_starts, soft_prompt_length):
    """Given logits with soft prompts, remove them to recover original shape"""

    # The first token to be removed should not be the first soft prompt,
    # But rather the token just before, and similarly the last token to be removed
    # Will be the second last soft token. This way, the last soft prompt's rep. is
    # used to predict the first label.
    label_starts = (label_starts - 1).clamp(min=0)

    bs, seq_len, vocab_size = logits.shape
    flat_logits = logits.view(bs * seq_len, vocab_size)

    bs_idx = torch.arange(bs, device=logits.device).view(-1, 1)
    soft_prompt_idx = label_starts.view(-1, 1) + torch.arange(
        soft_prompt_length, device=logits.device
    ).view(1, -1)
    flat_soft_prompt_idx = soft_prompt_idx + bs_idx * seq_len

    is_soft_prompt = torch.zeros(bs * seq_len, dtype=torch.bool, device=logits.device)
    is_soft_prompt[flat_soft_prompt_idx] = True
    not_soft_prompt = ~is_soft_prompt

    regular_tokens = flat_logits[not_soft_prompt].reshape(
        bs, seq_len - soft_prompt_length, vocab_size
    )

    soft_prompts = flat_logits[is_soft_prompt].reshape(
        bs, soft_prompt_length, vocab_size
    )

    return regular_tokens, soft_prompts


class ExtendedEmbedding(nn.Module):
    """Extends nn.Embedding to accomodate soft prompts"""

    def __init__(self, config, input_embeds, *args, **kwargs):
        super().__init__()
        self.config = config
        n_new_tokens = self.config.soft_prompt_length
        self.input_embeds = nn.Parameter(
            input_embeds.weight, requires_grad=input_embeds.weight.requires_grad
        )
        self.sparse = input_embeds.sparse

        # topk initialization
        self.new_embeds = nn.Parameter(
            input_embeds.weight[:n_new_tokens].clone().detach()
        )

    def forward(self, input_ids):
        # 1) concat the embeddings, and run fwd pass
        all_embeds = torch.cat((self.input_embeds, self.new_embeds), dim=0)
        out = F.embedding(input_ids, all_embeds, sparse=self.sparse)
        return out


class ExtendedLinear(nn.Module):
    """Extends Linear LM head to predict soft prompt routing"""

    def __init__(self, config, lm_head, *args, **kwargs):
        super().__init__()
        self.config = config
        n_new_tokens = self.config.n_skills
        self.lm_head = lm_head

        # topk initialization
        self.ext_weight = nn.Parameter(lm_head.weight[:n_new_tokens].clone().detach())
        if lm_head.bias is not None:
            self.ext_bias = nn.Parameter(lm_head.bias[:n_new_tokens].clone().detach())
        else:
            self.ext_bias = None

    def forward(self, x):
        # regular out
        weights = torch.cat((self.lm_head.weight, self.ext_weight), dim=0)
        if self.lm_head.bias is not None:
            bias = torch.cat((self.lm_head.bias, self.ext_bias), dim=0)
        else:
            bias = None

        return F.linear(x, weights, bias)


class DecoderPromptTuningWrapper(torch.nn.Module):
    def __init__(self, config, transformer, soft_prompt):
        super().__init__()
        self.transformer = transformer
        self.config = config
        self.soft_prompt = soft_prompt
        self.transformer_prepare_inputs_for_generation = (
            transformer.prepare_inputs_for_generation
        )

    # make sure we have all the attributes from `transformer`
    # by overwritting `getattr`
    def __getattr__(self, name):
        if name in ["transformer", "config", "soft_prompt"]:
            return super().__getattr__(name)
        else:
            return getattr(self.transformer, name)

    def forward(self, *args, **kwargs):
        # Should be padded (**right**)
        assert len(args) <= 1, "should have at most `input_ids`"

        if len(args) == 1:
            input_ids = args[0]
        else:
            input_ids = kwargs.pop("input_ids")

        attention_mask = kwargs.get("attention_mask", None)

        # soft prompt will return indices for the correct prompt
        soft_prompt_indices = self.soft_prompt(input_ids)

        # make sure we have the right padding
        assert (
            attention_mask[:, 0].sum() >= attention_mask[:, -1].sum()
        ), "expected right-padded input"

        # Assumes ExpertTrainer here. Removing the labels so as to not trigger an automatic loss computation
        info_container = InfoContainer.get()
        labels = info_container.routing_infos.labels

        # preprend the soft prompt
        if self.config.prompt_placement == "prefix":
            input_ids = torch.cat((soft_prompt_indices, input_ids), dim=1)
        elif self.config.prompt_placement == "suffix":
            assert labels is not None  # we are not generating.

            is_label = (labels != -100).int()
            # If there are multiple maximal values then the indices of the first maximal value are returned.
            label_starts = is_label.argmax(1)

            # put labels in front
            rolled_input_ids = roll_along(input_ids, -label_starts, 1)
            input_ids = torch.cat((soft_prompt_indices, rolled_input_ids), dim=1)

            # roll back, to put input in front
            input_ids = roll_along(input_ids, label_starts, 1)

        else:
            raise NotImplementedError(
                f"Unknown prompt placement: {self.config.prompt_placement}"
            )

        # expand the attention mask
        attention_mask = torch.cat(
            (torch.ones_like(soft_prompt_indices), attention_mask), dim=1
        )

        kwargs["attention_mask"] = attention_mask
        kwargs["input_ids"] = input_ids

        out = self.transformer(*(), **kwargs)
        # We need to remove remove the soft prompt so as to preserve
        # alignment with the labels
        if self.config.prompt_placement == "prefix":
            out.logits = out.logits[:, self.config.soft_prompt_length :, :]
        elif self.config.prompt_placement == "suffix":
            # remove the soft prompt logits
            out.logits, out.soft_prompt_logits = split_soft_prompts(
                out.logits, label_starts, self.config.soft_prompt_length
            )

        return out

    def generate(self, *args, **kwargs):
        self.transformer.prepare_inputs_for_generation = (
            self.prepare_inputs_for_generation
        )
        out = self.transformer.generate(*args, **kwargs)
        self.transformer.prepare_inputs_for_generation = (
            self.transformer_prepare_inputs_for_generation
        )
        return out

    def prepare_inputs_for_generation(self, *args, **kwargs):
        # Should be padded (**left**)
        model_kwargs = self.transformer_prepare_inputs_for_generation(*args, **kwargs)

        # Some useful information :
        # We expert `past_key_values` to already cache the soft prompts
        # However, the attention mask and position are not aware that we have extended the context

        # The last condition is specific to Phi-2
        if (
            model_kwargs["past_key_values"] is None
            or getattr(model_kwargs["past_key_values"], "seqlen_offset", -1) == 0
        ):
            input_ids, attn_mask = (
                model_kwargs["input_ids"],
                model_kwargs["attention_mask"],
            )
            bs = input_ids.size(0)
            soft_prompt_len = self.config.soft_prompt_length
            device = input_ids.device

            """ 1. Add soft prompt output """
            if self.config.prompt_placement == "prefix":
                # let's try a cleaner version with rolling
                ex_lens = model_kwargs["attention_mask"].sum(1)

                # pad left -> right
                right_padded_input_ids = roll_along(input_ids, ex_lens, 1)
                soft_prompt_indices = self.soft_prompt(right_padded_input_ids)
                input_ids = torch.cat(
                    (soft_prompt_indices, right_padded_input_ids), dim=1
                )

                # pad right -> left
                input_ids = roll_along(input_ids, -(ex_lens + soft_prompt_len), 1)

            elif self.config.prompt_placement == "suffix":
                # we are already padded left, so we can just concat right
                soft_prompt_indices = self.soft_prompt(input_ids)
                input_ids = torch.cat((input_ids, soft_prompt_indices), dim=1)

            model_kwargs["input_ids"] = input_ids

            """ 2. Adjust Decoder Attention Mask """
            model_kwargs["attention_mask"] = torch.cat(
                (attn_mask, attn_mask.new_ones(bs, soft_prompt_len)), dim=1
            )

            """ 3. Adjust Position ids """
            if "position_ids" in model_kwargs:
                position_ids = model_kwargs["position_ids"]
                model_kwargs["position_ids"] = torch.cat(
                    (
                        position_ids,
                        position_ids[:, -1].view(-1, 1)
                        + torch.arange(soft_prompt_len, device=device).view(1, -1),
                    ),
                    dim=1,
                )

        else:
            """1. Adjust Decoder Attention Mask"""
            attn_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat(
                (
                    attn_mask,
                    attn_mask.new_ones(
                        attn_mask.size(0), self.config.soft_prompt_length
                    ),
                ),
                dim=1,
            )

            """ 2. Adjust Position ids """
            if "position_ids" in model_kwargs:
                model_kwargs["position_ids"] += self.config.soft_prompt_length

            if hasattr(model_kwargs["past_key_values"], "seqlen_offset"):
                # Phi-2
                model_kwargs[
                    "past_key_values"
                ].seqlen_offset += self.config.soft_prompt_length

        return model_kwargs


def modify_with_prompt_tuning(soft_prompt_cls, embed_cls, transformer, config):
    """Wrap the forward pass of the model with a hook that takes extracts the embeddings
    from the `input_ids`, gets the embeddings from the soft prompt and stiches them together.
    """
    # assert isinstance(
    #    transformer, PreTrainedModel
    # ), "Transformer must be a PreTrainedModel."

    for param in transformer.parameters():
        param.requires_grad = False

    input_embeds = transformer.get_input_embeddings()

    # Create new embeddings and assign to transformer, replacing existing one
    ext_embeds = embed_cls(config, input_embeds)
    transformer.set_input_embeddings(ext_embeds)

    # Replace in the original model
    config.vocab_embed_dim = input_embeds.embedding_dim

    soft_prompt = soft_prompt_cls(
        config=config,
        base_input_embeddings=input_embeds,
    )

    return DecoderPromptTuningWrapper(
        config=config, transformer=transformer, soft_prompt=soft_prompt
    )


@dataclass
class PromptTuningConfig(KVAdapterConfig):
    prompt_placement: str = "prefix"


@register_modifier("prompt_tuning", config_cls=PromptTuningConfig)
class PromptTuning(Adapter, ModifyMixin):
    def __init__(self, base_input_embeddings, config, *args, **kwargs):
        super().__init__()

        self.config = config
        self.vocab_size = base_input_embeddings.num_embeddings
        self.embed_dim = config.vocab_embed_dim
        self.prompt_length = config.soft_prompt_length

        # build indices for the prompt
        self.prompt_indices = (
            torch.arange(self.prompt_length, dtype=torch.long) + self.vocab_size
        )

    def forward(self, input_ids, *args, **kwargs):
        prompt_indices = self.prompt_indices.to(input_ids.device)
        return prompt_indices.unsqueeze(0).expand(input_ids.size(0), -1)

    @classmethod
    def modify_transformer(cls, transformer, config):
        return modify_with_prompt_tuning(cls, ExtendedEmbedding, transformer, config)
