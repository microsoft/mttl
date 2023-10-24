import re
import torch
import torch.nn as nn
from mttl.models.modifiers import register_modifier
from mttl.models.adapters import LoRA, LN, IA3
from mttl.utils import logger
from transformers.modeling_utils import PreTrainedModel
from functools import partial
from mttl.models.modifiers.poly import PolytroponSelector

PromptTuningRouting = None


def modify_with_soft_prompt_cls(transformer, config, layer_type):
    """Wrap the forward pass of the model with a hook that takes extracts the embeddings
    from the `input_ids`, gets the embeddings from the soft prompt and stiches them together.
    """

    assert isinstance(
        transformer, PreTrainedModel
    ), "Transformer must be a PreTrainedModel."

    for param in transformer.parameters():
        param.requires_grad = False

    task_id_container = transformer.task_id_container

    input_embeddings = transformer.get_input_embeddings()
    config.vocab_embed_dim = input_embeddings.embedding_dim

    soft_prompt = layer_type(
        config=config,
        task_id_ptr=task_id_container,
        base_input_embeddings=input_embeddings,
    )

    return DecoderPromptTuningWrapper(
        config=config, transformer=transformer, soft_prompt=soft_prompt
    )


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
        input_embeds = self.transformer.get_input_embeddings()(input_ids)
        soft_prompts = self.soft_prompt(input_ids)

        # preprend the soft prompt
        input_embeds = torch.cat((soft_prompts, input_embeds), dim=1)

        # expand the attention mask
        attention_mask = torch.cat(
            (torch.ones_like(soft_prompts[:, :, 0]), attention_mask), dim=1
        )

        kwargs["attention_mask"] = attention_mask
        kwargs["inputs_embeds"] = input_embeds

        out = self.transformer(*(), **kwargs)

        # We need to remove remove the soft prompt so as to preserve
        # alignment with the labels
        out.logits = out.logits[:, self.config.soft_prompt_length :, :]

        # TODO: are we missing something else ?
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
        if model_kwargs["past_key_values"] is None:
            input_ids, attn_mask = (
                model_kwargs["input_ids"],
                model_kwargs["attention_mask"],
            )
            input_embeds = self.transformer.get_input_embeddings()(input_ids)
            soft_prompt = self.soft_prompt(model_kwargs["input_ids"])
            first_valid_idx = (attn_mask == 0).sum(1)
            bs, seq_len, embed_dim = input_embeds.size()
            soft_prompt_len = soft_prompt.size(1)

            """ 1. Add soft prompt output """
            # create container
            output = soft_prompt.new_zeros(bs, soft_prompt_len + seq_len, embed_dim)
            device = output.device

            # copy the soft prompt, padding left
            soft_prompt_idx = first_valid_idx.view(-1, 1) + torch.arange(
                soft_prompt_len, device=device
            ).view(1, -1)
            output.scatter_(
                dim=1,
                index=soft_prompt_idx.unsqueeze(-1).expand(-1, -1, embed_dim),
                src=soft_prompt,
            )

            # copy the input embeddings, padding left
            input_embeds_idx = (
                torch.arange(seq_len, device=device).view(1, -1).repeat(bs, 1)
            )
            # shift non-padding tokens to the right by `soft_prompt_len`
            input_embeds_idx[attn_mask == 1] += soft_prompt_len

            # don't copy the pad token embeddings from `input_embeds`
            output.scatter_(
                dim=1,
                index=input_embeds_idx.unsqueeze(-1).expand(-1, -1, embed_dim),
                src=input_embeds.type_as(output),
            )

            model_kwargs["inputs_embeds"] = output.type_as(input_embeds)
            model_kwargs["input_ids"] = None

            """ 2. Adjust Decoder Attention Mask """
            model_kwargs["attention_mask"] = torch.cat(
                (attn_mask, attn_mask.new_ones(bs, soft_prompt_len)), dim=1
            )

            """ 3. Adjust Position ids """
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
            model_kwargs["position_ids"] += self.config.soft_prompt_length

        return model_kwargs


@register_modifier("prompt_tuning")
def modify_with_soft_prompt_routing(transformer, config):
    return modify_with_soft_prompt_cls(transformer, config, PromptTuning)


@register_modifier("poly_prompt_tuning")
def modify_with_soft_prompt_routing(transformer, config):
    return modify_with_soft_prompt_cls(transformer, config, PolyPromptTuning)


class PromptTuning(nn.Module):
    def __init__(self, base_input_embeddings, config, *args, **kwargs):
        super().__init__()

        self.config = config
        self.embed_dim = config.vocab_embed_dim
        self.prompt_length = config.soft_prompt_length

        base_input_embeddings = base_input_embeddings.cpu()
        self.prompt_embedding = nn.Embedding(self.prompt_length, self.embed_dim)

        # initialize from the base model
        mean = base_input_embeddings.weight.mean(0)
        std = base_input_embeddings.weight.std(0)

        # TODO: Revisit this initialization
        new_weight = torch.randn_like(self.prompt_embedding.weight) * (std / 2) + mean
        self.prompt_embedding.weight.data.copy_(new_weight)

    def forward(self, input_ids, *args, **kwargs):
        bs, seq_len = input_ids.size()
        return self.prompt_embedding.weight.unsqueeze(0).expand(bs, -1, -1)


class PolyPromptTuning(nn.Module):
    def __init__(self, base_input_embeddings, config, task_id_ptr, *args, **kwargs):
        super().__init__()

        self.config = config
        self.task_id_ptr = task_id_ptr
        self.embed_dim = config.vocab_embed_dim
        self.prompt_length = config.soft_prompt_length

        base_input_embeddings = base_input_embeddings.cpu()
        self.selector = PolytroponSelector(config)

        # initialize from the base model
        mean = base_input_embeddings.weight.mean(0)
        std = base_input_embeddings.weight.std(0)

        # TODO: Revisit this initialization
        embedding = torch.randn(
            (
                config.n_skills,
                self.embed_dim,
                self.prompt_length,
            )
        ) * (std.view(1, -1, 1) / 2) + mean.view(1, -1, 1)

        # split dim according to `n_splits`
        embedding = embedding.reshape(
            config.n_skills,
            config.n_splits,
            self.embed_dim // config.n_splits,
            self.prompt_length,
        ).transpose(
            0, 1
        )  # (n_splits, n_skills, dim, prompt_length)

        self.prompt_embedding = nn.Parameter(embedding)

    def forward(self, input_ids, *args, **kwargs):
        bs, _ = input_ids.size()

        # get task ids
        routing_infos = self.task_id_ptr["routing_infos"]
        weights = self.selector(routing_infos)

        if weights.ndim == 1:
            raise NotImplementedError()
            # use indexing!
            # embed = self.prompt_embedding[:, weights.long(), :]
        else:
            embed = torch.einsum("bqs,qsdl->blqd", (weights, self.prompt_embedding))

        embed = embed.reshape(bs, self.prompt_length, self.embed_dim)
        return embed
