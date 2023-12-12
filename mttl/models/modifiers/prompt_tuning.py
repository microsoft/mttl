from dataclasses import dataclass
import torch
import torch.nn as nn
from mttl.models.modifiers import register_modifier
from mttl.models.modifiers.base import Adapter, ModifierConfig, ModifyMixin
from transformers.modeling_utils import PreTrainedModel
from mttl.models.modifiers.kv_adapter import (
    KVAdapterConfig,
    PolyKVAdapterConfig,
)
from mttl.models.modifiers.poly import PolytroponSelector
from mttl.models.modifiers.routing import RoutingMixin


PromptTuningRouting = None


def two_d_roll_along(data, offsets):
    bs, sq, dim = data.shape
    assert offsets.shape == (bs,), breakpoint()

    data_flat = data.view(bs, sq * dim)
    arr_flat = torch.arange(sq * dim, device=data.device).view(1, sq * dim)
    idx_flat = (arr_flat - offsets.view(-1, 1) * dim) % (sq * dim)
    out = torch.gather(data_flat, 1, idx_flat)
    return out.view(bs, sq, dim)


# assume left padding
def roll_along(arr, shifts, dim):
    assert arr.ndim - 1 == shifts.ndim
    dim %= arr.ndim
    shape = (1,) * dim + (-1,) + (1,) * (arr.ndim - dim - 1)
    dim_indices = torch.arange(arr.shape[dim]).reshape(shape).to(arr.device)
    indices = (dim_indices - shifts.unsqueeze(dim)) % arr.shape[dim]
    return torch.gather(arr, dim, indices)


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

        # make sure we have the right padding
        assert (
            attention_mask[:, 0].sum() >= attention_mask[:, -1].sum()
        ), "expected right-padded input"

        # Assumes ExpertTrainer here. Removing the labels so as to not trigger an automatic loss computation
        labels = self.task_id_container["routing_infos"].labels

        # preprend the soft prompt
        if self.config.prompt_placement == "prefix":
            input_embeds = torch.cat((soft_prompts, input_embeds), dim=1)
        elif self.config.prompt_placement == "suffix":
            assert labels is not None  # we are not generating.

            is_label = (labels != -100).int()
            # If there are multiple maximal values then the indices of the first maximal value are returned.
            label_starts = is_label.argmax(1)

            # put labels in front
            rolled_input_embeds = two_d_roll_along(input_embeds, -label_starts)
            new_input_embeds = torch.cat((soft_prompts, rolled_input_embeds), dim=1)

            # roll back, to put input in front
            new_input_embeds = two_d_roll_along(new_input_embeds, label_starts)

            input_embeds = new_input_embeds
        else:
            raise NotImplementedError(
                f"Unknown prompt placement: {self.config.prompt_placement}"
            )

        # expand the attention mask
        attention_mask = torch.cat(
            (torch.ones_like(soft_prompts[:, :, 0]), attention_mask), dim=1
        )

        kwargs["attention_mask"] = attention_mask
        kwargs["inputs_embeds"] = input_embeds

        # TODO: can we just expand the labels to handle soft prompts ?
        out = self.transformer(*(), **kwargs)

        # We need to remove remove the soft prompt so as to preserve
        # alignment with the labels
        if self.config.prompt_placement == "prefix":
            out.logits = out.logits[:, self.config.soft_prompt_length :, :]
        elif self.config.prompt_placement == "suffix":
            # remove the soft prompt logits
            rolled_logits = two_d_roll_along(out.logits, -label_starts)
            chunked_logits = rolled_logits[:, self.config.soft_prompt_length :, :]
            out.logits = two_d_roll_along(chunked_logits, label_starts)

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
            device = soft_prompt.device

            """ 1. Add soft prompt output """
            if self.config.prompt_placement == "prefix":
                # let's try a cleaner version with rolling
                ex_lens = model_kwargs["attention_mask"].sum(1)

                right_padded_input_ids = roll_along(input_ids, ex_lens, 1)
                right_input_embeds = self.transformer.get_input_embeddings()(
                    right_padded_input_ids
                )
                right_output = torch.cat((soft_prompt, right_input_embeds), dim=1)
                output = two_d_roll_along(right_output, -(ex_lens + soft_prompt_len))

            elif self.config.prompt_placement == "suffix":
                # we are already padded left, so we can just concat right
                output = torch.cat((input_embeds, soft_prompt), dim=1)

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


def modify_with_prompt_tuning(cls, transformer, config):
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

    soft_prompt = cls(
        config=config,
        task_id_ptr=task_id_container,
        base_input_embeddings=input_embeddings,
    )

    return DecoderPromptTuningWrapper(
        config=config, transformer=transformer, soft_prompt=soft_prompt
    )


class PromptTuningModifyMixin(ModifyMixin):
    @classmethod
    def modify_transformer(cls, transformer, config):
        return modify_with_prompt_tuning(cls, transformer, config)


@dataclass
class PromptTuningConfig(KVAdapterConfig):
    prompt_placement = "prefix"


@register_modifier("prompt_tuning", config_cls=PromptTuningConfig)
class PromptTuning(Adapter, PromptTuningModifyMixin):
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


@dataclass
class AlphaPromptTuningConfig(KVAdapterConfig):
    pass


@register_modifier("alpha_prompt_tuning", config_cls=AlphaPromptTuningConfig)
class AlphaPromptTuning(Adapter, PromptTuningModifyMixin):
    def __init__(self, base_input_embeddings, config, task_id_ptr, *args, **kwargs):
        super().__init__()

        self.config = config
        self.embed_dim = config.vocab_embed_dim
        self.prompt_length = config.soft_prompt_length
        self.n_splits = config.n_splits
        self.dim_per_split = base_input_embeddings.embedding_dim // self.n_splits
        self.task_id_ptr = task_id_ptr
        self.num_tokens = base_input_embeddings.num_embeddings

        n_splits = config.n_splits
        config.n_splits = config.soft_prompt_length * n_splits
        config.n_skills = base_input_embeddings.num_embeddings
        self.prompt_mixer = PolytroponSelector(config)
        config.n_splits = n_splits

        self.base_input_embeddings = base_input_embeddings.weight

    def forward(self, input_ids, *args, **kwargs):
        # build input for polytropon
        bs = self.task_id_ptr["routing_infos"].task_ids.size(0)
        embeds = self.base_input_embeddings.view(
            self.num_tokens, self.n_splits, self.dim_per_split
        )
        self.task_id_ptr["routing_infos"].task_ids.fill_(0)
        mixing_weights = self.prompt_mixer(self.task_id_ptr["routing_infos"])
        mixing_weights = mixing_weights.view(
            bs, self.prompt_length, self.n_splits, self.num_tokens
        )
        mixed = torch.einsum("BLSV,VSD->BLSD", (mixing_weights, embeds))
        return mixed.reshape(bs, self.prompt_length, -1)

    @classmethod
    def modify_transformer(cls, transformer, config):
        return modify_with_prompt_tuning(cls, transformer, config)


@dataclass
class PolyPromptTuningConfig(PolyKVAdapterConfig):
    pass


@register_modifier("poly_prompt_tuning", config_cls=PolyPromptTuningConfig)
class PolyPromptTuning(Adapter, RoutingMixin, PromptTuningModifyMixin):
    def __init__(self, base_input_embeddings, config, task_id_ptr, *args, **kwargs):
        super(Adapter, self).__init__()
        super(RoutingMixin, self).__init__(task_id_ptr)

        self.config = config
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
        weights = self.selector(self.routing_infos)

        if weights.ndim == 1:
            raise NotImplementedError()
            # use indexing!
            # embed = self.prompt_embedding[:, weights.long(), :]
        else:
            embed = torch.einsum("bqs,qsdl->blqd", (weights, self.prompt_embedding))

        embed = embed.reshape(bs, self.prompt_length, self.embed_dim)
        return embed
