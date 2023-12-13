from dataclasses import dataclass
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
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


def roll_along(arr, shifts, dim):
    """Roll each row/col of a specific dimension according to the specified shifts"""

    assert arr.ndim - 1 == shifts.ndim
    dim %= arr.ndim
    shape = (1,) * dim + (-1,) + (1,) * (arr.ndim - dim - 1)
    dim_indices = torch.arange(arr.shape[dim]).reshape(shape).to(arr.device)
    indices = (dim_indices - shifts.unsqueeze(dim)) % arr.shape[dim]
    return torch.gather(arr, dim, indices)


def two_d_roll_along(data, offsets):
    """Specific version of roll along to handle a `dim` axis"""

    bs, sq, dim = data.shape
    assert offsets.shape == (bs,), breakpoint()

    data_flat = data.view(bs, sq * dim)
    arr_flat = torch.arange(sq * dim, device=data.device).view(1, sq * dim)
    idx_flat = (arr_flat - offsets.view(-1, 1) * dim) % (sq * dim)
    out = torch.gather(data_flat, 1, idx_flat)
    return out.view(bs, sq, dim)


def recursive_getattr(obj, attr_str):
    """Given a state_dict string, recursively get the object"""

    # Regular expression to match 'attr', 'attr[index]', followed by '.attr[index]' etc.
    pattern = r"([^\[\].]+)(\[\d+\])?"
    matches = re.finditer(pattern, attr_str)

    for match in matches:
        attr, index = match.groups()
        obj = getattr(obj, attr)

        if index:
            obj = obj[int(index.strip("[]"))]

        # Check if the current match is at the end of the string
        if match.end() == len(attr_str):
            return obj

    return obj


def recursive_setattr(parent_obj, attr_str, value):
    """Given a state_dict string, recursively get and set the object"""

    # Regular expression to match 'attr', 'attr[index]', followed by '.attr[index]' etc.
    pattern = r"([^\[\].]+)(\[\d+\])?"
    matches = re.finditer(pattern, attr_str)
    obj = parent_obj

    n_matches = 0
    for match in matches:
        n_matches += 1
        attr, index = match.groups()
        parent_obj = obj
        obj = getattr(parent_obj, attr)

        if index:
            obj = obj[int(index.strip("[]"))]

        # Check if the current match is at the end of the string
        if match.end() == len(attr_str):
            setattr(parent_obj, attr, value)
            return

    raise AttributeError(f"attribute not found: {attr_str}")


def get_input_embeddings_pointer(transformer):
    embeds = transformer.get_input_embeddings()
    for parent_module in transformer.modules():
        for name, child_module in parent_module.named_modules():
            if child_module is embeds:
                name = re.sub(
                    r"\.(\d+)\.", r"[\1].", name
                )  # layer.0.wte -> layers[0].wte
                assert recursive_getattr(parent_module, name) is embeds
                return name, parent_module

    raise ValueError("Could not find input embeddings pointer")


class ExtendedEmbedding(nn.Module):
    """Extends nn.Embedding to accomodate soft prompts"""

    def __init__(self, config, input_embeds, n_new_tokens):
        super().__init__()
        self.config = config
        self.input_embeds = nn.Parameter(input_embeds)

        # topk initialization
        self.new_embeds = nn.Parameter(input_embeds[:n_new_tokens].clone().detach())

    def forward(self, input_ids):
        # 1) concat the embeddings, and run fwd pass
        all_embeds = torch.cat((self.new_embeds, self.input_embeds), dim=0)
        out = F.embedding(input_ids, all_embeds)
        return out


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
        old_input_ids = input_ids
        old_attn_mask = attention_mask
        old_kwargs = {"attention_mask": old_attn_mask, "input_ids": old_input_ids}

        # soft prompt will return indices for the correct prompt
        soft_prompt_indices = self.soft_prompt(input_ids)

        # make sure we have the right padding
        assert (
            attention_mask[:, 0].sum() >= attention_mask[:, -1].sum()
        ), "expected right-padded input"

        # Assumes ExpertTrainer here. Removing the labels so as to not trigger an automatic loss computation
        labels = self.task_id_container["routing_infos"].labels

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

        # TODO: can we just expand the labels to handle soft prompts ?
        # TODO: remove !
        self.transformer.eval()

        out = self.transformer(*(), **kwargs)

        def check_if_align(old_kw, new_kw, max_len, up_to=None):
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

            assert not self.transformer.training
            out_old = self.transformer(*(), **old_kw)
            out_new = self.transformer(*(), **new_kw)

            if up_to is not None:
                old_logits = out_old.logits[:, :up_to]
                new_logits = out_new.logits[:, :up_to]
            else:
                old_logits = out_old.logits
                new_logits = out_new.logits

            return torch.allclose(old_logits, new_logits)

        # We need to remove remove the soft prompt so as to preserve
        # alignment with the labels
        if self.config.prompt_placement == "prefix":
            out.logits = out.logits[:, self.config.soft_prompt_length :, :]
        elif self.config.prompt_placement == "suffix":
            # remove the soft prompt logits

            # This is always True
            print(
                "a",
                check_if_align(
                    old_kwargs,
                    kwargs,
                    input_ids.size(-1) - soft_prompt_indices.size(-1),
                    label_starts.min().item(),
                ),
            )

            # This is always False
            print(
                "b",
                check_if_align(
                    old_kwargs,
                    kwargs,
                    input_ids.size(-1) - soft_prompt_indices.size(-1) + 1,
                    label_starts.min().item(),
                ),
            )

            # So things go south as soon as we go beyond the original input length
            breakpoint()
            # check_if_align(old_kwargs, kwargs, 65, 65)
            # rolled_logits = two_d_roll_along(out.logits, -label_starts)
            # chunked_logits = rolled_logits[:, self.config.soft_prompt_length :, :]
            # out.logits = two_d_roll_along(chunked_logits, label_starts)

            # TODO: MAKE THIS OK (this is a temporary hack so that code compiles) !
            out.logits = out.logits[:, self.config.soft_prompt_length :, :]

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


def modify_with_prompt_tuning(soft_prompt_cls, embed_cls, transformer, config):
    """Wrap the forward pass of the model with a hook that takes extracts the embeddings
    from the `input_ids`, gets the embeddings from the soft prompt and stiches them together.
    """
    assert isinstance(
        transformer, PreTrainedModel
    ), "Transformer must be a PreTrainedModel."

    for param in transformer.parameters():
        param.requires_grad = False

    task_id_container = transformer.task_id_container

    embed_name, parent_mod = get_input_embeddings_pointer(transformer)
    input_embeds = recursive_getattr(parent_mod, embed_name)
    assert isinstance(input_embeds, nn.Embedding)

    # Create new embeddings and assign to transformer, replacing existing one
    ext_embeds = embed_cls(config, input_embeds.weight, config.soft_prompt_length)

    recursive_setattr(parent_mod, embed_name, ext_embeds)

    # Replace in the original model
    config.vocab_embed_dim = input_embeds.embedding_dim

    soft_prompt = soft_prompt_cls(
        config=config,
        task_id_ptr=task_id_container,
        base_input_embeddings=input_embeds,
    )

    return DecoderPromptTuningWrapper(
        config=config, transformer=transformer, soft_prompt=soft_prompt
    )


@dataclass
class PromptTuningConfig(KVAdapterConfig):
    prompt_placement = "prefix"


@register_modifier("prompt_tuning", config_cls=PromptTuningConfig)
class PromptTuning(Adapter, ModifyMixin):
    def __init__(self, base_input_embeddings, config, *args, **kwargs):
        super().__init__()

        self.config = config
        self.embed_dim = config.vocab_embed_dim
        self.prompt_length = config.soft_prompt_length

        # build indices for the prompt
        self.prompt_indices = (
            torch.arange(self.prompt_length, dtype=torch.long)
            + base_input_embeddings.num_embeddings
        )

    def forward(self, input_ids, *args, **kwargs):
        prompt_indices = self.prompt_indices.to(input_ids.device)
        return prompt_indices.unsqueeze(0).expand(input_ids.size(0), -1)

    @classmethod
    def modify_transformer(cls, transformer, config):
        return modify_with_prompt_tuning(cls, ExtendedEmbedding, transformer, config)


@register_modifier("alpha_prompt_tuning", config_cls=PromptTuningConfig)
class AlphaPromptTuning(Adapter, ModifyMixin):
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
class PolyPromptTuning(Adapter, RoutingMixin, ModifyMixin):
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
