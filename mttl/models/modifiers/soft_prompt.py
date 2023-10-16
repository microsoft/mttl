import re
import torch
import torch.nn as nn
from mttl.models.modifiers import register_modifier
from mttl.models.adapters import LoRA, LN, IA3
from mttl.utils import logger
from transformers.modeling_utils import PreTrainedModel

SoftPromptRouting = None


def modify_with_soft_prompt(transformer, config, layer_type):
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
    config.soft_prompt_length = 100

    soft_prompt = layer_type(
        config=config,
        task_id_ptr=task_id_container,
        base_input_embeddings=input_embeddings,
    )

    transformer.soft_prompt = soft_prompt

    def encoder_forward_wrapper(*args, **kwargs):
        assert args == (), "Only supports forward pass with named arguments."

        input_ids = kwargs.pop("input_ids")
        attention_mask = kwargs.get("attention_mask", None)

        input_embeds = transformer.get_input_embeddings()(input_ids)
        soft_prompts = soft_prompt(input_ids)

        # preprend the soft prompt
        input_embeds = torch.cat((soft_prompts, input_embeds), dim=1)

        # expand the attention mask
        attention_mask = torch.cat(
            (torch.ones_like(soft_prompts[:, :, 0]), attention_mask), dim=1
        )

        kwargs["attention_mask"] = attention_mask
        kwargs["inputs_embeds"] = input_embeds

        return old_enc_fwd(*args, **kwargs)

    def decoder_forward_wrapper(*args, **kwargs):
        assert args == (), "Only supports forward pass with named arguments."

        enc_out = kwargs.get("encoder_hidden_states", None)
        enc_attn_mask = kwargs.get("encoder_attention_mask", None)

        if (
            enc_out is not None
            and enc_attn_mask is not None
            and enc_out.size(1) != enc_attn_mask.size(1)
        ):
            assert enc_attn_mask.size(0) == enc_out.size(0), "batch sizes must match"
            assert (
                enc_out.size(1) - enc_attn_mask.size(1) == config.soft_prompt_length
            ), "prompt length must match"

            enc_attn_mask = torch.cat(
                (
                    torch.ones_like(enc_out[:, : config.soft_prompt_length, 0]),
                    enc_attn_mask,
                ),
                dim=1,
            )

            kwargs["encoder_attention_mask"] = enc_attn_mask

        return old_dec_fwd(*args, **kwargs)
    
    def forward_wrapper(*args, **kwargs):
        # get `input_ids`
        assert len(args) <= 1, "should have at most `input_ids`"

        if len(args) == 1:
            input_ids = args[0]
        else:
            input_ids = kwargs.pop("input_ids")
        
        attention_mask = kwargs.get("attention_mask", None)
        input_embeds = transformer.get_input_embeddings()(input_ids)

        soft_prompts = soft_prompt(input_ids)

        # preprend the soft prompt
        input_embeds = torch.cat((soft_prompts, input_embeds), dim=1)

        # expand the attention mask
        attention_mask = torch.cat(
            (torch.ones_like(soft_prompts[:, :, 0]), attention_mask), dim=1
        )

        kwargs["attention_mask"] = attention_mask
        kwargs["inputs_embeds"] = input_embeds

        out = old_fwd(*(), **kwargs)

        # We need to remove remove the soft prompt so as to preserve 
        # alignment with the labels
        out.logits = out.logits[:, config.soft_prompt_length:, :]

        # TODO: are we missing something else ?

        return out

    if hasattr(transformer, 'encoder'):
        # seq-2-seq model
        old_enc_fwd = transformer.encoder.forward
        old_dec_fwd = transformer.decoder.forward
        transformer.encoder.forward = encoder_forward_wrapper
        transformer.decoder.forward = decoder_forward_wrapper
    else:
        # decoder-only model 
        old_fwd = transformer.forward
        transformer.forward = forward_wrapper 

    return transformer

@register_modifier("soft_propmpt_routing")
def modify_with_soft_prompt_routing(transformer, config):
    return modify_with_soft_prompt(transformer, config, SoftPromptRouting)


@register_modifier("soft_prompt")
def modify_with_soft_prompt_routing(transformer, config):
    return modify_with_soft_prompt(transformer, config, SoftPrompt)


class SoftPrompt(nn.Module):
    def __init__(self, base_input_embeddings, config, *args, **kwargs):
        super().__init__()

        self.config = config
        self.embed_dim = config.vocab_embed_dim
        self.prompt_length = config.soft_prompt_length

        base_input_embeddings = base_input_embeddings.cpu()
        self.embedding = nn.Embedding(self.prompt_length, self.embed_dim)

        # initialize from the base model
        mean = base_input_embeddings.weight.mean(0)
        std = base_input_embeddings.weight.std(0)

        # TODO: Revisit this initialization
        new_weight = torch.randn_like(self.embedding.weight) * (std / 2) + mean
        self.embedding.weight.data.copy_(new_weight)

    def forward(self, input_ids, *args, **kwargs):
        bs, seq_len = input_ids.size()
        return self.embedding.weight.unsqueeze(0).expand(bs, -1, -1)


# Equivalent to SkilledLoRA
class SkilledSoftPrompt(SoftPrompt):
    pass


# Equivalent to PolyLoRA
class PolySoftPrompt(SoftPrompt):
    pass
