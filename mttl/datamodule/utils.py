import torch
from transformers import AutoTokenizer


def get_tokenizer(config):
    tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
    tokenizer.model_max_length = int(1e9)

    if config.model_family == 'gpt':
        tokenizer.padding_side = 'left'

    if tokenizer.pad_token_id is None:
        # no padding token, use EOS token instead!
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
