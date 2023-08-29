import logging
from transformers import AutoTokenizer, LlamaTokenizer


def get_tokenizer(config):
    if config.model=="yahma/llama-7b-hf":        
        tokenizer = LlamaTokenizer.from_pretrained(config.model)
        tokenizer.pad_token_id = 0 
        tokenizer.padding_side = 'left'
        return tokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
    tokenizer.model_max_length = int(1e9)

    if config.model_family == 'gpt':
        tokenizer.padding_side = 'left'

    if tokenizer.pad_token_id is None:
        # no padding token, use EOS token instead!
        logging.warn("Setting pad_token_id to 0, given that pad_token_id was not detected.")
        tokenizer.pad_token_id = 0

    tokenizer.add_eos_token = False
    return tokenizer
