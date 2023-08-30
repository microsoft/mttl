from transformers import AutoTokenizer, LlamaTokenizerFast

from mttl.utils import logger


def get_tokenizer(config):
    if "llama" in config.model: 
        tokenizer = LlamaTokenizerFast.from_pretrained(config.model)
        tokenizer.pad_token_id = 0 
        tokenizer.padding_side = 'left'
    else:
        tokenizer = AutoTokenizer.from_pretrained(config.model)
        tokenizer.model_max_length = int(1e9)

        if config.model_family == 'gpt':
            tokenizer.padding_side = 'left'

        if tokenizer.pad_token_id is None:
            logger.warn("Setting pad_token_id to 0, given that pad_token_id was not detected.")
            tokenizer.pad_token_id = 0

        tokenizer.add_eos_token = False
    return tokenizer
