from transformers import AutoTokenizer, LlamaTokenizerFast, LlamaTokenizer

from mttl.utils import logger


def tokenizer_enforces_eos(tokenizer):
    test = "this is a long text seq that should be truncated"

    # copy tokenizer with add_eos parameter set to True
    old_add_eos = None

    if hasattr(tokenizer, "add_eos_token"):
        old_add_eos = tokenizer.add_eos_token
        tokenizer.add_eos_token = True

    token_ids = tokenizer(test, truncation=True, max_length=3)
    enforce_eos = token_ids["input_ids"][-1] == tokenizer.eos_token_id    
    
    if old_add_eos is not None:
        tokenizer.add_eos_token = old_add_eos

    return enforce_eos


def get_tokenizer(config, for_generation=False):
    return get_tokenizer_with_args(
        config.model, config.model_family, config.padding_side, for_generation
    )


def get_tokenizer_with_args(
    model_name, model_family, padding_side="right", for_generation=False
):
    if "llama" in model_name:
        tokenizer.pad_token_id = 0 
        if not model_family == "gpt":
            raise ValueError(
                "We detected a Llama model, but model_family != 'gpt', fix your config!"
            )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.model_max_length = int(1e9)

    tokenizer.padding_side = padding_side
    logger.warn("Padding side is {}".format(tokenizer.padding_side))

    if model_family == "gpt":
        if for_generation:
            if padding_side == "right":
                logger.warn("Padding side is 'right', but we are in generation mode!")

            logger.warn(
                "for_generation is True, setting padding_side for tokenizer to 'left'."
            )

            tokenizer.padding_side = "left"

        # do not add eos token, we will add it accordingly *if* needed.
        tokenizer.add_eos_token = False

    if tokenizer.pad_token_id is None:
        logger.warn(
            "Setting pad_token_id to 0, given that pad_token_id was not detected."
        )
        tokenizer.pad_token_id = 0

    tokenizer.mttl_enforces_eos = tokenizer_enforces_eos(tokenizer)
    return tokenizer
