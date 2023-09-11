from transformers import AutoTokenizer, LlamaTokenizerFast

from mttl.utils import logger


def get_tokenizer(config, for_generation=False):
    return get_tokenizer_with_args(
        config.model, config.model_family, config.padding_side, for_generation
    )


def get_tokenizer_with_args(
    model_name, model_family, padding_side="right", for_generation=False
):
    if "llama" in model_name:
        tokenizer = LlamaTokenizerFast.from_pretrained(model_name)
        tokenizer.model_max_length = int(1e9)
        if not model_family == "gpt":
            raise ValueError(
                "We detected a Llama model, but model_family != 'gpt', fix your config!"
            )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.model_max_length = int(1e9)

    logger.warn("Setting padding side to {}".format(padding_side))
    tokenizer.padding_side = padding_side

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

    return tokenizer
