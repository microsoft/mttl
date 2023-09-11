from transformers import AutoTokenizer, LlamaTokenizerFast, LlamaTokenizer

from mttl.utils import logger


def get_tokenizer(config, for_generation=False):
    if "llama" in config.model:
        tokenizer = LlamaTokenizerFast.from_pretrained(config.model)
        # tokenizer.model_max_length = int(1e9)
        tokenizer.pad_token_id = 0
        if not config.model_family == "gpt":
            raise ValueError(
                "We detected a Llama model, but model_family != 'gpt', fix your config!"
            )
    else:
        tokenizer = AutoTokenizer.from_pretrained(config.model)
        tokenizer.model_max_length = int(1e9)

    if hasattr(config, "padding_side"):
        logger.warn("Setting padding side to {}".format(config.padding_side))

        tokenizer.padding_side = config.padding_side
    else:
        logger.warn("Padding side is {}".format(tokenizer.padding_side))

    if config.model_family == "gpt":
        if for_generation:
            if config.padding_side == "right":
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
