import bitsandbytes as bnb

from mttl.models.modifiers.base import Modifier, ModifierConfig


def get_modifier_name(config, model_modifier=None):
    model_modifier = model_modifier or getattr(config, "model_modifier", None)
    model_modifier = model_modifier or Modifier.get_name_by_config_class(type(config))
    return model_modifier


def modify_transformer(
    transformer,
    modifier_config: ModifierConfig,
    model_modifier: str = None,
    **modifier_kwargs,
):
    from mttl.logging import logger

    # set all params to not require grad
    for param in transformer.parameters():
        param.requires_grad = False

    if modifier_config is None or (
        hasattr(modifier_config, "model_modifier")
        and (modifier_config.model_modifier is None)
    ):
        # set all params to require grad
        for param in transformer.parameters():
            if not (
                isinstance(param, bnb.nn.modules.Params4bit)
                or isinstance(param, bnb.nn.Int8Params)
            ):
                param.requires_grad = True

    model_modifier = get_modifier_name(modifier_config, model_modifier=model_modifier)

    if model_modifier is None:
        logger.warning("Model modifier not set nor in config nor as an argument.")
        return transformer

    if model_modifier:
        if model_modifier in Modifier.registered_names():
            transformer = Modifier.get_class_by_name(model_modifier).modify_transformer(
                transformer, modifier_config
            )
        else:
            raise ValueError(f"Model modifier '{model_modifier}' not found.")
    return transformer
