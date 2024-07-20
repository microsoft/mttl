import bitsandbytes as bnb

# stores modifiers across the mttl lib
MODIFIERS = {}
# stores mapping from configs to modifiers
CONFIGS_TO_MODIFIERS = {}
# stores mapping from modifiers to configs
MODIFIERS_TO_CONFIGS = {}


def register_modifier(name, config_cls=None):
    print("Registering modifier..." + name)

    def _thunk(klass):
        if name in MODIFIERS:
            raise ValueError(f"Cannot register duplicate model modifier ({name})")
        MODIFIERS[name] = klass

        if config_cls is not None:
            CONFIGS_TO_MODIFIERS[config_cls] = name
            MODIFIERS_TO_CONFIGS[name] = config_cls
        return klass

    return _thunk


def get_modifier_type(config, model_modifier=None):
    model_modifier = model_modifier or getattr(config, "model_modifier", None)
    try:
        model_modifier = model_modifier or CONFIGS_TO_MODIFIERS.get(config, None)
    except:
        pass
    model_modifier = model_modifier or CONFIGS_TO_MODIFIERS.get(type(config), None)
    return model_modifier


def modify_transformer(
    transformer, modifier_config, model_modifier=None, **modifier_kwargs
):
    from mttl.utils import logger

    # create a shared container for the possible routers
    transformer.selectors = {}

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

    model_modifier = get_modifier_type(modifier_config, model_modifier=model_modifier)

    if model_modifier is None:
        logger.warning("Model modifier not set nor in config nor as an argument.")
        return transformer

    if model_modifier:
        if model_modifier in MODIFIERS:
            transformer = MODIFIERS[model_modifier].modify_transformer(
                transformer, modifier_config
            )
        else:
            raise ValueError(f"Model modifier '{model_modifier}' not found.")
    return transformer
