from mttl.utils import logger

MODIFIERS = {}
CONFIGS_TO_MODIFIERS = {}


def register_modifier(name, config_cls=None):
    print("Registering modifier..." + name)

    def _thunk(klass):
        if name in MODIFIERS:
            raise ValueError(f"Cannot register duplicate model modifier ({name})")
        MODIFIERS[name] = klass

        if config_cls is not None:
            CONFIGS_TO_MODIFIERS[config_cls] = name
        return klass

    return _thunk


def modify_transformer(transformer, modifier_config, model_modifier=None):
    import mttl.models.modifiers.lora  # noqa: F401
    import mttl.models.modifiers.poly  # noqa: F401
    import mttl.models.modifiers.routing  # noqa: F401
    import mttl.models.modifiers.prompt_tuning  # noqa: F401
    import mttl.models.modifiers.llama_adapter  # noqa: F401
    from mttl.utils import logger

    # import mttl.models.modifiers.prefix_tuning # noqa: F401

    # create a shared container for the task id
    transformer.task_id_container = {}

    # set all params to not require grad
    for param in transformer.parameters():
        param.requires_grad = False

    if model_modifier is None:
        model_modifier = getattr(modifier_config, "model_modifier", None)

    if model_modifier is None:
        model_modifier = CONFIGS_TO_MODIFIERS.get(type(modifier_config), None)

    if model_modifier is None:
        logger.warn("Model modifier not set nor in config nor as an argument.")
        return transformer

    if model_modifier:
        if model_modifier in MODIFIERS:
            transformer = MODIFIERS[model_modifier].modify_transformer(
                transformer, modifier_config
            )
        else:
            raise ValueError(f"Model modifier '{model_modifier}' not found.")
    return transformer
