import re


MODIFIERS = {}


def register_modifier(name):
    print("Registering modifier..." + name)

    def _thunk(fn):
        if name in MODIFIERS:
            raise ValueError(f"Cannot register duplicate model modifier ({name})")
        MODIFIERS[name] = fn
        return fn
    return _thunk


def modify_transformer(transformer, config):
    import mttl.models.modifiers.lora  # noqa: F401
    import mttl.models.modifiers.poly  # noqa: F401
    import mttl.models.modifiers.routing  # noqa: F401

    # create a shared container for the task id 
    transformer.task_id_container = {}

    if config.model_modifier:
        if config.model_modifier in MODIFIERS:
            transformer = MODIFIERS[config.model_modifier](transformer, config)
        else:
            raise ValueError(f"Model modifier '{config.model_modifier}' not found.")
    return transformer
