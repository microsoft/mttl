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


def patch_layers(transformer, config, layer_type, optional_wrapper=None):
    from mttl.models.routing import get_selector

    # How to "bin" different levels of selectors ?
    def _extract_identifier(string, match_on="coder"):
        """Returns a unique identifier for the "chunk" of layers sharing the
        same underlying selector
        # e.g. 'block' : 'encoder.block.0.layer.0.SelfAttention' -> 'encoder.block.0'
        """
        pattern_map = {
            "coarsegrained": None,
            "finegrained": None,
            "layerwise": "layer",
            "blockwise": "block",
            "coderwise": "coder",
        }
        assert match_on in pattern_map.keys()

        if match_on == "finegrained":
            return string
        if match_on == "coarsegrained":
            return ""

        match_on = pattern_map[match_on]
        left_idx = string.find(f"{match_on}.") + len(match_on) + 1
        right_idx = string[left_idx:].find(".")
        return string[: left_idx + right_idx]

    selectors = {}
    total_layers = 0

    for m_name, module in dict(transformer.named_modules()).items():
        if re.fullmatch(config.lora_modules, m_name):
            for c_name, layer in dict(module.named_children()).items():
                if re.fullmatch(config.lora_layers, c_name):
                    identifier = _extract_identifier(
                        f"{m_name}.{c_name}", config.poly_granularity
                    )
                    if identifier not in selectors.keys():
                        selectors[identifier] = get_selector(config)

                    selector = selectors[identifier]
                    total_layers += 1

                    print(f"Patching {m_name}.{c_name}...")

                    wrapper = layer_type(
                        config,
                        transformer.task_id_container,
                        layer,
                        selector=selector,
                    )
                    wrapper.__layer_name__ = f"{m_name}.{c_name}"

                    setattr(
                        module,
                        c_name,
                        wrapper,
                    )

    print(
        f"created {len(selectors)} selectors for a total of {total_layers} adapted layers"
    )

    if optional_wrapper is not None:
        return optional_wrapper.register_functions(transformer)
    return transformer


def modify_transformer(transformer, config):
    # create a shared container for the task id 
    transformer.task_id_container = {}

    if config.model_modifier:
        if config.model_modifier in MODIFIERS:
            transformer = MODIFIERS[config.model_modifier](transformer, config)
        else:
            raise ValueError(f"Model modifier '{config.model_modifier}' not found.")
    return transformer
