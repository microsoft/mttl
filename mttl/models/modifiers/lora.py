import re
from mttl.models.modifiers import register_modifier
from mttl.models.adapters import LoRA, LN, IA3


def modify_with_adapter(transformer, config, adapter_klass):
    for m_name, module in dict(transformer.named_modules()).items():
        if re.fullmatch(config.modify_modules, m_name):
            for c_name, layer in dict(module.named_children()).items():
                if re.fullmatch(config.modify_layers, c_name):
                    setattr(
                        module,
                        c_name,
                        adapter_klass(config, layer),
                    )
    return transformer


@register_modifier("ia3")
def modify_with_ia3(transformer, config):
    return modify_with_adapter(transformer, config, IA3)


@register_modifier("lora")
def modify_with_lora(transformer, config):
    return modify_with_adapter(transformer, config, LoRA)


@register_modifier("ln")
def modify_with_ln(transformer, config):
    return modify_with_adapter(transformer, config, LN)
