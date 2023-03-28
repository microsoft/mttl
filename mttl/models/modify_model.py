from mttl.models.lora import modify_with_gator, modify_with_lora, modify_with_ia3, modify_with_ln
from mttl.models.poly import modify_with_poly_lora, modify_with_poly_ia3


modifier_dict = {
    "lora": modify_with_lora,
    "ia3": modify_with_ia3,
    "gator": modify_with_gator,
    "ln": modify_with_ln,
    "poly_lora": modify_with_poly_lora,
    "poly_ia3": modify_with_poly_ia3,
}


def modify_transformer(transformer, config):
    # create a shared container for the task id 
    transformer.task_id_container = {}

    if config.model_modifier:
        if config.model_modifier in modifier_dict:
            transformer = modifier_dict[config.model_modifier](transformer, config)
        else:
            raise ValueError(f"Model modifier '{config.model_modifier}' not found.")
    return transformer
