from projects.instr_routing.models.routing import modify_with_poly_lora, modify_with_poly_ia3


modifier_dict = { 
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
