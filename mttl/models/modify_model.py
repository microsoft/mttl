from mttl.models.lora import (
    modify_with_lora,
    modify_with_ia3,
    modify_with_krona,
    modify_with_quantumIA3Linear,
    modify_with_dora,
)
from mttl.models.poly import (
    modify_with_poly_lora,
    modify_with_poly_ia3,
    modify_with_tensorpoly_lora,
    modify_with_tensororderpoly_lora,
    modify_with_tensorpoly_ia3,
    modify_with_tensororderpoly_ia3,
)


modifier_dict = {
    "lora": modify_with_lora,
    "ia3": modify_with_ia3,
    "krona": modify_with_krona,
    "poly_lora": modify_with_poly_lora,
    "poly_ia3": modify_with_poly_ia3,
    "tensorpoly_lora": modify_with_tensorpoly_lora,
    "tensororderpoly_lora": modify_with_tensororderpoly_lora,
    "tensorpoly_ia3": modify_with_tensorpoly_ia3,
    "tensororderpoly_ia3": modify_with_tensororderpoly_ia3,
    "quantum_ia3": modify_with_quantumIA3Linear,
    "dora": modify_with_dora,
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
