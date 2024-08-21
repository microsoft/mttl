import os
import pytest
from pytorch_lightning import seed_everything
from mttl.models.expert_model import MultiExpertModel
from mttl.models.containers import match_modules_to_modify, create_modif_regex


def test_add_expert_to_transformer():
    seed_everything(0)
    os.environ["COALESCED_LORA_CONTAINER"] = "0"
    # logic:
    # modify_modules -- will check if the module name contains the string
    # modify_layers -- will check if the module name ends on the string

    # modify_modules = .* -- modifies all modules
    # modify_layers = .* -- modifies all layers
    # will match any name that has a dot in it
    model = MultiExpertModel(model="EleutherAI/gpt-neo-125m", device_map="cpu")
    regex = create_modif_regex(modify_modules=".*", modify_layers=".*")
    matched_modules = [o[0] for o in match_modules_to_modify(model.model, regex)]
    assert len(matched_modules) == len(list(model.model.named_modules())) - 3

    # modify_modules = .* -- modifies all modules
    # modify_layers =
    # will match all modules
    regex = create_modif_regex(modify_modules=".*", modify_layers="")
    matched_modules = [o[0] for o in match_modules_to_modify(model.model, regex)]
    assert len(matched_modules) == len(list(model.model.named_modules()))

    # modify_modules=.*mlp -- only modifies modules edding with mlp
    regex = create_modif_regex(modify_modules=".*mlp", modify_layers="")
    matched_modules = [o[0] for o in match_modules_to_modify(model.model, regex)]
    assert all(name[-3:] == "mlp" for name in matched_modules)

    # same as above:
    regex = create_modif_regex(modify_modules="", modify_layers=".*mlp")
    matched_modules = [o[0] for o in match_modules_to_modify(model.model, regex)]

    # modify_modules=.*mlp -- only modifies modules called mlp
    # modify_layers=.* -- modifies all layers inside the mlp modules
    # will modify all layers of the module mlp, i.e. fc1, fc2 etc., but also act, dropout etc.
    regex = create_modif_regex(modify_modules=".*mlp", modify_layers=".*")
    matched_modules = [o[0] for o in match_modules_to_modify(model.model, regex)]
    assert len(matched_modules) == 48
    assert "transformer.h.7.mlp.dropout" in matched_modules

    # same as above but also matches the actual mlp parent modules
    # modify_layers=.*mlp.* -- modifies all layers of mlp modules
    # will modify all layers of the module mlp, i.e. fc1, fc2 etc., but also act, dropout etc.
    regex = create_modif_regex(modify_modules="", modify_layers=".*mlp.*")
    matched_modules = [o[0] for o in match_modules_to_modify(model.model, regex)]
    assert len(matched_modules) == 60
    assert "transformer.h.7.mlp.dropout" in matched_modules
    assert "transformer.h.7.mlp" in matched_modules

    # modify_modules = .*, modify_layers=.*out_proj -- only modifies layers called out_proj
    regex = create_modif_regex(
        modify_layers="k_proj|v_proj|q_proj", modify_modules=".*"
    )
    matched_modules = [o[0] for o in match_modules_to_modify(model.model, regex)]
    assert all(
        name[-6:] == "q_proj" or name[-6:] == "v_proj" or name[-6:] == "k_proj"
        for name in matched_modules
    )

    # but modify_modules=.*attn.* will match all modules containing attn
    regex = create_modif_regex(modify_layers="", modify_modules=".*attn.*")
    matched_modules = [o[0] for o in match_modules_to_modify(model.model, regex)]
    assert len(matched_modules) == 96
    assert "transformer.h.10.attn.attention.k_proj" in matched_modules
    assert "transformer.h.10.attn.attention" in matched_modules


if __name__ == "__main__":
    test_add_expert_to_transformer()
