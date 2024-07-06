import os
import sys
import re
import click
from copy import deepcopy

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from mttl.models.modifiers.expert_containers.expert_library import (
    ExpertLibrary,
)

from mttl.utils import logger


def translate_lib_to_hf_phi(
    library: ExpertLibrary, library_id_target, tie_params=False
) -> ExpertLibrary:
    """
    The new version of phi-2 on hugging face seperates W_qkv into k_proj, v_proj, q_proj, the previous version was using a single Wqkv matrix.
    This function transforms the library to be compatible with the new version of phi-2 by:
    - splitting W_qkv into k_proj, v_proj, q_proj: results in k_proj.lora_a, v_proj.lora_a, q_proj.lora_a being the same.
    - making sure layer count matches (it starts at 0 in the new version)
    - renames mixer into self_attn
    - renames out_proj into dense
    """
    destin_lib_class = ExpertLibrary._get_expert_lib_class(
        library_id_target, expert_library_type=None
    )
    new_library = destin_lib_class(
        library_id_target,
        create=True,
    )
    if len(new_library) > 0:
        raise ValueError(
            f"Library with id {library_id_target} already exists and it is not empty. Please provide a new library id."
        )
    with new_library.batched_commit():
        for expert_name in library.keys():
            expert_dump = library[expert_name]
            expert_dump.expert_config.modify_layers = (
                ".*k_proj.*|.*v_proj.*|.*q_proj.*|.*dense.*"
            )
            expert_dump.training_config.modify_layers = (
                ".*k_proj.*|.*v_proj.*|.*q_proj.*|.*dense.*"
            )
            # 1. split Wqkv into k_proj, v_proj, q_proj
            expert_weights = expert_dump.expert_weights
            new_expert_weights = {}
            for k, v in expert_weights.items():
                new_k = "model." + k
                new_k = new_k.replace("mixer", "self_attn")
                new_k = new_k.replace("out_proj", "dense")
                # regex to decrease the layer number by one
                new_k = re.sub(r"\d+", lambda m: str(int(m.group()) - 1), new_k)
                if "Wqkv.lora_a" in k:
                    lora_a = v
                    lora_b = expert_weights[k.replace("lora_a", "lora_b")]
                    in_d = lora_a.shape[0]
                    for i, attn_key in enumerate(["q_proj", "k_proj", "v_proj"]):
                        module_name = new_k.split(".Wqkv.lora_a")[0]
                        new_expert_weights[f"{module_name}.{attn_key}.lora_a"] = lora_a
                        new_expert_weights[f"{module_name}.{attn_key}.lora_b"] = lora_b[
                            :, i * in_d : (i + 1) * in_d
                        ]
                elif "Wqkv.lora_b" in k:
                    continue
                else:
                    new_expert_weights[new_k] = v
            expert_dump.expert_weights = new_expert_weights
            if tie_params:
                # makes sure that params are tied for arrow prototype calculation: q,k and v should get the same prototype.
                expert_dump.training_config.tie_params = (
                    "q_proj.*\\.lora_a|k_proj.*\\.lora_a|v_proj.*\\.lora_a"
                )
            new_library.add_expert(expert_name=expert_name, expert_dump=expert_dump)
            # make sure arrow routing uses same routing for q, k, v as in older version of phi-2 implementation.
        return new_library


@click.command()
@click.option(
    "--library_id_source"
)  # e.g. hf://ostapeno/library-phi_2-v3-10-flan-clusters-fromlucas
@click.option(
    "--library_id_target"
)  # e.g. hf://ostapeno/library-phi_2-v3-10-flan-clusters-fromlucas_transformed
@click.option("--remote_token", default=None)
@click.option(
    "--tie_params",
    default=True,
    help="If 'True' sets tie_params argument of experts to 'q_proj.*\\.lora_a|k_proj.*\\.lora_a|v_proj.*\\.lora_a'. This makes sure that later same arrow routing is computed for q,k,v.",
)
def main(library_id_source, library_id_target, remote_token, tie_params=True):
    library = ExpertLibrary.get_expert_library(
        repo_id=library_id_source,
        token=remote_token if remote_token else None,
    )
    an_expert = library[next(iter(library.keys()))]
    train_cfg = deepcopy(an_expert.training_config)
    assert train_cfg.model == "phi-2", "This script is only for phi-2 models."
    assert (
        "mixer" in list(an_expert.expert_weights.keys())[0]
    ), "This script is only for converting old phi-2 versions using mixer."
    new_library = translate_lib_to_hf_phi(
        library, library_id_target, tie_params=tie_params
    )
    assert len(new_library) == len(library), "The number of experts should be the same."


if __name__ == "__main__":
    main()
