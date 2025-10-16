import glob
import json
import os

import click
import torch

from mttl.models.library.expert_library import ExpertLibrary, LocalExpertLibrary


def find_directories_with_config(start_dir, model_name="best_model"):
    matching_dirs = set()

    # this dir is a list of ids
    # loop through the first level of directories
    missing = []
    for id in glob.glob(os.path.join(start_dir, "*")):
        # loop through the second level of directories
        checkpoint_dirs = []
        for ckpt in glob.glob(os.path.join(id, "*")):
            if "checkpoint" in ckpt or model_name in ckpt:
                checkpoint_dirs.append(os.path.join(start_dir, id, f"{model_name}"))

        has_best_model = [ckpt for ckpt in checkpoint_dirs if model_name in ckpt]
        if has_best_model:
            matching_dirs.add(has_best_model[0])
        else:
            missing.append(id)

    return list(matching_dirs), missing


@click.command()
@click.option("--ckpt_path", type=str)
@click.option("--library_path", type=str)
@click.option("--model_name", type=str, default="best_model")
def create(ckpt_path, library_path, model_name):
    library = ExpertLibrary.get_expert_library(library_path, create=True)
    expert_paths, missing = find_directories_with_config(ckpt_path)
    if missing:
        print("Missing documents:")
        print(missing)

    library.add_experts_from_ckpts(expert_paths, force=False, update=True)

    # Outputs from `train_km.py`
    if os.path.exists(os.path.join(expert_paths[0], "mttl_args.bin")):
        training_args = json.dumps(
            torch.load(
                os.path.join(expert_paths[0], "mttl_args.bin"), weights_only=False
            ),
            indent=4,
        )
    # Outputs form `train_km_simple.py`
    else:
        training_args = json.load(
            open(os.path.join(expert_paths[0], "config.json"), "r")
        )
    library.update_readme(
        extra_info=f"Training arguments:\n```json\n{training_args}\n```"
    )
    print("Library uploaded to: ", f"{library_path}")


if __name__ == "__main__":
    create()
