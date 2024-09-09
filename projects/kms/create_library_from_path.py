import os

import click

from mttl.models.library.expert_library import ExpertLibrary, LocalExpertLibrary


def find_directories_with_config(start_dir):
    matching_dirs = []

    for root, dirs, files in os.walk(start_dir):
        if "mttl_config.json" in files:
            matching_dirs.append(root)

    return matching_dirs


@click.command()
@click.option("--ckpt_path", type=str)
@click.option("--local_path", type=str)
def create(ckpt_path, local_path):
    library = ExpertLibrary.get_expert_library("local://" + local_path, create=True)
    expert_paths = find_directories_with_config(ckpt_path)

    print(expert_paths)
    for expert_path in expert_paths:
        library.add_expert_from_ckpt(expert_path)
        print("Length...", len(library))


if __name__ == "__main__":
    create()
