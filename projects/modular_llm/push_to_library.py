from mttl.models.library.expert_library import HFExpertLibrary
from mttl.models.library.expert import load_expert
import glob
import huggingface_hub

huggingface_token = "your_token"
huggingface_hub.login(token=huggingface_token)


import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--library",
    type=str,
    required=True,
    default="zhan1993/default",
)

parser.add_argument("--input_path", type=str, required=True)
parser.add_argument("--expert_name", type=str, required=True)

args = parser.parse_args()
library_dest = HFExpertLibrary(f"{args.library}", create=True)
keys = library_dest.keys()

ckpt = args.input_path

with library_dest.batched_commit():
    expert = load_expert(ckpt)
    expert_name = args.expert_name
    if expert_name in keys:
        print(expert_name, "is in the library")
        exit()
    expert.expert_info.expert_name = expert_name
    if expert.name not in library_dest:
        library_dest.add_expert(expert)
