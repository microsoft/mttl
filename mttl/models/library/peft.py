import json
import math
import os
from typing import Dict, Tuple

import safetensors
import torch
from huggingface_hub import snapshot_download

from mttl.models.library.expert import Expert, ExpertInfo
from mttl.models.modifiers.base import ModifierConfig
from mttl.models.modifiers.lora import LoRAConfig


def load_peft_weights(peft_dir) -> Dict[str, torch.Tensor]:
    """
    Load the PEFT adapter from the given path.

    Args:
        peft_path (str): Path to the PEFT adapter.

    Returns:
        Adapter: PEFT adapter.
    """
    tensor_path = os.path.join(peft_dir, "adapter_model.safetensors")
    bin_file_path = os.path.join(peft_dir, "adapter_model.bin")
    new_embeddings_tensor_path = os.path.join(peft_dir, "new_embeddings.safetensors")
    new_embeddings_bin_file_path = os.path.join(peft_dir, "new_embeddings.bin")

    if os.path.isfile(tensor_path):
        tensors: Dict[str, torch.Tensor] = {}
        with safetensors.safe_open(lora_tensor_path, framework="pt") as f:  # type: ignore
            for module in f.keys():
                tensors[module] = f.get_tensor(module)

    elif os.path.isfile(bin_file_path):
        tensors = torch.load(bin_file_path, map_location="cpu")
    else:
        raise ValueError(f"{peft_dir} doesn't contain tensors")

    embeddings = None
    if os.path.isfile(new_embeddings_tensor_path):
        raise ValueError(f"{peft_dir} contains new embeddings tensors, not supported!")

    return tensors


def create_expert_from_peft_path(peft_path: str, expert_name: str = None) -> Expert:
    """
    Load the PEFT adapter from the given path.

    Args:
        peft_path (str): Path to the PEFT adapter.

    Returns:
        Adapter: PEFT adapter.
    """
    peft_dir = snapshot_download(peft_path)
    config_path = os.path.join(peft_dir, "adapter_config.json")

    base_model = config["base_model_name_or_path"]
    with open(config_path) as f:
        config = json.load(f)

    if config["peft_type"] == "LORA":
        rank = config["r"]
        lora_alpha = config["lora_alpha"]
        lora_dropout = config["lora_dropout"]
        target_modules = config["target_modules"]

        modifier_config = LoRAConfig(
            lora_rank=rank,
            lora_alpha=lora_alpha,
            lora_dropout=0.0,
            modify_modules=".*",
            modify_layers="|".join(target_modules),
        )
    else:
        raise ValueError(f"Not supported PEFT type {config['peft_type']}")

    tensors = load_peft_weights(peft_dir)
    expert_info = ExpertInfo(
        expert_config=modifier_config,
        expert_name=expert_name or peft_path.replace("/", "_"),
        training_config=None,
        expert_model=base_model,
    )
    expert = Expert(
        expert_info=expert_info,
        expert_weights=tensors,
    )
    return expert
