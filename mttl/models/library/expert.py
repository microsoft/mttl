from dataclasses import dataclass
from typing import Any, Dict, Union

import torch

from mttl.logging import logger
from mttl.models.modifiers.base import (
    AutoModifierConfig,
    ModifierConfig,
    get_target_2_source_param_mapping,
)
from mttl.serializable import Serializable
from mttl.utils import get_checkpoint_path


@dataclass
class ExpertInfo(Serializable):
    expert_name: str
    expert_task_name: str = None
    parent_node: str = None
    # configuration for this expert, i.e. a modifier config
    expert_config: AutoModifierConfig = None
    # arguments with which the expert was trained, i.e. the full training config
    training_config: Dict[str, Any] = None
    expert_model: str = None

    @property
    def model(self):
        """Returns the expert model associated with the expert. Tries to get it
        from training_config if expert_model is None for back-compatibility.
        """
        if self.expert_model:
            return self.expert_model
        elif self.training_config:
            # fallback to training config
            return self.training_config.get("model")

    @property
    def dataset(self) -> str:
        """Returns the dataset name from training config or an empty string."""
        training_config = getattr(self, "training_config", {})
        if training_config is not None:
            return training_config.get("dataset", "")
        return "undefined"

    @property
    def modifier_name(self):
        if self.expert_config is None:
            # fallback in the training config
            logger.warning(
                "No expert config found for this expert, this is probably a legacy checkpoint, and will be deprecated."
            )

            return getattr(self.training_config, "modifier_name", None)
        return self.expert_config.modifier_name


class Expert:
    def __init__(
        self,
        expert_info: ExpertInfo,
        expert_weights: Dict[str, torch.Tensor] = None,
        expert_optimizer_state: Dict[str, torch.Tensor] = None,
    ):
        self.expert_info = expert_info
        self._expert_weights = expert_weights
        self.expert_optimizer_state = expert_optimizer_state
        self._tied_expert_weights = None

    @property
    def expert_weights(self):
        if (
            self.expert_info.expert_config is not None
            and hasattr(self.expert_info.expert_config, "tie_params")
            and self.expert_info.expert_config.tie_params is not None
            and isinstance(self._expert_weights, Dict)
        ):
            # make sure tied params are in the state dict.
            if self._tied_expert_weights is not None:
                return self._tied_expert_weights
            target_2_source_params_map = get_target_2_source_param_mapping(
                self._expert_weights.items(),
                self.expert_info.expert_config.tie_params,
                expand_if_targets_are_missing=True,
            )
            expert_weights = self._expert_weights
            for target_name, source_name in target_2_source_params_map.items():
                if not target_name in expert_weights:
                    # tie weights
                    expert_weights[target_name] = expert_weights[source_name]
                assert torch.allclose(
                    expert_weights[target_name], expert_weights[source_name]
                ), f"Weight tying failed for {target_name} and {source_name}"
            self._tied_expert_weights = expert_weights
            return expert_weights
        return self._expert_weights

    @expert_weights.setter
    def expert_weights(self, weights):
        self._expert_weights = weights

    def clone(self):
        import copy

        exp_copy = Expert(
            expert_info=copy.deepcopy(self.expert_info),
            expert_weights=copy.deepcopy(self._expert_weights),
            expert_optimizer_state=copy.deepcopy(self.expert_optimizer_state),
        )
        return exp_copy

    @classmethod
    def fromdict(cls, data):
        data["expert_info"] = ExpertInfo.fromdict(data["expert_info"])
        return cls(**data)

    @property
    def training_config(self) -> Dict:
        # back-compatibility, returns expert config
        return self.expert_info.training_config

    @property
    def expert_config(self) -> ModifierConfig:
        return self.expert_info.expert_config

    def asdict(self):
        return {
            "expert_info": self.expert_info.asdict(),
            "expert_weights": self._expert_weights,
            "expert_optimizer_state": self.expert_optimizer_state,
        }

    @property
    def name(self):
        return self.expert_info.expert_name

    @name.setter
    def name(self, name):
        self.expert_info.expert_name = name


def load_expert(
    expert_path: str,
    expert_library: Union[Dict, "ExpertLibrary"] = None,
    expert_name: str = None,
    **kwargs,
):
    """Load expert directly from the checkpoint.

    Supports pytorch lightning and huggingface checkpoints.
    """
    import os

    from huggingface_hub import list_repo_files

    from mttl.models.base_model import WEIGHTS_NAME
    from mttl.models.library.peft import load_expert_from_peft_checkpoint
    from mttl.models.lightning.base_module import CHECKPOINT_PATH_IN_HUB

    if expert_library is not None and expert_path in expert_library:
        return expert_library[expert_path]

    if os.path.isdir(expert_path) and os.path.isfile(
        os.path.join(expert_path, WEIGHTS_NAME)
    ):
        return load_expert_from_hf_checkpoint(expert_path, expert_name)
    elif os.path.isdir(expert_path) and os.path.isfile(expert_path):
        return load_expert_from_pl_checkpoint(expert_path, expert_name)

    # this is not a local path, try to download from hub
    try:
        files = list_repo_files(expert_path)
    except:
        raise ValueError(
            f"Could not list {expert_path}, are you sure it's a HF repository?"
        )

    if "adapter_config.json" in files:
        # PEFT expert!
        return load_expert_from_peft_checkpoint(expert_path, expert_name)
    elif WEIGHTS_NAME in files:
        # MTTL expert trained with HFTrainer
        return load_expert_from_hf_checkpoint(expert_path, expert_name)
    elif CHECKPOINT_PATH_IN_HUB in files:
        # MTTL expert trained with PytorchLightning
        return load_expert_from_pl_checkpoint(expert_path, expert_name)


def load_expert_from_pl_checkpoint(
    expert_path: str,
    expert_name: str = None,
    **kwargs,
):
    import os

    from mttl.models.lightning.utils import download_from_hub

    if os.path.isfile(expert_path) or os.path.isdir(expert_path):
        expert_checkpoint = get_checkpoint_path(expert_path)
    else:
        expert_checkpoint = download_from_hub(expert_path)

    logger.info(f"Loading expert from {expert_checkpoint}...")
    expert_checkpoint = torch.load(
        expert_checkpoint, weights_only=False, map_location="cpu"
    )

    if "hyper_parameters" in expert_checkpoint:
        # this is a PL checkpoint
        if "tokenizer" in expert_checkpoint["hyper_parameters"]:
            del expert_checkpoint["hyper_parameters"]["tokenizer"]

        expert_info_data = expert_checkpoint.get("expert_info", {})

        # fix bug in checkpoints
        if "tokenizer" in expert_checkpoint["hyper_parameters"]:
            expert_checkpoint["hyper_parameters"].pop("tokenizer")

        if not expert_info_data.get("expert_config", None):
            expert_info_data["expert_config"] = expert_checkpoint["hyper_parameters"]
        else:
            if "tokenizer" in expert_info_data["expert_config"]:
                expert_info_data["expert_config"].pop("tokenizer")

        if not expert_info_data.get("expert_name", None):
            expert_info_data["expert_name"] = expert_checkpoint["hyper_parameters"][
                "expert_name"
            ]
        if not expert_info_data.get("expert_task_name", None):
            expert_info_data["expert_task_name"] = expert_checkpoint[
                "hyper_parameters"
            ]["finetune_task_name"]

        # back-compatibility, we removed this
        expert_info_data.pop("expert_embeddings", None)
        expert_info_data.pop("expert_scores", None)
        # in case of deepspeed checkpoints, the state_dict is named "module"
        expert_weights = (
            expert_checkpoint["state_dict"]
            if "state_dict" in expert_checkpoint
            else expert_checkpoint["module"]
        )
        expert_weights = {
            k.replace("model.", "", 1): v for k, v in expert_weights.items()
        }
        expert = Expert.fromdict(
            {
                "expert_info": expert_info_data,
                "expert_weights": expert_weights,
            }
        )
    else:
        expert = Expert.fromdict(expert_checkpoint)

    # override expert name
    if expert_name is not None:
        expert.name = expert_name
    return expert


def load_expert_from_hf_checkpoint(
    expert_path: str,
    expert_name: str = None,
    **kwargs,
):
    """Transforms an expert model trained with MTTL or PEFT into an Expert object."""
    # load the expert weights
    import os

    from mttl.models.expert_model import ExpertModel
    from mttl.models.hf.trainer import MTTL_ARGS_NAME

    logger.info(f"Loading expert from {expert_path}...")

    # gather mttl training arguments from the checkpoint if available
    mttl_args_file = os.path.join(expert_path, MTTL_ARGS_NAME)

    if os.path.isfile(mttl_args_file):
        training_config = torch.load(mttl_args_file, weights_only=False)
    else:
        training_config = None

    # we assume it's an expert model, which is used to train single experts
    expert: Expert = ExpertModel.from_pretrained(
        expert_path, device_map="cpu"
    ).as_expert(training_config=training_config)

    # override expert name
    if expert_name is not None:
        expert.name = expert_name
    return expert
