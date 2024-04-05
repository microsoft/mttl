import re
import torch
from typing import Dict, Union
import re
from string import Template
from mttl.models.modifiers.base import ModifierConfig
from mttl.models.modifiers.modify_model import CONFIGS_TO_MODIFIERS
from mttl.models.expert_config import ExpertConfig

from mttl.models.utils import download_from_hub
from mttl.models.modifiers.base import get_target_2_source_param_mapping
from mttl.utils import get_checkpoint_path, logger
from mttl.config import Config
from dataclasses import dataclass


@dataclass
class ExpertInfo:
    """
    Stuff that we want to save about experts but will never be passed from command line
    """

    expert_name: str
    expert_task_name: str = None
    parent_node: str = None
    # configuration for this expert, i.e. a modifier config
    expert_config: ModifierConfig = None
    # configuration with which the expert was trained, i.e. a training config
    training_config: ExpertConfig = None

    @classmethod
    def fromdict(cls, data):
        expert_config = None
        training_config = None

        if "expert_config" in data:
            try:
                expert_config = ModifierConfig.fromdict(data["expert_config"])
            except:
                # back-compatibility: in the previously stored checkpoints, expert_config was an object
                if isinstance(data["expert_config"], Config):
                    expert_config = ModifierConfig.from_training_config(
                        data["expert_config"]
                    )
                else:
                    # we are probably in the old format
                    training_config = Config.fromdict(data["expert_config"])
                    expert_config = ModifierConfig.from_training_config(training_config)

        if "training_config" in data:
            # convert it to the generic Config object
            training_config = ExpertConfig.fromdict(data["training_config"])

        if training_config is None:
            training_config = expert_config

        kwargs = {}
        for key in cls.__dataclass_fields__.keys():
            kwargs[key] = data.get(key, None)

        kwargs["expert_config"] = expert_config
        kwargs["training_config"] = training_config
        return cls(**kwargs)

    def asdict(self) -> Dict:
        data = {
            "expert_name": self.expert_name,
            "expert_task_name": self.expert_task_name,
            "parent_node": self.parent_node,
        }
        if self.expert_config:
            data["expert_config"] = self.expert_config.asdict()
        if self.training_config:
            data["training_config"] = self.training_config.asdict()
        return data

    @property
    def model(self):
        return self.training_config.model

    @property
    def dataset(self):
        return self.training_config.dataset

    @property
    def model_modifier(self):
        if self.expert_config is not None:
            return CONFIGS_TO_MODIFIERS.get(type(self.expert_config), None)
        return self.training_config.model_modifier


@dataclass
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
    def training_config(self) -> ExpertConfig:
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
    """Transforms a potentially lightning checkpoint into an Expert object."""
    # load the expert weights
    import os

    if expert_library is not None and expert_path in expert_library:
        return expert_library[expert_path]

    if os.path.isfile(expert_path) or os.path.isdir(expert_path):
        expert_checkpoint = get_checkpoint_path(expert_path)
    else:
        expert_checkpoint = download_from_hub(expert_path)

    logger.info(f"Loading expert from {expert_checkpoint}...")
    expert_checkpoint = torch.load(expert_checkpoint, map_location="cpu")

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

        expert_weights = expert_checkpoint["state_dict"]
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
