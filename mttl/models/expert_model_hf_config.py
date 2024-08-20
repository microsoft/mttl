import dataclasses
import importlib
import inspect
import json
import math
import os
import re
import threading
from collections import defaultdict
from dataclasses import MISSING, asdict, dataclass
from functools import partial
from typing import Any, Dict, List, Optional, Type, Union

from huggingface_hub import hf_hub_download

from mttl.configuration import AutoConfig, SerializableConfig
from mttl.logging import logger

CONFIG_NAME = "mttl_config.json"


@dataclass
class BaseExpertModelConfig(SerializableConfig):
    base_model: str

    def save_pretrained(self, save_directory, **kwargs):
        """Bare bone save pretrained function that saves the model and config."""
        if os.path.isfile(save_directory):
            logger.error(
                f"Provided path ({save_directory}) should be a directory, not a file"
            )
            return

        os.makedirs(save_directory, exist_ok=True)
        output_config_file = os.path.join(save_directory, CONFIG_NAME)

        with open(output_config_file, "w") as f:
            data = self.asdict()
            f.write(json.dumps(data))
        return output_config_file

    @classmethod
    def from_pretrained(
        cls,
        model_id: Union[str, os.PathLike],
        **kwargs: Any,
    ):
        if os.path.isfile(os.path.join(model_id, CONFIG_NAME)):
            config_file = os.path.join(model_id, CONFIG_NAME)
        else:
            try:
                config_file = hf_hub_download(model_id, CONFIG_NAME)
            except Exception as exc:
                raise ValueError(f"Can't find {CONFIG_NAME} at '{model_id}'") from exc

        with open(config_file, "r") as f:
            config = cls.fromdict(json.load(f))
        return config


class AutoModelConfig(AutoConfig, BaseExpertModelConfig):
    pass
