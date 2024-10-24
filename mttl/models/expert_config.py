import json
import os
from dataclasses import dataclass
from typing import Any, Union

from huggingface_hub import hf_hub_download

from mttl.logging import logger
from mttl.serializable import AutoSerializable, Serializable

CONFIG_NAME = "mttl_config.json"


@dataclass
class BaseExpertModelConfig(Serializable):
    base_model: str = None

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


class AutoModelConfig(AutoSerializable, BaseExpertModelConfig):
    pass
