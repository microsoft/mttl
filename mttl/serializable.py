import dataclasses
import importlib
from dataclasses import dataclass
from typing import Any, Dict

from mttl.logging import logger


@dataclass
class Serializable:
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self.asdict() == other.asdict()

    @classmethod
    def fromdict(cls, data: Dict[str, Any]) -> "Serializable":
        from typing import get_args, get_origin

        data_ = {}
        for field in dataclasses.fields(cls):
            if field.name not in data:
                if (
                    field.default is dataclasses.MISSING
                    and field.default_factory is dataclasses.MISSING
                ):
                    raise ValueError(
                        f"Required {field.name} is missing from the data provided."
                    )
                continue

            field_type = field.type

            if type(field_type) == str:
                field_type = AutoSerializable.dynamic_class_resolution(field_type)
                logger.debug(f"Resolved {field.name} to {field_type}.")

            value = data[field.name]
            if value is None:
                data_[field.name] = None
            # handle the case of a config
            elif hasattr(field_type, "fromdict"):
                data_[field.name] = field_type.fromdict(value)
            # handle the case of a list of configs
            elif get_origin(field_type) == list and hasattr(
                get_args(field_type)[0], "fromdict"
            ):
                data_[field.name] = [get_args(field_type)[0].fromdict(v) for v in value]
            # handle the case of a dict of configs
            elif get_origin(field_type) == dict and hasattr(
                get_args(field_type)[0], "asdict"
            ):
                data_[field.name] = {
                    k: get_args(field_type)[0].fromdict(v) for k, v in value.items()
                }
            # simple value
            else:
                data_[field.name] = value
        return cls(**data_)

    def to_json_string(self) -> str:
        import json

        return json.dumps(self.asdict())

    def from_json_string(self, str) -> str:
        import json

        return self.from_dict(json.loads(str))

    def from_json(self, str) -> str:
        return self.from_json_string(str)

    def to_json(self) -> str:
        return self.to_json_string()

    @classmethod
    def from_dict(cls, data) -> "Serializable":
        return cls.fromdict(data)

    def to_dict(self) -> Dict[str, Any]:
        return self.asdict()

    def asdict(self, skip_fields=None) -> Dict[str, Any]:
        """Serialize the config to a dictionary.

        Args:
            skip_fields: List of fields to skip when serializing the config.

        Returns:
            A dictionary representation of the config.
        """

        data = {}
        for field in dataclasses.fields(self):
            if skip_fields and field.name in skip_fields:
                if (
                    field.default is dataclasses.MISSING
                    and field.default_factory is dataclasses.MISSING
                ):
                    raise ValueError("Cannot skip required field in dataclass!")
                continue

            value = getattr(self, field.name)
            if value is None:
                data[field.name] = None
            elif hasattr(value, "asdict"):
                data[field.name] = value.asdict()
            elif isinstance(value, list) and all(hasattr(v, "asdict") for v in value):
                data[field.name] = [v.asdict() for v in value]
            elif isinstance(value, dict) and all(
                hasattr(k, "asdict") for k in value.keys()
            ):
                data[field.name] = {k: v.asdict() for k, v in value.items()}
            else:
                data[field.name] = value
            data["class_name"] = (
                f"{self.__class__.__module__}.{self.__class__.__name__}"
            )
        return data


@dataclass
class AutoSerializable:
    @classmethod
    def fromdict(cls, data: Dict[str, Any]):
        # try to infer the class from the data
        class_name = data.pop("class_name", None)
        if not class_name:
            raise ValueError(
                f"`class_name` is missing from the data provided. Cannot use `{cls.__name__}.fromdict`."
            )

        dataclass_cls: Serializable = AutoSerializable.dynamic_class_resolution(
            class_name
        )
        if not issubclass(dataclass_cls, Serializable):
            raise ValueError(f"Class {class_name} is not a subclass of Serializable")

        return dataclass_cls.fromdict(data)

    @staticmethod
    def dynamic_class_resolution(class_name: str) -> Any:
        try:
            # First, attempt to import using the original module path
            module_name, class_name = class_name.rsplit(".", 1)
            module = importlib.import_module(module_name)
            return getattr(module, class_name)
        except (ModuleNotFoundError, AttributeError, ValueError):
            import sys

            # If it fails, try to find the class in any loaded module
            for module in sys.modules.values():
                if hasattr(module, class_name):
                    return getattr(module, class_name)
            raise ImportError(f"Cannot find class {class_name}")
