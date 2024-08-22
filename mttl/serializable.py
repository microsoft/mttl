import dataclasses
import importlib
from dataclasses import dataclass
from typing import Any, Dict, List, Type


@dataclass
class Serializable:
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self.asdict() == other.asdict()

    @classmethod
    def fromdict(cls, data: Dict[str, Any]) -> Type:
        from copy import deepcopy
        from typing import get_args, get_origin

        data = deepcopy(data)
        for field in dataclasses.fields(cls):
            if field.name not in data and field.default is dataclasses.MISSING:
                raise ValueError(
                    f"Required {field.name} is missing from the data provided."
                )
            elif field.name not in data:
                # field is not in data
                continue

            value = data[field.name]
            if value is None:
                continue

            # handle the case of a config
            if hasattr(field.type, "fromdict"):
                data[field.name] = field.type.fromdict(data[field.name])
            # handle the case of a list of configs
            elif get_origin(field.type) == list and hasattr(
                get_args(field.type)[0], "fromdict"
            ):
                data[field.name] = [
                    get_args(field.type)[0].fromdict(value)
                    for value in data[field.name]
                ]
            # simple value
            elif get_origin(field.type) == dict and hasattr(
                get_args(field.type)[0], "asdict"
            ):
                for k, v in value.items():
                    data[field.name][k] = v.asdict()
            else:
                data[field.name] = data.get(field.name, field.default)
        return cls(**data)

    @classmethod
    def from_dict(cls, data) -> Type:
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
        from typing import get_args, get_origin

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
                continue

            if hasattr(value, "asdict"):
                data[field.name] = value.asdict()
            elif get_origin(field.type) == list and hasattr(
                get_args(field.type)[0], "asdict"
            ):
                data[field.name] = []
                for inner_value in value:
                    data[field.name].append(inner_value.asdict())
            elif get_origin(field.type) == dict and hasattr(
                get_args(field.type)[0], "asdict"
            ):
                data[field.name] = {}
                for k, v in value.items():
                    data[field.name][k] = v.asdict()
            else:
                data[field.name] = value

        data["class_name"] = f"{self.__module__}.{self.__class__.__name__}"
        return data


@dataclass
class AutoSerializable(Serializable):
    @classmethod
    def fromdict(cls, data: Dict[str, Any]):
        # try to infer the class from the data
        class_name = data.pop("class_name", None)
        if not class_name:
            raise ValueError(
                f"`class_name` is missing from the data provided. Cannot use `{cls.__name__}.fromdict`."
            )

        dataclass_cls = AutoSerializable.dynamic_class_resolution(class_name)
        return dataclass_cls.fromdict(data)

    @staticmethod
    def dynamic_class_resolution(class_name: str) -> Any:
        try:
            # First, attempt to import using the original module path
            module_name, class_name = class_name.rsplit(".", 1)
            module = importlib.import_module(module_name)
            return getattr(module, class_name)
        except (ModuleNotFoundError, AttributeError):
            import sys

            # If it fails, try to find the class in any loaded module
            for module in sys.modules.values():
                if hasattr(module, class_name):
                    return getattr(module, class_name)
            raise ImportError(f"Cannot find class {class_name}")
