import dataclasses
import importlib
from dataclasses import dataclass
from typing import Any, Dict, List, Type


class SerializableConfig:
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self.asdict() == other.asdict()

    @classmethod
    def fromdict(cls, data: Dict[str, Any]):
        data_ = {}
        for field in dataclasses.fields(cls):
            # handle the case of a config
            if hasattr(field.type, "fromdict"):
                data_[field.name] = field.type.fromdict(data[field.name])
            # handle the case of a list of configs
            elif field.type == List and hasattr(field.type.__args__[0], "fromdict"):
                for i, value in enumerate(data[field.name]):
                    data[field.name][i] = field.type.__args__[0].fromdict(value)
            # simple value
            else:
                data_[field.name] = data.get(field.name, field.default)
        return cls(**data_)

    def asdict(self) -> Type:
        data = {}
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if value is not None and hasattr(value, "asdict"):
                data[field.name] = value.asdict()
            elif field.type == List and hasattr(field.type.__args__[0], "asdict"):
                for i, inner_value in enumerate(value):
                    data[field.name][i] = inner_value.asdict()
            else:
                data[field.name] = value
        data["class_name"] = f"{self.__module__}.{self.__class__.__name__}"
        return data


@dataclass
class AutoConfig(SerializableConfig):
    @classmethod
    def fromdict(cls, data: Dict[str, Any]):
        # try to infer the class from the data
        class_name = data.pop("class_name", None)
        if not class_name:
            raise ValueError(
                "`class_name` is missing from the data provided. Cannot use `AutoModelConfig.fromdict`"
            )

        dataclass_cls = AutoConfig.dynamic_class_resolution(class_name)
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
