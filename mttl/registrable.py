from collections import defaultdict
from typing import Any, Callable, List, Optional, Type

from mttl.logging import logger


class Registrable:
    _registry = defaultdict(dict)

    @classmethod
    def register(cls, name: str, config_cls: Optional[Any] = None) -> Callable:
        registry = Registrable._registry[cls]

        def add_to_registry(subclass):
            if name in registry:
                raise ValueError(f"Cannot register {name} multiple times.")

            logger.info(
                "Registering %s: adding %s as %s", cls.__name__, subclass.__name__, name
            )
            registry[name] = (subclass, config_cls)
            return subclass

        return add_to_registry

    @classmethod
    def get_config_class_by_name(cls, name: str) -> Type:
        subclass, config_cls = Registrable._registry[cls].get(name)
        return config_cls

    @classmethod
    def get_class_by_name(cls, name: str) -> Type:
        subclass, config_cls = Registrable._registry[cls].get(name)
        return subclass

    @classmethod
    def get_name_by_config_class(cls, config_class: Type) -> str:
        for key, value in Registrable._registry[cls].items():
            subclass, config_cls = value
            if config_cls == config_class:
                return key

    @classmethod
    def get_class_by_config_class(cls, config_class: Type) -> Type:
        return cls.get_class_by_name(cls.get_name_by_config_class(config_class))

    @classmethod
    def registered_names(cls) -> List[str]:
        return list(Registrable._registry[cls].keys())

    @classmethod
    def registered_configs(cls) -> List[Type]:
        return [
            config
            for _, config in Registrable._registry[cls].values()
            if config is not None
        ]
