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
                if subclass != registry[name][0]:
                    raise ValueError(
                        f"Cannot register '{name}' multiple times for class '{cls.__name__}'."
                    )
                else:
                    return subclass

            logger.info(
                "Registering %s: adding %s as %s", cls.__name__, subclass.__name__, name
            )
            registry[name] = (subclass, config_cls)

            # add the registered name to the subclass as a field
            subclass.registered_name = name
            return subclass

        return add_to_registry

    @classmethod
    def get_config_class_by_name(cls, name: str) -> Type:
        result = Registrable._registry[cls].get(name)
        if result is None:
            raise ValueError(f"'{name}' is not registered for class '{cls.__name__}'.")
        _, config_cls = result
        if config_cls is None:
            raise ValueError(
                f"No config class associated with name '{name}' for class '{cls.__name__}'."
            )
        return config_cls

    @classmethod
    def get_class_by_name(cls, name: str) -> Type:
        result = Registrable._registry[cls].get(name)
        if result is None:
            raise ValueError(f"'{name}' is not registered for class '{cls.__name__}'.")
        subclass, _ = result
        return subclass

    @classmethod
    def get_name_by_config_class(cls, config_class: Type) -> str:
        for key, (subclass, config_cls) in Registrable._registry[cls].items():
            if config_cls == config_class:
                return key
        raise ValueError(
            f"Config class '{config_class}' is not registered for class '{cls.__name__}'."
        )

    @classmethod
    def get_class_by_config_class(cls, config_class: Type) -> Type:
        name = cls.get_name_by_config_class(config_class)
        return cls.get_class_by_name(name)

    @classmethod
    def registered_names(cls) -> List[str]:
        return list(Registrable._registry[cls].keys())

    @classmethod
    def registered_configs(cls) -> List[Type]:
        return [
            config_cls
            for _, config_cls in Registrable._registry[cls].values()
            if config_cls is not None
        ]
