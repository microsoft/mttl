import functools
import threading
from abc import ABC
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Union

import torch
from pyparsing import abstractmethod
from torch import nn

from mttl.logging import logger
from mttl.models.containers.selectors.selector_output import (
    BatchExpertsAndWeightsSelectorOutput,
    BatchExpertsSelectorOutput,
    BatchExpertsSplitsAndWeightsSelectorOutput,
    BatchSequenceExpertsAndWeightsSelectorOutput,
    ExpertsAndWeightsSelectorOutput,
    ExpertsSplitsAndWeightsSelectorOutput,
    SelectorOutput,
)
from mttl.models.library.expert import ExpertInfo
from mttl.models.modifiers.base import Modifier
from mttl.models.ranker.adapter_ranker import AdapterRankerHelper
from mttl.models.ranker.classifier_ranker import ClusterPredictor
from mttl.models.utils import MetricLogger
from mttl.registrable import Registrable
from mttl.serializable import AutoSerializable, Serializable

EPS = 1e-8


def get_selector(selector_config: "SelectorConfig", **kwargs):
    """Returns a selector object for the given routing_config."""
    return Selector.get_class_by_config_class(selector_config.__class__)(
        config=selector_config, **kwargs
    )


@dataclass
class SelectorConfig(Serializable):
    # the granularity of the selector (which layers use the same selectors)
    router_granularity: str = "*"
    lora_merge_after: bool = False
    selector_logging: bool = True
    num_experts: int = 0

    def __eq__(self, other):
        # compare all the attributes
        return self.__dict__ == other.__dict__

    @property
    def selector_name(self):
        return Selector.get_name_by_config_class(type(self))

    @classmethod
    def from_training_config(
        cls,
        training_config: Union["Config", "SelectorConfig"],
    ) -> Union["SelectorConfig", None]:
        """Build modifier config from the training config.

        Returns None if no modifier is set.
        """
        from mttl.arguments import create_config_class_from_args

        if isinstance(training_config, SelectorConfig):
            # nothing to do here
            return training_config

        if cls == SelectorConfig:
            # if called on the base class, we need to find the correct subclass
            if training_config.router_selector is None:
                return None

            if training_config.router_selector not in Selector.registered_names():
                raise ValueError(
                    f"Selector '{training_config.router_selector}' not found, has it been registered?"
                )

            config_klass = Selector.get_config_class_by_name(
                training_config.router_selector
            )
        else:
            config_klass = cls

        return create_config_class_from_args(config_klass, training_config)


class AutoSelectorConfig(AutoSerializable):
    pass


@dataclass
class MultiSelectorConfig(Serializable):
    selectors: Dict[str, AutoSelectorConfig] = field(default_factory=dict)

    def __len__(self):
        return len(self.selectors)

    def __getitem__(self, key):
        return self.selectors[key]

    def get(self, key):
        return self.selectors.get(key, TaskNameSelectorConfig())

    def keys(self):
        return self.selectors.keys()

    def values(self):
        return self.selectors.values()

    def items(self):
        return self.selectors.items()

    def __setitem__(self, key, value):
        self.selectors[key] = value

    @property
    def selector_name(self):
        import json

        return json.dumps({k: v.selector_name for k, v in self.selectors.items()})

    @classmethod
    def from_training_config(
        cls,
        training_config: "Config",
    ) -> Union[SelectorConfig, "MultiSelectorConfig"]:
        import copy
        import json

        if training_config.router_selector is None:
            return None

        try:
            router_selector = json.loads(training_config.router_selector)
        except:
            # if not a json, assume it's a single selector
            return SelectorConfig.from_training_config(training_config)

        selector_configs = cls()
        for modifier_name, selector_name in router_selector.items():
            config_clone = copy.deepcopy(training_config)
            config_clone.router_selector = selector_name

            selector_configs.selectors[modifier_name] = (
                SelectorConfig.from_training_config(config_clone)
            )
        return selector_configs


class SelectorsCache:
    """Keep a cache of all added selectors indexed by both modifier and selector name."""

    def __init__(self):
        self.cache = defaultdict(dict)
        self.clear()

    def clear(self, modifier_name: str = None):
        # initialize cache for all registered modifiers
        if modifier_name is None:
            for modifier_name in Modifier.registered_names():
                self.cache[modifier_name] = {}
        else:
            self.cache[modifier_name] = {}

    def insert(self, modifier_name: str, selector_name: str, selector: "Selector"):
        self.cache[modifier_name][selector_name] = selector

    def get(
        self, modifier_name: str, selector_name: str = None
    ) -> Union["Selector", Dict]:
        if selector_name is None:
            return self.cache[modifier_name]
        return self.cache[modifier_name].get(selector_name, None)

    def keys(self):
        return self.cache.keys()

    def items(self):
        return iter(self.cache.items())

    def __setitem__(self, key, value):
        if key not in Modifier.registered_names():
            raise ValueError(f"Modifier '{key}' not found, has it been registered?")

        self.cache[key] = value


@dataclass
class LoadableSelectorConfig(SelectorConfig):
    """Adds support for library_id and data_id, which specifies the unique identifier to load."""

    library_id: str = None
    selector_data_id: str = None

    @property
    def artifacts_hash(self):
        """Returns an unique key identifying the artifacts for this selector."""
        return f"{self.library_id}_{self.selector_data_id}"


def forward_with_cache(func):
    def wrapper(self: Selector, input, **kwargs):
        if self.forward_cache is not None and not self.clear_cache:
            self.count_call()
            return self.forward_cache

        output = func(self, input, **kwargs)
        self.forward_cache = output
        self.count_call()
        return output

    return wrapper


def safe_logging(func):
    def wrapper(selector, *args, **kwargs):
        if not selector.config.selector_logging:
            return None
        try:
            result = func(selector, *args, **kwargs)
        except Exception as e:
            if str(e)[:100] != getattr(logger, "previous_error", ""):
                logger.exception(f"An error occurred in {func.__name__}: {e}")
                logger.previous_error = str(e)[:100]
            result = None
        return result

    return wrapper


def artifacts_cache(func):
    cache = {}
    lock = threading.Lock()

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # wrapper expects
        key = args[1].artifacts_hash

        with lock:
            if key in cache:
                return cache[key]
            result = func(*args, **kwargs)
            cache[key] = result
        return result

    return wrapper


class Selector(nn.Module, Registrable):
    metric_logger: MetricLogger = MetricLogger()

    def __init__(self, config=None, **kwargs):
        nn.Module.__init__(self)

        self.config = config
        self.expert_infos = {}
        self.selector_views = []
        self.forward_cache = None
        self.default_expert_name = None
        self.total_calls_per_forward = 0
        self._calls_counter = 0
        self._task_to_expert_name = {}
        # dependency injection filled from ExpertContainer
        self.__layer_name__ = None
        self.device = None

    @property
    def expert_names(self) -> list:
        return list(self.expert_infos.keys())

    @property
    def clear_cache(self):
        reset_cache = self._calls_counter >= self.total_calls_per_forward
        if reset_cache:
            self._calls_counter = 0
        return reset_cache

    def count_call(self):
        self._calls_counter += 1

    @abstractmethod
    def forward(self, input, **kwargs) -> SelectorOutput:
        pass

    def create_view(self) -> "SelectorView":
        self.selector_views.append(SelectorView(self))
        return self.selector_views[-1]

    @property
    def views(self):
        return self.selector_views

    @abstractmethod
    def get_merging_weights(self, **selector_kwargs) -> Dict:
        """
        Returns a dictionary of the form {expert_name: weight} for each expert in the container.
        raises ValueError if not supported, e.g. because routing depends on the input x.
        """
        pass

    @property
    def layer_name(self):
        if not hasattr(self, "__layer_name__"):
            raise ValueError(
                "Layer name not available, dependency injection not done properly?"
            )

        return self.__layer_name__

    @property
    def task_to_expert_name(self):
        return getattr(self, "_task_to_expert_name", {})

    @property
    def n_experts(self):
        return len(self.expert_names)

    @property
    def info_container(self):
        from mttl.models.expert_context import InfoContainer

        return InfoContainer.get()

    @property
    def routing_infos(self):
        info_container = self.info_container
        if not info_container:
            return None
        return info_container.routing_infos

    @abstractmethod
    def on_add_expert(
        self, expert_name: str, expert_info: ExpertInfo = None, is_default=False
    ):
        pass

    def add_expert(
        self, expert_name: str, expert_info: ExpertInfo = None, is_default=False
    ):
        if expert_info is None or expert_info.expert_task_name is None:
            logger.debug(
                "Expert's task_name not set, assume task name corresponds to expert name!"
            )
            self._task_to_expert_name[expert_name] = expert_name
        else:
            for task_name in expert_info.expert_task_name.split(","):
                if task_name in self._task_to_expert_name:
                    logger.warning(
                        f"Task name {task_name} already assigned to expert {self._task_to_expert_name[task_name]}"
                    )
                self._task_to_expert_name[task_name] = expert_name

        # standard bookkeeping for all selectors
        if is_default:
            self.default_expert_name = expert_name

        self.expert_infos[expert_name] = expert_info

        # call custom logic for add expert
        self.on_add_expert(expert_name, expert_info, is_default)


class SelectorView:
    """A view on a selector that allows it to call forward but doesn't act on add_expert.

    This is because add expert is to be called only on the main instance of this selector
    and not on the multiple views across the network.
    """

    def __init__(self, selector_instance):
        self.selector_instance = selector_instance

    @property
    def config(self):
        return self.selector_instance.config

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.selector_instance.forward(*args, **kwargs)

    def get_merging_weights(self, **selector_kwargs) -> Dict:
        return self.selector_instance.get_merging_weights(**selector_kwargs)

    def add_expert(
        self,
        expert_name: str,
        expert_info: ExpertInfo = None,
        is_default=False,
        **kwargs,
    ):
        pass


@dataclass
class TaskPredictorSelectorConfig(SelectorConfig):
    ranker_path: str = None
    ranker_model: str = None
    ranker_top_k: int = 1


@Selector.register("task_predictor_selector", TaskPredictorSelectorConfig)
class TaskPredictorSelector(Selector):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.ranker_top_k = self.config.ranker_top_k
        # get the routing model

        self.expert_ranker = AdapterRankerHelper.get_ranker_instance(
            ranker_model=self.config.ranker_model,
            ranker_path=self.config.ranker_path,
        )

        if isinstance(self.expert_ranker, ClusterPredictor):
            self.expert_ranker.init_clusters(kwargs["training_config"].library_id)

    @forward_with_cache
    def forward(self, input, **kwargs) -> BatchExpertsAndWeightsSelectorOutput:
        # get the sources_texts from routing_infos
        routing_infos = self.routing_infos

        if hasattr(routing_infos, "sources_texts"):
            sources_texts = routing_infos.sources_texts
            self.expert_ranker.set_available_tasks(self.expert_names)

            experts, weights = self.expert_ranker.predict_task(
                sources_texts, n=self.ranker_top_k
            )
            logger.debug(f"Predicted tasks: {experts} with weights {weights}")

            weights = torch.tensor(weights, device=input.device)
            return BatchExpertsAndWeightsSelectorOutput(experts, weights)
        else:
            raise ValueError(
                "Routing infos does not contain sources_texts, cannot predict tasks."
            )

    def get_merging_weights(self, **selector_kwargs) -> Dict:
        raise ValueError(
            f"Not supported for {self.__class__} since routing depends on input."
        )

    def on_add_expert(
        self, expert_name: str, expert_info: ExpertInfo = None, is_default=False
    ):
        pass


class LoadableLibraryMixin(ABC):
    @classmethod
    @abstractmethod
    def load_from_library(cls, config):
        pass


def get_expert_prototype_from_library_artifacts(
    expert_name: str, layer_name: str, library_artifacts: Dict
) -> torch.Tensor:
    """Utils function that returns the expert prototype stored in the library.

    This is used by Arrow, PhatGoose and Avg Activation selectors.
    """
    import numpy as np

    patched_layer_name = layer_name.replace(".selector", "")

    if expert_name not in library_artifacts:
        raise ValueError(
            f"Cannot load prototypes for expert `{expert_name}`, was not found in library.\n"
            f"Please recompute selector prototypes with the correct library transform."
        )

    layer_names = library_artifacts[expert_name].keys()
    valid_layer_names = [
        k
        for k in layer_names
        if patched_layer_name in k  # k.startswith(patched_layer_name)
    ]

    key = sorted(valid_layer_names)[0]
    proto = library_artifacts[expert_name][key]
    if isinstance(proto, np.ndarray):
        proto = torch.from_numpy(proto)
    return proto


@dataclass
class TaskNameSelectorConfig(SelectorConfig):
    pass


@Selector.register("task_selector", TaskNameSelectorConfig)
class TaskNameSelector(Selector):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    @forward_with_cache
    def forward(self, input, **kwargs) -> BatchExpertsSelectorOutput:
        if not self.routing_infos or not self.routing_infos.task_names:
            # try to infer batch size
            if "input_ids" in kwargs:
                batch_size = kwargs["input_ids"].size(0)
            else:
                batch_size = input.shape[0]

            if not self.default_expert_name:
                raise ValueError("No default expert name set and no task names given!")

            experts = [self.default_expert_name for _ in range(batch_size)]
        else:
            task_names = self.routing_infos.task_names

            if (
                all(
                    task_name not in self.task_to_expert_name
                    for task_name in task_names
                )
                and not self.default_expert_name
                and len(self.task_to_expert_name)
            ):
                raise ValueError(
                    "Experts for all tasks have not been loaded! Set a default expert?"
                )
            experts = [
                self.task_to_expert_name.get(task_name, self.default_expert_name)
                for task_name in task_names
            ]

        return BatchExpertsSelectorOutput(experts)

    def get_merging_weights(self, **selector_kwargs) -> Dict:
        raise NotImplementedError(
            "Not required for TaskNameSelector as it performs hard selection. Use 'get_expert_instance' instead."
        )


@dataclass
class UniformSelectorConfig(SelectorConfig):
    pass


@Selector.register("uniform_selector", UniformSelectorConfig)
class UniformSelector(Selector):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    @forward_with_cache
    def forward(self, input, **kwargs) -> BatchExpertsSelectorOutput:
        return ExpertsAndWeightsSelectorOutput(
            experts=self.expert_names,
            weights=torch.ones(
                len(self.expert_names), device=input.device, dtype=input.dtype
            )
            / len(self.expert_names),
        )
