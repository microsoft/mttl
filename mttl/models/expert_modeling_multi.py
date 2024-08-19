import re
import threading
from dataclasses import dataclass
from functools import partial
from typing import Dict, List, Union

import torch

from mttl.logging import logger
from mttl.models.containers import add_expert_to_transformer
from mttl.models.containers.base import ExpertContainer
from mttl.models.containers.selectors.base import (
    AutoSelectorConfig,
    LoadableSelectorConfig,
    Selector,
    SelectorConfig,
    SelectorsCache,
)
from mttl.models.expert_configuration import BaseExpertModelConfig
from mttl.models.expert_modeling_base import BaseExpertModel
from mttl.models.library.expert import Expert, ExpertInfo
from mttl.models.library.expert_library import ExpertLibrary
from mttl.models.modifiers.base import ModifierConfig
from mttl.models.modifiers.modify_model import modify_transformer
from mttl.models.utils import model_loader_helper


@dataclass
class MultiExpertModelConfig(BaseExpertModelConfig):
    default_expert_name: str = None
    expert_infos: List[ExpertInfo] = None
    selector_config: AutoSelectorConfig = None


@BaseExpertModel.register("multi_expert_model", config_cls=MultiExpertModelConfig)
class MultiExpertModel(BaseExpertModel):
    """Adds all functions and properties for a multi-expert model."""

    def __init__(self, config, **loading_kwargs):
        super().__init__(config, **loading_kwargs)

        # config about the routing
        self.selector_config = config.selector_config
        self.selector_cache = SelectorsCache()
        self.experts_infos = {}

        if self.config.expert_infos is not None:
            for expert_info in self.config.expert_infos:
                expert_info: ExpertInfo
                self.add_empty_expert(
                    expert_info.expert_name,
                    expert_info.expert_config,
                )

            if self.config.default_expert_name:
                self.set_default_expert(self.config.default_expert_name)

    @property
    def experts_names(self):
        return list(self.experts_infos.keys())

    def set_default_expert(self, expert_name):
        """Propagate default expert to all containers that contain it."""
        if expert_name not in self.experts_infos:
            raise ValueError(f"Expert {expert_name} not found in the model.")

        for container in self.experts_containers:
            if expert_name in container.expert_infos:
                container.default_expert_name = expert_name

    @classmethod
    def from_pretrained_library(
        cls,
        library_id: Union[str, ExpertLibrary],
        selector_config: Union[SelectorConfig, Dict[str, SelectorConfig]] = None,
        remote_token: str = None,
        default_expert_name: str = None,
        **loading_kwargs,
    ):
        from copy import deepcopy

        if not isinstance(library_id, ExpertLibrary):
            library = ExpertLibrary.get_expert_library(
                repo_id=library_id,
                token=remote_token,
            )
            repo_id = library_id
        else:
            library = library_id
            repo_id = library_id.uri

        # get a config file from the library, and initialize the expert model
        an_expert = library[next(iter(library.keys()))]

        # set selector for the added experts
        if selector_config is not None:
            if isinstance(selector_config, LoadableSelectorConfig):
                selector_config.library_id = repo_id

            elif isinstance(selector_config, dict):
                for modifier_name, cfg in selector_config.items():
                    # inject the library id if it is None
                    if (
                        isinstance(cfg, LoadableSelectorConfig)
                        and cfg.library_id is None
                    ):
                        cfg.library_id = repo_id
        else:
            logger.info("No selector config provided, assuming expert name selector!")

        config = MultiExpertModelConfig(
            an_expert.expert_info.model,
            default_expert_name=default_expert_name,
            selector_config=selector_config,
        )
        model = cls(config, **loading_kwargs)
        model.add_experts_from_library(library)
        return model

    @property
    def lock(self):
        if not hasattr(self, "_lock"):
            self._lock = threading.Lock()
        return self._lock

    @property
    def experts_containers(self) -> List[ExpertContainer]:
        containers = []
        for _, module in self.model.named_modules():
            for _, child in dict(module.named_children()).items():
                if isinstance(child, ExpertContainer) and len(child.experts) > 0:
                    containers.append(child)
        return containers

    @property
    def selectors(self) -> Dict[str, List[Selector]]:
        return {
            key: list(self.selector_cache.get(key).values())
            for key in self.selector_cache.keys()
        }

    def delete_expert_container(self):
        """
        Replaces the expert container with the expert with the given name.
        """
        for _, module in self.model.named_modules():
            for c_name, child in dict(module.named_children()).items():
                if isinstance(child, ExpertContainer) and len(child.experts) > 0:
                    setattr(module, c_name, child.layer)

        self.selector_cache.clear()
        self.experts_infos.clear()

    def add_experts_from_library(self, library):
        import concurrent.futures

        import tqdm

        def add_module(self, module_name):
            expert_dump = library[module_name]
            self.add_expert_instance(expert_dump)

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            # Create a list to hold the futures
            futures = []
            for element in library.keys():
                futures.append(executor.submit(partial(add_module, self), element))

            # Progress bar setup
            with tqdm.tqdm(
                total=len(library), desc="Adding experts...", unit="expert"
            ) as progress_bar:
                for result in concurrent.futures.as_completed(futures):
                    # raise exception
                    if result.exception():
                        raise result.exception()
                    progress_bar.update(1)

    def add_experts_from_dict(self, experts_dict, action="route"):
        for expert_name, expert_dump in experts_dict.items():
            self.add_expert_instance(expert_dump, expert_name, action=action)

    def add_empty_expert(
        self,
        expert_name,
        expert_config=None,
        is_default=False,
    ) -> Expert:
        """Adds a new empty expert to the model."""
        new_expert = Expert(
            expert_info=ExpertInfo(
                expert_name,
                expert_config=expert_config,
                expert_model=self.hparams.model,
                training_config=self.training_config,
            ),
        )

        new_expert = self.add_expert_instance(new_expert, is_default=is_default)
        logger.info("Added empty expert: {}".format(expert_name))
        return new_expert

    def load_expert(
        self,
        expert_path: str,
        expert_name: str = None,
        action: str = "merge",
        is_default: bool = False,
        expert_library: ExpertLibrary = None,
    ):
        from mttl.models.library.expert import load_expert

        expert = load_expert(
            expert_path,
            expert_name=expert_name,
            expert_library=expert_library,
        )

        if self.hparams.model != expert.training_config.model:
            raise ValueError(
                "The expert has been trained on top of a different model!"
                " Detected: {} - Expected: {}".format(
                    expert.training_config.model, self.hparams.model
                )
            )

        logger.info(
            f"Adding expert with name {expert.name}... with action ... {action}!"
        )
        self.add_expert_instance(expert, action=action, is_default=is_default)

    def _get_selector_config(self, model_modifier: str) -> SelectorConfig:
        if not self.selector_config:
            return None
        if isinstance(self.selector_config, dict):
            return self.selector_config.get(model_modifier)
        else:
            return self.selector_config

    def add_expert_instance(
        self,
        expert_instance: Expert,
        expert_name=None,
        action="route",
        is_default=False,
    ) -> Expert:
        """
        If action is merge, then the expert is merged with the existing model, and we return None.
        """
        if expert_name is not None:
            # we want to load expert instance with a given name (might be different from the one in the expert instance)
            # we dont want to change expert instance though!
            # will create a copy for now (maybe safer), alternatively can change the name and set it back at the end of the function
            expert_instance = expert_instance.clone()
            expert_instance.name = expert_name

        with self.lock:
            modifier_name = expert_instance.expert_config.modifier_name
            selector_config = self._get_selector_config(modifier_name)

            add_expert_to_transformer(
                self.model,
                expert_instance,
                action=action,
                is_default=is_default,
                selector_config=selector_config,
                selector_cache=self.selector_cache,
            )

            if action != "merge":
                self.experts_infos[expert_instance.name] = expert_instance.expert_info
                # reload the expert instance to fill the weights properly if this was an empty expert
                expert_instance = self.get_expert_instance(expert_instance.name)
            return expert_instance

    def set_selector(
        self,
        modifier_name: str,
        selector_config: SelectorConfig,
    ):
        from mttl.models.containers import replace_selector_for_container

        n_selectors, n_selectors_views = replace_selector_for_container(
            self.model,
            modifier_name,
            selector_config,
            self.selector_cache,
            force_replace=True,
        )
        logger.info(
            "Created {} selectors and {} views.".format(n_selectors, n_selectors_views)
        )

    def extract_parameters(self, p_name_pattern=".*lora.*"):
        """
        Extracts task embeddings for parameters matching the given pattern.

        Args:
            p_name_pattern (str, optional): Regular expression pattern to match parameter names.
                Defaults to ".*lora.*".

        Returns:
            torch.Tensor: Concatenated tensor of task embeddings for the matched parameters.
        """
        para_list = []
        for name, param in self.model.named_parameters():
            if re.fullmatch(p_name_pattern, name):
                para_list.append(param.reshape(-1))
        return torch.cat(para_list)

    def get_expert_instance(self, expert_name):
        """
        Retrieves an instance of the specified expert from the model.

        Args:
            expert_name (str): The name of the expert to retrieve.
            silent (bool, optional): If True, suppresses the ValueError exception when the expert is not found.
                Defaults to True.

        Returns:
            expert: An instance of the specified expert.

        Raises:
            AssertionError: If the expert name is not found in the model, if no expert containers are found,
                or if the expert names are not unique.
            ValueError: If the expert is not found in the model and silent is False.
        """
        assert (
            expert_name in self.experts_names
        ), f"Expert {expert_name} not found in the model."
        assert (
            len(self.experts_containers) > 0
        ), "No expert containers found in the model."
        assert len(set(self.experts_names)) == len(
            self.experts_names
        ), "Expert names are not unique."

        expert_params = {}
        for container in self.experts_containers:
            if expert_name in container.expert_infos:
                expert_info = container.expert_infos[expert_name]
                expert_weights = container[expert_name].state_dict()
                expert_weights = {
                    f"{container.layer_name}.{k}": v for k, v in expert_weights.items()
                }
                expert_params.update(expert_weights)

        retrieved_expert = Expert(expert_info=expert_info, expert_weights=expert_params)
        return retrieved_expert

    def save_to_library(self, library_id):
        """
        Saves the current loaded experts to the specified library.

        Args:
            library_id (str): The ID of the library to save the experts to.
        """
        library = ExpertLibrary.get_expert_library(library_id, create=True)
        for expert_name in self.experts_names:
            expert = self.get_expert_instance(expert_name)
            library.add_expert(expert)
        return library

    def as_expert(self):
        raise NotImplementedError(
            "This method is not implemented for MultiExpertModel."
        )