import copy
from functools import partial
import re
from tempfile import TemporaryDirectory
import threading
from typing import Any, Dict, List
import numpy as np
import torch
import tqdm

from transformers import PreTrainedModel
from mttl.models.llama_patch import replace_attn_with_flash_attn
from mttl.models.modifiers.expert_containers import add_expert_to_transformer
from mttl.models.modifiers.expert_containers.expert_library import ExpertLibrary
from mttl.models.modifiers.routing import RoutingInfo
from mttl.models.utils import prepare_model_for_kbit_training
from mttl.utils import logger
from mttl.models.modifiers.expert_containers.expert import Expert, ExpertInfo
from mttl.models.modifiers.expert_containers.expert_containers import (
    ExpertContainer,
    LoRAExpertContainer,
)
from mttl.models.modifiers.expert_containers.selectors import Selector, SelectorConfig


class MultiExpertModel:
    """Wrapper for a HF model that allows loading and managing multiple experts."""

    def __init__(
        self,
        model=None,
        model_object=None,
        load_in_8bit=False,
        device_map="auto",
        routing_config=None,
    ):
        if model is None and model_object is None:
            raise ValueError("Either model or model_object must be provided.")

        self.model: PreTrainedModel = None
        if model_object is not None:
            self.model = model_object
        else:
            from mttl.models.utils import model_loader_helper

            self.model = model_loader_helper(
                model,
                load_in_8bit=load_in_8bit,
                device_map=device_map,
            )

        if load_in_8bit:
            self.model = prepare_model_for_kbit_training(self.model)

        replace_attn_with_flash_attn(self.model)

        # patch model to contain info about routing
        self.model.info_container = {}
        self.model.selectors = {}

        self.experts_names = []
        self.routing_config = routing_config
        self.lock = threading.Lock()

    def deepcopy(self):
        lock = self.lock
        self.lock = None
        _copy = copy.deepcopy(self)
        _copy.lock = lock
        self.lock = lock
        return _copy

    @property
    def experts_containers(self) -> List[ExpertContainer]:
        containers = []
        for _, module in self.model.named_modules():
            for _, child in dict(module.named_children()).items():
                if isinstance(child, ExpertContainer) and len(child.experts) > 0:
                    containers.append(child)
        return containers

    @property
    def selectors(self) -> Dict[str, Selector]:
        return self.model.selectors

    def get_router_weights(self):
        weights = {}
        for _, selector in self.selectors.items():
            weights[selector.layer_name] = selector.get_routing_weights()
        return weights

    def delete_expert_container(self):
        """
        Replaces the expert container with the expert with the given name.
        """
        for _, module in self.model.named_modules():
            for c_name, child in dict(module.named_children()).items():
                if isinstance(child, ExpertContainer) and len(child.experts) > 0:
                    setattr(module, c_name, child.layer)
        self.experts_names.clear()

    def add_experts_from_library(self, library):
        import tqdm
        import concurrent.futures

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
                total=len(library), desc="Processing", unit="module"
            ) as progress_bar:
                for _ in concurrent.futures.as_completed(futures):
                    progress_bar.update(1)

    def load_from_module_dict(self, module_dict, action="route"):
        for module_name, destination in module_dict.items():
            if isinstance(destination, str):
                self.load_expert(
                    destination,
                    module_name,
                    action=action,
                    is_default=module_name == "default",
                )
            elif isinstance(destination, Expert):
                self.add_expert_instance(destination, module_name, action=action)

    def add_empty_expert(
        self,
        expert_name,
        expert_config=None,
    ):
        """Adds a new empty expert to the model."""
        new_expert = Expert(
            expert_info=ExpertInfo(
                expert_name,
                expert_config=expert_config,
            ),
        )

        self.add_expert_instance(new_expert)
        logger.info("Added empty expert: {}".format(expert_name))

    def load_expert(
        self,
        expert_path: str,
        expert_name: str = None,
        action: str = "merge",
        is_default: bool = False,
        expert_library: ExpertLibrary = None,
    ):
        from mttl.models.modifiers.expert_containers.expert import load_expert

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

    def add_expert_instance(
        self,
        expert_instance: Expert,
        expert_name=None,
        action="route",
        is_default=False,
    ):
        if expert_name is not None:
            # we want to load expert instance with a given name (might be different from the one in the expert instance)
            # we dont want to change expert instance though!
            # will create a copy for now (maybe safer), alternatively can change the name and set it back at the end of the function
            expert_instance = expert_instance.clone()
            expert_instance.name = expert_name

        with self.lock:
            self.model = add_expert_to_transformer(
                self.model,
                expert_instance,
                action=action,
                is_default=expert_instance.name == "default" or is_default,
                routing_config=self.routing_config,
            )
            if action != "merge":
                self.experts_names.append(expert_instance.name)

    def load_from_library(self, library, subsample_library_experts=0):
        import copy

        keys = list(library.keys())
        if self.hparams.subsample_library_experts > 0:
            keys = np.random.permutation(keys)[:subsample_library_experts]

        for expert_name in tqdm.tqdm(keys, desc="Loading experts..."):
            expert_dump = library.get_expert(expert_name, with_auxiliary_data=True)
            self.add_expert_instance(expert_dump)

    def set_selector(
        self,
        modifier_type: str,
        selector_config: SelectorConfig,
        selector_weights: dict = None,
    ):
        from mttl.models.modifiers.expert_containers import (
            replace_selector_for_container,
        )

        n_selectors, n_selectors_views = replace_selector_for_container(
            self.model,
            modifier_type,
            selector_config,
            selector_weights,
            force_replace=True,
        )
        assert self.model.selectors[modifier_type]
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

    def get_task_embeddings(self):
        """
        Retrieves the task embeddings for the loaded experts.

        This method assumes that the names of the loaded experts correspond to the tasks they are made for.

        Returns:
        embeddings (dict): A dictionary containing the task embeddings for each expert.
                           The keys are the expert names and the values are the corresponding embeddings.
        """
        if len(self.experts_names) == 0:
            return self.extract_parameters()

        embeddings = {}
        for exp_name in self.experts_names:
            embeddings[exp_name] = (
                self.extract_parameters(p_name_pattern=rf".*{exp_name}\..*lora.*")
                .detach()
                .cpu()
            )
        return embeddings

    def set_routing_infos(self, batch, generate=False):
        self.model.info_container["routing_infos"] = RoutingInfo.from_batch(batch)

    def forward(self, batch):
        input_ids, labels = batch["input_ids"], batch["labels"]

        self.set_routing_infos(batch)

        return self.model.forward(input_ids, attention_mask=batch["attention_mask"])

    def generate(
        self,
        batch,
        **kwargs,
    ):
        self.set_routing_infos(batch, generate=True)

        generations = self.model.generate(
            inputs=batch["input_ids"], attention_mask=batch["attention_mask"], **kwargs
        )
        return generations


def to_expert(
    model: MultiExpertModel,
    training_config: Any,
    expert_name: str = None,
    finetune_task_name: str = None,
    weights: dict = None,
    with_global_names=True,
) -> Expert:
    """
    Converts the current expert model into an instance of the Expert class.

    Args:
        weights (dict, optional): A dictionary of weights to merge the experts. If not provided, the router's weights will be used.
        with_global_names (bool, optional): Whether to include global names in the merged weights. Defaults to True.

    Returns:
        Expert: An instance of the Expert class.

    Raises:
        None

    Example:
        model = ExpertModel()
        expert = model.to_expert(weights={'expert1': 0.5, 'expert2': 0.5}, with_global_names=True)
    """

    expert_weights = {}
    for container in model.experts_containers:
        assert isinstance(container, LoRAExpertContainer)

        if hasattr(container, "get_merged_weights"):
            expert_config, _weights = container.get_merged_weights(
                with_global_names=with_global_names, weights=weights
            )
            expert_weights.update(_weights)

    if len(expert_weights) == 0:
        return None

    expert_info = ExpertInfo(
        expert_name=expert_name,
        expert_task_name=finetune_task_name or expert_name,
        training_config=training_config,
        expert_config=expert_config,
    )
    return Expert(expert_info=expert_info, expert_weights=expert_weights)


def replace_container_with_expert(
    model: MultiExpertModel, expert_name, get_expert_instance=True
):
    raise NotImplementedError()
