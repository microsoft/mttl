import math
import os
import re
import threading
from collections import defaultdict
from functools import partial
from typing import Any, Dict, List, Optional, Union

import torch
from torch.optim.optimizer import Optimizer
from transformers import PreTrainedModel
from transformers.utils import PushToHubMixin, cached_file

from mttl.logging import logger
from mttl.models.containers import add_expert_to_transformer
from mttl.models.containers.base import ExpertContainer
from mttl.models.containers.selectors import Selector, SelectorConfig
from mttl.models.containers.selectors.arrow_selector import ArrowSelectorConfig
from mttl.models.containers.selectors.base import (
    LoadableLibraryMixin,
    LoadableSelectorConfig,
    MultiSelectorConfig,
    Selector,
    SelectorConfig,
    SelectorsCache,
)
from mttl.models.containers.selectors.moe_selector import MOERKHSSelectorConfig
from mttl.models.containers.selectors.poly_selector import PolySelectorConfig
from mttl.models.expert_config import ExpertConfig
from mttl.models.expert_context import InfoContainer
from mttl.models.library.expert import Expert, ExpertInfo
from mttl.models.library.expert_library import ExpertLibrary
from mttl.models.llama_patch import replace_attn_with_flash_attn
from mttl.models.modifiers import modify_transformer
from mttl.models.modifiers.base import Modifier, ModifierConfig
from mttl.models.modifiers.lora import SkilledLoRAConfig
from mttl.models.modifiers.modify_model import get_modifier_name
from mttl.models.utils import (
    CHECKPOINT_PATH_IN_HUB,
    EfficientCheckpointModule,
    model_loader_helper,
    prepare_model_for_kbit_training,
)
from mttl.utils import get_checkpoint_path

torch.set_float32_matmul_precision("high")


class ExpertModel(torch.nn.Module):
    """Base model to load a single model from HF and modify it with a modifier config."""

    def __init__(
        self,
        model_name_or_object: Union[str, PreTrainedModel],
        modifier_config: ModifierConfig = None,
        expert_info: ExpertInfo = None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        device_map: str = "cpu",
        attn_implementation: str = None,
    ):
        super().__init__()

        if modifier_config is not None and expert_info is not None:
            raise ValueError(
                "Both ``modifier_config`` and ``expert_info`` cannot be provided at the same time."
            )

        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.device_map = device_map

        self.model: PreTrainedModel = (
            model_name_or_object
            if isinstance(model_name_or_object, PreTrainedModel)
            else model_loader_helper(
                model_name_or_object,
                load_in_4bit=self.load_in_4bit,
                load_in_8bit=self.load_in_8bit,
                device_map=device_map,
                attn_implementation=attn_implementation,
            )
        )
        self.base_model_name = (
            self.model.config._name_or_path
            if isinstance(self.model, PreTrainedModel)
            else model_name_or_object
        )

        # initialize an empty expert info, fields will be filled by either loading or training
        self.expert_info = expert_info or ExpertInfo(
            expert_name=None,
            expert_task_name=None,
            expert_config=modifier_config,
            training_config=None,
        )

        self.modifier_config = self.expert_info.expert_config

        if self.modifier_config is not None:
            self.model = modify_transformer(self.model, self.modifier_config)

    @property
    def generation_config(self):
        return self.model.generation_config

    @InfoContainer.wrap_forward
    def forward(self, batch, reduction="mean"):
        input_ids = batch["input_ids"]
        labels = batch["labels"]

        outputs = self.model.forward(input_ids, attention_mask=batch["attention_mask"])

        # calculate loss, could also be done inside of the model
        bs = input_ids.size(0)
        logits = outputs.logits
        vocab_size = logits.size(-1)
        labels = labels.squeeze(-1)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten the tokens
        loss_fct = torch.nn.CrossEntropyLoss(reduction=reduction)
        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)

        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

        # reshape back
        if reduction == "none":
            loss = loss.view((bs, -1))
            # mean only non-zero
            non_zero_loss = (loss != 0).sum(dim=-1)
            non_zero_loss[non_zero_loss == 0] = 1
            loss = loss.sum(dim=-1) / non_zero_loss

        del outputs, shift_logits, shift_labels
        return loss

    @InfoContainer.wrap_forward
    def generate(
        self,
        batch,
        **kwargs,
    ):
        generations = self.model.generate(
            inputs=batch["input_ids"], attention_mask=batch["attention_mask"], **kwargs
        )
        return generations

    @classmethod
    def from_pretrained_expert(
        cls,
        expert: Expert,
        **kwargs,
    ):
        model = expert.expert_info.model
        modifier_config = expert.expert_config
        instance = cls(
            model,
            modifier_config=modifier_config,
            expert_info=expert.expert_info,
            **kwargs,
        )
        instance.load_state_dict(expert.expert_weights, strict=False)
        return instance

    @classmethod
    def load_from_checkpoint(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        **kwargs,
    ):
        """Assumes the model has been trained with the Pytorch lightning and ExpertModelLightningWrapper."""
        from mttl.models.expert_trainer import ExpertModelLightningWrapper

        return ExpertModelLightningWrapper.load_from_checkpoint(
            pretrained_model_name_or_path, *model_args, **kwargs
        ).model

    def get_expert_instance(self):
        """Returns the model as an expert instance."""
        expert_params = {}
        for mname, module in self.named_modules().items():
            for cname, children in module.named_children().items():
                if isinstance(children, Modifier):
                    expert_weights = children.state_dict()
                    expert_weights = {
                        f"{mname}.{cname}.{k}": v for k, v in expert_weights.items()
                    }
                    expert_params.update(expert_weights)

        expert = Expert(expert_info=self.expert_info, expert_weights=expert_params)
        return expert

    @classmethod
    def from_training_config(cls, training_config: ExpertConfig):
        return cls(
            model_name_or_object=training_config.model,
            modifier_config=ModifierConfig.from_training_config(training_config),
            load_in_4bit=training_config.load_in_4bit,
            load_in_8bit=training_config.load_in_8bit,
            device_map=training_config.device_map,
        )


class MultiExpertModel(ExpertModel):
    """Adds all functions and properties for a multi-expert model."""

    def __init__(
        self,
        model_name_or_object: Union[str, PreTrainedModel],
        selector_config: Union[SelectorConfig, MultiSelectorConfig] = None,
        experts_info: List[ExpertInfo] = None,
        **kwargs,
    ):
        super().__init__(model_name_or_object, modifier_config=None, **kwargs)

        # inject memory for adding selectors
        self.selector_config = selector_config
        self.selector_cache = SelectorsCache()
        self.experts_infos = {}

        # initialize the model with the empty experts
        if experts_info is not None:
            for expert_info in experts_info:
                self.add_empty_expert(
                    expert_info.expert_name, expert_info.expert_config
                )

    @classmethod
    def from_training_config(cls, training_config: ExpertConfig):
        return cls(
            model_name_or_object=training_config.model,
            selector_config=SelectorConfig.from_training_config(training_config),
            load_in_4bit=training_config.load_in_4bit,
            load_in_8bit=training_config.load_in_8bit,
            device_map=training_config.device_map,
        )

    @property
    def experts_names(self):
        return self.experts_infos.keys()

    @classmethod
    def load_from_checkpoint(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        **kwargs,
    ):
        """Assumes the model has been trained with the Pytorch lightning and ExpertModelLightningWrapper."""
        from mttl.models.expert_trainer import MultiExpertModelLightningWrapper

        return MultiExpertModelLightningWrapper.load_from_checkpoint(
            pretrained_model_name_or_path, **kwargs
        ).model

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
        **kwargs,
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
        an_expert: Expert = library[next(iter(library.keys()))]

        model_name = an_expert.expert_info.model

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

        model = cls(
            model,
            selector_config=selector_config,
            device_map=kwargs.get("device_map", "cpu"),
        )
        model.add_experts_from_library(library)
        return model

    @property
    def experts(self):
        for expert_name in self.experts_names:
            yield self.get_expert_instance(expert_name)

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
                expert_model=self.base_model_name,
                training_config=None,
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

        if self.base_model_name != expert.training_config.model:
            raise ValueError(
                "The expert has been trained on top of a different model!"
                " Detected: {} - Expected: {}".format(
                    expert.training_config.model, self.base_model_name
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

        if expert_instance.name in self.experts_names:
            raise ValueError(
                f"Expert with name {expert_instance.name} already exists in the model!"
            )

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
            selector_cache=self.selector_cache,
            selector_config=selector_config,
            force_replace=True,
        )
        assert n_selectors > 0, f"No selector added for modifier {modifier_name}."

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
        with library.batched_commit():
            for expert_name in self.experts_names:
                expert = self.get_expert_instance(expert_name)
                library.add_expert(expert)
        return library


class LoRAMoEModel(MultiExpertModel):
    """A MoE model with LoRA experts."""

    __supported_selectors__ = [
        PolySelectorConfig,
        MOERKHSSelectorConfig,
        ArrowSelectorConfig,
    ]

    def __init__(
        self,
        model,
        selector_config: SelectorConfig,
        modifier_config: SkilledLoRAConfig = None,
        experts_library_id: str = None,
        experts_info: List[ExpertInfo] = None,
        **kwargs,
    ):
        if type(selector_config) not in self.__supported_selectors__:
            raise ValueError(
                f"Compatible selector config is required for MoE model: {self.__supported_selectors__}."
            )

        if (
            sum(
                [
                    experts_info is not None,
                    experts_library_id is not None,
                    modifier_config is not None,
                ]
            )
            > 1
        ):
            raise ValueError(
                "Only one of `experts_info` or `expert_library` or `modifier_config` can be provided."
            )

        super().__init__(
            model, selector_config=selector_config, experts_info=experts_info
        )

        self.experts_library_id = experts_library_id

        if experts_library_id is None and experts_info is None:
            if not modifier_config:
                raise ValueError("Modifier config is required for MoE model.")

            if not selector_config.num_experts:
                raise ValueError(
                    "`num_experts` is None in SelectorConfig, this requires prior specification of the number of experts."
                )

            for i in range(selector_config.num_experts):
                # Adding a Skilled LoRA with 1 skill.
                exp_config = SkilledLoRAConfig(
                    n_skills=1,
                    modify_layers=modifier_config.modify_layers,
                    modify_modules=modifier_config.modify_modules,
                    lora_alpha=modifier_config.lora_alpha,
                    lora_dropout=modifier_config.lora_dropout,
                    lora_rank=modifier_config.lora_rank,
                    lora_init_b_random=True,
                    n_splits=modifier_config.n_splits,
                    phi_2_align_heads=modifier_config.phi_2_align_heads,
                )
                self.add_empty_expert(f"e{i}", exp_config)

            self.num_experts = selector_config.num_experts
        elif experts_library_id is not None:
            expert_library = ExpertLibrary.get_expert_library(experts_library_id)

            for i, expert in enumerate(sorted(list(self.expert_library.keys()))):
                self.add_expert_instance(expert_library[expert], expert_name=f"e{i}")

            self.selector_config.num_experts = i + 1

    def training_step(self, batch, _):
        loss, context = self.forward(batch, return_context=True)
        total_loss = loss

        self.log(f"{self._log_pref}train/loss", loss, on_step=True, prog_bar=True)
        self.log(
            f"{self._log_pref}train/total_loss", total_loss, on_step=True, prog_bar=True
        )

        for i, pg in enumerate(self.optimizers().optimizer.param_groups):
            self.log(f"train/lr_{i}", pg["lr"])

        total_loss = loss.clone()
        routing_gates = context["routing_gates"]

        if routing_gates:
            num = 0.0
            entropy_of_avg = 0.0
            entropy_of_route = 0.0

            for values in routing_gates:
                # compute MI loss
                values = values.to(torch.float32)
                values = values.view(-1, values.shape[-1])
                probs = torch.softmax(values, -1)
                entropy_of_avg += -(
                    probs.mean(0) * torch.log(probs.mean(0) + 1e-6)
                ).sum(-1)
                entropy_of_route += -(probs * torch.log(probs + 1e-6)).sum(-1).mean(0)
                num += 1.0

            entropy_of_avg = entropy_of_avg / num
            entropy_of_route = entropy_of_route / num
            mi_loss = -entropy_of_avg + entropy_of_route

            self.log(
                f"{self._log_pref}train/entropy_of_route",
                entropy_of_route,
                on_step=True,
                prog_bar=True,
            )
            self.log(
                f"{self._log_pref}train/entropy_of_avg",
                entropy_of_avg,
                on_step=True,
                prog_bar=True,
            )
            self.log(
                f"{self._log_pref}train/mi_loss",
                mi_loss,
                on_step=True,
                prog_bar=True,
            )

    @classmethod
    def load_from_checkpoint(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        **kwargs,
    ):
        """Assumes the model has been trained with the Pytorch lightning and ExpertModelLightningWrapper."""
        from mttl.models.expert_trainer import LoRAMoELightningWrapper

        return LoRAMoELightningWrapper.load_from_checkpoint(
            pretrained_model_name_or_path, **kwargs
        ).model
