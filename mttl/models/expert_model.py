import math
import os
import re
import threading
from collections import defaultdict
from functools import partial
from typing import Dict, List, Optional, Union

import torch
from torch.optim.optimizer import Optimizer
from transformers import PreTrainedModel
from transformers.utils import cached_file

from mttl.logging import logger
from mttl.models.containers import add_expert_to_transformer
from mttl.models.containers.base import ExpertContainer
from mttl.models.containers.selectors import Selector, SelectorConfig
from mttl.models.containers.selectors.base import (
    LoadableLibraryMixin,
    LoadableSelectorConfig,
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


class SingleExpertModel(torch.nn.Module):
    """Base model to load a single model from HF and modify it with a modifier config."""

    def __init__(
        self, base_model_name, modifier_config: ModifierConfig = None, **kwargs
    ):
        super().__init__()

        self.load_in_4bit = kwargs.get("load_in_4bit", None) or False
        self.load_in_8bit = kwargs.get("load_in_8bit", None) or False
        self.model: PreTrainedModel = None
        self.base_model_name = base_model_name
        self.accumulate_metrics_batch = defaultdict(list)

        self.model = model_loader_helper(
            base_model_name,
            load_in_4bit=self.load_in_4bit,
            load_in_8bit=self.load_in_8bit,
            device_map=kwargs.get("device_map", "cpu"),
        )
        if modifier_config is not None:
            self.model = modify_transformer(self.model, modifier_config)

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
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        **kwargs,
    ):
        # Load model
        instantiate_model = kwargs.pop("instantiate_model", True)
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", False)
        use_auth_token = kwargs.pop("use_auth_token", None)

        user_agent = {
            "file_type": "model",
            "framework": "pytorch",
            "from_auto_class": False,
        }

        if pretrained_model_name_or_path is not None:
            pretrained_model_name_or_path = str(pretrained_model_name_or_path)

            if os.path.isfile(pretrained_model_name_or_path) or os.path.isdir(
                pretrained_model_name_or_path
            ):
                resolved_archive_file = get_checkpoint_path(
                    pretrained_model_name_or_path
                )
            else:
                try:
                    # Load from URL or cache if already cached
                    resolved_archive_file = cached_file(
                        pretrained_model_name_or_path,
                        CHECKPOINT_PATH_IN_HUB,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        proxies=proxies,
                        resume_download=resume_download,
                        local_files_only=local_files_only,
                        use_auth_token=use_auth_token,
                        user_agent=user_agent,
                    )
                except EnvironmentError as err:
                    logger.error(err)
                    msg = (
                        f"Can't load weights for '{pretrained_model_name_or_path}'. Make sure that:\n\n"
                        f"- '{pretrained_model_name_or_path}' is a correct model identifier listed on 'https://huggingface.co/models'\n\n"
                        f"- or '{pretrained_model_name_or_path}' is the correct path to a directory containing a file named one of checkpoint.ckpt\n\n"
                    )
                    raise EnvironmentError(msg)

                if resolved_archive_file == pretrained_model_name_or_path:
                    logger.info(f"loading weights file {resolved_archive_file}")
                else:
                    logger.info(
                        f"loading weights file {pretrained_model_name_or_path} from cache at {resolved_archive_file}"
                    )
        else:
            resolved_archive_file = None

        if instantiate_model:
            ckpt = torch.load(resolved_archive_file, map_location="cpu")

            # assume this has been trained with pytorch lightning, which dumps all the
            # hyperparameters in the checkpoint
            model_base_name = ckpt["hyper_parameters"].get("model")

            if tokenizer is None and "model" in ckpt["hyper_parameters"]:
                tokenizer = get_tokenizer_with_args(
                    model_name=ckpt["hyper_parameters"]["model"],
                    model_family=ckpt["hyper_parameters"]["model_family"],
                    padding_side=ckpt["hyper_parameters"]["padding_side"],
                )

            model = cls()
            model.load_state_dict(ckpt["state_dict"], strict=False)
            return model
        else:
            ckpt = torch.load(resolved_archive_file, map_location="cpu")

            return ckpt["state_dict"], ckpt["hyper_parameters"]

    def from_pretrained_library(self, library_path: str, expert_name: str):
        pass


class MultiExpertModel(SingleExpertModel):
    """Adds all functions and properties for a multi-expert model."""

    def __init__(
        self,
        base_model_name,
        selector_config: Union[SelectorConfig, Dict[str, SelectorConfig]] = None,
        **kwargs,
    ):
        super().__init__(base_model_name, modifier_config=None, **kwargs)

        self.selector_config = selector_config
        # inject memory for adding selectors
        self.model.selectors = {}
        self.experts_names = []

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
        else:
            library = library_id

        # get a config file from the library, and initialize the expert model
        an_expert: Expert = library[next(iter(library.keys()))]

        train_cfg: ExpertConfig = deepcopy(an_expert.training_config)
        train_cfg.device_map = "cpu"

        model = cls(train_cfg.model)
        model.add_experts_from_library(library)

        # set selector for the added experts
        if selector_config is not None:
            # assume "lora" is the default modifier type
            if type(selector_config) is SelectorConfig:
                logger.info(
                    "Assuming provided selector config is for `lora` modifier type."
                )
                selector_config = {"lora": selector_config}

            for modifier_type, selector_config_ in selector_config.items():
                # inject the library id if it is None
                if (
                    isinstance(selector_config_, LoadableSelectorConfig)
                    and selector_config_.library_id is None
                ):
                    selector_config_.library_id = library_id

                model.set_selector(modifier_type, selector_config_)
        else:
            logger.info("No selector config provided, assuming expert name selector!")
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
        selectors = defaultdict(list)
        for modifier, selectors_dict in self.model.selectors.items():
            for selector in selectors_dict.values():
                if isinstance(selector, Selector):
                    selectors[modifier].append(selector)
        return selectors

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
                self.add_expert_instance(
                    destination,
                    module_name,
                    action=action,
                    is_default=module_name == "default",
                )

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
            add_expert_to_transformer(
                self.model,
                expert_instance,
                action=action,
                is_default=is_default,
                routing_config=self.selector_config,
                training_config=expert_instance.training_config,
            )

            if action != "merge":
                self.experts_names.append(expert_instance.name)
                # reload the expert instance to fill the weights properly if this was an empty expert
                expert_instance = self.get_expert_instance(expert_instance.name)
            return expert_instance

    def set_selector(
        self,
        modifier_type: str,
        selector_config: SelectorConfig,
    ):
        from mttl.models.containers import replace_selector_for_container

        n_selectors, n_selectors_views = replace_selector_for_container(
            self.model,
            modifier_type,
            selector_config,
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

    def get_merged_expert(
        self, modifier_type: str = "lora", with_global_names=True, **kwargs
    ) -> Expert:
        """
        Converts the current expert model into an instance of the Expert class by merging the experts in the containers using weights from the corresponding selectors.

        Args:
            with_global_names (bool, optional): Whether to include global names in the merged weights. Defaults to True.

        Returns:
            Expert: An instance of the Expert class.

        Raises:
            None

        Example:
            model = ExpertModel()
            expert = model.get_merged_expert(weights={'expert1': 0.5, 'expert2': 0.5}, with_global_names=True)
        """
        expert_params = {}
        assert (
            modifier_type in self.selectors
        ), f"Modifier type {modifier_type} not in model."

        for container in self.experts_containers:
            config_modifier = get_modifier_name(container.config)
            if config_modifier != modifier_type:
                logger.info(
                    f"Skipping container {container.layer_name} with modifier type {config_modifier}"
                )
                continue

            params = container.get_merged_params(
                with_global_names=with_global_names, **kwargs
            )
            expert_params.update(params)
            expert_config = container.config
        if len(expert_params) == 0:
            raise ValueError(
                "No experts to merge found. Make sure the 'modifier_type' is correct."
            )

        expert_info = ExpertInfo(
            expert_name=self.training_config.finetune_task_name,
            expert_task_name=self.training_config.finetune_task_name,
            training_config=self.training_config,
            expert_config=expert_config,
        )
        return Expert(expert_info=expert_info, expert_weights=expert_params)


class LoRAMoEModel(MultiExpertModel):
    __supported_selectors__ = [PolySelectorConfig, MOERKHSSelectorConfig]

    def __init__(
        self,
        base_model_name,
        modifier_config: SkilledLoRAConfig,
        selector_config: SelectorConfig,
        expert_library: ExpertLibrary = None,
        **kwargs,
    ):
        if type(selector_config) not in self.__supported_selectors__:
            raise ValueError(
                f"Compatible selector config is required for MoE model: {self.__supported_selectors__}."
            )

        super().__init__(
            base_model_name=base_model_name, selector_config=selector_config
        )

        if expert_library is None:
            if not modifier_config:
                raise ValueError("Modifier config is required for MoE model.")

            if not selector_config.num_experts:
                raise ValueError("Required specification of the number of experts.")

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
        else:
            if expert_library is None:
                expert_library = ExpertLibrary.get_expert_library(
                    self.hparams.library_id
                )
            for i, expert in enumerate(sorted(list(expert_library.keys()))):
                self.add_expert_instance(expert_library[expert], expert_name=f"e{i}")

            self.num_experts = i + 1
