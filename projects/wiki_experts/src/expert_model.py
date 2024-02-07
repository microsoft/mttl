from functools import partial
import math
import threading
import torch
import re
import os
import numpy as np
from typing import Dict, List
from tempfile import TemporaryDirectory

import tqdm
from mttl.models.modifiers.expert_containers.expert_containers import (
    LoRAExpertContainer,
)
from mttl.models.modifiers.expert_containers.expert_library import (
    ExpertLibrary,
    HFExpertLibrary,
)
import copy

from mttl.models.modifiers.lora import LoRAConfig, SkilledLoRAConfig
from mttl.models.modifiers.routing import RoutingInfo
from mttl.utils import logger
from mttl.models.modifiers.expert_containers import ExpertContainer
from mttl.models.modifiers.expert_containers.selectors import Selector
from mttl.models.modifiers.expert_containers import (
    add_expert_to_transformer,
)

from projects.wiki_experts.src.expert_trainer import ExpertTrainer
from projects.wiki_experts.src.ranker.adapter_ranker import AdapterRankerHelper

from mttl.models.modifiers.expert_containers.expert import (
    Expert,
    ExpertInfo,
    load_expert,
)


def push_expert_to_hub(
    ckpt_path,
    hf_user_id,
    auto_search=True,
    use_last=False,
    expert_name=None,
) -> None:
    from mttl.models.utils import convert_and_push_to_hub

    """Searches into local path for the checkpoint with lowest validation loss,
    then uploads that.

    if use_last is True, then uses the last checkpoint `last.ckpt` instead
    of the one with lowest validation loss.
    """
    from mttl.models.modifiers.expert_containers.module_graph import load_expert
    from mttl.utils import get_checkpoint_path

    expert = load_expert(get_checkpoint_path(ckpt_path, use_last=use_last))

    dataset_name = expert.training_config.dataset
    # handle the case where dataset is from huggingface
    if "/" in dataset_name:
        dataset_name = dataset_name.partition("/")[-1]

    # model is definitely from HF
    model_name = expert.training_config.model
    if "/" in model_name:
        model_name = model_name.partition("/")[-1]

    if expert_name is not None:
        expert.expert_info.expert_name = expert_name
    else:
        expert_name = expert.expert_info.expert_name

    assert expert_name is not None

    repo_id = f"{hf_user_id}/expert__{model_name}__{dataset_name}__{expert_name}"

    logger.info("Uploading checkpoint {} --> {}".format(ckpt_path, repo_id))
    convert_and_push_to_hub(expert, repo_id)


class MultiExpertModel(ExpertTrainer):
    def __init__(self, **kwargs: dict):
        # we dont use any  model modifier for MultiExpertModel model by default.
        # If you want to use a model modifier, use one of the 'self.modify_with...' methods.
        kwargs["model_modifier"] = None
        super().__init__(**kwargs)

        self.experts_names = []
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
            weights[selector.name] = selector.get_routing_weights()
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

    def replace_container_with_expert(self, expert_name, get_expert_instance=True):
        """
        Replaces the expert container with the expert with the given name.
        """
        if len(self.experts_names) == 0:
            return
        expert = None
        for _, module in self.model.named_modules():
            for c_name, child in dict(module.named_children()).items():
                if isinstance(child, ExpertContainer) and len(child.experts) > 0:
                    setattr(module, c_name, child.experts[expert_name])
                    if expert is None:
                        expert = child.experts[expert_name]
        # make sure hparams reflect the loaded expert
        if expert:
            self.hparams.update(expert.config.__dict__)
        if get_expert_instance:
            td = TemporaryDirectory()
            expert_checkpoint = MultiExpertModel.save_pretrained(self, td.name)
            expert: Expert = load_expert(expert_checkpoint)
            # remove the temporary directory
            os.remove(expert_checkpoint)
            return expert
        return

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
                expert_config=expert_config or self.hparams.__dict__,
                expert_task_name=self.hparams.finetune_task_name,
            ),
        )
        self.add_expert_instance(new_expert)
        logger.info("Added empty expert: {}".format(expert_name))

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
                training_config=self.training_config,
            )
            if action != "merge":
                self.experts_names.append(expert_instance.name)

    def load_from_library(self, library, subsample_library_experts=0):
        import copy

        keys = list(library.keys())
        if self.hparams.subsample_library_experts > 0:
            keys = np.random.permutation(keys)[:subsample_library_experts]

        # fill all the weights with zeros after deep copying the weights
        # TODO: clean this in some way
        expert = library[keys[0]]
        expert = copy.deepcopy(expert)
        for _, value in expert.expert_weights.items():
            value.fill_(0)
        expert.name = "default"

        self.add_expert_instance(expert, is_default=True)
        for expert_name in tqdm.tqdm(keys, desc="Loading experts..."):
            expert_dump = library.get_expert(expert_name, with_auxiliary_data=True)
            self.add_expert_instance(expert_dump)

    def load_expert(
        self,
        expert_path: str,
        expert_name: str = None,
        action: str = "merge",
        is_default: bool = False,
        expert_library: ExpertLibrary = None,
    ):
        from mttl.models.modifiers.expert_containers.module_graph import load_expert

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

    def extract_task_embeddings_lora(self, p_name_pattern=".*lora.*"):
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
            return self.extract_task_embeddings_lora()
        embeddings = {}
        for exp_name in self.experts_names:
            embeddings[exp_name] = (
                self.extract_task_embeddings_lora(
                    p_name_pattern=rf".*{exp_name}\..*lora.*"
                )
                .detach()
                .cpu()
            )
        return embeddings

    def forward(self, batch, reduction="mean"):
        return super().forward(batch, reduction)

    def to_expert(self, weights: dict = None, with_global_names=True) -> Expert:
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
        for container in self.experts_containers:
            assert isinstance(container, LoRAExpertContainer)

            if hasattr(container, "get_merged_weights"):
                expert_config, _weights = container.get_merged_weights(
                    with_global_names=with_global_names, weights=weights
                )
                expert_weights.update(_weights)

        if len(expert_weights) == 0:
            return None

        expert_info = ExpertInfo(
            expert_name=self.hparams.finetune_task_name,
            expert_task_name=self.hparams.finetune_task_name,
            training_config=self.training_config,
            expert_config=expert_config,
        )
        return Expert(expert_info=expert_info, expert_weights=expert_weights)


class MultiExpertModelRanker(MultiExpertModel):
    def __init__(self, **kwargs):
        kwargs["router_selector"] = "info_selector"
        super().__init__(**kwargs)

        self.expert_ranker = AdapterRankerHelper.get_ranker_instance(
            ranker_model=kwargs["ranker_model"],
            ranker_path=kwargs["ranker_path"],
        )

    def set_routing_infos(self, batch, generate=False):
        self.model.info_container["routing_infos"] = RoutingInfo.from_batch(batch)

        self.expert_ranker.set_available_tasks(self.experts_names)
        mod_names, mod_weights = self.expert_ranker.predict_batch(
            batch,
            n=self.hparams.ranker_top_k,
        )

        # fill in the weights for the routing selector, for now just take the first one
        # mod_names = [['mod1', 'mod2'], ['mod3', 'mod4']]
        # mod_wgths = [[0.5, 0.5], [0.3, 0.7]]
        # mod_names = [['default', 'mod1']]
        # mod_wgths = [[0.7, 0.3]]
        self.model.info_container["routing_infos"].routing_modules = mod_names
        self.model.info_container["routing_infos"].routing_weights = mod_weights

        # infos
        logger.info(f"Most similar: {str(mod_names)}")
        logger.info(f"Most similar weights: {str(mod_weights)}")


class MoETrainer(MultiExpertModel):
    def __init__(self, **kwargs):
        kwargs["top_k"] = kwargs["moe_top_k"]
        kwargs["emb_dim"] = kwargs["moe_emb_dim"]
        kwargs["rkhs_dim"] = kwargs["moe_rkhs_dim"]
        library = kwargs.pop("expert_library", None)
        super().__init__(**kwargs)

        # 8 experts
        if library is None:
            if not self.hparams.hf_lib_id:
                for i in range(self.hparams.moe_num_experts):
                    # Adding a Skilled LoRA with 1 skill.
                    exp_config = SkilledLoRAConfig(
                        n_skills=1,
                        modify_layers=self.hparams.modify_layers,
                        modify_modules=self.hparams.modify_modules,
                        lora_alpha=self.hparams.lora_alpha,
                        lora_dropout=self.hparams.lora_dropout,
                        lora_rank=self.hparams.lora_rank,
                        lora_init_b_random=True,
                        n_splits=self.hparams.n_splits,
                        phi_2_align_heads=self.hparams.phi_2_align_heads,
                    )
                    self.add_empty_expert(f"e{i}", exp_config)
                self.moe_num_experts = kwargs["moe_num_experts"]
            else:
                library = HFExpertLibrary(self.hparams.hf_lib_id)

        if library is not None:
            for i, expert in enumerate(sorted(list(library.keys()))):
                self.add_expert_instance(library[expert], expert_name=f"e{i}")
            self.moe_num_experts = i + 1

    def training_step(self, batch, _):
        loss = self.forward(batch)
        total_loss = loss.clone()

        if (
            "routing_gates" in self.model.info_container
            and self.model.info_container["routing_gates"]
        ):
            num = 0.0
            entropy_of_avg = 0.0
            entropy_of_route = 0.0

            for values in self.model.info_container["routing_gates"]:
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

            if self.hparams.moe_ent_reg > 0.0:
                total_loss += self.hparams.moe_ent_reg * mi_loss

            elif self.hparams.moe_ent_free_bits > 0.0:
                normalized_entropy = entropy_of_route / math.log(self.moe_num_experts)
                total_loss += (
                    (1.0 - normalized_entropy) >= self.hparams.moe_ent_free_bits
                ) * -entropy_of_route

            self.model.info_container["routing_gates"].clear()

        self.log(f"{self._log_pref}train/loss", loss, on_step=True, prog_bar=True)
        self.log(
            f"{self._log_pref}train/total_loss", total_loss, on_step=True, prog_bar=True
        )

        for i, pg in enumerate(self.optimizers().optimizer.param_groups):
            self.log(f"train/lr_{i}", pg["lr"])
        return total_loss


class RoutedMultiExpertModel(MultiExpertModel):
    """
    TODO: This is depreciated. Use MultiExpertModel instead.
    Class that allows to route to different experts with a learned router from mttl.models.modifiers.experts.Router.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def to_expert(self, weights: dict = None, with_global_names=True) -> Expert:
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
        for container in self.experts_containers:
            assert type(container) == LoRAExpertContainer

            expert_config, _weights = container.get_merged_weights(
                with_global_names=with_global_names, weights=weights
            )
            expert_weights.update(_weights)

        expert_info = ExpertInfo(
            expert_name=self.hparams.finetune_task_name,
            expert_task_name=self.hparams.finetune_task_name,
            training_config=self.training_config,
            expert_config=expert_config,
        )
        return Expert(expert_info=expert_info, expert_weights=expert_weights)

    def on_save_checkpoint(self, ckpt):
        expert: Expert = self.to_expert()
        ckpt["expert_dumps"] = expert.asdict()
        ckpt["merging_weights"] = self.get_router_weights()
