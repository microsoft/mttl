import torch
import copy
import re
import numpy as np
from typing import Dict
from tempfile import TemporaryDirectory

import tqdm
from mttl.models.modifiers.routing import RoutingInfo
from mttl.utils import logger
from mttl.models.modifiers.expert_containers import ExpertContainer
from mttl.models.modifiers.expert_containers import Selector
from mttl.models.modifiers.expert_containers import (
    add_expert_to_transformer,
)

from projects.wiki_experts.src.expert_trainer import ExpertTrainer
from projects.wiki_experts.src.ranker.adapter_ranker import AdapterRankerHelper

from mttl.models.modifiers.expert_containers.module_graph import Expert, ExpertInfo
from mttl.models.modifiers.expert_containers.module_graph import (
    ModuleGraph,
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

    dataset_name = expert.expert_config.dataset
    # handle the case where dataset is from huggingface
    if "/" in dataset_name:
        dataset_name = dataset_name.partition("/")[-1]

    # model is definitely from HF
    model_name = expert.expert_config.model
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
        # If you want to use a model modifier, use one of the 'self.modify_weith...' methods.
        kwargs["model_modifier"] = None
        super().__init__(**kwargs)

        self.experts = []

    def get_router_weights(self):
        weights = {}
        for _, selector in self.selectors.items():
            weights[selector.name] = selector.get_routing_weights()
        return weights

    def load_from_graph(self, graph: ModuleGraph, action="route", **kwargs):
        for _, module in graph.create_modules(
            base_hparams=self.hparams, **kwargs
        ).items():
            print("Loading module: {}".format(module.name))
            self.model = add_expert_to_transformer(
                self.model,
                module,
                action=action,
                is_default=module.name == "default",
                config=self.hparams,
            )
            self.experts.append(module.name)

    def delete_expert_container(self):
        """
        Replaces the expert container with the expert with the given name.
        """
        for _, module in self.model.named_modules():
            for c_name, child in dict(module.named_children()).items():
                if isinstance(child, ExpertContainer) and len(child.experts) > 0:
                    setattr(module, c_name, child.layer)
        self.experts = []

    def replace_container_with_expert(self, expert_name, get_expert_instance=True):
        """
        Replaces the expert container with the expert with the given name.
        """
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
            return expert
        return

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

    def add_expert_instance(
        self,
        expert_instance: Expert,
        expert_name=None,
        action="route",
        is_default=False,
    ):
        if expert_name is not None:
            expert_instance.name = expert_name

        self.model = add_expert_to_transformer(
            self.model,
            expert_instance,
            action=action,
            is_default=expert_name == "default" or is_default,
        )
        if action != "merge":
            self.experts.append(expert.name)

    def load_from_graph_string(self, s, action="route", expert_library=None):
        from mttl.models.modifiers.expert_containers.module_graph import ModuleGraph

        graph = ModuleGraph.from_string(s, expert_library=expert_library)
        self.load_from_graph(graph, action=action)

    def load_from_library(self, library, subsample_library_experts=0):
        import copy

        keys = list(library.keys())
        if self.hparams.subsample_library_experts > 0:
            keys = np.random.permutation(keys)[:subsample_library_experts]

        # fill all the weights with zeros after deep copying the weights
        expert = library[keys[0]]
        expert = copy.deepcopy(expert)
        for _, value in expert.expert_weights.items():
            value.fill_(0)
        expert.name = "default"

        add_expert_to_transformer(
            self.model,
            expert,
            action="route",
            is_default=True,
            config=self.hparams,
        )

        for expert_name in tqdm.tqdm(keys, desc="Loading experts..."):
            expert_dump = library[expert_name]
            add_expert_to_transformer(
                self.model,
                expert_dump,
                action="route",
                is_default=expert_name == "default",
                config=self.hparams,
            )
            self.experts.append(expert_name)

    def load_expert(
        self,
        expert_path: str,
        expert_name: str = None,
        action: str = "merge",
        is_default: bool = False,
        load_only_layers: str = None,
    ):
        from mttl.models.modifiers.expert_containers.module_graph import load_expert

        expert = load_expert(expert_path, expert_name=expert_name)

        if self.hparams.model != expert.expert_config.model:
            raise ValueError(
                "The expert has been trained on top of a different model!"
                " Detected: {} - Expected: {}".format(
                    expert.expert_config.model, self.hparams.model
                )
            )

        logger.info(
            f"Adding expert with name {expert_name}... with action ... {action}!"
        )

        self.model = add_expert_to_transformer(
            self.model,
            expert,
            action=action,
            is_default=is_default,
            load_only_layers=load_only_layers,
        )
        if action != "merge":
            self.experts.append(expert.name)

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
        if len(self.experts) == 0:
            return self.extract_task_embeddings_lora()
        embeddings = {}
        for exp_name in self.experts:
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

    @property
    def generation_config(self):
        return self.model.generation_config

    def generate(
        self,
        batch,
        **kwargs,
    ):
        if hasattr(self.model, "task_id_container"):
            self.model.task_id_container["routing_infos"] = RoutingInfo.from_batch(
                batch
            )
        generations = self.model.generate(
            inputs=batch["input_ids"], attention_mask=batch["attention_mask"], **kwargs
        )
        return generations


class MultiExpertModelRanker(MultiExpertModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.expert_ranker = AdapterRankerHelper.get_ranker_instance(
            ranker_model=kwargs["ranker_model"],
            ranker_path=kwargs["ranker_path"],
        )
        self.hparams.selector = "info_selector"

    def generate(
        self,
        batch,
        **kwargs,
    ):
        if hasattr(self.model, "task_id_container"):
            self.model.task_id_container["routing_infos"] = RoutingInfo.from_batch(
                batch
            )

        self.expert_ranker.set_available_tasks(self.experts)
        mod_names, mod_weights = self.expert_ranker.predict_batch(
            batch,
            n=self.hparams.ranker_top_k,
        )

        # fill in the weights for the routing selector, for now just take the first one
        # mod_names = [['mod1', 'mod2'], ['mod3', 'mod4']]
        # mod_wgths = [[0.5, 0.5], [0.3, 0.7]]
        # mod_names = [['default', 'mod1']]
        # mod_wgths = [[0.7, 0.3]]
        self.model.task_id_container["routing_infos"].routing_modules = mod_names
        self.model.task_id_container["routing_infos"].routing_weights = mod_weights

        # infos
        logger.info(f"Most similar: {str(mod_names)}")
        logger.info(f"Most similar weights: {str(mod_weights)}")

        generations = self.model.generate(
            inputs=batch["input_ids"], attention_mask=batch["attention_mask"], **kwargs
        )
        return generations


class RoutedMultiExpertModel(MultiExpertModel):
    """
    Class that allows to route to different experts with a learned router from mttl.models.modifiers.experts.Router.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.selectors: Dict[str:Selector] = torch.nn.ModuleDict()

    def load_expert(
        self,
        expert_path: str,
        expert_name: str = None,
        action: str = "merge",
        is_default: bool = False,
        load_only_layers: str = None,
    ):
        from mttl.models.modifiers.expert_containers.module_graph import load_expert

        expert = load_expert(expert_path, expert_name=expert_name)

        if self.hparams.model != expert.expert_config.model:
            raise ValueError(
                "The expert has been trained on top of a different model!"
                " Detected: {} - Expected: {}".format(
                    expert.expert_config.model, self.hparams.model
                )
            )

        logger.info(
            f"Adding expert with name {expert.name}... with action ... {action}!"
        )

        self.model = add_expert_to_transformer(
            self.model,
            expert,
            action=action,
            is_default=is_default,
            load_only_layers=load_only_layers,
            selectors=self.selectors,
            config=self.hparams,
        )
        if action != "merge":
            self.experts.append(expert.name)

    def load_from_graph(self, graph, action="route", **kwargs):
        for _, module in graph.create_modules(
            base_hparams=self.hparams, **kwargs
        ).items():
            self.model = add_expert_to_transformer(
                self.model,
                module,
                action=action,
                is_default=module.name == "default",
                selectors=self.selectors,
                config=self.hparams,
            )
            self.experts.append(module.name)

    def get_router_weights(self):
        weights = {}
        for _, selector in self.selectors.items():
            weights[selector.name] = selector.get_routing_weights()
        return weights

    def add_expert_instance(self, expert_instance, expert_name=None, action="route"):
        if expert_name is not None:
            expert_instance.name = expert_name

        self.model = add_expert_to_transformer(
            self.model,
            expert_instance,
            action=action,
            is_default=expert_name == "default",
            selectors=self.selectors,
            config=self.hparams,
        )

        if action != "merge":
            self.experts.append(expert_instance.name)

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
        for _, module in self.model.named_modules():
            for _, child in dict(module.named_children()).items():
                if isinstance(child, ExpertContainer) and len(child.experts) > 0:
                    # creates a single Lora
                    exp_config, _weights = child.get_merged_weights(
                        with_global_names=with_global_names, weights=weights
                    )
                    expert_weights.update(_weights)

        config_merged = copy.deepcopy(self.hparams)
        config_merged.model_modifier = exp_config.model_modifier
        expert_info = ExpertInfo(
            self.hparams.finetune_task_name,
            expert_task_name=self.hparams.finetune_task_name,
            expert_config=config_merged,
        )
        return Expert(expert_info=expert_info, expert_weights=expert_weights)

    def on_save_checkpoint(self, ckpt):
        expert: Expert = self.to_expert()
        ckpt["expert_dumps"] = expert.dumps()
        ckpt["merging_weights"] = self.get_router_weights()
