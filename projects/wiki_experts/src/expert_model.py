import torch
import numpy as np
from typing import Dict
from tempfile import TemporaryDirectory
from mttl.models.modifiers.routing import RoutingInfo
from mttl.utils import logger
from mttl.models.modifiers.expert_containers import ExpertContainer
from mttl.models.modifiers.expert_containers import Selector
from mttl.models.modifiers.expert_containers import add_expert_to_transformer

from projects.wiki_experts.src.expert_trainer import ExpertTrainer
from projects.wiki_experts.src.ranker.adapter_ranker import ExpertRanker
from mttl.models.modifiers.expert_containers.module_graph import Expert
from mttl.models.modifiers.expert_containers.module_graph import (
    ModuleGraph,
    load_expert,
)
from projects.wiki_experts.src.ranker.classification_module import ids_to_tasks_names


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
    from mttl.utils import get_checkpoint_path

    if auto_search:
        ckpt_path = get_checkpoint_path(ckpt_path, use_last=use_last)

    ckpt = torch.load(ckpt_path)

    if expert_name is None:
        for key in ["expert_name", "finetune_task_name"]:
            expert_name = ckpt["hyper_parameters"].get(key)
            if expert_name is not None:
                break

    dataset_name = ckpt["hyper_parameters"]["dataset"]
    # handle the case where dataset is from huggingface
    if "/" in dataset_name:
        dataset_name = dataset_name.partition("/")[-1]

    # model is definitely from HF
    model_name = ckpt["hyper_parameters"]["model"]
    model_name = model_name.partition("/")[-1]

    repo_id = f"{hf_user_id}/expert__{model_name}__{dataset_name}__{expert_name}"

    logger.info("Uploading checkpoint {} --> {}".format(ckpt_path, repo_id))
    convert_and_push_to_hub(ckpt_path, repo_id, auto_search=False, use_last=False)


class MultiExpertModel(ExpertTrainer):
    """
    MultiExpert models handels multiple experts with ExpertContainer and allows to route to different experts.
    You can add modifiers using one of the 'self.modify_weith...' methods.
    """

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
        for module_name, module_data in graph.create_modules(
            base_hparams=self.hparams, **kwargs
        ).items():
            print("Loading module: {}".format(module_name))
            self.model = add_expert_to_transformer(
                self.model,
                module_name,
                module_data.expert_config,
                module_data.expert_weights,
                action=action,
                is_default=module_name == "default",
                config=self.hparams,
            )
            self.experts.append(module_name)

    def convert_container_to_expert(self, expert_name, get_expert_instance=True):
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

    def add_expert_instance(self, expert_instance: Expert, expert_name, action="route"):
        self.model = add_expert_to_transformer(
            self.model,
            expert_name,
            expert_instance.expert_config,
            expert_instance.expert_weights,
            action=action,
            is_default=expert_name == "default",
        )
        if action != "merge":
            self.experts.append(expert_name)

    def load_from_graph_string(self, s, action="route"):
        from mttl.models.modifiers.expert_containers.module_graph import ModuleGraph

        graph = ModuleGraph.from_string(s)
        self.load_from_graph(graph, action=action)

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
            expert_name,
            expert.expert_config,
            expert.expert_weights,
            action=action,
            is_default=is_default,
            load_only_layers=load_only_layers,
        )
        if action != "merge":
            self.experts.append(expert_name)

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
        if self.hparams.routing == "first":
            batch["task_names"] = [
                self.experts[0] for _ in range(batch["input_ids"].shape[0])
            ]
        elif self.hparams.routing == "random":
            import numpy as np

            batch["task_names"] = np.random.choice(
                self.experts, batch["input_ids"].shape[0], replace=True
            ).tolist()

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

        self.classifier = ExpertRanker(
            num_labels=kwargs["num_labels"],
            classifer_repo_id=kwargs["classifer_repo_id"],
        ).get_classifer()

    def get_predicted_experts(self, batch):
        if "inputs" in batch:
            input_texts = batch["inputs"]
        elif "sources_texts" in batch:
            input_texts = batch["sources_texts"]
        else:
            raise ValueError("No inputs found in batch!")
        expert_logits = self.classifier(input_texts)
        expert_indices = expert_logits.argmax(dim=1).cpu()
        expert_prediction = [ids_to_tasks_names[i.item()] for i in expert_indices]
        return expert_prediction

    def expert_retrieval(self, batch, **kwargs):
        expert_selection = []
        # get the expert predictions
        expert_prediction = self.get_predicted_experts(batch)
        print("predicted experts: {}".format(expert_prediction))
        for expert in expert_prediction:
            if expert in self.experts:
                expert_selection.append(expert)
            else:
                # randomly select an expert
                expert_selection.append(
                    self.experts[np.random.randint(len(self.experts))]
                )

        return expert_selection

    def generate(
        self,
        batch,
        **kwargs,
    ):
        if self.hparams.routing == "first":
            batch["task_names"] = [
                self.experts[0] for _ in range(batch["input_ids"].shape[0])
            ]
        elif self.hparams.routing == "random":
            import numpy as np

            batch["task_names"] = np.random.choice(
                self.experts, batch["input_ids"].shape[0], replace=True
            ).tolist()
        elif self.hparams.routing == "retrieval":
            logger.info("retrieval routing")
            batch["task_names"] = self.expert_retrieval(batch)

        if hasattr(self.model, "task_id_container"):
            self.model.task_id_container["routing_infos"] = RoutingInfo.from_batch(
                batch
            )

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

        expert = load_expert(expert_path)
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
            expert_name,
            expert.expert_config,
            expert.expert_weights,
            action=action,
            is_default=is_default,
            load_only_layers=load_only_layers,
            selectors=self.selectors,
            config=self.hparams,
        )
        if action != "merge":
            self.experts.append(expert_name)

    def load_from_graph(self, graph, action="route", **kwargs):
        for module_name, module_data in graph.create_modules(
            base_hparams=self.hparams, **kwargs
        ).items():
            self.model = add_expert_to_transformer(
                self.model,
                module_name,
                module_data.expert_config,
                module_data.expert_weights,
                action=action,
                is_default=module_name == "default",
                selectors=self.selectors,
                config=self.hparams,
            )
            self.experts.append(module_name)

    def get_router_weights(self):
        weights = {}
        for _, selector in self.selectors.items():
            weights[selector.name] = selector.get_routing_weights()
        return weights

    def add_expert_instance(self, expert_instance, expert_name, action="route"):
        self.model = add_expert_to_transformer(
            self.model,
            expert_name,
            expert_instance.expert_config,
            expert_instance.expert_weights,
            action=action,
            is_default=expert_name == "default",
            selectors=self.selectors,
            config=self.hparams,
        )
        if action != "merge":
            self.experts.append(expert_name)

    def to_expert(self, weights: dict = None) -> Expert:
        """
        Merges current experts together according to weights if given, otherwise uses router's weights
        """
        for _, module in self.model.named_modules():
            for c_name, child in dict(module.named_children()).items():
                if isinstance(child, ExpertContainer) and len(child.experts) > 0:
                    # creates a single Lora
                    child.merge_experts_together(weights)
        return self.convert_container_to_expert("merged_expert")

    def on_save_checkpoint(self, ckpt):
        expert: Expert = self.to_expert()
        ckpt["expert_dumps"] = expert.dumps()
        ckpt["merging_weights"] = self.get_router_weights()
