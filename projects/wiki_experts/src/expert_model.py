import torch

from mttl.models.modifiers.routing import RoutingInfo
from mttl.models.modifiers.experts import add_expert_to_transformer, Router
from mttl.utils import logger
from projects.wiki_experts.src.expert_trainer import ExpertTrainer
from typing import Dict
from projects.wiki_experts.adapter_ranker import get_classifier
from mttl.models.modifiers.experts import ExpertContainer


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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.selectors: Dict[str, Router] = {}
        self.experts = []

        self.classifier = get_classifier()

    def get_router_weights(self):
        weights = {}
        for _, selector in self.selectors.items():
            weights[selector.name] = selector.get_routing_weights()
        return weights

    def load_from_graph(self, graph, action="route", **kwargs):
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
                selectors=self.selectors,
            )
            self.experts.append(module_name)

        for _, selector in self.selectors.items():
            selector.resize_module_logits(self.experts)

    def convert_container_to_expert(self, expert_name):
        loaded_expert = None
        for _, module in self.model.named_modules():
            for c_name, child in dict(module.named_children()).items():
                if isinstance(child, ExpertContainer) and len(child.experts) > 0:
                    setattr(module, c_name, child.experts[expert_name])
                    if loaded_expert is None:
                        loaded_expert = child.experts[expert_name]
        # make sure hparams reflect the loaded expert
        if loaded_expert:
            self.hparams.update(loaded_expert.config.__dict__)

    def load_from_module_dict(self, module_dict, action="route"):
        for module_name, destination in module_dict.items():
            self.load_expert(destination, module_name, action=action)

    def load_from_graph_string(self, s, action="route"):
        from projects.wiki_experts.src.graph.module_graph import ModuleGraph

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
        from projects.wiki_experts.src.graph.module_graph import load_expert

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

    def expert_retrieval(self, batch, **kwargs):
        if "inputs" in batch:
            input_texts = batch["inputs"]
        elif "sources_texts" in batch:
            input_texts = batch["sources_texts"]
        else:
            raise ValueError("No inputs found in batch!")
        expert_selection = []
        # get the expert predictions
        expert_predictions = self.classifier(input_texts)
        expert_indices = expert_predictions.argmax(dim=1).cpu()
        for i in expert_indices:
            if i < len(self.experts):
                expert_selection.append(self.experts[i])
            else:
                expert_selection.append(self.experts[0])
        return expert_selection

    def expert_choice(self, batch, **kwargs):
        input_ids = batch["input_ids"]
        mask = batch["attention_mask"]

        # convert left to right padding here
        def roll_along(arr, shifts, dim):
            assert arr.ndim - 1 == shifts.ndim
            dim %= arr.ndim
            shape = (1,) * dim + (-1,) + (1,) * (arr.ndim - dim - 1)
            dim_indices = torch.arange(arr.shape[dim]).reshape(shape).to(arr.device)
            indices = (dim_indices - shifts.unsqueeze(dim)) % arr.shape[dim]
            return torch.gather(arr, dim, indices)

        input_ids = roll_along(input_ids, mask.sum(1), 1)
        mask = input_ids.ne(0)
        labels = torch.masked_fill(input_ids, ~mask, -100)

        scores = []
        for expert in self.experts:
            batch["task_names"] = [expert for _ in range(batch["input_ids"].shape[0])]
            self.model.task_id_container["routing_infos"] = RoutingInfo.from_batch(
                batch
            )
            outputs = self.model.forward(
                input_ids,
                attention_mask=mask,
            )
            # calculate loss, could also be done inside of the model
            bs = input_ids.size(0)
            logits = outputs.logits
            vocab_size = logits.size(-1)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            shift_logits = shift_logits.view(-1, vocab_size)
            shift_labels = shift_labels.view(-1)

            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            loss = loss.view((bs, -1)).sum(1)
            # mean only non-zero
            scores.append(loss.cpu())

        scores = torch.stack(scores, 0)
        expert_indices = scores.argmin(0)
        return [self.experts[i] for i in expert_indices]

    # def forward(self, batch, reduction="mean"):

    def generate(
        self,
        batch,
        **kwargs,
    ):
        if self.hparams.routing == "auto":
            logger.info(
                "Auto-routing... ground-truth tasks: {}".format(batch["task_names"])
            )
            batch["task_names"] = self.expert_choice(batch)
            logger.info("Auto-route tasks: {}".format(batch["task_names"]))
        elif self.hparams.routing == "first":
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

        generations = self.model.generate(inputs=batch["input_ids"], **kwargs)
        return generations


class RoutedMultiExpertModel(MultiExpertModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.selectors = torch.nn.ModuleDict()
        self.load_from_module_dict(self.hparams.module_dict, action="route")

    def load_expert(
        self,
        expert_path: str,
        expert_name: str = None,
        action: str = "merge",
        is_default: bool = False,
        load_only_layers: str = None,
    ):
        from projects.wiki_experts.src.graph.module_graph import load_expert

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
        self.resize_selector_logits()

    def load_from_module_dict(self, module_dict, action="route"):
        out = super().load_from_module_dict(module_dict, action)
        self.resize_selector_logits()
        return out

    def resize_selector_logits(self):
        for _, selector in self.selectors.items():
            if selector:
                selector.resize_module_logits(self.experts)

    def get_router_weights(self):
        weights = {}
        for _, selector in self.selectors.items():
            weights[selector.name] = selector.get_routing_weights()
        return weights

    def merge_experts_together(self, weights: dict = None):
        """
        Merges current experts together according to weights if given, otherwise uses router's weights
        """
        for _, module in self.model.named_modules():
            for c_name, child in dict(module.named_children()).items():
                if isinstance(child, ExpertContainer) and len(child.experts) > 0:
                    # creates a single Lora
                    child.merge_experts_together(weights)
        self.convert_container_to_expert("merged_expert")
