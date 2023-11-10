from abc import abstractproperty
import re
from pyparsing import abstractmethod
import torch
from torch import nn
from typing import Any, Dict
from mttl.models.modifiers.base import MergeableAdapter
from mttl.models.modifiers.lora import LoRA, SkilledLoRA
from mttl.utils import logger


MULTI_EXPERT_ROUTERS = {}


def register_multi_expert_selector(name):
    print("Registering multi-expert selector..." + name)

    def _thunk(fn):
        if name in MULTI_EXPERT_ROUTERS:
            raise ValueError(
                f"Cannot register duplicate multi-expert selector ({name})"
            )
        MULTI_EXPERT_ROUTERS[name] = fn
        return fn

    return _thunk


def get_selector(config, **kwargs):
    if config.expert_routing:
        if config.expert_routing not in MULTI_EXPERT_ROUTERS:
            raise ValueError(f"Cannot find selector: {config.expert_routing}")
        return MULTI_EXPERT_ROUTERS[config.expert_routing](config, **kwargs)
    else:
        return None


class Router:
    @abstractmethod
    def forward(self, input, **kwargs):
        pass

    @abstractmethod
    def get_routing_weights(self):
        pass

    @abstractproperty
    def name(self):
        pass


@register_multi_expert_selector("poly_router")
class Multi_ExpertRouter(torch.nn.Module, Router):
    """
    Implements routing at a per-layer or pe-model level
    """

    def __init__(self, config, expert_names=[]):
        super().__init__()
        self.config = config
        self.expert_names: list = expert_names

        self.module_logits = nn.Parameter(
            torch.empty(len(expert_names)).uniform_(-1e-3, 1e-3)
        )

        self.__layer_name__ = f"poly_router"

    def resize_module_logits(self, expet_names: list):
        self.expert_names += expet_names
        self.module_logits.data = torch.empty(len(self.expert_names)).uniform_(
            -1e-3, 1e-3
        )

    @property
    def name(self):
        return f"{self.__layer_name__}"

    def forward(self, *args, **kwargs):
        module_logits = torch.sigmoid(self.module_logits)
        module_weights = module_logits / (module_logits.sum(dim=-1, keepdim=True) + EPS)
        return {k: v for k, v in zip(self.expert_names, module_weights)}

    def get_routing_weights(self):
        return self.forward()


def add_expert_to_transformer(
    transformer,
    expert_name,
    expert_config,
    expert_weights,
    action="route",
    is_default=False,
    load_only_layers=None,
    selectors={},
):
    # create a shared container for the task id
    if not hasattr(transformer, "task_id_container"):
        transformer.task_id_container = {}

    total_layers = 0
    added_layers = []

    for m_name, module in dict(transformer.named_modules()).items():
        if re.fullmatch(expert_config.modify_modules, m_name):
            for c_name, layer in dict(module.named_children()).items():
                if re.fullmatch(expert_config.modify_layers, c_name):
                    total_layers += 1
                    layer_name = f"{m_name}.{c_name}"

                    if type(layer) != ExpertContainer:
                        # create an expert lora container
                        expert_container = ExpertContainer(
                            expert_config,
                            transformer.task_id_container,
                            layer,
                        )
                        expert_container.__layer_name__ = layer_name
                        setattr(
                            module,
                            c_name,
                            expert_container,
                        )
                    else:
                        expert_container = layer

                    # subset the relevant expert weights starting w __layer_name__
                    subset_expert_weights = {
                        k.replace(expert_container.__layer_name__ + ".", ""): v
                        for k, v in expert_weights.items()
                        if k.startswith(expert_container.__layer_name__)
                    }

                    layer_num = int(expert_container.__layer_name__.split(".")[2])

                    if load_only_layers:
                        pos = load_only_layers.find("-")
                        sel = int(load_only_layers.replace("-", ""))

                        if pos == 0:
                            # add until layer number excluded
                            if layer_num >= sel:
                                continue
                        else:
                            if layer_num < sel:
                                continue

                    added_layers.append(expert_container.__layer_name__)
                    expert_container.add_expert(
                        expert_name,
                        expert_config,
                        subset_expert_weights,
                        action=action,
                        is_default=is_default,
                    )

    logger.info("Adding expert to layers %s", added_layers)
    return transformer


class ExpertContainer(MergeableAdapter):
    def __init__(self, config, task_id_container, layer, selector=None):
        super().__init__()
        self.config = config
        self.layer = layer
        self.selector = selector

        if not isinstance(self.layer, nn.Linear):
            raise ValueError(
                "Expert containers for layers other than nn.Linear have not been implemented."
            )

        self.info_container = task_id_container
        self.default_expert_name = None
        self.merged_expert_names = []
        self.experts = nn.ModuleDict({})

    def add_expert(
        self,
        name: str,
        expert_config: Any,
        expert_weights: Dict[str, torch.Tensor],
        action="merge",
        is_default=False,
    ) -> None:
        if name in self.experts:
            raise ValueError("An expert with name {} already exists.".format(name))

        if is_default and action == "merge":
            raise ValueError(
                "Cannot set is_default if this expert is merged, change to 'route'."
            )

        # hack this for now, but build a proper config for each module
        if expert_config.model_modifier == "lora":
            expert_module = LoRA(expert_config, self.layer)
            expert_module.load_lora_weights(expert_weights)
        else:
            raise NotImplementedError("ExpertContainer only supports LoRA experts.")

        if action == "merge":
            # weight is merged with layer so we can discard it now
            if expert_config.model_modifier == "lora":
                expert_module.merge_with_layer()
                self.merged_expert_names.append(name)
            else:
                raise NotImplementedError("Merging experts only supports LoRA experts.")
        else:
            # we keep track of the expert weights
            self.experts[name] = expert_module
        if is_default:
            self.default_expert_name = name

    def merge_with_layer(self):
        if len(self.experts) > 0:
            for name, expert_module in self.experts.items():
                assert isinstance(
                    expert_module, LoRA
                ), "Only LoRA experts can be merged with the layer for now."
                expert_module.merge_with_layer()
                self.merged_expert_names.append(name)
                self.experts.pop(name)

    def weighted_route(self, input, task_weights):
        """
        Route all examples according to the weights given in the weights dictionary: expert_name -> weight
        """
        load_experts = []
        weights = []

        for task_name, weight in task_weights.items():
            load_experts.append(self.experts[task_name])
            weights.append(weight)
        # assume all experts are loras
        output = SkilledLoRA.weighted_merge_forward(
            input, load_experts, weights, merge_after=True
        )
        return output

    def route_with_task_name(self, input, task_names):
        """
        Route according to the task name information
        """
        load_experts = []

        for task_name in task_names:
            if task_name not in self.experts:
                if not self.default_expert_name:
                    raise ValueError(
                        "The expert for this task {} does not exists. Consider setting a default expert!".format(
                            task_name
                        )
                    )
                else:
                    selected_expert = self.default_expert_name
            else:
                selected_expert = task_name
            load_experts.append(self.experts[selected_expert])
        # assume all experts are loras
        output = LoRA.parallel_linear_forward(input, load_experts)
        return output

    def forward(self, input, **kwargs):
        task_names = self.info_container["routing_infos"].task_names

        if task_names and (
            any(task_name not in self.experts for task_name in task_names)
            and not self.default_expert_name
            and len(self.experts)
        ):
            raise ValueError(
                "Experts for all tasks have not been loaded! Set a default expert?"
            )

        # if it has some routing experts *and* task names, then we can route
        if len(self.experts) and self.selector is None:
            assert (
                task_names is not None
            ), "Task names are not given: set router or merge experts into the layer."
            output = self.route_with_task_name(input, task_names)
        elif len(self.experts) and self.selector is not None:
            weights: Dict = self.selector(input)
            output = self.weighted_route(input, weights)
        else:
            ###############################################################
            ## no experts -- no routing, experts were merged into the layer
            output = self.layer(input, **kwargs)
        return output
