from abc import abstractproperty
from pyparsing import abstractmethod
import torch
import math
from torch import nn
from typing import Any, Dict
from mttl.utils import logger
import torch.nn.functional as F
from mttl.models.modifiers.expert_containers.module_graph import (
    ExpertInfo,
    ExpertConfig,
)


MULTI_EXPERT_ROUTERS = {}
EPS = 1e-8


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


class Selector:
    @abstractmethod
    def forward(self, input, **kwargs) -> list:
        pass

    @abstractmethod
    def get_routing_weights(self):
        pass

    @abstractproperty
    def name(self):
        pass

    def add_expert(self, expert_name: str):
        if expert_name not in self.expert_names:
            self.expert_names.append(expert_name)

    def add_experts(self, expet_names: list):
        for expert_name in expet_names:
            self.add_expert(expert_name)


@register_multi_expert_selector("poly_router")
class MultiExpertSelector(torch.nn.Module, Selector):
    """
    Implements routing at a per-layer or pe-model level
    """

    def __init__(self, config, info_container=None, **kwargs) -> None:
        super().__init__()
        self.config = config
        self.expert_names: list = []

        self.module_logits = nn.Parameter(torch.empty(1).uniform_(-1e-3, 1e-3))

        self.__layer_name__ = f"poly_router"

    @property
    def name(self):
        return f"{self.__layer_name__}"

    def forward(self, *args, **kwargs):
        module_logits = torch.sigmoid(self.module_logits)
        module_weights = module_logits / (module_logits.sum(dim=-1, keepdim=True) + EPS)
        return [{k: v for k, v in zip(self.expert_names, module_weights)}]

    def get_routing_weights(self):
        weights = self.forward()[0]
        return {k: v.detach().item() for k, v in weights.items()}

    def add_expert(self, expert_name: str, **kwargs):
        if expert_name not in self.expert_names:
            self.expert_names.append(expert_name)
            self.module_logits.data = torch.empty(len(self.expert_names)).uniform_(
                -1e-3, 1e-3
            )


@register_multi_expert_selector("poly_router_dir")
class MultiExpertSelector(torch.nn.Module, Selector):
    """
    Implements routing at a per-layer or pe-model level. Initialized all experts' weights to 0 with the exception of the expert responsible for the curent task.'
    """

    def __init__(self, config, device="cuda", **kwargs) -> None:
        super().__init__()
        self.config: ExpertConfig = config
        self.device = device

        self.module_logits = torch.nn.ParameterDict()

        self.__layer_name__ = f"poly_router_dir"

        self.init_gap = [0, 0]
        self.main_m = 1

    @property
    def name(self):
        return f"{self.__layer_name__}"

    def forward(self, *args, **kwargs):
        return [self.module_logits]

    def get_routing_weights(self):
        weights = self.forward()[0]
        return {k: v.detach().item() for k, v in weights.items()}

    def add_expert(self, expert_name: str, expert_task_name: str, **kwargs):
        """
        Assume:
        expert_task_name -- task name expert is pecialized in
        self.config.finetune_task_name -- name of the task the model is currently trained on
        """
        init_gap = [0, 0]
        main_m = 1
        if expert_name not in self.module_logits:
            if self.config.finetune_task_name == expert_task_name:
                self.module_logits[expert_name] = torch.nn.Parameter(
                    torch.ones(1).to(self.device)
                )
                self.module_logits[expert_name].data *= main_m
            else:
                self.module_logits[expert_name] = torch.nn.Parameter(
                    torch.empty(1).uniform_(*init_gap).to(self.device)
                )


@register_multi_expert_selector("task_selector")
class TaskNameSelector(torch.nn.Module, Selector):
    def __init__(self, config=None, info_container=None, **kwargs) -> None:
        super().__init__()
        self.info_container = info_container
        self.__layer_name__ = f"task_selector"
        self.expert_names = []
        self.default_expert_name = None

    def forward(self, *args, **kwargs):
        # try to infer batch size
        if "routing_infos" not in self.info_container:
            if "input_ids" in kwargs:
                batch_size = kwargs["input_ids"].size(0)
            else:
                raise ValueError(
                    "routing_infos not in info_container and cannot infer batch size."
                )

            if not self.default_expert_name:
                raise ValueError("No default expert name set and no task names given!")

            routing_weights = [
                {self.default_expert_name: torch.tensor(1.0)} for _ in range(batch_size)
            ]
        else:
            task_names = self.info_container["routing_infos"].task_names

            if (
                any(task_name not in self.expert_names for task_name in task_names)
                and not self.default_expert_name
                and len(self.expert_names)
            ):
                raise ValueError(
                    "Experts for all tasks have not been loaded! Set a default expert?"
                )

            routing_weights = [
                {task_name: torch.tensor(1.0)} for task_name in task_names
            ]
        return routing_weights

    def add_expert(self, expert_name: str, *args, **kwargs):
        # here we experts based on their name, which can be different from the task name
        if expert_name not in self.expert_names:
            self.expert_names.append(expert_name)

    @property
    def name(self):
        return f"{self.__layer_name__}"


class KVSelector(Selector):
    """Selector specific to KV adapters. The KV Adapter modifies the self-attention
    call, adding the following execution :

    1. adapter_k, adapter_v = adapter.get_kv_weights(k_proj, v_proj)
    2. adapter_weights = adapter.route(query_states, adapter_k, self)
        2.1 `adapter.route` calls get_gate(adapter_weights)
    3. adapter_output = adapter.aggregate(adapter_weights, adapter_v)

    To enable custom routing, one typically needs to modify `get_kv_weights` and `get_gate`.
    For example, see `KVTaskNameSelector` for an example.

    You can also overwrite the `route` method; the `KVExpertContainer` will call it instead of
    the default `route` method (see lines 199-201 in `KVExpertContainer`)
    """

    @abstractmethod
    def get_kv_weights(self, k_proj, v_proj):
        pass

    @abstractmethod
    def get_gate(self, adapter_weights):
        pass

    @property
    def name(self):
        return f"{self.__layer_name__}"

    @property
    def n_experts(self):
        return len(self.expert_names)


@register_multi_expert_selector("kv_task_selector")
class KVTaskNameSelector(KVSelector):
    """Selects KVAdapters based on the task name."""

    def __init__(self, config=None, info_container=None, **kwargs) -> None:
        super().__init__()
        self.info_container = info_container
        self.__layer_name__ = f"task_selector"
        self.expert_names = []
        self.default_expert_name = None

    def get_kv_weights(self, experts, k_proj, v_proj):
        task_names = self.info_container["routing_infos"].task_names

        if task_names is None:
            task_names = [self.default_expert_name]

        if len(set(task_names)) == 1:
            # broadcast along batch dim if only 1 task
            adapter_k, adapter_v = experts[task_names[0]].get_kv_weights(k_proj, v_proj)
            return adapter_k, adapter_v

        out = zip(
            *[experts[name].get_kv_weights(k_proj, v_proj) for name in task_names]
        )
        out = (torch.cat(tensors, dim=0) for tensors in out)
        return out

    def get_gate(self, experts, adapter_weights):
        task_names = self.info_container["routing_infos"].task_names

        if task_names is None:
            task_names = [self.default_expert_name]

        if len(set(task_names)) == 1:
            # broadcast along batch dim if only 1 task
            out = experts[task_names[0]].get_gate(adapter_weights)
            return out

        return torch.cat(
            [experts[name].get_gate(adapter_weights) for name in task_names],
        )


@register_multi_expert_selector("kv_concat_selector")
class KVConcatSelector(KVSelector, nn.Module):
    """Concatenates along the sequence dim. all the adapters, and lets the
    model's internal attention mechanism take care of routing in a task agnostic way
    """

    def __init__(self, config=None, info_container=None, **kwargs) -> None:
        super().__init__()
        self.config = config
        self.info_container = info_container
        self.__layer_name__ = f"task_selector"
        self.expert_names = []
        self.default_expert_name = None

    def get_kv_weights(self, experts, k_proj, v_proj):
        out = zip(
            *[
                kv_adapter.get_kv_weights(k_proj, v_proj)
                for kv_adapter in experts.values()
            ]
        )
        # NO (1, n_experts, n_heads, soft_prompt_len, head_dim)
        # (n_experts, n_heads, soft_prompt_len, head_dim)
        adapter_k, adapter_v = (torch.cat(tensors, dim=0) for tensors in out)
        n_experts, n_heads, soft_prompt_len, head_dim = adapter_k.size()

        # (n_heads, n_experts * soft_prompt_len, head_dim)
        adapter_k = adapter_k.transpose(0, 1).reshape(
            1, n_heads, n_experts * soft_prompt_len, head_dim
        )
        adapter_v = adapter_v.transpose(0, 1).reshape(
            1, n_heads, n_experts * soft_prompt_len, head_dim
        )

        return adapter_k, adapter_v

    def get_gate(self, experts, adapter_weights):
        bsz, n_heads, q_len, n_exp_kv_len = adapter_weights.size()
        adapter_weights = adapter_weights.view(bsz, n_heads, q_len, len(experts), -1)

        # sum probs over all `soft_prompt_len` keys, to get (bsz, n_heads, q_len, n_experts)
        per_expert_weight = adapter_weights.sum(dim=-1)

        # (n_experts, n_heads, 1, 1)
        all_gates = torch.cat(
            [kv_adapter.get_gate(adapter_weights) for kv_adapter in experts.values()]
        )
        # output : (bsz, n_heads, q_len, 1)
        out = torch.einsum("bhqe,ehab->bhqa", per_expert_weight, all_gates)
        return out


class KVNormSelector(KVSelector):
    def route(self, experts, query, keys, attn_layer):
        """(2) Compute The Standard Attention Scores in augmented attention"""

        query = F.normalize(query, dim=-1, p=2)
        keys = F.normalize(keys, dim=-1, p=2)

        adapter_logits = torch.matmul(
            query, keys.transpose(2, 3).type_as(query)
        ) / math.sqrt(attn_layer.head_dim)

        adapter_weights = F.softmax(adapter_logits, dim=-1, dtype=torch.float32)
        gate_out = self.get_gate(experts, adapter_weights)
        out = gate_out * adapter_weights.type_as(query)

        return out


@register_multi_expert_selector("kv_concat_norm_selector")
class KVConcatNormSelector(KVConcatSelector, KVNormSelector):
    pass


@register_multi_expert_selector("kv_task_norm_selector")
class KVTaskNameNormSelector(KVTaskNameSelector, KVNormSelector):
    pass
