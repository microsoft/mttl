from abc import abstractproperty
from pyparsing import abstractmethod
import torch
import math
from torch import nn
from typing import Any, Dict
from mttl.utils import logger
import torch.nn.functional as F


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

    @abstractmethod
    def add_expert(self, expert_name: str):
        pass

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
        return self.forward()[0]

    def add_expert(self, expert_name: str):
        if expert_name not in self.expert_names:
            self.expert_names.append(expert_name)
            self.module_logits.data = torch.empty(len(self.expert_names)).uniform_(
                -1e-3, 1e-3
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
                {self.default_expert_name: 1.0} for _ in range(batch_size)
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

            routing_weights = [{task_name: 1.0} for task_name in task_names]
        return routing_weights

    def add_expert(self, expert_name: str):
        if expert_name not in self.expert_names:
            self.expert_names.append(expert_name)

    @property
    def name(self):
        return f"{self.__layer_name__}"


class KVSelector(Selector):
    """KV Specific Stuff"""

    @abstractmethod
    def get_kv_weights(self, k_proj, v_proj):
        pass

    @abstractmethod
    def get_gate(self, adapter_weights):
        pass

    def add_expert(self, expert_name: str):
        if expert_name not in self.expert_names:
            self.expert_names.append(expert_name)

    @property
    def name(self):
        return f"{self.__layer_name__}"

    @property
    def n_experts(self):
        return len(self.expert_names)


@register_multi_expert_selector("kv_task_selector")
class KVTaskNameSelector(KVSelector):
    def __init__(self, config=None, info_container=None, **kwargs) -> None:
        super().__init__()
        self.info_container = info_container
        self.__layer_name__ = f"task_selector"
        self.expert_names = []
        self.default_expert_name = None

    def get_kv_weights(self, experts, k_proj, v_proj):
        task_names = self.info_container["routing_infos"].task_names
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
        if len(set(task_names)) == 1:
            # broadcast along batch dim if only 1 task
            out = experts[task_names[0]].get_gate(adapter_weights)
            return out

        return torch.cat(
            [experts[name].get_gate(adapter_weights) for name in task_names],
        )


@register_multi_expert_selector("kv_concat_selector")
class KVConcatSelector(KVSelector, nn.Module):
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

        """
        # NO (1, n_heads, n_experts * soft_prompt_len, head_dim)
        adapter_k = adapter_k.transpose(1, 2).reshape(
            1, n_heads, n_experts * soft_prompt_len, head_dim
        )
        adapter_v = adapter_v.transpose(1, 2).reshape(
            1, n_heads, n_experts * soft_prompt_len, head_dim
        )
        """
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


@register_multi_expert_selector("kv_mean_selector")
class KVMeanSelector(KVConcatSelector):
    def route(self, experts, query, keys, attn_layer):
        """(2) Compute The Standard Attention Scores in augmented attention"""

        adapter_logits = torch.matmul(
            query, keys.transpose(2, 3).type_as(query)
        ) / math.sqrt(attn_layer.head_dim)

        shp = adapter_logits.size()
        adapter_logits = adapter_logits.view(
            *adapter_logits.shape[:-1], self.n_experts, -1
        )
        # uniform over experts
        adapter_weights = (
            F.softmax(adapter_logits, dim=-1, dtype=torch.float32).view(shp)
            / self.n_experts
        )
        gate_out = self.get_gate(experts, adapter_weights)
        out = gate_out * adapter_weights.type_as(query)

        return out


@register_multi_expert_selector("kv_temp_selector")
class KVTempSelector(KVConcatSelector):
    def route(self, experts, query, keys, attn_layer):
        """(2) Compute The Standard Attention Scores in augmented attention"""

        adapter_logits = (
            torch.matmul(query, keys.transpose(2, 3).type_as(query))
            / math.sqrt(attn_layer.head_dim)
            / 0.005
        )

        adapter_weights = F.softmax(adapter_logits, dim=-1, dtype=torch.float32)
        gate_out = self.get_gate(experts, adapter_weights)
        out = gate_out * adapter_weights.type_as(query)

        return out
