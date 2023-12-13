from abc import abstractproperty
from dataclasses import dataclass
from typing import Dict, List, Union
from pyparsing import abstractmethod
import torch
import math
from torch import nn
import torch.nn.functional as F
from mttl.models.modifiers.routing import RoutingMixin

from mttl.utils import logger


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
        fn.__layer_name__ = name
        return fn

    return _thunk


@dataclass
class RoutingConfig:
    # what type of selector to use
    router_selector: str
    # the granularity of the selector (which layers use the same selectors)
    router_granularity: str = "*"


@dataclass
class SelectorOutput:
    pass


@dataclass
class ModulesAndWeightsSelectorOutput(SelectorOutput):
    modules: List[str]
    weights: Union[List[float], torch.Tensor]


@dataclass
class BatchModulesAndWeightsSelectorOutput(SelectorOutput):
    modules: List[List[str]]
    weights: Union[List[List[float]], torch.Tensor]


@dataclass
class BatchAndSequenceModulesAndWeightsSelectorOutput(SelectorOutput):
    indices: torch.Tensor
    weights: Union[List[List[float]], torch.Tensor]


@dataclass
class ModulesSelectorOutput(SelectorOutput):
    modules: List[str]


class Selector(RoutingMixin, nn.Module):
    def __init__(self, info_container, **kwargs):
        nn.Module.__init__(self)
        RoutingMixin.__init__(self, info_container)

        self.expert_names = []

    @abstractmethod
    def forward(self, input, **kwargs) -> SelectorOutput:
        pass

    def get_merged_weights(self, container, **selector_kwargs) -> Dict:
        return {}

    @property
    def name(self):
        return f"{self.__layer_name__}"

    @abstractmethod
    def add_expert(self, expert_name: str, **kwargs):
        pass


@register_multi_expert_selector("poly_router")
class PolySelector(Selector):
    """
    Implements routing at a per-layer or per-model level
    """

    def __init__(self, info_container=None, **kwargs) -> None:
        super().__init__(info_container)

        self.module_logits = nn.Parameter(torch.empty(1).uniform_(-1e-3, 1e-3))

    def _get_weights(self):
        module_logits = torch.sigmoid(self.module_logits)
        module_weights = module_logits / (module_logits.sum(dim=-1, keepdim=True) + EPS)
        return module_weights

    def forward(self, input, **kwargs) -> ModulesAndWeightsSelectorOutput:
        weights = self._get_weights()
        modules = self.expert_names
        return ModulesAndWeightsSelectorOutput(modules, weights)

    def get_merged_weights(self, container, **selector_kwargs) -> Dict:
        """Return the merged weights for the experts in the container."""
        if selector_kwargs.get("weights", None) is None:
            weights = self.get_routing_weights()

        merged_weights = {}
        for name, expert in container.experts.items():
            assert name in weights, f"Weight for expert {name} is not given"
            expert_state_dict = expert.state_dict()
            weight = weights[name]

            for k, v in expert_state_dict.items():
                if not "lora" in k:
                    continue
                value = weight * v
                if k in merged_weights:
                    merged_weights[k] += value
                else:
                    merged_weights[k] = value
        return merged_weights

    def get_routing_weights(self):
        return {
            k: v.detach().item() for k, v in zip(self.expert_names, self._get_weights())
        }

    def add_expert(self, expert_name: str, **kwargs):
        if expert_name not in self.expert_names:
            self.expert_names.append(expert_name)
            self.module_logits.data = torch.empty(len(self.expert_names)).uniform_(
                -1e-3, 1e-3
            )


@register_multi_expert_selector("moe_rkhs_router")
class MOERKHSSelector(Selector):
    def load_balancing_loss_func(
        self, gate_logits: torch.Tensor, num_experts: torch.Tensor, top_k=2
    ) -> float:
        if gate_logits is None:
            return 0

        if isinstance(gate_logits, tuple):
            # cat along the layers?
            gate_logits = torch.cat(gate_logits, dim=0)

        routing_weights, selected_experts = torch.topk(gate_logits, top_k, dim=-1)
        routing_weights = routing_weights.softmax(dim=-1)

        # cast the expert indices to int64, otherwise one-hot encoding will fail
        if selected_experts.dtype != torch.int64:
            selected_experts = selected_experts.to(torch.int64)

        if len(selected_experts.shape) == 2:
            selected_experts = selected_experts.unsqueeze(2)

        expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

        # For a given token, determine if it was routed to a given expert.
        expert_mask = torch.max(expert_mask, axis=-2).values

        # cast to float32 otherwise mean will fail
        expert_mask = expert_mask.to(torch.float32)
        tokens_per_group_and_expert = torch.mean(expert_mask, axis=-1)
        router_prob_per_group_and_expert = torch.mean(routing_weights, axis=-1)

        return torch.mean(
            tokens_per_group_and_expert * router_prob_per_group_and_expert
        ) * (num_experts**2)

    def __init__(self, info_container=None, **kwargs) -> None:
        super().__init__(info_container)

        if "layer" not in kwargs:
            raise ValueError(
                "MOERKHSSelector requires a layer to be passed in kwargs to infer the input dimension."
            )

        self.topk = 2
        self.input_dim = kwargs["layer"].weight.data.shape[-1]
        self.rkhs_exp = nn.Linear(128, 1024)
        self.rkhs_hid = nn.Linear(self.input_dim, 1024)
        self.rkhs_embeddings = nn.Parameter(torch.empty((0, 128)))

    def _get_weights(self, input):
        input_view = input.view(-1, input.shape[-1])
        return self.rkhs_hid(input_view).reshape(input.shape[0], input.shape[1], -1)

    def forward(
        self, input, **kwargs
    ) -> BatchAndSequenceModulesAndWeightsSelectorOutput:
        rkhs_enc = self._get_weights(input)
        rkhs_emb = self.rkhs_exp(self.rkhs_embeddings)

        router_logits = torch.matmul(rkhs_enc, rkhs_emb.T)
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.topk, dim=-1
        )
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        # we cast back to the input dtype
        routing_weights = routing_weights.to(input.dtype)

        aux_losses = self.routing_infos.aux_losses.get(f"moe_balance_loss", [])
        aux_losses.append(
            self.load_balancing_loss_func(
                router_logits, self.rkhs_embeddings.data.shape[0], self.topk
            )
        )
        self.routing_infos.aux_losses[f"moe_balance_loss"] = aux_losses

        return BatchAndSequenceModulesAndWeightsSelectorOutput(
            indices=selected_experts, weights=routing_weights
        )

    def get_merged_weights(self, container, **selector_kwargs) -> Dict:
        raise ValueError("Not supported for MOESelector.")

    def get_routing_weights(self):
        raise ValueError("Not supported for MOESelector.")

    def add_expert(self, expert_name: str, **kwargs):
        self.rkhs_embeddings.data = torch.cat(
            [
                self.rkhs_embeddings.data,
                torch.zeros(1, 128, device=self.rkhs_embeddings.device).uniform_(
                    -0.02, 0.02
                ),
            ],
            dim=0,
        )


@register_multi_expert_selector("poly_router_dir")
class PolySelectorDirect(PolySelector):
    def __init__(self, config, info_container=None, **kwargs) -> None:
        super().__init__(info_container)

        self.module_logits = nn.ParameterDict()
        self.training_config = kwargs["training_config"]

    def _get_weights(self):
        weights = torch.tensor([self.module_logits[k] for k in self.expert_names])
        return weights

    def add_expert(self, expert_name: str, expert_task_name: str, **kwargs):
        """
        Assume:
        expert_task_name -- task name expert is pecialized in
        self.config.finetune_task_name -- name of the task the model is currently trained on
        """
        init_gap = [0, 0]
        main_m = 1

        if expert_name not in self.module_logits:
            # TODO: this is very error prone, e.g. when you reload this model, this cannot be inited
            # @Oleksiy do you need this?
            if self.training_config.finetune_task_name == expert_task_name:
                self.module_logits[expert_name] = torch.nn.Parameter(
                    torch.ones(1).to(self.device)
                )
                self.module_logits[expert_name].data *= main_m
            else:
                self.module_logits[expert_name] = torch.nn.Parameter(
                    torch.empty(1).uniform_(*init_gap).to(self.device)
                )


@register_multi_expert_selector("info_selector")
class RoutingInfosContainerSelector(Selector):
    """A simple selector that looks for routing information in the info container."""

    def __init__(self, info_container=None, **kwargs) -> None:
        super().__init__(info_container)

        self.default_expert_name = None

    def forward(self, input, **kwargs) -> BatchModulesAndWeightsSelectorOutput:
        if not hasattr(self.routing_infos, "routing_modules"):
            raise ValueError("routing_modules not in routing_infos")

        if not hasattr(self.routing_infos, "routing_weights"):
            raise ValueError("routing_weights not in routing_infos")

        routing_mods = self.routing_infos.routing_modules
        routing_weights = self.routing_infos.routing_weights
        return BatchModulesAndWeightsSelectorOutput(routing_mods, routing_weights)


@register_multi_expert_selector("task_selector")
class TaskNameSelector(Selector):
    def __init__(self, info_container, **kwargs) -> None:
        super().__init__(info_container)

        self.default_expert_name = None

    def forward(self, input, **kwargs) -> ModulesSelectorOutput:
        # try to infer batch size
        if not self.routing_infos.task_names:
            if "input_ids" in kwargs:
                batch_size = kwargs["input_ids"].size(0)
            else:
                batch_size = input.shape[0]

            if not self.default_expert_name:
                raise ValueError("No default expert name set and no task names given!")

            modules = [self.default_expert_name for _ in range(batch_size)]
        else:
            task_names = self.routing_infos.task_names

            if (
                any(task_name not in self.expert_names for task_name in task_names)
                and not self.default_expert_name
                and len(self.expert_names)
            ):
                raise ValueError(
                    "Experts for all tasks have not been loaded! Set a default expert?"
                )
            modules = task_names

        return ModulesSelectorOutput(modules)

    def add_expert(self, expert_name: str, **kwargs):
        # here we experts based on their name, which can be different from the task name
        if expert_name not in self.expert_names:
            self.expert_names.append(expert_name)


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

    def __init__(self, info_container=None, **kwargs) -> None:
        super().__init__(info_container)

        self.default_expert_name = None

    def get_kv_weights(self, experts, k_proj, v_proj):
        task_names = self.routing_infos.task_names

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
        task_names = self.routing_infos.task_names

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

    def __init__(self, info_container=None, **kwargs) -> None:
        super().__init__(info_container)

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
