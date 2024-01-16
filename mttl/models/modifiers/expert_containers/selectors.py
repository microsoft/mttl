from dataclasses import dataclass
from typing import Dict, List, Union
from pyparsing import abstractmethod
import torch
import math
from torch import nn
import torch.nn.functional as F
from mttl.models.modifiers.routing import RoutingMixin


SELECTORS_NAME_TO_KLASS = {}
SELECTORS_CONFIG_TO_NAME = {}
SELECTORS_NAME_TO_CONFIG = {}


EPS = 1e-8


def register_multi_expert_selector(name, config_cls):
    print("Registering multi-expert selector..." + name)

    def _thunk(fn):
        if name in SELECTORS_NAME_TO_KLASS:
            raise ValueError(
                f"Cannot register duplicate multi-expert selector ({name})."
            )

        if config_cls in SELECTORS_CONFIG_TO_NAME:
            raise ValueError(f"Cannot register with config class ({config_cls}).")

        SELECTORS_NAME_TO_KLASS[name] = fn
        SELECTORS_CONFIG_TO_NAME[config_cls] = name
        SELECTORS_NAME_TO_CONFIG[name] = config_cls
        fn.__layer_name__ = name
        return fn

    return _thunk


def get_selector(routing_config: "SelectorConfig", info_container: Dict, **kwargs):
    """Returns a selector object for the given routing_config."""
    return SELECTORS_NAME_TO_KLASS[SELECTORS_CONFIG_TO_NAME[routing_config.__class__]](
        info_container, config=routing_config, **kwargs
    )


@dataclass
class SelectorConfig:
    # the granularity of the selector (which layers use the same selectors)
    router_granularity: str = "*"

    def __eq__(self, other):
        # compare all the attributes
        return self.__dict__ == other.__dict__

    def asdict(self) -> Dict:
        """Dump the config to a string."""
        from dataclasses import asdict

        data = asdict(self)
        # store the model modifier for easy loading
        data["__selector_name__"] = SELECTORS_CONFIG_TO_NAME[type(self)]
        return data

    @classmethod
    def fromdict(cls, dumped: Dict) -> "SelectorConfig":
        if "__selector_name__" not in dumped:
            raise ValueError(
                "Cannot load SelectorConfig from dict, missing '__selector_name__' key."
            )
        name = dumped.pop("__selector_name__")
        return SELECTORS_NAME_TO_CONFIG[name](**dumped)

    @staticmethod
    def from_training_config(
        training_config: Union["Config", "SelectorConfig"]
    ) -> Union["SelectorConfig", None]:
        """Build modifier config from the training config.

        Returns None if no modifier is set.
        """
        if isinstance(training_config, SelectorConfig):
            # nothing to do here
            return training_config

        if training_config.router_selector is None:
            return None

        if training_config.router_selector not in SELECTORS_NAME_TO_KLASS:
            raise ValueError(
                f"Selector '{training_config.router_selector}' not found, has it been registered?"
            )

        config_klass = SELECTORS_NAME_TO_CONFIG[training_config.router_selector]
        kwargs = {}
        for key, _ in config_klass.__dataclass_fields__.items():
            if hasattr(training_config, key):
                kwargs[key] = getattr(training_config, key)
        return config_klass(**kwargs)


@dataclass
class SelectorOutput:
    pass


@dataclass
class ModulesSelectorOutput(SelectorOutput):
    """A selector output that contains a list of modules without weights.

    modules: names of the selected modules
    """

    modules: List[str]


@dataclass
class ModulesAndWeightsSelectorOutput(SelectorOutput):
    """A selector output that contains a list of modules and weights shared across the batch.

    modules: names of the selected modules
    weights: their weights
    """

    modules: List[str]
    weights: Union[List[float], torch.Tensor]


@dataclass
class BatchModulesAndWeightsSelectorOutput(SelectorOutput):
    """A selector output that contains a list of modules and weights for each example.

    modules: either names or indices of the selected modules
    weights: their weights
    """

    modules: Union[List[List[str]], torch.Tensor]
    weights: Union[List[List[float]], torch.Tensor]


@dataclass
class BatchSequenceModulesAndWeightsSelectorOutput(SelectorOutput):
    """A selector output that contains a list of modules and weights for each example and token.

    modules: indices of the selected modules
    weights: their weights
    """

    modules: torch.Tensor
    weights: Union[List[List[float]], torch.Tensor]


def forward_with_cache(func):
    def wrapper(self: Selector, input, **kwargs):
        if self.forward_cache is not None and not self.clear_cache:
            self.count_call()
            return self.forward_cache

        output = func(self, input, **kwargs)
        self.forward_cache = output
        self.count_call()
        return output

    return wrapper


class Selector(RoutingMixin, nn.Module):
    def __init__(self, info_container, config=None, **kwargs):
        nn.Module.__init__(self)
        RoutingMixin.__init__(self, info_container)

        self.config = config
        self.expert_names = []
        self.selector_views = []
        self.forward_cache = None
        self.total_calls_per_forward = 0
        self._calls_counter = 0

    @property
    def clear_cache(self):
        reset_cache = self._calls_counter >= self.total_calls_per_forward
        if reset_cache:
            self._calls_counter = 0
        return reset_cache

    def count_call(self):
        self._calls_counter += 1

    @abstractmethod
    def forward(self, input, **kwargs) -> SelectorOutput:
        pass

    def create_view(self) -> "SelectorView":
        self.selector_views.append(SelectorView(self))
        return self.selector_views[-1]

    @property
    def views(self):
        return self.selector_views

    def get_merged_weights(self, container, **selector_kwargs) -> Dict:
        return {}

    @property
    def name(self):
        return f"{self.__layer_name__}"

    @abstractmethod
    def add_expert(self, expert_name: str, **kwargs):
        pass


class SelectorView:
    """A view on a selector that allows it to call forward but doesn't act on add_expert.

    This is because add expert is to be called only on the main instance of this selector
    and not on the multiple views across the network.
    """

    def __init__(self, selector_instance):
        self.selector_instance = selector_instance

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.selector_instance.forward(*args, **kwargs)

    def get_merged_weights(self, container, **selector_kwargs) -> Dict:
        return self.selector_instance.get_merged_weights(container, **selector_kwargs)

    def add_expert(self, expert_name: str, **kwargs):
        pass


@dataclass
class PolySelectorConfig(SelectorConfig):
    pass


@register_multi_expert_selector("poly_router", PolySelectorConfig)
class PolySelector(Selector):
    """
      eval_every=10_000
    Implements routing at a per-layer or per-model level
    """

    def __init__(self, info_container, **kwargs) -> None:
        super().__init__(info_container)

        self.module_logits = nn.Parameter(torch.empty(1).uniform_(-1e-3, 1e-3))

    def _get_weights(self):
        module_logits = torch.sigmoid(self.module_logits)
        module_weights = module_logits / (module_logits.sum(dim=-1, keepdim=True) + EPS)
        return module_weights

    @forward_with_cache
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
        self.expert_names.append(expert_name)
        self.module_logits.data = torch.empty(len(self.expert_names)).uniform_(
            -1e-3, 1e-3
        )


@dataclass
class MOERKHSSelectorConfig(SelectorConfig):
    rkhs_dim: int = 512
    emb_dim: int = 128
    top_k: int = -1


@register_multi_expert_selector("moe_rkhs_router", MOERKHSSelectorConfig)
class MOERKHSSelector(Selector):
    def __init__(self, info_container, config, **kwargs) -> None:
        super().__init__(info_container, config)

        if "layer" not in kwargs:
            raise ValueError(
                "MOERKHSSelector requires a layer to be passed in kwargs to infer the input dimension."
            )

        self.top_k = config.top_k
        self.input_dim = kwargs["layer"].weight.data.shape[-1]
        self.rkhs_dim = config.rkhs_dim
        self.emb_dim = config.emb_dim

        device = kwargs["layer"].weight.device

        self.rkhs_exp = nn.Linear(self.emb_dim, self.rkhs_dim, device=device)
        self.rkhs_hid = nn.Linear(self.input_dim, self.rkhs_dim, device=device)
        self.rkhs_embeddings = nn.Parameter(
            torch.empty((0, self.emb_dim), device=device)
        )

    def _get_weights(self, input):
        input_view = input.view(-1, input.shape[-1])
        return self.rkhs_hid(input_view).reshape(input.shape[0], input.shape[1], -1)

    @forward_with_cache
    def forward(self, input, **kwargs) -> BatchSequenceModulesAndWeightsSelectorOutput:
        # do routing business on fp32
        input = input.to(dtype=self.rkhs_exp.weight.dtype)

        rkhs_enc = self._get_weights(input)
        rkhs_emb = self.rkhs_exp(self.rkhs_embeddings)

        router_logits = torch.matmul(rkhs_enc, rkhs_emb.T)
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)

        if self.top_k > 0:
            routing_weights, selected_experts = torch.topk(
                routing_weights, self.top_k, dim=-1
            )
            # we cast back to the input dtype
            routing_weights = routing_weights.to(input.dtype)
        else:
            # soft routing
            selected_experts = None

        g = self.info_container.get("routing_gates", [])
        g.append(router_logits)
        self.info_container["routing_gates"] = g

        return BatchSequenceModulesAndWeightsSelectorOutput(
            modules=selected_experts, weights=routing_weights
        )

    def get_merged_weights(self, container, **selector_kwargs) -> Dict:
        raise ValueError("Not supported for MOESelector.")

    def get_routing_weights(self):
        raise ValueError("Not supported for MOESelector.")

    def add_expert(self, expert_name: str, **kwargs):
        """It is important to guard against multiple calls as this can be called multiple times."""
        self.expert_names.append(expert_name)
        self.rkhs_embeddings.data = torch.cat(
            [
                self.rkhs_embeddings.data,
                torch.zeros(
                    1, self.emb_dim, device=self.rkhs_embeddings.device
                ).uniform_(-0.02, 0.02),
            ],
            dim=0,
        )


@dataclass
class ZeroSelectorConfig(SelectorConfig):
    top_k: int = -1


@register_multi_expert_selector("zero_router", ZeroSelectorConfig)
class ZeroSelector(Selector):
    def __init__(self, info_container, config, **kwargs) -> None:
        super().__init__(info_container, config)

        if "layer" not in kwargs:
            raise ValueError(
                "ZeroSelector requires a layer to be passed in kwargs to infer the input dimension."
            )

        self.top_k = config.top_k
        self.input_dim = kwargs["layer"].weight.data.shape[-1]
        # dependency injection
        self._container = None

    @forward_with_cache
    def forward(
        self, input, container, **kwargs
    ) -> BatchModulesAndWeightsSelectorOutput:
        from mttl.models.modifiers.expert_containers.expert_containers import (
            CoalescedLoRAExpertContainer,
        )

        if not isinstance(container, CoalescedLoRAExpertContainer):
            raise ValueError(
                "ZeroSelector requires a coalesced LoRA container. Set COALESCED_LORA_CONTAINER=1 as env variable."
            )

        # do routing business on fp32
        input = input.to(dtype=container.experts.lora_a.dtype)

        logits = (
            torch.einsum("bsd,kpdr->bskrp", input, container.experts.lora_a)
            .squeeze(-1)
            .pow(2.0)
            .sum(-1)
            .sqrt()
        ).mean(
            1
        )  # bk
        routing_weights = torch.softmax(logits, dim=-1)

        if self.top_k > 0:
            routing_weights, selected_experts = torch.topk(
                routing_weights, self.top_k, dim=-1
            )
            # we cast back to the input dtype
            routing_weights = routing_weights.to(input.dtype)
        else:
            # soft routing
            selected_experts = None

        g = self.info_container.get("routing_gates", [])
        g.append(torch.log(routing_weights + 1e-6))
        self.info_container["routing_gates"] = g

        return BatchModulesAndWeightsSelectorOutput(
            modules=selected_experts, weights=routing_weights
        )

    def get_merged_weights(self, container, **selector_kwargs) -> Dict:
        raise ValueError("Not supported for MOESelector.")

    def get_routing_weights(self):
        raise ValueError("Not supported for MOESelector.")


@dataclass
class ZeroPerTokenSelectorConfig(SelectorConfig):
    top_k: int = -1


@register_multi_expert_selector("zero_per_token_router", ZeroPerTokenSelectorConfig)
class ZeroPerTokenSelector(Selector):
    def __init__(self, info_container, config, **kwargs) -> None:
        super().__init__(info_container, config)

        if "layer" not in kwargs:
            raise ValueError(
                "MOERKHSSelector requires a layer to be passed in kwargs to infer the input dimension."
            )

        self.top_k = config.top_k
        self.input_dim = kwargs["layer"].weight.data.shape[-1]
        # dependency injection
        self._container = None

    @forward_with_cache
    def forward(
        self, input, container, **kwargs
    ) -> BatchSequenceModulesAndWeightsSelectorOutput:
        # do routing business on fp32
        input = input.to(dtype=container.experts.lora_a.dtype)

        logits = (
            torch.einsum("bsd,kpdr->bskrp", input, container.experts.lora_a)
            .squeeze(-1)
            .pow(2.0)
            .sum(-1)
            .sqrt()
        )
        routing_weights = torch.softmax(logits, dim=-1)

        if self.top_k > 0:
            routing_weights, selected_experts = torch.topk(
                routing_weights, self.top_k, dim=-1
            )
            # we cast back to the input dtype
            routing_weights = routing_weights.to(input.dtype)
        else:
            # soft routing
            selected_experts = None

        g = self.info_container.get("routing_gates", [])
        g.append(torch.log(routing_weights + 1e-6))
        self.info_container["routing_gates"] = g

        return BatchSequenceModulesAndWeightsSelectorOutput(
            modules=selected_experts, weights=routing_weights
        )

    def get_merged_weights(self, container, **selector_kwargs) -> Dict:
        raise ValueError("Not supported for MOESelector.")

    def get_routing_weights(self):
        raise ValueError("Not supported for MOESelector.")


@dataclass
class PolySelectorDirectConfig(SelectorConfig):
    pass


@register_multi_expert_selector("poly_router_dir", PolySelectorDirectConfig)
class PolySelectorDirect(PolySelector):
    def __init__(self, info_container, **kwargs) -> None:
        super().__init__(info_container)

        self.module_logits_dict = nn.ParameterDict()

        self.training_config = kwargs["training_config"]
        self.init_gap = [-1e-3, 1e-3]

        self.device = kwargs["layer"].weight.device

    def _get_weights(self):
        weights = torch.cat(
            [self.module_logits_dict[k] for k in self.module_logits_dict.keys()]
        )
        return weights

    def get_routing_weights(self):
        return {k: v.detach().item() for k, v in self.module_logits_dict.items()}

    def add_expert(self, expert_name: str, **kwargs):
        """
        Assume:
        expert_task_name -- task name expert is pecialized at
        self.config.finetune_task_name -- name of the task the model is currently trained on

        If we eocounter a module for the current task, we init it with one hot, otherwise with uniform.


        """
        main_m = 1

        expert_task_name = kwargs["expert_info"].expert_task_name
        if expert_name not in self.module_logits_dict:
            if self.training_config.finetune_task_name == expert_task_name:
                self.init_gap = [
                    0,
                    0,
                ]  # encountered module for current task, init one hot
                self.module_logits_dict[expert_name] = torch.nn.Parameter(
                    torch.ones(1).to(self.device)
                )
                self.init_logits_uniform()
                self.module_logits_dict[expert_name].data *= main_m

            else:
                self.module_logits_dict[expert_name] = torch.nn.Parameter(
                    torch.empty(1).uniform_(*self.init_gap).to(self.device)
                )

    def load_state_dict(self, state_dict, strict=True):
        self._initialized = True
        return super().load_state_dict(state_dict, strict=strict)

    def init_logits_uniform(self):
        if sum([p for p in self.module_logits_dict.values()]) == 0:
            for name, param in self.module_logits_dict.items():
                self.module_logits_dict[name].data = (
                    torch.empty(1).uniform_(-1e-3, 1e-3).to(self.device)
                )
        self._initialized = True

    @forward_with_cache
    def forward(self, *args, **kwargs):
        weights = self._get_weights()
        modules = list(self.module_logits_dict.keys())
        return ModulesAndWeightsSelectorOutput(modules, weights)


@dataclass
class RoutingInfoContainerConfig(SelectorConfig):
    pass


@register_multi_expert_selector("info_selector", RoutingInfoContainerConfig)
class RoutingInfosContainerSelector(Selector):
    """A simple selector that looks for routing information in the info container."""

    def __init__(self, info_container, **kwargs) -> None:
        super().__init__(info_container)

        self.default_expert_name = None

    @forward_with_cache
    def forward(self, input, **kwargs) -> BatchModulesAndWeightsSelectorOutput:
        if not hasattr(self.routing_infos, "routing_modules"):
            raise ValueError("routing_modules not in routing_infos")

        if not hasattr(self.routing_infos, "routing_weights"):
            raise ValueError("routing_weights not in routing_infos")

        routing_mods = self.routing_infos.routing_modules
        routing_weights = self.routing_infos.routing_weights
        return BatchModulesAndWeightsSelectorOutput(routing_mods, routing_weights)


@dataclass
class TaskNameSelectorConfig(SelectorConfig):
    pass


@register_multi_expert_selector("task_selector", TaskNameSelectorConfig)
class TaskNameSelector(Selector):
    def __init__(self, info_container, **kwargs) -> None:
        super().__init__(info_container)

        self.default_expert_name = None

    @forward_with_cache
    def forward(self, input, **kwargs) -> ModulesSelectorOutput:
        # try to infer batch size
        if not self.routing_infos or not self.routing_infos.task_names:
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


@dataclass
class KVTaskNameSelectorConfig(SelectorConfig):
    pass


@register_multi_expert_selector("kv_task_selector", KVTaskNameSelectorConfig)
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


@dataclass
class KVConcatSelectorConfig(SelectorConfig):
    pass


@register_multi_expert_selector("kv_concat_selector", KVConcatSelectorConfig)
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


@dataclass
class KVNormSelectorConfig(SelectorConfig):
    pass


@register_multi_expert_selector("kv_norm_selector", KVNormSelectorConfig)
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


@dataclass
class KVConcatNormSelectorConfig(SelectorConfig):
    pass


@register_multi_expert_selector("kv_concat_norm_selector", KVConcatNormSelectorConfig)
class KVConcatNormSelector(KVConcatSelector, KVNormSelector):
    pass


@dataclass
class KVTaskNameNormSelectorConfig(SelectorConfig):
    pass


@register_multi_expert_selector("kv_task_norm_selector", KVTaskNameNormSelectorConfig)
class KVTaskNameNormSelector(KVTaskNameSelector, KVNormSelector):
    pass
