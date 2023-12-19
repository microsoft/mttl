from abc import abstractproperty
from dataclasses import dataclass
from typing import Dict, List, Union
from pyparsing import abstractmethod
import torch
import math
from torch import nn
import torch.nn.functional as F
from mttl.models.modifiers.routing import RoutingMixin


SELECTORS_NAME_TO_KLASS = {}
SELECTORS_CONFIG_TO_KLASS = {}
SELECTORS_NAME_TO_CONFIG = {}


EPS = 1e-8


def register_multi_expert_selector(name, config_cls):
    print("Registering multi-expert selector..." + name)

    def _thunk(fn):
        if name in SELECTORS_NAME_TO_KLASS:
            raise ValueError(
                f"Cannot register duplicate multi-expert selector ({name})."
            )

        if config_cls in SELECTORS_CONFIG_TO_KLASS:
            raise ValueError(f"Cannot register with config class ({config_cls}).")

        SELECTORS_NAME_TO_KLASS[name] = fn
        SELECTORS_CONFIG_TO_KLASS[config_cls] = fn
        SELECTORS_NAME_TO_CONFIG[name] = config_cls
        fn.__layer_name__ = name
        return fn

    return _thunk


def get_selector(routing_config: "SelectorConfig", info_container: Dict, **kwargs):
    """Returns a selector object for the given routing_config."""
    return SELECTORS_CONFIG_TO_KLASS[routing_config.__class__](
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
        data["selector_config_klass"] = self.__class__.__name__
        return data

    @classmethod
    def fromdict(cls, dumped: Dict) -> "SelectorConfig":
        klass = dumped.pop("selector_config_klass")
        return eval(klass)(**dumped)

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


def cache_if_views(func):
    def wrapper(self, input, **kwargs):
        output = func(self, input, **kwargs)

        if self.forward_cache is None:
            self.forward_cache = output

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

    @abstractmethod
    def forward(self, input, **kwargs) -> SelectorOutput:
        pass

    @property
    def views(self):
        return self.selector_views

    def get_merged_weights(self, container, **selector_kwargs) -> Dict:
        return {}

    def create_view(self) -> "SelectorView":
        """
        Create a view on an existing selector for all the "shared" layers
        these don't hold params, are not registered as modules, and read from the cache
        of their parent "real" selector.
        """
        selector_view = SelectorView(self, len(self.selector_views))
        self.selector_views.append(selector_view)
        return selector_view

    def clear_cache(self):
        self.forward_cache = None

    @property
    def name(self):
        return f"{self.__layer_name__}"

    @abstractmethod
    def add_expert(self, expert_name: str, **kwargs):
        pass


class SelectorView:
    """A view on a selector, handles selector caching during forward pass, in the case
    this selector is shared across layers. First layer gets the real deal, the others get a view on it.
    """

    def __init__(self, selector, id):
        self._id = id
        self.selector = selector
        # only the last view clears the cache
        for view in self.selector.views:
            view.clear_cache = False
        self.clear_cache = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, _, **__):
        output = self.selector.forward_cache

        if output is None:
            raise ValueError("No forward cache set, this view seems dangling!")

        if self.clear_cache:
            self.selector.forward_cache = None
        return output

    def get_merged_weights(self, container, **selector_kwargs) -> Dict:
        return self.selector.get_merged_weights(container, **selector_kwargs)

    def add_expert(self, *args, **kwargs):
        """Do not add experts! This is just a view of the real thing."""
        pass


@dataclass
class PolySelectorConfig(SelectorConfig):
    pass


@register_multi_expert_selector("poly_router", PolySelectorConfig)
class PolySelector(Selector):
    """
    Implements routing at a per-layer or per-model level
    """

    def __init__(self, info_container, **kwargs) -> None:
        super().__init__(info_container)

        self.module_logits = nn.Parameter(torch.empty(1).uniform_(-1e-3, 1e-3))

    def _get_weights(self):
        module_logits = torch.sigmoid(self.module_logits)
        module_weights = module_logits / (module_logits.sum(dim=-1, keepdim=True) + EPS)
        return module_weights

    @cache_if_views
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


@register_multi_expert_selector("moe_rkhs_router", MOERKHSSelectorConfig)
class MOERKHSSelector(Selector):
    def __init__(self, info_container, config, **kwargs) -> None:
        super().__init__(info_container, config)

        if "layer" not in kwargs:
            raise ValueError(
                "MOERKHSSelector requires a layer to be passed in kwargs to infer the input dimension."
            )

        self.topk = 2
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

    @cache_if_views
    def forward(
        self, input, **kwargs
    ) -> BatchAndSequenceModulesAndWeightsSelectorOutput:
        # do routing business on fp32
        input = input.to(dtype=self.rkhs_exp.weight.dtype)

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

        g = self.info_container.get("routing_gates", [])
        g.append(router_logits)
        self.info_container["routing_gates"] = g

        return BatchAndSequenceModulesAndWeightsSelectorOutput(
            indices=selected_experts, weights=routing_weights
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
class PolySelectorDirectConfig(SelectorConfig):
    pass


@register_multi_expert_selector("poly_router_dir", PolySelectorDirectConfig)
class PolySelectorDirect(PolySelector):
    def __init__(self, info_container, **kwargs) -> None:
        super().__init__(info_container)

        self.module_logits = nn.ParameterDict()
        self._initialized = False
        self.training_config = kwargs["training_config"]

    def _get_weights(self):
        weights = torch.tensor([self.module_logits[k] for k in self.expert_names])
        return weights

    def add_expert(self, expert_name: str, **kwargs):
        """
        Assume:
        expert_task_name -- task name expert is pecialized in
        self.config.finetune_task_name -- name of the task the model is currently trained on
        """
        init_gap = [0, 0]
        main_m = 1

        expert_task_name = kwargs["expert_info"].expert_task_name
        if expert_name not in self.module_logits:
            # TODO: this is very error prone, e.g. when you reload this model, this cannot be inited
            # @Oleksiy do you need this?
            if self.training_config.finetune_task_name == expert_task_name:
                self.module_logits[expert_name] = torch.nn.Parameter(
                    torch.ones(1).to(self.device)
                )
                self.module_logits[expert_name].data *= main_m
                self._initialized = True
            else:
                self.module_logits[expert_name] = torch.nn.Parameter(
                    torch.empty(1).uniform_(*init_gap).to(self.device)
                )

    def load_state_dict(self, state_dict, strict=True):
        self._initialized = True
        return super().load_state_dict(state_dict, strict=strict)

    def init_logits_uniform(self):
        if sum([p for p in self.module_logits.values()]) == 0:
            for name, param in self.module_logits.items():
                self.module_logits[name].data = (
                    torch.empty(1).uniform_(-1e-3, 1e-3).to(self.device)
                )
        self._initialized = True

    def forward(self, *args, **kwargs):
        if not self._initialized:
            self.init_logits_uniform()
        return super().forward(*args, **kwargs)


@dataclass
class RoutingInfoContainerConfig(SelectorConfig):
    pass


@register_multi_expert_selector("info_selector", RoutingInfoContainerConfig)
class RoutingInfosContainerSelector(Selector):
    """A simple selector that looks for routing information in the info container."""

    def __init__(self, info_container, **kwargs) -> None:
        super().__init__(info_container)

        self.default_expert_name = None

    @cache_if_views
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

    @cache_if_views
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
