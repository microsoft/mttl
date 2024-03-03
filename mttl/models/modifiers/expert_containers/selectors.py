from dataclasses import dataclass
from typing import Dict, List, Union
from pyparsing import abstractmethod
import torch
import math
import wandb
import numpy as np
from torch import nn
import torch.nn.functional as F
from mttl.models.modifiers.expert_containers.expert import ExpertInfo
from mttl.models.modifiers.routing import RoutingInfo
from torch.distributions import Bernoulli, Categorical

# from mttl.models.modifiers.routing import RoutingMixin
from mttl.utils import logger


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
    lora_merge_after: bool = False

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


class Selector(nn.Module):
    def __init__(self, info_container, config=None, **kwargs):
        nn.Module.__init__(self)

        self.config = config
        self.expert_names = []
        self.selector_views = []
        self.forward_cache = None
        self.total_calls_per_forward = 0
        self._calls_counter = 0
        self.info_container = info_container

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

    @abstractmethod
    def get_merging_weights(self, **selector_kwargs) -> Dict:
        """
        Returns a dictionary of the form {expert_name: weight} for each expert in the container.
        raises ValueError if not supported, e.g. because routing depends on the input x.
        """
        pass

    @property
    def layer_name(self):
        if not hasattr(self, "__layer_name__"):
            raise ValueError(
                "Layer name not available, dependency injection not done properly?"
            )

        return self.__layer_name__

    @property
    def n_experts(self):
        return len(self.expert_names)

    @property
    def routing_infos(self):
        return self.info_container.get("routing_infos", None)

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

    @property
    def config(self):
        return self.selector_instance.config

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.selector_instance.forward(*args, **kwargs)

    def get_merging_weights(self, **selector_kwargs) -> Dict:
        return self.selector_instance.get_merging_weights(**selector_kwargs)

    def add_expert(self, expert_name: str, **kwargs):
        pass


@dataclass
class PolySelectorConfig(SelectorConfig):
    n_splits: int = 1
    task_names: List[str] = None


@register_multi_expert_selector("poly_router", PolySelectorConfig)
class PolySelector(Selector):
    """
    Implements routing at a per-layer or per-model level
    """

    avg_selector_warned: bool = False

    def __init__(self, info_container, **kwargs) -> None:
        super().__init__(info_container, **kwargs)

        self.n_tasks = len(self.config.task_names) if self.config.task_names else 0

        # We add an extra task for the default (average) expert if not found
        self.module_logits = nn.Parameter(
            torch.empty(self.n_tasks + 1, self.config.n_splits).uniform_(-1e-3, 1e-3)
        )

    def _get_weights(self):
        # Poly used for finetuning a single task
        if self.n_tasks == 0:
            task_ids = [0]
        else:
            # Poly for pretraining with multiple tasks
            if hasattr(self.info_container["routing_infos"], "task_ids_from_name"):
                task_ids = self.info_container["routing_infos"].task_ids_from_name
            else:
                task_ids = [
                    (
                        self.config.task_names.index(t)
                        if t in self.config.task_names
                        else self.n_tasks
                    )
                    for t in self.info_container["routing_infos"].task_names
                ]
                task_ids = torch.LongTensor(task_ids).to(self.module_logits.device)
                self.info_container["routing_infos"].task_ids_from_name = task_ids
            if task_ids.max() < self.n_tasks:
                if PolySelector.avg_selector_warned:
                    logger.warning(
                        f"Task ids were found. Reverting to default task-based routing"
                    )

                PolySelector.avg_selector_warned = False
            else:
                if not PolySelector.avg_selector_warned:
                    not_found_tasks = set(
                        [
                            t
                            for t in self.info_container["routing_infos"].task_names
                            if t not in self.config.task_names
                        ]
                    )
                    logger.warning(
                        f"Tasks {not_found_tasks} not in taining tasks. Defaulting to average selector."
                    )
                    PolySelector.avg_selector_warned = True

                assert not self.training, "Unknown tasks during training"

        module_logits = torch.sigmoid(self.module_logits[task_ids])
        module_logits = module_logits.view(
            module_logits.size(0), self.config.n_splits, self.n_experts
        )
        module_weights = module_logits / (module_logits.sum(dim=-1, keepdim=True) + EPS)

        return module_weights

    @forward_with_cache
    def forward(self, input, **kwargs) -> ModulesAndWeightsSelectorOutput:
        weights = self._get_weights()
        modules = self.expert_names
        return ModulesAndWeightsSelectorOutput(modules, weights)

    def get_merging_weights(self, **selector_kwargs) -> Dict:
        return self.get_routing_weights(**selector_kwargs)

    def get_routing_weights(self, task_name, **selector_kwargs) -> Dict:
        assert task_name in self.config.task_names, f"Task {task_name} not found."
        self.info_container["routing_infos"] = RoutingInfo(task_names=[task_name])
        weights = self._get_weights()
        return {k: v.detach().item() for k, v in zip(self.expert_names, weights[0][0])}

    def add_expert(self, expert_name: str, **kwargs):
        self.expert_names.append(expert_name)
        self.module_logits.data = torch.empty(
            self.n_tasks + 1, self.config.n_splits * len(self.expert_names)
        ).uniform_(-1e-3, 1e-3)

        # Last expert is exactly uniform
        self.module_logits.data[-1] = 0.0


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

    def get_merging_weights(self, **selector_kwargs) -> Dict:
        raise ValueError(
            f"Not supported for {self.__class__} since routing depends on input."
        )

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
class ClownRouterConfig(SelectorConfig):
    router_granularity: str = "finegrained"
    router_temp: float = 1.0
    moe_top_k: int = -1
    clown_mode: str = "window"  # "last", "mean", "per_token", "window"
    router_window_size: int = 3
    proto_init: str = "hidden"
    normalize_router_input: bool = True


@register_multi_expert_selector("clown_router", ClownRouterConfig)
class ClownSelector(Selector):
    def __init__(self, info_container, config, **kwargs) -> None:
        super().__init__(info_container, config, **kwargs)

        if "layer" not in kwargs:
            raise ValueError(
                "Selector requires a layer to be passed in kwargs to infer the input dimension."
            )

        layer = kwargs["layer"]
        self.output_dim, self.input_dim = layer.weight.data.shape

        self.prototypes = nn.Parameter(
            torch.empty((0, self.input_dim), device=layer.weight.device)
        )

        if config.clown_mode == "window":
            # Build conv kernel to compute mean over window
            router_window_size = self.config.router_window_size
            avg_1d_conv_kernel = (
                torch.ones(1, 1, router_window_size) / router_window_size
            )
            self.register_buffer("avg_1d_conv_kernel", avg_1d_conv_kernel)

    def overwrite_prototypes(self, prototypes):
        self.prototypes.data = prototypes.type_as(self.prototypes)

    @forward_with_cache
    def forward(self, input, **kwargs) -> BatchSequenceModulesAndWeightsSelectorOutput:
        # do routing business on fp32
        temp = (
            self.config.router_temp
            if self.config.router_temp > 0
            else np.sqrt(input.shape[-1])
        )
        if self.prototypes.size(0) != len(self.expert_names):
            raise ValueError("Prototypes not initialized correctly.")

        input = input.to(dtype=self.prototypes.dtype)

        if self.config.normalize_router_input:
            input /= input.norm(dim=-1, p=2, keepdim=True).clamp(min=EPS)

        input_ids = self.info_container["routing_infos"].input_ids
        attn_mask = self.info_container["routing_infos"].attention_mask
        bs, sq, D = input.shape

        if self.config.clown_mode == "per_token":
            router_logits = F.linear(input, self.prototypes)
            if self.config.proto_init == "svd":
                router_logits = router_logits.abs()

            routing_weights = F.softmax(router_logits / temp, dim=-1, dtype=torch.float)
        else:
            attn_mask = self.info_container["routing_infos"].attention_mask
            if sq == attn_mask.size(1):
                if self.config.clown_mode == "mean":
                    # teacher force mode. router input is the mean over valid tokens
                    router_input = (input * attn_mask.unsqueeze(-1)).sum(dim=1) / (
                        attn_mask.sum(dim=1, keepdim=True) + EPS
                    )
                elif self.config.clown_mode == "last":
                    last_idx = attn_mask.sum(1) - 1
                    router_input = input[torch.arange(bs), last_idx]
                elif self.config.clown_mode == "window":
                    flat_input = input.transpose(1, 2).reshape(bs * D, 1, sq)
                    left_pad_input = torch.cat(
                        (
                            flat_input[:, :, [0]].expand(
                                -1, -1, self.config.router_window_size - 1
                            ),
                            flat_input,
                        ),
                        dim=-1,
                    )
                    conv_out = F.conv1d(left_pad_input, self.avg_1d_conv_kernel).view(
                        bs, D, sq
                    )
                    router_input = conv_out.transpose(1, 2)
            else:
                assert sq == 1
                # we are in generation mode
                router_input = input.squeeze(1)

            router_logits = F.linear(router_input, self.prototypes)
            if self.config.proto_init == "svd":
                router_logits = router_logits.abs()
            routing_weights = F.softmax(router_logits / temp, dim=-1, dtype=torch.float)

        if routing_weights.ndim == 2:
            routing_weights = routing_weights.unsqueeze(1).expand(
                -1, input.shape[1], -1
            )

        # uniform routing entropy
        ent_routing = -1 * (routing_weights * torch.log(routing_weights + 1e-6)).sum(-1)
        if sq == 1:
            ent_routing = ent_routing.sum()
            valid_ps = routing_weights
        else:
            ent_routing = (ent_routing * attn_mask).sum() / attn_mask.sum()
            valid_ps = routing_weights[attn_mask == 1]

        max_p, min_p = valid_ps.max(), valid_ps.min()
        to_store = {
            "ent_uniform": np.log(len(self.expert_names)),
            "ent_routing": ent_routing.item(),
            "max_p": max_p.item(),
            "min_p": min_p.item(),
        }

        # Keep running statistics of routing
        task = self.info_container["routing_infos"].task_names[0]
        task_container = self.info_container.get(task, {})
        count = task_container.get("routing_count", 0)

        for name, value in to_store.items():
            old_value = task_container.get(name, 0)
            task_container[name] = (old_value * count + value) / (count + 1)

        task_container["routing_count"] = count + 1
        self.info_container[task] = task_container

        if self.config.moe_top_k > 0:
            # TODO: mask and renormalize the routing_weights, so that it's still differentiable
            _, selected_experts = torch.topk(
                routing_weights, self.config.moe_top_k, dim=-1
            )

            routing_weights = torch.zeros_like(routing_weights)
            value = (
                torch.ones(
                    size=(1,),
                    device=routing_weights.device,
                    dtype=routing_weights.dtype,
                )
                / self.config.moe_top_k
            )
            value = value.expand_as(selected_experts)
            routing_weights.scatter_(dim=-1, index=selected_experts, src=value)

        return BatchSequenceModulesAndWeightsSelectorOutput(
            modules=None, weights=routing_weights
        )

    def get_merging_weights(self, **selector_kwargs) -> Dict:
        raise ValueError(
            f"Not supported for {self.__class__} since routing depends on input."
        )

    def get_routing_weights(self):
        raise ValueError("Not supported for ClownSelector.")

    def add_expert(self, expert_name: str, **kwargs):
        self.expert_names.append(expert_name)


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

    def get_merging_weights(self, **selector_kwargs) -> Dict:
        raise ValueError(
            f"Not supported for {self.__class__} since routing depends on input."
        )

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

    def get_merging_weights(self, **selector_kwargs) -> Dict:
        raise ValueError(
            f"Not supported for {self.__class__} since routing depends on input."
        )

    def get_routing_weights(self):
        raise ValueError("Not supported for MOESelector.")


@dataclass
class PhatgooseSelectorConfig(SelectorConfig):
    moe_top_k: int = 2  # negative number means we take all
    router_temp: float = (
        -1
    )  # negative value means we use the default \sqrt(d) temp, where n is the input dimension.


class SigmoidGate(nn.Module):
    def __init__(self, input_dim, output_dim=1, **kwargs):
        super().__init__()
        self.v = nn.Parameter(torch.zeros(output_dim, input_dim))

    def forward(self, x):
        return torch.sigmoid(F.linear(x, self.v, bias=None))


class Router(nn.Module):
    def __init__(
        self, gates: nn.ModuleList, hard=False, t=-1, top_k=2, device="cuda", **kwargs
    ):
        super().__init__()
        self.hard = hard
        self.moe_top_k = top_k
        self.temperature = t
        self.expert_embeddings = torch.stack([g.v.T.squeeze() for g in gates]).to(
            device
        )  # (n_experts, input_dim)
        # standardize the expert embeddings: sub mean and divide by std
        self.standardizer = nn.LayerNorm(
            self.expert_embeddings.shape[1], elementwise_affine=False
        )
        self.expert_embeddings = self.standardizer(self.expert_embeddings)

    def forward(self, input):
        # x: (batch_size, seq, d)
        # expert_embeddings: (n_experts, d)
        # perform routing business in fp32 as in Clown
        input = input.to(self.expert_embeddings.dtype)
        # standardize the input
        input = self.standardizer(input)
        # cosine similarity of each token with each expert
        sim = torch.einsum(
            "bsd,ed->bse", input, self.expert_embeddings
        )  # (batch_size, seq, n_experts)
        temp = self.temperature if self.temperature > 0 else np.sqrt(input.shape[-1])

        if self.hard:
            weights, module_indices = torch.topk(sim, self.moe_top_k, dim=-1)
            return module_indices, F.softmax(weights / temp, dim=-1)
        else:
            return None, F.softmax(sim / temp, dim=-1)


@register_multi_expert_selector("phatgoose_selector", PhatgooseSelectorConfig)
class PhatgooseSelector(Selector):
    """
    Selector from https://arxiv.org/abs/2402.05859
    """

    def __init__(
        self, info_container, config: PhatgooseSelectorConfig, **kwargs
    ) -> None:
        super().__init__(info_container, config)

        if "layer" not in kwargs:
            raise ValueError(
                "PhatgooseSelector requires a layer to be passed in kwargs to infer the input dimension."
            )

        self.top_k = config.moe_top_k
        self.input_dim = kwargs["layer"].weight.data.shape[-1]
        self.device = kwargs["layer"].weight.device

        self.gates = nn.ParameterDict()
        self.router = None
        self.layer = kwargs["layer"]
        self.default_expert_name = None
        self.routing_gates = []  # for logging purposes at training time
        self.log_routing_stats = False

    def get_prototypes(self):
        return {k: gate.v.detach().cpu().numpy() for k, gate in self.gates.items()}

    def set_prototypes(self, prototypes):
        """
        Sets prototypes for the gates and re-initializes the router with the new gates
        Input:
            - prototypes: a dictionary with expert names as keys mapped to another dict with layer names as keys and values as the prototypes
        """
        for task_name, gates in prototypes.items():
            gate_v = gates[f"model.{self.__layer_name__}.{task_name}.v"]
            assert self.gates[task_name].v.shape == gate_v.shape
            self.gates[task_name].v.data = torch.tensor(gate_v).to(self.device)

        self.router = Router(
            self.gates.values(),
            hard=self.top_k > 0,
            t=self.config.router_temp,
            top_k=self.top_k,
            device=self.device,
        )

    @forward_with_cache
    def forward(self, input, **kwargs) -> BatchSequenceModulesAndWeightsSelectorOutput:
        if len(self.gates) == 1:
            # selectors for tasks are trained independently
            # all samples go through the same selector
            scores = self.gates[self.default_expert_name](input)
            # log the scores
            container = kwargs.get("container", None)
            if container is not None:
                self.routing_gates.append(scores.detach().cpu().float())
            return BatchSequenceModulesAndWeightsSelectorOutput(
                torch.zeros_like(scores, dtype=torch.int8), scores
            )
        else:
            # the inference procedure: we have multiple gates
            # parallel forward + top-k selection
            assert len(self.gates) == len(self.expert_names)
            modules, scores = self.router(input)
            routings = torch.zeros(
                (input.shape[0], input.shape[1], len(self.expert_names)),
                dtype=torch.float,
                device=self.device,
            )

            if self.log_routing_stats:
                # for logging purposes: we transform it into tokens x experts
                _modules = modules.view(-1, self.router.moe_top_k)
                _routings = routings.view(-1, len(self.expert_names))
                _scores = scores.view(-1, self.router.moe_top_k)
                _routings = _routings.scatter(1, _modules, _scores)
                # calculate entropy and MI over the toekns dimention and log
                layer_routing_dist = _routings
                dims = layer_routing_dist.shape[1]
                layer_routing_mean = layer_routing_dist.mean(0)
                h_mean = Categorical(probs=layer_routing_mean).entropy() / math.log(
                    dims
                )
                mean_h = Categorical(
                    probs=layer_routing_dist
                ).entropy().mean() / math.log(dims)
                mi = h_mean - mean_h
                logger.info(f"Layer {self.__layer_name__} MI: {mi}, mean_H: {mean_h}")
                # TODO: figure out a way of how to actually log it to somewhere
                # at first glance there is no collapse here, i.e. MI is low and entropy is not too low.

            return BatchSequenceModulesAndWeightsSelectorOutput(
                modules=modules, weights=scores
            )

    def add_expert(self, expert_name: str, **kwargs):
        self.expert_names.append(expert_name)
        expert_info = kwargs["expert_info"]
        self.default_expert_name = expert_name

        self.gates[expert_name] = SigmoidGate(self.input_dim)

        if len(self.gates) > 1:
            self.router = Router(
                self.gates.values(),
                hard=self.top_k > 0,
                t=self.config.router_temp,
                top_k=self.top_k,
                device=self.device,
            )

    def get_merging_weights(self, **selector_kwargs) -> Dict:
        raise ValueError(
            f"Not supported for {self.__class__}  since routing depends on input."
        )


@dataclass
class PolySelectorDirectConfig(PolySelectorConfig):
    pass


@dataclass
class PolySelectorDirectConfigUniform(PolySelectorConfig):
    pass


@dataclass
class PolySelectorDirectConfigUniform(SelectorConfig):
    pass


@register_multi_expert_selector("poly_router_dir", PolySelectorDirectConfig)
class PolySelectorDirect(PolySelector):
    def __init__(self, info_container, **kwargs) -> None:
        super().__init__(info_container, **kwargs)

        self.module_logits_dict = nn.ParameterDict()
        self.training_config = kwargs["training_config"]
        self.init_gap = [-1e-3, 1e-3]

        self.device = kwargs["layer"].weight.device

    def _get_weights(self):
        weights = torch.cat(
            [self.module_logits_dict[k] for k in self.module_logits_dict.keys()]
        )
        return weights

    def get_merging_weights(self, **selector_kwargs) -> Dict:
        return self.get_routing_weights(**selector_kwargs)

    def get_routing_weights(self):
        return {k: v.detach().item() for k, v in self.module_logits_dict.items()}

    def add_expert(self, expert_name: str, **kwargs):
        """
        Assume:
        expert_task_name -- task name expert is pecialized at
        self.config.finetune_task_name -- name of the task the model is currently trained on

        If we encounter a module for the current task, we init it with one hot, otherwise with uniform.
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


@register_multi_expert_selector("uniform", PolySelectorDirectConfigUniform)
class PolyUniform(PolySelectorDirect):
    """
    Currently only used for uniform merging of experts.
    """

    def add_expert(self, expert_name: str, **kwargs):
        if expert_name not in self.module_logits_dict:
            self.module_logits_dict[expert_name] = torch.nn.Parameter(
                torch.ones(1).to(self.device)
            )
            for name in self.module_logits_dict.keys():
                self.module_logits_dict[name].data = torch.ones(1).to(self.device)
                self.module_logits_dict[name].data /= len(self.module_logits_dict)


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

    def get_merging_weights(self, **selector_kwargs) -> Dict:
        raise NotImplementedError(
            "Not implemented yet for RoutingInfosContainerSelector."
        )


@dataclass
class TaskNameSelectorConfig(SelectorConfig):
    pass


@register_multi_expert_selector("task_selector", TaskNameSelectorConfig)
class TaskNameSelector(Selector):
    def __init__(self, info_container, **kwargs) -> None:
        super().__init__(info_container)

        self.default_expert_name = None
        self.task2expert_name = {}

    @forward_with_cache
    def forward(self, input, **kwargs) -> ModulesSelectorOutput:
        if not self.routing_infos or not self.routing_infos.task_names:
            # try to infer batch size
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
                any(task_name not in self.task2expert_name for task_name in task_names)
                and not self.default_expert_name
                and len(self.task2expert_name)
            ):
                raise ValueError(
                    "Experts for all tasks have not been loaded! Set a default expert?"
                )
            modules = [
                self.task2expert_name.get(task_name, self.default_expert_name)
                for task_name in task_names
            ]

        return ModulesSelectorOutput(modules)

    def add_expert(self, expert_name: str, expert_info: ExpertInfo = None, **kwargs):
        if expert_info is None or expert_info.expert_task_name is None:
            logger.warn(
                "Expert's task_name not set, assume task name corresponds to expert name!"
            )
            self.task2expert_name[expert_name] = expert_name
        else:
            for task_name in expert_info.expert_task_name.split(","):
                self.task2expert_name[task_name] = expert_name

    def get_merging_weights(self, **selector_kwargs) -> Dict:
        raise NotImplementedError(
            "Not required for TaskNameSelector as it performs hard selection. Use 'get_expert_instance' instead."
        )


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

    def get_merging_weights(self, **selector_kwargs) -> Dict:
        raise ValueError(
            f"Not supported for {self.__class__}  since routing depends on input."
        )


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
