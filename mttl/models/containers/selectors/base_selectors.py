import math
import threading
from abc import ABC
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from pyparsing import abstractmethod
from torch import nn
from torch.distributions import Categorical

from mttl.logging import logger, warn_once
from mttl.models.expert_context import InfoContainer
from mttl.models.library.expert import ExpertInfo
from mttl.models.ranker.adapter_ranker import AdapterRankerHelper
from mttl.models.ranker.classifier_ranker import ClusterPredictor
from mttl.models.utils import MetricLogger

SELECTORS_NAME_TO_KLASS = {}
SELECTORS_CONFIG_TO_NAME = {}
SELECTORS_NAME_TO_CONFIG = {}


EPS = 1e-8


def register_multi_expert_selector(name, config_cls):
    print("Registering muti-expert selector..." + name)

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


def get_selector(selector_config: "SelectorConfig", **kwargs):
    """Returns a selector object for the given routing_config."""
    return SELECTORS_NAME_TO_KLASS[SELECTORS_CONFIG_TO_NAME[selector_config.__class__]](
        config=selector_config, **kwargs
    )


@dataclass
class SelectorConfig:
    # the granularity of the selector (which layers use the same selectors)
    router_granularity: str = "*"
    lora_merge_after: bool = False
    selector_logging: bool = True

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

    @classmethod
    def from_training_config(
        cls, training_config: Union["Config", "SelectorConfig"]
    ) -> Union["SelectorConfig", None]:
        """Build modifier config from the training config.

        Returns None if no modifier is set.
        """
        if isinstance(training_config, SelectorConfig):
            # nothing to do here
            return training_config

        # if called on the base class, we need to find the correct subclass
        if training_config.router_selector is None:
            return None

        if training_config.router_selector not in SELECTORS_NAME_TO_KLASS:
            raise ValueError(
                f"Selector '{training_config.router_selector}' not found, has it been registered?"
            )

        config_klass = SELECTORS_NAME_TO_CONFIG[training_config.router_selector]

        kwargs = {}
        for key in config_klass.__dataclass_fields__.keys():
            # only overwrite default if value exists and is not None
            train_cfg_value = getattr(training_config, key, None)
            if train_cfg_value is not None:
                kwargs[key] = getattr(training_config, key)
        return config_klass(**kwargs)


@dataclass
class LoadableSelectorConfig(SelectorConfig):
    """Adds support for library_id and data_id, which specifies the unique identifier to load."""

    library_id: str = None
    selector_data_id: str = None


@dataclass
class SelectorOutput:
    ALL_EXPERTS = "all"

    def __post_init__(self):
        if hasattr(self, "weights") and self.weights.ndim != len(self.dim_names):
            raise ValueError(
                "Weights should have the same number of dimensions as dim_names for this SelectorOutput."
            )

    @property
    def dim_names(self):
        raise NotImplementedError("dim_names not implemented for this selector output.")


@dataclass
class BatchExpertsSelectorOutput(SelectorOutput):
    """A selector output that contains a list of experts without weights.

    experts: names of the selected experts for each element in the batch
    """

    experts: List[str]


@dataclass
class ExpertsAndWeightsSelectorOutput(SelectorOutput):
    """A selector output that contains a list of experts and weights shared across the batch.

    experts: names of the selected experts
    weights: their weights
    """

    experts: List[str]
    weights: torch.Tensor

    @property
    def dim_names(self):
        return ["experts"]


@dataclass
class BatchExpertsAndWeightsSelectorOutput(SelectorOutput):
    """A selector output that contains a list of experts and weights for each example.

    experts: either names or indices of the selected experts
    weights: their weights
    """

    experts: Union[List[List[str]], torch.Tensor]
    weights: torch.Tensor

    @property
    def dim_names(self):
        return ["batch", "experts"]


@dataclass
class ExpertsSplitsAndWeightsSelectorOutput(ExpertsAndWeightsSelectorOutput):
    """A selector output that contains a list of experts and weights for each split (MHR) and expert shared across the batch.

    experts: names of the selected experts
    weights: their weights
    """

    @property
    def dim_names(self):
        return ["splits", "experts"]


@dataclass
class BatchExpertsSplitsAndWeightsSelectorOutput(BatchExpertsAndWeightsSelectorOutput):
    """A selector output that contains a list of experts and weights for each split (MHR) and expert shared across the batch.

    experts: names of the selected experts
    weights: their weights
    """

    @property
    def dim_names(self):
        return ["batch", "splits", "experts"]


@dataclass
class BatchSequenceExpertsAndWeightsSelectorOutput(SelectorOutput):
    """A selector output that contains a list of experts and weights for each example and token.

    experts: indices of the selected experts
    weights: their weights
    """

    experts: torch.Tensor
    weights: torch.Tensor

    @property
    def dim_names(self):
        return ["batch", "sequence", "experts"]


@dataclass
class BatchSequenceExpertsSplitsAndWeightsSelectorOutput(
    BatchSequenceExpertsAndWeightsSelectorOutput
):
    """A selector output that contains a list of experts and weights for each example, token and split (MHR)."""

    @property
    def dim_names(self):
        return ["batch", "sequence", "splits", "experts"]


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


def safe_logging(func):
    def wrapper(selector, *args, **kwargs):
        if not selector.config.selector_logging:
            return None
        try:
            result = func(selector, *args, **kwargs)
        except Exception as e:
            if str(e)[:100] != getattr(logger, "previous_error", ""):
                logger.exception(f"An error occurred in {func.__name__}: {e}")
                logger.previous_error = str(e)[:100]
            result = None
        return result

    return wrapper


class Selector(nn.Module):
    metric_logger: MetricLogger = MetricLogger()

    def __init__(self, config=None, **kwargs):
        nn.Module.__init__(self)

        self.config = config
        self.expert_infos = {}
        self.expert_names = []
        self.selector_views = []
        self.forward_cache = None
        self.default_expert_name = None
        self.total_calls_per_forward = 0
        self._calls_counter = 0
        # dependency injection filled from ExpertContainer
        self.__layer_name__ = None

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
        info_container = InfoContainer.get()
        if not info_container:
            return None
        return info_container.routing_infos

    @abstractmethod
    def on_add_expert(
        self, expert_name: str, expert_info: ExpertInfo = None, is_default=False
    ):
        pass

    def add_expert(
        self, expert_name: str, expert_info: ExpertInfo = None, is_default=False
    ):
        self.on_add_expert(expert_name, expert_info, is_default)

        # standard bookkeeping for all selectors
        if is_default:
            self.default_expert_name = expert_name

        self.expert_infos[expert_name] = expert_info
        self.expert_names.append(expert_name)


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

    def add_expert(
        self,
        expert_name: str,
        expert_info: ExpertInfo = None,
        is_default=False,
        **kwargs,
    ):
        pass


@dataclass
class TaskPredictorSelectorConfig(SelectorConfig):
    ranker_path: str = None
    ranker_model: str = None
    ranker_top_k: int = 1


@register_multi_expert_selector("task_predictor_selector", TaskPredictorSelectorConfig)
class TaskPredictorSelector(Selector):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.ranker_top_k = self.config.ranker_top_k
        # get the routing model

        self.expert_ranker = AdapterRankerHelper.get_ranker_instance(
            ranker_model=self.config.ranker_model,
            ranker_path=self.config.ranker_path,
        )

        if isinstance(self.expert_ranker, ClusterPredictor):
            self.expert_ranker.init_clusters(kwargs["training_config"].library_id)

    @forward_with_cache
    def forward(self, input, **kwargs) -> BatchExpertsAndWeightsSelectorOutput:
        # get the sources_texts from routing_infos
        routing_infos = self.routing_infos

        if hasattr(routing_infos, "sources_texts"):
            sources_texts = routing_infos.sources_texts
            self.expert_ranker.set_available_tasks(self.expert_names)

            experts, weights = self.expert_ranker.predict_task(
                sources_texts, n=self.ranker_top_k
            )
            logger.debug(f"Predicted tasks: {experts} with weights {weights}")

            weights = torch.tensor(weights, device=input.device)
            return BatchExpertsAndWeightsSelectorOutput(experts, weights)
        else:
            raise ValueError(
                "Routing infos does not contain sources_texts, cannot predict tasks."
            )

    def get_merging_weights(self, **selector_kwargs) -> Dict:
        raise ValueError(
            f"Not supported for {self.__class__} since routing depends on input."
        )

    def on_add_expert(
        self, expert_name: str, expert_info: ExpertInfo = None, is_default=False
    ):
        pass


@dataclass
class MOERKHSSelectorConfig(SelectorConfig):
    rkhs_dim: int = 512
    emb_dim: int = 128
    top_k: int = -1


@register_multi_expert_selector("moe_rkhs_router", MOERKHSSelectorConfig)
class MOERKHSSelector(Selector):
    def __init__(self, config, **kwargs) -> None:
        super().__init__(config)

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
    def forward(self, input, **kwargs) -> BatchSequenceExpertsAndWeightsSelectorOutput:
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
            selected_experts = SelectorOutput.ALL_EXPERTS

        g = getattr(self.info_container, "routing_gates", [])
        g.append(router_logits)
        self.info_container.routing_gates = g

        return BatchSequenceExpertsAndWeightsSelectorOutput(
            experts=selected_experts, weights=routing_weights
        )

    def get_merging_weights(self, **selector_kwargs) -> Dict:
        raise ValueError(
            f"Not supported for {self.__class__} since routing depends on input."
        )

    def get_routing_weights(self):
        raise ValueError("Not supported for MOESelector.")

    def on_add_expert(
        self, expert_name: str, expert_info: ExpertInfo = None, is_default=False
    ):
        # just initialize the expert embeddings
        self.rkhs_embeddings.data = torch.cat(
            [
                self.rkhs_embeddings.data,
                torch.zeros(
                    1, self.emb_dim, device=self.rkhs_embeddings.device
                ).uniform_(-0.02, 0.02),
            ],
            dim=0,
        )


class TaskToExpertMixin:
    """
    Builds `task_to_expert_name` mapping on add_expert, useful for
    routing (as in TaskNameSelector) or for logging in-distribution stats (PerTokenSelector)
    """

    @property
    def task_to_expert_name(self):
        return getattr(self, "_task_to_expert_name", {})

    def on_add_expert(
        self, expert_name: str, expert_info: ExpertInfo = None, is_default=False
    ):
        _task_to_expert_name = self.task_to_expert_name

        if expert_info is None or expert_info.expert_task_name is None:
            logger.warning(
                "Expert's task_name not set, assume task name corresponds to expert name!"
            )
            _task_to_expert_name[expert_name] = expert_name
        else:
            for task_name in expert_info.expert_task_name.split(","):
                if task_name in _task_to_expert_name:
                    logger.warning(
                        f"Task name {task_name} already assigned to expert {_task_to_expert_name[task_name]}"
                    )
            _task_to_expert_name[task_name] = expert_name

        self._task_to_expert_name = _task_to_expert_name


@dataclass
class PerTokenSelectorConfig(LoadableSelectorConfig):
    router_temp: float = None
    moe_top_k: int = None
    proto_init: str = None
    input_norm_fn: str = None
    proto_norm_fn: str = None


class LoadableLibraryMixin(ABC):

    cache = threading.local()
    cache.library_artifacts = None

    @property
    def library_artifacts(self) -> Optional[Dict]:
        return LoadableLibraryMixin.cache.library_artifacts

    @abstractmethod
    def _load_from_library(self):
        pass

    def load_from_library(self):

        if LoadableLibraryMixin.cache.library_artifacts is None:
            LoadableLibraryMixin.cache.library_artifacts = self._load_from_library()

            if not LoadableLibraryMixin.cache.library_artifacts:
                raise ValueError(f"Could not load library artifacts for selector.")


def get_expert_prototype_from_library_artifacts(
    expert_name: str, layer_name: str, library_artifacts: Dict
) -> torch.Tensor:
    """Utils function that returns the expert prototype stored in the library.

    This is used by Arrow, PhatGoose and Avg Activation selectors.
    """
    import numpy as np

    patched_layer_name = layer_name.replace(".selector", "")

    if expert_name not in library_artifacts:
        raise ValueError(
            f"Cannot load prototypes for expert `{expert_name}`, was not found in library.\n"
            f"Please recompute selector prototypes with the correct library transform."
        )

    layer_names = library_artifacts[expert_name].keys()
    valid_layer_names = [
        k
        for k in layer_names
        if patched_layer_name in k  # k.startswith(patched_layer_name)
    ]

    key = sorted(valid_layer_names)[0]
    proto = library_artifacts[expert_name][key]
    if isinstance(proto, np.ndarray):
        proto = torch.from_numpy(proto)
    return proto


@register_multi_expert_selector("per_token_router", PerTokenSelectorConfig)
class PerTokenSelector(Selector, TaskToExpertMixin, LoadableLibraryMixin):
    def __init__(self, config, **kwargs) -> None:
        super().__init__(config, **kwargs)

        if "layer" not in kwargs:
            raise ValueError(
                "Selector requires a layer to be passed in kwargs to infer the input dimension."
            )

        layer = kwargs["layer"]
        self.output_dim, self.input_dim = layer.out_features, layer.in_features

        self.prototypes = nn.Parameter(
            torch.empty((0, self.input_dim), device=layer.weight.device)
        )

        # validate args
        assert self.config.proto_init is not None
        assert self.config.input_norm_fn in ["id", "norm_d", "unit"]
        assert self.config.proto_norm_fn in ["id", "norm_d", "norm_p", "unit"]

        def _get_norm_layer(norm_fn):
            """helper for normalizing input and expert embeddings"""

            if norm_fn == "norm_d":
                return nn.LayerNorm(self.input_dim, elementwise_affine=False)
            elif norm_fn == "norm_p":

                def _unit_norm(x):
                    x_ = x.transpose(0, 1)  # (d, n_exp)
                    return F.layer_norm(x_, x_.shape[1:]).transpose(0, 1)

                return _unit_norm

            elif norm_fn == "unit":

                def _unit_norm(x):
                    return x / x.norm(dim=-1, p=2, keepdim=True).clamp(min=EPS)

                return _unit_norm
            else:
                return nn.Identity()

        self.input_norm = _get_norm_layer(self.config.input_norm_fn)
        self.proto_norm = _get_norm_layer(self.config.proto_norm_fn)

        # init selector from library if needed
        if self.config.library_id is not None:
            self.load_from_library()

    def overwrite_prototypes(self, prototypes: torch.tensor):
        """Overwrites the prototypes with the given tensor."""
        if (
            prototypes.shape[0] != self.prototypes.shape[0]
            or self.prototypes.shape[1] != prototypes.shape[1]
        ):
            raise ValueError("Prototypes shape are mismatched!")

        self.prototypes.data = prototypes.to(
            dtype=self.prototypes.dtype, device=self.prototypes.device
        )

    @safe_logging
    def _log_angle(self, angle):
        bs, sq, n_exp = angle.size()

        if sq > 1:
            attn_mask = self.routing_infos.attention_mask == 1.0
            mean_angle = angle[attn_mask].sum() / attn_mask.sum() / n_exp
        else:
            mean_angle = angle.mean()

        task = self.routing_infos.task_sources[0]
        to_store = {"angle": mean_angle.item()}
        self.metric_logger.update(prefix=f"task_{task}", value_dict=to_store)
        self.metric_logger.update(prefix=self.__layer_name__, value_dict=to_store)

    @safe_logging
    def _log_entropy(self, logits):
        # uniform routing entropy
        bs, sq, dim = logits.size()

        dist = Categorical(logits=logits)
        entropy = dist.entropy()

        if sq > 1:
            attn_mask = self.routing_infos.attention_mask == 1.0
            mean_entropy = entropy[attn_mask].sum() / attn_mask.sum()
        else:
            mean_entropy = entropy.mean()

        task = self.routing_infos.task_sources[0]
        to_store = {"ent_routing": mean_entropy.item()}
        self.metric_logger.update(prefix=f"task_{task}", value_dict=to_store)
        self.metric_logger.update(prefix=self.__layer_name__, value_dict=to_store)

        to_store["ent_uniform"] = np.log(len(self.expert_names))
        self.metric_logger.update(value_dict=to_store)

    @safe_logging
    def _maybe_log_in_dist(self, logits):
        probs = F.softmax(logits, dim=-1)
        bs, seq_len, _ = probs.size()
        task_names = self.routing_infos.task_names

        if all([t in self.task_to_expert_name for t in task_names]):
            expert_names = [self.task_to_expert_name[t] for t in task_names]

            expert_ids = torch.LongTensor(
                [self.expert_names.index(e) for e in expert_names]
            ).to(logits.device)

            expert_p = torch.gather(
                probs, index=expert_ids.view(bs, 1, 1).expand(-1, seq_len, -1), dim=-1
            )

            attn_mask = self.routing_infos.attention_mask == 1.0

            # are we teacher forcing or generating ?
            if seq_len == 1:
                mean_correct_p = expert_p.mean()
            else:
                mean_correct_p = expert_p[attn_mask].mean()

            to_store = {"expert_p": mean_correct_p.item()}
            self.metric_logger.update(
                prefix=f"task_{task_names[0]}", value_dict=to_store
            )
            self.metric_logger.update(prefix=self.__layer_name__, value_dict=to_store)

    @forward_with_cache
    def forward(self, input, **kwargs) -> BatchSequenceExpertsAndWeightsSelectorOutput:
        # do routing business on fp32
        temp = (
            self.config.router_temp
            if self.config.router_temp > 0
            else np.sqrt(input.shape[-1])
        )
        if self.prototypes.size(0) != len(self.expert_names):
            raise ValueError("Prototypes not initialized correctly.")

        input = input.to(dtype=self.prototypes.dtype)
        input = self.input_norm(input)
        prototypes = self.proto_norm(self.prototypes)

        # logit computation
        router_logits = F.linear(input, prototypes)
        if self.config.proto_init == "arrow":
            router_logits = router_logits.abs()

        # log angle between input and prototypes
        angle = router_logits / input.norm(p=2, dim=-1, keepdim=True).clamp(min=EPS)
        angle = angle / prototypes.norm(p=2, dim=-1).view(1, 1, -1).clamp(min=EPS)

        self._log_angle(angle)

        # control entropy of distribution
        router_logits /= temp

        if self.config.moe_top_k > 0:
            # For now, we always renormalize the routing weights for hard routing
            top_k_logits, experts = torch.topk(
                router_logits, self.config.moe_top_k, dim=-1
            )
            router_probs = F.softmax(top_k_logits, dim=-1)

            # Adjust router_logits accordingly for logging
            chosen = torch.zeros_like(router_logits, dtype=torch.bool)
            chosen.scatter_add_(
                dim=-1, index=experts, src=torch.ones_like(experts).bool()
            )
            router_logits = router_logits.masked_fill(~chosen, -1e9)
        else:
            experts = SelectorOutput.ALL_EXPERTS
            router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float)

        self._log_entropy(router_logits)
        self._maybe_log_in_dist(router_logits)

        return BatchSequenceExpertsAndWeightsSelectorOutput(
            experts=experts, weights=router_probs
        )

    def on_add_expert(
        self, expert_name: str, expert_info: ExpertInfo = None, is_default=False
    ):
        if self.library_artifacts is not None:
            proto = get_expert_prototype_from_library_artifacts(
                expert_name, self.layer_name, self.library_artifacts
            ).view(1, -1)
        else:
            warn_once(
                f"Library artifacts not loaded for {self.__class__}, using zero initialization."
            )
            proto = torch.zeros(
                1,
                self.prototypes.size(1),
                dtype=self.prototypes.dtype,
                device=self.prototypes.device,
            )

        dev = self.prototypes.device
        self.prototypes.data = torch.cat([self.prototypes.data, proto.to(dev)])


@dataclass
class TaskNameSelectorConfig(SelectorConfig):
    pass


@register_multi_expert_selector("task_selector", TaskNameSelectorConfig)
class TaskNameSelector(Selector, TaskToExpertMixin):
    def __init__(self, **kwargs) -> None:
        super().__init__()

    @forward_with_cache
    def forward(self, input, **kwargs) -> BatchExpertsSelectorOutput:
        if not self.routing_infos or not self.routing_infos.task_names:
            # try to infer batch size
            if "input_ids" in kwargs:
                batch_size = kwargs["input_ids"].size(0)
            else:
                batch_size = input.shape[0]

            if not self.default_expert_name:
                raise ValueError("No default expert name set and no task names given!")

            experts = [self.default_expert_name for _ in range(batch_size)]
        else:
            task_names = self.routing_infos.task_names

            if (
                any(
                    task_name not in self.task_to_expert_name
                    for task_name in task_names
                )
                and not self.default_expert_name
                and len(self.task_to_expert_name)
            ):
                raise ValueError(
                    "Experts for all tasks have not been loaded! Set a default expert?"
                )
            experts = [
                self.task_to_expert_name.get(task_name, self.default_expert_name)
                for task_name in task_names
            ]

        return BatchExpertsSelectorOutput(experts)

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

    def on_add_expert(
        self, expert_name: str, expert_info: ExpertInfo = None, is_default=False
    ):
        pass


@dataclass
class KVTaskNameSelectorConfig(SelectorConfig):
    pass


@register_multi_expert_selector("kv_task_selector", KVTaskNameSelectorConfig)
class KVTaskNameSelector(KVSelector):
    """Selects KVAdapters based on the task name."""

    def __init__(self, **kwargs) -> None:
        super().__init__()

    def get_kv_weights(self, experts, k_proj, v_proj):
        routing_infos = self.routing_infos

        task_names = routing_infos.task_names

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
        routing_infos = self.routing_infos

        task_names = routing_infos.task_names

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

    def __init__(self, **kwargs) -> None:
        super().__init__()

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
