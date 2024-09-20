from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from mttl.logging import warn_once
from mttl.models.containers.selectors.base import (
    EPS,
    LoadableLibraryMixin,
    LoadableSelectorConfig,
    Selector,
    forward_with_cache,
    get_expert_prototype_from_library_artifacts,
    safe_logging,
)
from mttl.models.containers.selectors.selector_output import (
    BatchSequenceExpertsAndWeightsSelectorOutput,
    SelectorOutput,
)
from mttl.models.library.expert import ExpertInfo


@dataclass
class PerTokenSelectorConfig(LoadableSelectorConfig):
    router_temp: float = None
    top_k: int = None
    proto_init: str = None
    input_norm_fn: str = None
    proto_norm_fn: str = None


@Selector.register("per_token_router", PerTokenSelectorConfig)
class PerTokenSelector(Selector, LoadableLibraryMixin):
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
            self.library_artifacts = self.load_from_library(self.config)
        else:
            self.library_artifacts = None

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

        task = self.routing_infos.task_names[0] if self.routing_infos.task_names is not None and len(self.routing_infos.task_names) > 0 else 'default'

        to_store = {"angle": mean_angle.item()}
        self.metric_logger.update(prefix=f"task_{task}", value_dict=to_store)
        self.metric_logger.update(prefix=self.__layer_name__, value_dict=to_store)

    @safe_logging
    def _log_entropy(self, logits):
        # uniform routing entropy
        bs, sq, dim = logits.size()

        dist = torch.distributions.Categorical(logits=logits)
        entropy = dist.entropy()

        if sq > 1:
            attn_mask = self.routing_infos.attention_mask == 1.0
            mean_entropy = entropy[attn_mask].sum() / attn_mask.sum()
        else:
            mean_entropy = entropy.mean()

        task = self.routing_infos.task_names[0]

        to_store = {"ent_routing": mean_entropy.item()}
        self.metric_logger.update(prefix=f"task_{task}", value_dict=to_store)
        self.metric_logger.update(prefix=self.__layer_name__, value_dict=to_store)

        to_store["ent_uniform"] = np.log(len(self.expert_names))
        self.metric_logger.update(value_dict=to_store)

    @safe_logging
    def log_in_dist(self, logits):
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

        task_names = self.routing_infos.task_names 
        in_dist = task_names is not None and all([t in self.task_to_expert_name for t in task_names])

        # control entropy of distribution
        router_logits /= temp

        if self.config.top_k > 0:
            # For now, we always renormalize the routing weights for hard routing
            top_k_logits, experts = torch.topk(router_logits, self.config.top_k, dim=-1)
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
        if in_dist:
            self._log_angle(angle)
            self._log_in_dist(router_logits)

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
                f"Library artifacts not loaded for {self.__class__.__name__}, using zero initialization."
            )
            proto = torch.zeros(
                1,
                self.prototypes.size(1),
                dtype=self.prototypes.dtype,
                device=self.prototypes.device,
            )

        dev = self.prototypes.device
        self.prototypes.data = torch.cat([self.prototypes.data, proto.to(dev)])
