from dataclasses import dataclass
import numpy as np
from mttl.models.containers.selectors.base import Selector, artifacts_cache
from mttl.models.containers.selectors.per_token_selector import (
    PerTokenSelector,
    PerTokenSelectorConfig,
)

from mttl.models.containers.selectors.base import forward_with_cache
from mttl.models.containers.selectors.selector_output import (
    BatchSequenceExpertsAndWeightsSelectorOutput,
)

from torch.nn import functional as F


@dataclass
class PriorSelectorConfig(PerTokenSelectorConfig):
    alpha: float = 1.0


@Selector.register("arrow_prior_router", PriorSelectorConfig)
class PriorPerTokenSelector(PerTokenSelector):
    def __init__(self, config, **kwargs) -> None:
        super().__init__(config, **kwargs)
        self.alpha = config.alpha

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
        in_dist = task_names is not None and all(
            [t in self.task_to_expert_name for t in task_names]
        )

        # control entropy of distribution
        router_logits /= temp

        ## get the prior distribution
        posterior_router_probs, prior, likelihood = self.prior.get_posterior(
            router_logits,
            temp,
            input=input,
            layer_name=self.__layer_name__,
            k=len(self.expert_names),
            device=self.device,
            info_container=self.info_container,
        )

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
