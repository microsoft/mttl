from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn

from mttl.logging import logger
from mttl.models.containers.selectors.base import (
    BatchSequenceExpertsAndWeightsSelectorOutput,
    Selector,
    SelectorConfig,
    artifacts_cache,
    forward_with_cache,
)
from mttl.models.containers.selectors.per_token_selector import (
    PerTokenSelector,
    PerTokenSelectorConfig,
)


def compute_phatgoose_embeddings(
    library,
    selector_data_id=None,
    n_steps_pg=100,
    learning_rate_pg=0.001,
    recompute_prototypes=False,
    default_args=None,
) -> str:
    """Computes Phatgoose embeddings for the given library."""
    from mttl.models.library.library_transforms import (
        PhatgooseConfig,
        PhatgooseTransform,
    )

    cfg = PhatgooseConfig(
        n_steps=n_steps_pg,
        learning_rate=learning_rate_pg,
        name=selector_data_id,
    )

    phatgoose_transform = PhatgooseTransform(cfg)
    phatgoose_transform.transform(
        library,
        default_args=default_args,
        recompute=recompute_prototypes,
        persist=True,
    )
    return cfg.save_name


@dataclass
class PhatgooseSelectorConfig(PerTokenSelectorConfig):
    router_temp: float = -1
    top_k: int = 2
    proto_init: str = "phatgoose"
    input_norm_fn: str = "norm_d"
    proto_norm_fn: str = "norm_d"
    lora_merge_after: bool = True


@Selector.register("phatgoose_router", PhatgooseSelectorConfig)
class PhatgooseSelector(PerTokenSelector):
    def __init__(self, config, **kwargs) -> None:
        super().__init__(config, **kwargs)

        if not self.config.lora_merge_after:
            logger.warning("PhatgooseSelector should have lora_merge_after=True")

    @classmethod
    @artifacts_cache
    def load_from_library(cls, config):
        """Fetches prototypes from the library."""
        from mttl.models.library.library_transforms import (
            PhatgooseConfig,
            PhatgooseTransform,
        )

        return PhatgooseTransform(PhatgooseConfig(name=config.selector_data_id)).fetch(
            config.library_id
        )


@dataclass
class PhatgooseTrainerSelectorConfig(SelectorConfig):
    pass


class SigmoidGate(nn.Module):
    def __init__(self, input_dim, output_dim=1, **kwargs):
        super().__init__()
        self.v = nn.Parameter(torch.zeros(output_dim, input_dim))

    def forward(self, x):
        return torch.sigmoid(torch.nn.functional.linear(x, self.v, bias=None))


@Selector.register("phatgoose_trainer_selector", PhatgooseTrainerSelectorConfig)
class PhatgooseTrainerSelector(Selector):
    """
    Selector from https://arxiv.org/abs/2402.05859
    """

    def __init__(self, config: PhatgooseTrainerSelectorConfig, **kwargs) -> None:
        super().__init__(config)

        if "layer" not in kwargs:
            raise ValueError(
                "PhatgooseTrainerSelector requires a layer to be passed in kwargs to infer the input dimension."
            )

        self.input_dim = kwargs["layer"].weight.data.shape[-1]
        self.device = kwargs["layer"].weight.device

        self.gates = nn.ParameterDict()
        self.layer = kwargs["layer"]
        self.default_expert_name = None
        self.routing_gates = []  # for logging purposes at training time

    @forward_with_cache
    def forward(self, input, **kwargs) -> BatchSequenceExpertsAndWeightsSelectorOutput:
        # selectors for tasks are trained independently
        # all samples go through the same selector
        scores = self.gates[self.default_expert_name](input)
        self.routing_gates.append(scores.detach().cpu().float())

        return BatchSequenceExpertsAndWeightsSelectorOutput(
            torch.zeros_like(scores, dtype=torch.long), scores
        )

    def on_add_expert(
        self, expert_name: str, expert_info: "ExpertInfo", is_default: bool = False
    ):
        self.gates[expert_name] = SigmoidGate(self.input_dim)

    def get_merging_weights(self, **selector_kwargs) -> Dict:
        raise ValueError(
            f"Not supported for {self.__class__}  since routing depends on input."
        )

    def get_prototypes(self):
        return {
            k: gate.v.detach().float().cpu().numpy() for k, gate in self.gates.items()
        }
