import functools
import os

import numpy as np
import pytest
import torch
from pytorch_lightning import seed_everything

from mttl.config import MoEExpertConfig, MultiExpertConfig
from mttl.models.containers.lora_containers import (
    CoalescedLoRAExpertContainer,
    LoRAExpertContainer,
)
from mttl.models.containers.selectors.base import LoadableLibraryMixin
from mttl.models.containers.selectors.moe_selector import MOERKHSSelector
from mttl.models.containers.selectors.per_token_selector import PerTokenSelector
from mttl.models.containers.selectors.poly_selector import (
    PolySelector,
    PolySelectorConfig,
    PolySelectorDirect,
)
from mttl.models.containers.selectors.selector_output import (
    BatchSequenceExpertsAndWeightsSelectorOutput,
    SelectorOutput,
)
from mttl.models.expert_model import MoEModel, MultiExpertModel
from mttl.models.library.expert import Expert
from mttl.models.modifiers.base import ModifierConfig
from mttl.models.modifiers.lora import LoRA


def test_peer_moe(tmp_peer_moe_config, dummy_batch):
    config: MoEExpertConfig = tmp_peer_moe_config
    module = MoEModel(**vars(config))

    output = module(dummy_batch)
    assert np.allclose(output.item(), 18.0, atol=0.1)


if __name__ == "__main__":
    pytest.main([__file__])
