import numpy as np
import pytest

from mttl.arguments import MoEExpertConfig
from mttl.models.expert_model import MoEModel, MoEModelConfig


def test_peer_moe(tmp_peer_moe_config, dummy_batch):
    config: MoEExpertConfig = tmp_peer_moe_config
    module = MoEModel(
        MoEModelConfig(
            base_model=config.model,
            moe_num_experts=config.moe_num_experts,
            selector_config=config.selector_config,
            modifier_config=config.modifier_config,
        )
    )

    output = module(**dummy_batch).loss
    assert np.allclose(output.item(), 18.0, atol=0.1)


if __name__ == "__main__":
    pytest.main([__file__])
