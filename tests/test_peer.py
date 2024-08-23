import numpy as np
import pytest

from mttl.config import MoEExpertConfig
from mttl.models.expert_model import MoEModel


def test_peer_moe(tmp_peer_moe_config, dummy_batch):
    config: MoEExpertConfig = tmp_peer_moe_config
    module = MoEModel(**vars(config))

    output = module(dummy_batch)
    assert np.allclose(output.item(), 18.0, atol=0.1)


if __name__ == "__main__":
    pytest.main([__file__])
