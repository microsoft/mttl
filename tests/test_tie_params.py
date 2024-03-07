import torch
from torch import nn
import pytest
import logging
from mttl.utils import logger


def test_torch_param_tie():
    logger.setLevel(logging.DEBUG)

    class model_cls(nn.Module):
        def __init__(self):
            super().__init__()
            A1 = torch.randn(10, 20)
            A2 = torch.randn(10, 20)

            self.param_A1 = torch.nn.Parameter(A1)
            self.param_A2 = torch.nn.Parameter(A2)

    model = model_cls()

    assert model.param_A1.sum().item() != model.param_A2.sum().item()
    new_param = torch.nn.Parameter(model.param_A1.data)
    model.param_A1 = new_param
    model.param_A2 = new_param
    assert model.param_A1.sum().item() == model.param_A2.sum().item()

    # make sure all params are in named_parameters
    assert len(list(model.named_parameters())) == 2
    assert len(list(model.parameters())) == 2
    assert "param_A1" in dict(model.named_parameters())
    assert "param_A2" in dict(model.named_parameters())

    optim = torch.optim.SGD(model.parameters(), lr=0.1)

    for _ in range(100):
        optim.zero_grad()
        loss = (0.1 * model.param_A1 + -2 * model.param_A2).sum()
        loss.backward()
        optim.step()

    assert model.param_A1.sum().item() == model.param_A2.sum().item()
    # is nto the same, so the grads are stored separately and update the same underlying .data.
    # also the optimizer keeps separate stats per parameter
    assert model.param_A1.grad.sum().item() != model.param_A2.grad.sum().item()
    assert "param_A1" in model.state_dict().keys()
    assert "param_A2" in model.state_dict().keys()


if __name__ == "__main__":
    pytest.main([__file__])
