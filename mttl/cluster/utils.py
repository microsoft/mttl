from torch import Tensor
import tempfile
import torch
import numpy as np

ch = torch

def vectorize(g, arr=None, device="cuda") -> Tensor:
    """
    records result into arr

    gradients are given as a dict :code:`(name_w0: grad_w0, ... name_wp:
    grad_wp)` where :code:`p` is the number of weight matrices. each
    :code:`grad_wi` has shape :code:`[batch_size, ...]` this function flattens
    :code:`g` to have shape :code:`[batch_size, num_params]`.
    """
    if arr is None:
        g_elt = g[list(g.keys())[0]]
        batch_size = g_elt.shape[0]
        num_params = 0
        for param in g.values():
            assert param.shape[0] == batch_size
            num_params += int(param.numel() / batch_size)
        arr = ch.empty(size=(batch_size, num_params), dtype=g_elt.dtype, device=device)

    pointer = 0
    for param in g.values():
        if len(param.shape) < 2:
            num_param = 1
            p = param.data.reshape(-1, 1)
        else:
            num_param = param[0].numel()
            p = param.flatten(start_dim=1).data

        arr[:, pointer : pointer + num_param] = p.to(device)
        pointer += num_param

    return arr