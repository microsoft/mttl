import torch
import numpy as np
from huggingface_hub import hf_hub_download
from types import SimpleNamespace


def dict_to_config(d):
    if isinstance(d, dict):
        return SimpleNamespace(**d)
    return d  # or raise an error


def topk_multiple_experts(expert_vectors, topk, TH_type=None):
    assert TH_type != None
    n_tasks = expert_vectors.shape[0]
    values = []
    for t in range(n_tasks):
        print("topk expert", t)
        v, _ = torch.topk(expert_vectors[t, :], topk)
        if TH_type == "lower":
            values.append(v[-1])
        elif TH_type == "upper":
            values.append(v[0])
        del v
    values = torch.stack(values, dim=0)  # Shape will be (n_tasks,)
    return values


def load_mask(expert):
    config = dict_to_config(expert.training_config)
    try:
        print("trying to load mask from hf")
        library_id = config.library_id
        destination_type, f_name = library_id.split("://")
        repo_id = ("/").join(f_name.split("/")[:2])
        filename = f"{expert.expert_info.expert_name}_mask.npz"
        f_path = hf_hub_download(repo_id=repo_id, filename=filename)
        Mask = np.load(f_path, allow_pickle=True)["arr"].item()
    except:
        print("trying to load mask from local dir")
        m_loc = f"experiment/{config.exp_name}/mask.npz"
        Mask = np.load(m_loc, allow_pickle=True)["arr"].item()
    return Mask


def convert_idx_2_mask(weight_idx, mat_dim):
    m = np.zeros(mat_dim)
    m[tuple(zip(*weight_idx))] = 1
    return torch.FloatTensor(m)


def get_nested_model(module):
    current = module
    while hasattr(current, "model"):
        current = current.model
    return current
