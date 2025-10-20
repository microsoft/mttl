import copy
import torch
import numpy as np
from dataclasses import dataclass
from huggingface_hub import hf_hub_download
from mttl.models.library.expert_library import ExpertLibrary
from mttl.models.library.library_transforms import (
    LibraryTransform,
    LibraryTransformConfig,
)
from mttl.models.expert_model import ExpertModel
from mttl.models.library.expert import Expert


# -----------------------------------
# SLERP and LERP implementation
# -----------------------------------
def lerp(t, v0, v1, origin_data_type=None):
    v2 = (1 - t) * v0 + t * v1
    return torch.from_numpy(v2).to(origin_data_type)


# SLERP
def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    """
    Spherical linear interpolation
    Args:
        t (float/np.ndarray): Float value between 0.0 and 1.0
        v0 (np.ndarray): Starting vector
        v1 (np.ndarray): Final vector
        DOT_THRESHOLD (float): Threshold for considering the two vectors as
                               colineal. Not recommended to alter this.
    Returns:
        v2 (np.ndarray): Interpolation vector between v0 and v1
    """
    origin_data_type = v0.dtype
    v0 = v0.detach().cpu().float().numpy()
    v1 = v1.detach().cpu().float().numpy()

    # Copy the vectors to reuse them later
    v0_copy = np.copy(v0)
    v1_copy = np.copy(v1)
    # Normalize the vectors to get the directions and angles
    v0 = v0 / np.linalg.norm(v0)
    v1 = v1 / np.linalg.norm(v1)
    # Dot product with the normalized vectors (can't use np.dot in W)
    dot = np.sum(v0 * v1)
    # If absolute value of dot product is almost 1, vectors are ~colineal, so use lerp
    if np.abs(dot) > DOT_THRESHOLD:
        return lerp(t, v0_copy, v1_copy, origin_data_type=origin_data_type)
    # Calculate initial angle between v0 and v1
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    # Angle at timestep t
    theta_t = theta_0 * t
    sin_theta_t = np.sin(theta_t)
    # Finish the slerp algorithm
    s0 = np.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0
    v2 = s0 * v0_copy + s1 * v1_copy

    return torch.from_numpy(v2).to(origin_data_type)


@dataclass
class SLERPMergeConfig(LibraryTransformConfig):
    weights: dict = None


class SLERPMerge(LibraryTransform):
    """
    Computes a uniform weight mixture across experts of a given library
    """

    def __init__(self, config: SLERPMergeConfig = None):
        super().__init__(config or SLERPMergeConfig())

    def load_mask(self, expert):
        try:
            print("trying to load mask from hf")
            library_id = expert.training_config.library_id
            destination_type, f_name = library_id.split("://")
            repo_id = ("/").join(f_name.split("/")[:2])
            filename = f"{expert.expert_info.expert_name}_mask.npz"
            f_path = hf_hub_download(repo_id=repo_id, filename=filename)
            Mask = np.load(f_path, allow_pickle=True)["arr"].item()
        except:
            print("trying to load mask from local dir")
            m_loc = f"experiment/{expert.training_config.exp_name}/mask.npz"
            Mask = np.load(m_loc, allow_pickle=True)["arr"].item()
        return Mask

    def convert_idx_2_mask(self, weight_idx, mat_dim):
        m = np.zeros(mat_dim)
        m[tuple(zip(*weight_idx))] = 1
        return torch.FloatTensor(m)

    def sparse_SLERP(self, model, experts, base_expert):
        base_expert_mask = self.load_mask(base_expert)
        for layer, _ in base_expert_mask.items():
            mod_layer = ".".join(layer.split(".")[1:])
            base_expert_mask[layer] = self.convert_idx_2_mask(
                weight_idx=base_expert_mask[layer],
                mat_dim=base_expert.expert_weights[f"{mod_layer}.weight"].shape,
            )
        weight_names = [n for n in model.state_dict().keys() if "sparse_layer" in n]

        for expert in experts:
            mask = self.load_mask(expert)
            # for expert_name, expert in library.items():
            for layer, weight in model.state_dict().items():
                if layer in weight_names:
                    common_name = ".".join(layer.split(".")[1:-1])
                    param_type = layer.split(".")[-1]

                    if param_type == "weight":
                        # get mask-m for layer-l: convert the weight_indx to convert sparse-mask
                        m = self.convert_idx_2_mask(
                            weight_idx=mask[f"model.{common_name}"],
                            mat_dim=expert.expert_weights[
                                f"{common_name}.weight"
                            ].shape,
                        )
                        bm = base_expert_mask[f"model.{common_name}"]
                    else:
                        m = 1.0
                        bm = 1.0
                    base_expert._expert_weights[f"{common_name}.{param_type}"] = slerp(
                        float(1.0) - 0.5,
                        v0=base_expert._expert_weights[f"{common_name}.{param_type}"]
                        * bm,
                        v1=expert._expert_weights[f"{common_name}.{param_type}"] * m,
                    )
                    if param_type == "weight":
                        base_expert_mask[f"model.{common_name}"] = torch.logical_or(
                            m, bm
                        ).float()

        updated_state_dict = {}
        for layer, weight in model.state_dict().items():
            if layer in weight_names:
                mod_layer = ".".join(layer.split(".")[1:])
                updated_state_dict[layer] = base_expert._expert_weights[mod_layer]
            else:
                updated_state_dict[layer] = weight
        # load state_dict into model
        model.load_state_dict(updated_state_dict)
        return model

    def FFT_SLERP(self, model, experts, base_expert):
        for expert in experts:
            for layer, _ in model.state_dict().items():
                common_name = ".".join(layer.split(".")[1:-1])
                param_type = layer.split(".")[-1]
                base_expert._expert_weights[f"{common_name}.{param_type}"] = slerp(
                    float(1.0) - 0.5,
                    v0=base_expert._expert_weights[f"{common_name}.{param_type}"],
                    v1=expert._expert_weights[f"{common_name}.{param_type}"],
                )
        updated_state_dict = {}
        for layer, _ in model.state_dict().items():
            mod_layer = ".".join(layer.split(".")[1:])
            updated_state_dict[layer] = base_expert._expert_weights[mod_layer]
        # load state_dict into model
        model.load_state_dict(updated_state_dict)
        return model

    @torch.no_grad()
    def transform(self, library) -> Expert:
        if type(library) == str:
            library = ExpertLibrary.get_expert_library(library)

        expert_names = list(library.keys())
        experts = [library[name] for name in expert_names]
        base_expert = copy.deepcopy(experts[0])
        training_config = copy.deepcopy(base_expert.training_config)
        from mttl.models.expert_model import ExpertModel

        model = ExpertModel(**vars(training_config))
        if training_config.model_modifier == None:
            # skip the first expert as it's now acting as base_expert
            model = self.FFT_SLERP(model, experts[1:], base_expert)
        elif training_config.model_modifier == "sparse_mask_adapter":
            model = self.sparse_SLERP(model, experts[1:], base_expert)
        return model
