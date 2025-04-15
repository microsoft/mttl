import os
import json
import collections
import torch
from huggingface_hub import hf_hub_download
from copy import deepcopy
import numpy as np
from dataclasses import dataclass
from mttl.models.library.expert_library import ExpertLibrary
from mttl.models.library.expert import Expert
from mttl.models.library.library_transforms import (
    LibraryTransform,
    LibraryTransformConfig,
)
from mttl.models.expert_model import ExpertModel
from mttl.models.library.merge_models.utils import load_mask, convert_idx_2_mask


@dataclass
class SparseWeightLinearMergeConfig(LibraryTransformConfig):
    weights: dict = None


class SparseWeightLinearMerge(LibraryTransform):
    """
    Computes a uniform weight mixture across experts of a given library
    """

    def __init__(self, config: SparseWeightLinearMergeConfig = None):
        super().__init__(config or SparseWeightLinearMergeConfig())

    def update_module_mask(self, module, expert):
        Mask = load_mask(expert)
        for m_name, m in dict(module.named_modules()).items():
            if "sparse_layer" in m_name:
                keep_mask = convert_idx_2_mask(
                    weight_idx=Mask[m_name], mat_dim=m.weight_mask.shape
                )

                m.weight_mask = keep_mask.data.clone()

    @torch.no_grad()
    def transform(self, library) -> Expert:
        if type(library) == str:
            library = ExpertLibrary.get_expert_library(library)
        # get expert config
        an_expert = library[next(iter(library.keys()))]
        training_config = deepcopy(an_expert.training_config)
        # create a ExpertModel
        from mttl.models.expert_model import ExpertModel

        print("Trying to load base model from:", training_config.model)
        model = ExpertModel(**vars(training_config))
        sparse_layer_names = [
            n for n in model.state_dict().keys() if ("sparse_layer" in n)
        ]  # allow to add weights and bias
        assert sparse_layer_names != [], print("could not find sparse-layer modules")

        # iterate over the library
        store_W = collections.defaultdict(dict)
        store_m = collections.defaultdict(dict)
        expert_weight_hist = collections.defaultdict(dict)  # weight stats

        for expert_name, expert in library.items():
            print(f"Merging sparse weight for task: {expert_name}")
            Mask = self.load_mask(expert)
            expert_weight_hist[expert_name] = collections.defaultdict(
                dict
            )  # weight stats
            # for each layer compute the "average of the overlapped weight"
            for l in sparse_layer_names:
                common_name = ".".join(l.split(".")[1:-1])
                param_type = l.split(".")[-1]

                if param_type == "weight":
                    # get mask-m for layer-l: convert the weight_indx to convert sparse-mask
                    m = self.convert_idx_2_mask(
                        weight_idx=Mask[f"model.{common_name}"],
                        mat_dim=expert.expert_weights[f"{common_name}.weight"].shape,
                    )
                else:
                    m = 1.0
                # Check if entry exists
                if l in store_W:
                    # store weight
                    store_W[l] += (
                        expert.expert_weights[f"{common_name}.{param_type}"] * m
                    )
                    # store mask
                    store_m[l] += m
                # new entry for expert 1
                else:
                    store_W[l] = (
                        expert.expert_weights[f"{common_name}.{param_type}"] * m
                    )
                    store_m[l] = m
                expert_weight_hist[expert_name][common_name] = (
                    float(
                        expert.expert_weights[f"{common_name}.weight"]
                        .mean()
                        .data.numpy()
                    ),
                    float(
                        expert.expert_weights[f"{common_name}.weight"]
                        .std()
                        .data.numpy()
                    ),
                )  # weight stats

        # we sum the total per-layer weight overlap and devide the count as an alternate to calculate accurate weight average
        for l in sparse_layer_names:
            param_type = l.split(".")[-1]
            if param_type == "weight":
                store_m[l][
                    store_m[l] == 0
                ] = 1  # assigning 1 to the zero-masked weights positions to avoid numerical error in the next step
                store_W[l] /= store_m[l]
            else:
                store_W[l] /= len(library)
        new_state_dict = {}
        # add the averaged Ws to the model
        for key, value in model.state_dict().items():
            if key in sparse_layer_names:
                print(f"added {key}")
                new_state_dict[key] = value + store_W[key]
            else:
                new_state_dict[key] = value

        # load state_dict into model
        model.load_state_dict(new_state_dict)
        # save weights stats
        exp_temp = training_config.library_id.split("/")[-1]
        file_loc = f"Weight_Stats/{exp_temp}"
        os.makedirs(file_loc, exist_ok=True)
        with open(f"{file_loc}/weight_stats.json", "w") as json_file:
            json.dump(expert_weight_hist, json_file)

        return model

    @torch.no_grad()
    def transform_dummy(self, library, get_expert) -> Expert:
        if type(library) == str:
            library = ExpertLibrary.get_expert_library(library)
        # =============================
        # get expert config
        from copy import deepcopy

        an_expert = library[next(iter(library.keys()))]
        training_config = deepcopy(an_expert.training_config)
        # create a ExpertModel
        from mttl.models.expert_model import ExpertModel

        model = ExpertModel(**vars(training_config))
        # filter the weight names
        # weight_names = [n for n in model.state_dict().keys() if ('Wqkv.weight' in n or 'out_proj.weight' in n)]
        weight_names = [
            n
            for n in model.state_dict().keys()
            if ("Wqkv.sparse_layer.weight" in n or "out_proj.sparse_layer.weight" in n)
        ]

        # iterate over the library
        import collections

        store_W = collections.defaultdict(dict)
        store_m = collections.defaultdict(dict)
        for expert_name, expert in library.items():
            # TODO: only consider the weights that matches the given `get_expert` input
            if expert_name == get_expert:
                print(f"Merging sparse weight for task: {expert_name}")
                Mask = self.load_mask(expert)
                # for each layer compute the "average of the overlapped weight"
                for l in weight_names:
                    common_name = ".".join(l.split(".")[1:-1])
                    # get mask-m for layer-l: convert the weight_indx to convert sparse-mask
                    m = self.convert_idx_2_mask(
                        weight_idx=Mask[f"model.{common_name}"],
                        mat_dim=expert.expert_weights[f"{common_name}.weight"].shape,
                    )
                    if l in store_W.keys():
                        # store weight
                        store_W[l] += expert.expert_weights[f"{common_name}.weight"] * m
                        # store mask
                        store_m[l] += m
                    else:
                        store_W[l] = expert.expert_weights[f"{common_name}.weight"] * m
                        store_m[l] = m
        # we sum the total per-layer weight overlap and devide the count as an alternate to calculate accurate weight average
        for l in weight_names:
            store_m[l][
                store_m[l] == 0
            ] = 1  # assigning 1 to the zero-masked weights positions to avoid numerical error in the next step
            store_W[l] /= store_m[l]

        new_state_dict = {}
        # add the averaged Ws to the model
        for key, value in model.state_dict().items():
            if key in weight_names:
                print(f"added {key}")
                new_state_dict[key] = value + store_W[key]
            else:
                new_state_dict[key] = value

        # load state_dict into model
        model.load_state_dict(new_state_dict)
        return model
