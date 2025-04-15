from dataclasses import dataclass
import torch
from mttl.models.library.expert_library import ExpertLibrary
from mttl.models.library.expert import Expert
from mttl.models.library.library_transforms import (
    LibraryTransform,
    LibraryTransformConfig,
)
from mttl.models.library.expert import Expert


@dataclass
class LoRA_ab_LinearMergeConfig(LibraryTransformConfig):
    weights: dict = None


class LoRA_ab_LinearMerge(LibraryTransform):
    """
    Computes a uniform weight mixture across experts of a given library
    """

    def __init__(self, config: LoRA_ab_LinearMergeConfig = None):
        super().__init__(config or LoRA_ab_LinearMergeConfig())

    @torch.no_grad()
    def transform(self, library) -> Expert:
        if type(library) == str:
            library = ExpertLibrary.get_expert_library(library)
        # get expert config
        from copy import deepcopy

        an_expert = library[next(iter(library.keys()))]
        training_config = deepcopy(an_expert.training_config)
        # create a ExpertModel
        from mttl.models.expert_model import ExpertModel

        model = ExpertModel(**vars(training_config))
        # filter the weight names
        weight_names = [
            n
            for n in model.state_dict().keys()
            if ("Wqkv.weight" in n or "out_proj.weight" in n)
        ]

        # iterate over the library
        import collections

        store_W = collections.defaultdict(dict)
        for expert_name, expert in library.items():
            # iterate over the expert weights
            for l in weight_names:
                common_name = ".".join(l.split(".")[1:-1])
                A, B = (
                    expert.expert_weights[f"{common_name}.lora_a"],
                    expert.expert_weights[f"{common_name}.lora_b"],
                )
                W = A @ B
                store_W[l][expert_name] = W

        store_average_W = collections.defaultdict(dict)
        # iterate over all the layers of W
        for l in weight_names:
            average_W = 0
            for k, v in store_W[l].items():
                average_W += v
            average_W /= len(store_W[l])
            store_average_W[l] = average_W
            # average the Ws for each layer

        new_state_dict = {}
        # add the averaged Ws to the model
        for key, value in model.state_dict().items():
            if key in weight_names:
                print(f"added {key}")
                new_state_dict[key] = value + store_average_W[key].T
            else:
                new_state_dict[key] = value

        # load state_dict into model
        model.load_state_dict(new_state_dict)
        return model
