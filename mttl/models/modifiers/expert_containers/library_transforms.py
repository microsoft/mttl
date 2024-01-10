from dataclasses import dataclass
from mttl.models.modifiers.expert_containers.expert_library import HFExpertLibrary
from mttl.models.modifiers.expert_containers.module_graph import Expert
from mttl.utils import logger
from mttl.models.modifiers.modify_model import get_modifier_type
from typing import Optional

import copy
import torch
from tqdm import tqdm
import numpy as np
import sklearn.decomposition


class LibraryTransform:
    """Defines a transformation of a library of experts."""

    def __init__(self, config):
        self.config = config

    def transform(library):
        raise NotImplementedError()


@dataclass
class SVDEmbeddingTransformConfig:
    name: str = "svd"
    n_components: int = 64
    sparsity_threshold: float = 0.8


class SVDEmbeddingTransform(LibraryTransform):
    """Creates adapter embeddings by low-rank decomposition of a sparsified version
    of the adapter modules.
    """

    def transform(self, library, upload_to_hf=True):
        if type(library) == str:
            library = HFExpertLibrary(library)

        svd = sklearn.decomposition.TruncatedSVD(
            n_components=self.config.n_components,
            algorithm="randomized",
            n_iter=5,
            n_oversamples=10,
            power_iteration_normalizer="auto",
            random_state=None,
            tol=0.0,
        )

        names = []
        array = []
        for name in tqdm(list(library.keys())):
            dump = library[name]
            flat = []
            for _, p in dump.expert_weights.items():
                flat.append(p.flatten().cpu())
            array.append(torch.concatenate(flat, 0).numpy())
            names.append(name)

        array = np.array(array)
        for i, _ in enumerate(array):
            if self.config.sparsity_threshold > 0.0:
                for thr in [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
                    ar_copy = array[i].copy()
                    ar_copy[np.abs(ar_copy) <= thr] = 0.0
                    ratio = float((ar_copy == 0.0).sum()) / ar_copy.size

                    if ratio >= self.config.sparsity_threshold:
                        logger.info("Found sparsity threshold: {}".format(thr))
                        break
                array[i] = ar_copy

        experts_embeddings = svd.fit_transform(array)
        experts_embeddings = (
            experts_embeddings / np.linalg.norm(experts_embeddings, 2, axis=1)[:, None]
        )

        if upload_to_hf:
            # add embeddings to the library
            with library.batched_commit():
                for i, name in enumerate(names):
                    library.add_embeddings(
                        name,
                        self.config.__dict__,
                        experts_embeddings[i],
                    )
        return experts_embeddings


class WeightedExpert(LibraryTransform):
    """
    Computes a uniform weight mixture across experts of a given library
    """

    def __init__(self, config, weights: Optional[dict] = None):
        super().__init__(config)

        if config.hf_lib_id is None:
            raise ValueError("AverageExpert requires a library id!")

        self.weights = weights
        self.library = HFExpertLibrary(config.hf_lib_id)
        self._expert = None

    @torch.no_grad()
    def compute(self, return_expert=False):
        if self._expert is None:
            expert_names = list(self.library.keys())
            experts = [self.library[name] for name in expert_names]

            logger.info("Averaging {} experts".format(len(experts)))

            if self.weights is not None:
                assert set(self.weights.keys()) == set(
                    expert_names
                ), "Weights must have the same keys as the experts"
                if sum(self.weights.values()) != 1.0:
                    logger.warn(
                        "Weights do not sum to 1.0, please make sure this is intended"
                    )

            base_expert = copy.deepcopy(experts[0])
            base_expert.name = "weighted_expert"

            for expert_name, expert in zip(expert_names[1:], experts[1:]):
                # Validate that the expert is compatible

                assert type(expert.expert_info.expert_config) == type(
                    base_expert.expert_info.expert_config
                ), "Expert configs must be the same type"
                assert set(expert.expert_weights.keys()) == set(
                    base_expert.expert_weights.keys()
                ), "Expert weights must have the same keys"

                for k, v in expert.expert_weights.items():
                    base_expert.expert_weights[k] += v

            # Normalize the final expert
            for k, v in base_expert.expert_weights.items():
                base_expert.expert_weights[k] /= len(experts)

            self._expert = base_expert

        if return_expert:
            return self._expert
