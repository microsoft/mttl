from dataclasses import dataclass
from mttl.models.modifiers.expert_containers.expert_library import HFExpertLibrary
from mttl.models.modifiers.expert_containers.module_graph import Expert
from mttl.utils import logger
from mttl.models.modifiers.modify_model import get_modifier_type

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
