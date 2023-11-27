from dataclasses import dataclass
from mttl.utils import logger
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
    n_components: int = 64
    sparsity_threshold: float = 0.8


class SVDEmbeddingTransform(LibraryTransform):
    """Creates adapter embeddings by low-rank decomposition of a sparsified version
    of the adapter modules.
    """

    def transform(self, library, upload_to_hf=True):
        experts_weights = []
        experts_names = list(library.keys())

        logger.info(f"Factorizing library with {len(experts_names)} experts.")

        bar = tqdm(experts_names)
        for key in bar:
            model = library[key]
            flattened = []

            for _, p in model.expert_weights.items():
                flattened = flattened + list(p.flatten().cpu().numpy())

            experts_weights.append(flattened)
            bar.set_description("Processed %s" % key)

        experts_weights = np.asarray(experts_weights)

        svd = sklearn.decomposition.TruncatedSVD(
            n_components=self.config.n_components,
            algorithm="randomized",
            n_iter=5,
            n_oversamples=10,
            power_iteration_normalizer="auto",
            random_state=None,
            tol=0.0,
        )

        if self.config.sparsity_threshold > 0.0:
            for thr in [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
                ew_copy = experts_weights.copy()
                ew_copy[np.abs(ew_copy) <= thr] = 0.0
                ratio = float(np.count_nonzero(ew_copy)) / ew_copy.size
                if ratio >= self.config.sparsity_threshold:
                    break

        experts_embeddings = svd.fit_transform(ew_copy)
        experts_embeddings = (
            experts_embeddings / np.linalg.norm(experts_embeddings, 2, axis=1)[:, None]
        )

        if upload_to_hf:
            # add embeddings to the library
            with library.batched_commit():
                for name, embedding in zip(experts_names, experts_embeddings):
                    library.add_embedding("svd", name, embedding, config=self.config)
        return experts_embeddings
