from dataclasses import dataclass
from mttl.models.modifiers.expert_containers.expert_library import HFExpertLibrary
from mttl.models.modifiers.expert_containers.module_graph import Expert
from mttl.utils import logger
from mttl.models.modifiers.modify_model import get_modifier_type

from datasets import Dataset
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
        if type(library) == str:
            library = HFExpertLibrary(library)

        experts_weights = []
        experts_names = list(library.keys())

        logger.info(f"Factorizing library with {len(experts_names)} experts.")

        def get_weights(example):
            expert = library[example["name"]]
            model_modifier = get_modifier_type(expert.expert_config)

            flattened = []
            if model_modifier == "lora":
                for _, p in expert.expert_weights.items():
                    flattened = flattened + list(p.flatten().cpu().numpy())
                return {"weights": flattened}
            else:
                return {"weights": None}

        dataset = Dataset.from_list([{"name": n} for n in experts_names])
        dataset = dataset.map(get_weights, num_proc=16)
        experts_weights = np.asarray([d["weights"] for d in dataset])

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
            library.add_embeddings(
                "svd", experts_names, experts_embeddings, config=self.config
            )
        return experts_embeddings
