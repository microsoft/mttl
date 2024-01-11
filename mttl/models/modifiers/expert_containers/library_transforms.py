from dataclasses import dataclass
from mttl.models.modifiers.expert_containers.expert_library import HFExpertLibrary
from mttl.models.modifiers.expert_containers.module_graph import Expert
from projects.wiki_experts.src.expert_model import MultiExpertModel
from mttl.utils import logger
from mttl.models.modifiers.modify_model import get_modifier_type
from mttl.models.utils import model_loader_helper
from typing import Optional
from mttl.models.utils import EfficientCheckpointModule, transfer_batch_to_device

import copy
import torch
import torch.nn.functional as F
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


@dataclass
class WeightedLinearMergeConfig:
    weights: dict = None


class WeightedLinearMerge(LibraryTransform):
    """
    Computes a uniform weight mixture across experts of a given library
    """

    def __init__(self, config: WeightedLinearMergeConfig = None):
        super().__init__(config or WeightedLinearMergeConfig())

    @torch.no_grad()
    def transform(self, library) -> Expert:
        if type(library) == str:
            library = HFExpertLibrary(library)

        expert_names = list(library.keys())
        experts = [library[name] for name in expert_names]

        logger.info("Averaging {} experts".format(len(experts)))

        if self.config.weights is not None:
            assert set(self.config.weights.keys()) == set(
                expert_names
            ), "Weights must have the same keys as the experts"
            if sum(self.config.weights.values()) != 1.0:
                logger.warn(
                    "Weights do not sum to 1.0, please make sure this is intended"
                )

        base_expert = copy.deepcopy(experts[0])
        base_expert.name = "weighted_expert"

        for _, expert in zip(expert_names[1:], experts[1:]):
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

        return base_expert


@dataclass
class PrototypeComputerConfig:
    use_base_model_only: bool = (
        False  # This computes sentence embeddings without the adapter
    )
    model: str = None  # If `use_base_model_only`, can pass a specific model to compute embeddings with
    max_samples_per_task: int = 25
    upload_to_hf: bool = False
    name: str = "dataset_centroids"


class DatasetCentroidComputer(LibraryTransform):
    """
    Encodes a dataset and computes the average embedding
    """

    def __init__(self, config: PrototypeComputerConfig = None):
        super().__init__(config or PrototypeComputerConfig())

    def _fill_missing_args(self, args, default_args):
        # TODO: put in library utils
        for k, v in vars(default_args).items():
            if not hasattr(args, k):
                setattr(args, k, v)

    @torch.no_grad()
    def transform(self, library, upload_to_hf=False, default_args=None) -> Expert:
        # TODO: remove project import
        from projects.wiki_experts.train_experts_main import get_datamodule

        if type(library) == str:
            library = HFExpertLibrary(library)

        # try to fetch auxiliary data
        output = library.get_auxiliary_data(data_type=self.config.name)
        if len(output) == len(library):
            logger.info("Found {} precomputed centroids".format(len(output)))
            return output

        logger.info("Computing centroids for {} experts".format(len(library)))
        output = {}

        for e_id, (expert_name, expert) in enumerate(library.items()):
            training_config = expert.training_config
            if default_args is not None:
                self._fill_missing_args(training_config, default_args)

            if self.config.use_base_model_only and self.config.model_name is not None:
                training_config.model = self.config.model

            model = MultiExpertModel(**vars(training_config)).to("cuda")

            if not self.config.use_base_model_only:
                model.add_expert_instance(expert, is_default=True)

            train_tasks = expert.expert_info.expert_task_name.split(",")

            centroid, count = 0, 0
            for t_id, train_task in enumerate(train_tasks):
                # get datamodule
                training_config.subsample_train = self.config.max_samples_per_task
                training_config.dataset = expert.expert_info.dataset
                training_config.finetune_task_name = train_task
                training_config.train_batch_size = (
                    default_args.predict_batch_size if default_args is not None else 4
                )

                dm = get_datamodule(training_config)
                dataloader = dm.train_dataloader()

                pbar = tqdm(enumerate(dataloader), total=len(dataloader))
                device = next(model.parameters()).device

                container = {}

                def fetch_pre_logit_output(module, input, output):
                    container["hidden_states"] = input[0].detach()

                model.model.get_output_embeddings().register_forward_hook(
                    fetch_pre_logit_output
                )

                for num_batch, batch in pbar:
                    batch = transfer_batch_to_device(batch, device)

                    if isinstance(model, EfficientCheckpointModule):
                        model.forward(batch, reduction="none")
                    else:
                        model.forward(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                        )

                    bs = batch["input_ids"].size(0)
                    hidden_states = container["hidden_states"]
                    last_token_state = hidden_states[
                        torch.arange(bs, device=device),
                        batch["attention_mask"].sum(1) - 1,
                    ]
                    centroid += last_token_state.sum(0)
                    count += bs

            # average over all batches
            centroid /= count
            output[expert_name] = F.normalize(centroid, p=2, dim=-1).cpu()

        if upload_to_hf:
            logger.info("Uploading centroids to HF")
            # add embeddings to the library
            with library.batched_commit():
                for i, name in enumerate(output.keys()):
                    library.add_auxiliary_data(
                        data_type=self.config.name,
                        expert_name=expert_name,
                        config=self.config.__dict__,
                        data=output[name],
                        force=True,
                    )

        return output
