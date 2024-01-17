from dataclasses import dataclass
from mttl.models.modifiers.expert_containers.expert_library import HFExpertLibrary
from mttl.models.modifiers.expert_containers.module_graph import Expert
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
from collections import defaultdict


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
    recompute: bool = False


class SVDEmbeddingTransform(LibraryTransform):
    """Creates adapter embeddings by low-rank decomposition of a sparsified version
    of the adapter modules.
    """

    def transform(self, library, upload_to_hf=True):
        if type(library) == str:
            library = HFExpertLibrary(library)

        # try to fetch auxiliary data
        output = library.get_auxiliary_data(data_type=self.config.name)
        if len(output) > 0 and not self.config.recompute:
            logger.info("Found {} precomputed SVD Embeddings".format(len(output)))
            return (
                np.stack(
                    [
                        output[expert_name][self.config.name][self.config.name]
                        for expert_name in library.keys()
                    ]
                ),
                None,
            )

        logger.info("Computing SVD Embeddings for {} experts".format(len(library)))
        svd = sklearn.decomposition.TruncatedSVD(
            n_components=self.config.n_components,
            algorithm="randomized",
            n_iter=5,
            n_oversamples=10,
            power_iteration_normalizer="auto",
            random_state=None,
            tol=0.0,
        )

        array, names = [], []
        for name in tqdm(list(library.keys())):
            expert = library[name]
            array += [
                torch.nn.utils.parameters_to_vector(
                    [p for p in expert.expert_weights.values()]
                )
            ]
            names += [name]
        array = torch.stack(array).cpu().numpy()

        # Use quantiles to fit the exact threshold
        thr = np.quantile(np.abs(array), self.config.sparsity_threshold, axis=1)
        array[np.abs(array) <= thr.reshape(-1, 1)] = 0.0
        logger.info("Sparsity threshold: {}".format(str([f"{x:.4f}" for x in thr])))
        assert (
            np.abs(
                (array == 0).sum(axis=1) / np.prod(array.shape[1])
                - self.config.sparsity_threshold
            ).max()
            < 1e-4
        )

        experts_embeddings = svd.fit_transform(array)
        experts_embeddings = (
            experts_embeddings / np.linalg.norm(experts_embeddings, 2, axis=1)[:, None]
        )

        if upload_to_hf:
            logger.info("Uploading SVD Embeddings to HF")
            # add embeddings to the library
            with library.batched_commit():
                for i, name in enumerate(names):
                    library.add_auxiliary_data(
                        data_type=self.config.name,
                        expert_name=name,
                        config=self.config.__dict__,
                        data=experts_embeddings[i],
                        force=True,  # make sure we overwrite
                    )
        return experts_embeddings, svd


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

        base_expert = copy.deepcopy(experts[0])
        base_expert.name = "weighted_expert"

        if self.config.weights is not None:
            assert set(self.config.weights.keys()) == set(
                expert_names
            ), "Weights must have the same keys as the experts"
            if not (1 - 1e-6) <= sum(self.config.weights.values()) <= (1 + 1e-6):
                logger.warn(
                    "Weights do not sum to 1.0, please make sure this is intended"
                )

            # scale the base expert
            for k, v in base_expert.expert_weights.items():
                base_expert.expert_weights[k] *= self.config.weights[expert_names[0]]

        for _, expert in zip(expert_names[1:], experts[1:]):
            # Validate that the expert is compatible
            assert type(expert.expert_info.expert_config) == type(
                base_expert.expert_info.expert_config
            ), "Expert configs must be the same type"
            assert set(expert.expert_weights.keys()) == set(
                base_expert.expert_weights.keys()
            ), "Expert weights must have the same keys"

            weight = 1.0
            if self.config.weights is not None:
                weight = self.config.weights[expert.expert_info.expert_name]

            for k, v in expert.expert_weights.items():
                base_expert.expert_weights[k] += v * weight

        # Normalize the final expert
        if self.config.weights is None:
            for k, v in base_expert.expert_weights.items():
                base_expert.expert_weights[k] /= len(experts)

        return base_expert


@dataclass
class TiesMergeConfig:
    top_k: float = 0.2
    only_sparsify: bool = False


class TiesMerge(LibraryTransform):
    """
    Computes a uniform weight mixture across experts of a given library
    """

    def __init__(self, config: TiesMergeConfig = None):
        super().__init__(config or TiesMergeConfig())

        assert self.config.top_k > 0.0 and self.config.top_k <= 1.0

    @torch.no_grad()
    def transform(self, library) -> Expert:
        if type(library) == str:
            library = HFExpertLibrary(library)

        expert_names = list(library.keys())
        experts = [library[name] for name in expert_names]

        logger.info("Averaging {} experts".format(len(experts)))

        base_expert = copy.deepcopy(experts[0])
        base_expert.name = "ties_weighted_expert"

        state_dict_keys = list(base_expert.expert_weights.keys())

        # Build n_tasks x D experts
        # TODO: No need to build this matrix, can be done 1 expert at a time
        expert_vectors = []
        for expert in experts:
            expert_vectors += [
                torch.nn.utils.parameters_to_vector(
                    list(expert.expert_weights[k] for k in state_dict_keys)
                )
            ]

        expert_vectors = torch.stack(expert_vectors, dim=0)
        per_exp_th = expert_vectors.abs().quantile(1.0 - self.config.top_k, dim=1)
        keep_param = expert_vectors.abs() >= per_exp_th.view(-1, 1)

        mean_valid_per_task = keep_param.float().mean(1)
        assert torch.all((mean_valid_per_task - self.config.top_k).abs() < 1e-4)

        used, kept, total = 0, 0, 0

        for param_name in state_dict_keys:
            # stack the expert weights
            expert_weights = torch.stack(
                [expert.expert_weights[param_name] for expert in experts], dim=0
            )

            # keep weights over the threshold
            TH = per_exp_th.view(-1, *((1,) * (expert_weights.ndim - 1)))
            keep_mask = expert_weights.abs() >= TH
            expert_weights = expert_weights * keep_mask

            if self.config.only_sparsify:
                final_param = expert_weights.mean(0)
                used += keep_mask.sum().item()
            else:
                # sign majority vote
                sign_per_dim = expert_weights.sign().sum(0, keepdim=True).sign()
                sign_per_dim = expert_weights.sum(0, keepdim=True).sign()

                # keep only weights whose sign agree with the majority
                use_for_avg = expert_weights.sign() == sign_per_dim

                deno = use_for_avg.sum(0).clamp(min=1.0)
                sum_param = (expert_weights * use_for_avg).sum(0)
                final_param = sum_param / deno
                used += (use_for_avg & (sign_per_dim != 0.0)).sum().item()

            kept += (expert_weights.abs() > TH).sum()
            total += expert_weights.numel()

            base_expert.expert_weights[param_name].data.copy_(final_param)

        logger.info(
            "Params not reset to 0 in TIES merge: {:.10f}%".format(100.0 * kept / total)
        )
        logger.info(
            "Params used to compute TIES mean: {:.10f}%".format(100.0 * used / total)
        )

        return base_expert


@dataclass
class HiddenStateComputerConfig:
    use_base_model_only: bool = (
        False  # This computes sentence embeddings without the adapter
    )
    model: str = None  # If `use_base_model_only`, can pass a specific model to compute embeddings with
    max_samples_per_task: int = 10
    upload_to_hf: bool = False
    name: str = "dataset_centroids"
    recompute: bool = False
    track: str = "each_layer"  # last layer, or each layer
    pool: str = "last"  # last, or mean


class HiddenStateComputer(LibraryTransform):
    """
    Encodes a dataset and computes the average embedding
    """

    def __init__(self, config: HiddenStateComputerConfig = None):
        super().__init__(config or HiddenStateComputerConfig())

    def _fill_missing_args(self, args, default_args):
        # TODO: put in library utils
        for k, v in vars(default_args).items():
            if not hasattr(args, k):
                setattr(args, k, v)

    def _get_parent_from_name(self, model, name):
        parts = name.split(".")
        for part in parts:
            if part.isdigit():
                new_model = model[int(part)]
            else:
                new_model = getattr(model, part, None)

            if new_model is None:
                return model

            model = new_model

        return model

    def _track_hidden_states(self, model, keys=None):
        self.container = {}

        if self.config.track == "last_layer":
            # Add a hook to the last layer
            def fetch_input(module, input, output):
                self.container["last_layer"] = input[0].detach()

            model.model.get_output_embeddings().register_forward_hook(fetch_input)
        elif self.config.track == "each_layer":
            # add a hook for all the layers that an expert modifies
            def build_hook(name):
                def retrieve_input(module, input, output):
                    self.container[name] = input[0].detach()

                return retrieve_input

            for key in keys:
                module = self._get_parent_from_name(model.model, key)
                module.register_forward_hook(build_hook(key))
        else:
            raise NotImplementedError()

    def _retrieve_hidden_states(self):
        keys = list(self.container.keys())
        values = [self.container[k] for k in keys]
        for key in keys:
            del self.container[key]

        return {k: v for k, v in zip(keys, values)}

    @torch.no_grad()
    def transform(self, library, default_args=None) -> Expert:
        # TODO: remove project import
        from projects.wiki_experts.train_experts_main import get_datamodule
        from projects.wiki_experts.src.expert_model import MultiExpertModel

        if type(library) == str:
            library = HFExpertLibrary(library)

        # try to fetch auxiliary data
        output = library.get_auxiliary_data(data_type=self.config.name)
        if len(output) == len(library) and not self.config.recompute:
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

            self._track_hidden_states(model, keys=expert.expert_weights.keys())

            training_config.dataset = expert.expert_info.dataset
            train_tasks = expert.expert_info.expert_task_name.split(",")
            training_config.subsample_train = self.config.max_samples_per_task * len(
                train_tasks
            )
            training_config.finetune_task_name = ",".join(train_tasks)
            training_config.train_batch_size = (
                default_args.predict_batch_size if default_args is not None else 4
            )

            # get datamodule
            dm = get_datamodule(training_config)
            dataloader = dm.train_dataloader()

            centroid, count = defaultdict(lambda: 0.0), 0

            pbar = tqdm(enumerate(dataloader), total=len(dataloader))
            device = next(model.parameters()).device

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
                bs_idx = torch.arange(bs, device=device)
                last_token_idx = batch["attention_mask"].sum(1) - 1
                hidden_states = self._retrieve_hidden_states()

                for layer, hidden_state in hidden_states.items():
                    assert hidden_state.ndim == 3

                    if self.config.pool == "last":
                        centroid[layer] += hidden_state[bs_idx, last_token_idx].sum(0)
                    elif self.config.pool == "mean":
                        deno = batch["attention_mask"].sum(1, keepdim=True)
                        centroid[layer] += (
                            (hidden_state * batch["attention_mask"].unsqueeze(-1)).sum(
                                1
                            )
                            / deno
                        ).sum(0)
                    else:
                        raise NotImplementedError()

                count += bs

            # average over all batches
            for layer in centroid.keys():
                centroid[layer] /= count
                centroid[layer] = F.normalize(centroid[layer], p=2, dim=-1).cpu()

            # can't pickle a defaultdict
            output[expert_name] = {k: v for k, v in centroid.items()}

        if self.config.upload_to_hf:
            logger.info("Uploading centroids to HF")
            # add embeddings to the library
            with library.batched_commit():
                for i, expert_name in enumerate(output.keys()):
                    library.add_auxiliary_data(
                        data_type=self.config.name,
                        expert_name=expert_name,
                        config=self.config.__dict__,
                        data=output[expert_name],
                        force=True,  # make sure we overwrite
                    )
        return output


@dataclass
class ExpertProjectorConfig:
    name: str = "expert_projector"
    granularity: str = "finegrained"  # whether to use the same coefficients for all parameters or per `nn.Parameter` instance
    project_over_all_experts: bool = (
        False  # whether to project over all experts or just the ones in the cluster
    )


class ExpertProjector(LibraryTransform):
    """
    Given a library of clustered experts, project each one onto the basis generated
    by the individual experts of each cluster.
    """

    def __init__(self, config: ExpertProjectorConfig = None):
        super().__init__(config or ExpertProjectorConfig())

    def _project(self, source_expert, expert_basis, granularity="coarsegrained"):
        source_sd = source_expert.expert_weights
        state_dict_keys = list(source_sd.keys())

        assert set(state_dict_keys) == set(
            expert_basis[0].expert_weights.keys()
        ), breakpoint()

        if granularity == "coarsegrained":
            # build a n_experts x D matrix of concatenated parameters
            basis_vectors = []
            for expert in expert_basis:
                basis_vectors += [
                    torch.nn.utils.parameters_to_vector(
                        list(expert.expert_weights[k] for k in state_dict_keys)
                    )
                ]
            basis_vector = torch.stack(basis_vectors)
            project_vector = torch.nn.utils.parameters_to_vector(
                list(source_sd[k] for k in state_dict_keys)
            )

            # Treat as a min-squares problem
            global_alpha = torch.linalg.lstsq(
                basis_vector.T, project_vector.view(-1, 1)
            ).solution
        else:
            assert granularity == "finegrained"

        projected_expert = copy.deepcopy(source_expert)
        for key in state_dict_keys:
            basis_vector = torch.stack(
                [expert.expert_weights[key].flatten() for expert in expert_basis]
            )

            if granularity == "coarsegrained":
                alpha = global_alpha
            else:
                alpha = torch.linalg.lstsq(
                    basis_vector.T, source_sd[key].view(-1, 1)
                ).solution

            # project the source expert onto the basis
            projected = (basis_vector.T @ alpha).view(source_sd[key].shape)
            projected_expert.expert_weights[key].data.copy_(projected)

        return projected_expert

    @torch.no_grad()
    def transform(self, expert_library, cluster_library) -> Expert:
        if isinstance(expert_library, str):
            expert_library = HFExpertLibrary(expert_library)

        if isinstance(cluster_library, str):
            cluster_library = HFExpertLibrary(cluster_library)

        output = {}
        for cluster_name, cluster_exp in cluster_library.items():
            logger.info(f"processing cluster {cluster_name}")
            if self.config.project_over_all_experts:
                task_experts = [
                    expert_library[expert_name] for expert_name in expert_library.keys()
                ]
            else:
                tasks = cluster_exp.expert_info.expert_task_name.split(",")
                task_experts = [expert_library[expert_name] for expert_name in tasks]
            projected_expert = self._project(
                cluster_exp, task_experts, granularity=self.config.granularity
            )
            output[cluster_name] = projected_expert

        return output


@dataclass
class ClusteringConfig:
    n_clusters: int = 10


class LibraryClusterer(LibraryTransform):
    """
    Given a library of experts, cluster them and return the clusters
    """

    def __init__(self, config: ClusteringConfig = None):
        super().__init__(config or ClusteringConfig())

    def transform(self, library) -> Expert:
        if isinstance(library, str):
            library = HFExpertLibrary(library)
