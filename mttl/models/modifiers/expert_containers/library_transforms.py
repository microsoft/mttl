from dataclasses import dataclass
from typing import Any
from mttl.models.modifiers.expert_containers.expert_library import HFExpertLibrary
from mttl.models.modifiers.expert_containers.module_graph import Expert
from mttl.utils import logger
from mttl.models.modifiers.modify_model import get_modifier_type
from mttl.models.utils import model_loader_helper
from typing import Optional
from mttl.models.utils import EfficientCheckpointModule, transfer_batch_to_device
from mttl.models.modifiers.lora import LoRAConfig

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

    def __init__(self, config, random_state=None):
        super().__init__(config)
        self.random_state = random_state

    def transform(self, library, upload_to_hf=True, force=False):
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
            random_state=self.random_state,
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
    name: str = "dataset_centroids_2"
    recompute: bool = False
    track: str = "each_layer"  # last layer, or each layer
    pool: str = "last"  # last, or mean
    compute_delta: str = False


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
        model.container = {}

        if self.config.track == "last_layer":
            # Add a hook to the last layer
            def fetch_input(module, input, output):
                model.container["last_layer"] = input[0].detach()

            model.model.get_output_embeddings().register_forward_hook(fetch_input)
        elif self.config.track == "each_layer":
            # add a hook for all the layers that an expert modifies
            def build_hook(name):
                def retrieve_input(module, input, output):
                    model.container[name] = input[0].detach()

                return retrieve_input

            for key in keys:
                module = self._get_parent_from_name(model.model, key)
                module.register_forward_hook(build_hook(key))
        else:
            raise NotImplementedError()

    def _retrieve_hidden_states(self, model):
        keys = list(model.container.keys())
        values = [model.container[k] for k in keys]
        for key in keys:
            del model.container[key]

        return {k: v for k, v in zip(keys, values)}

    @torch.no_grad()
    def transform(self, library, default_args=None) -> Expert:
        # TODO: remove project import
        from projects.wiki_experts.train_experts_main import get_datamodule
        from projects.wiki_experts.src.expert_model import MultiExpertModel

        if type(library) == str:
            library = HFExpertLibrary(library)

        args_in_name = [
            "name",
            "model",
            "use_base_model_only",
            "max_samples_per_task",
            "track",
            "pool",
        ]
        save_name = "-".join(
            [
                "" if x is None else str(x)
                for x in [getattr(self.config, k) for k in args_in_name]
            ]
        )

        if self.config.compute_delta:
            save_name += "-delta"

        print("save_name", save_name)

        # try to fetch auxiliary data
        output = library.get_auxiliary_data(data_type=save_name)
        if len(output) == len(library) and not self.config.recompute:
            logger.info("Found {} precomputed centroids".format(len(output)))
            # format the output to be dict[expert_name][layer_name] = embedding
            output = {
                expert_name: {
                    k: v for k, v in expert_data[self.config.name][save_name].items()
                }
                for expert_name, expert_data in output.items()
            }

            return output

        logger.info("Computing centroids for {} experts".format(len(library)))
        output = {}

        for e_id, (expert_name, expert) in enumerate(library.items()):
            training_config = expert.training_config
            if default_args is not None:
                self._fill_missing_args(training_config, default_args)

            if self.config.use_base_model_only and self.config.model is not None:
                training_config.model = self.config.model

            if self.config.compute_delta:
                model = MultiExpertModel(**vars(training_config)).to("cuda")
                expert_model = MultiExpertModel(**vars(training_config)).to("cuda")
                expert_model.add_expert_instance(expert, is_default=True)
            else:
                model = MultiExpertModel(**vars(training_config)).to("cuda")

            if not self.config.use_base_model_only:
                model.add_expert_instance(expert, is_default=True)

            self._track_hidden_states(model, keys=expert.expert_weights.keys())
            if self.config.compute_delta:
                self._track_hidden_states(
                    expert_model, keys=expert.expert_weights.keys()
                )

            training_config.dataset = expert.expert_info.dataset
            training_config.subsample_train = self.config.max_samples_per_task
            if expert.expert_info.expert_task_name:
                train_tasks = expert.expert_info.expert_task_name.split(",")
                training_config.finetune_task_name = ",".join(train_tasks)
                training_config.subsample_train *= len(train_tasks)
            else:
                train_tasks = None

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
                    if self.config.compute_delta:
                        expert_model.forward(batch, reduction="none")
                else:
                    model.forward(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    )
                    if self.config.compute_delta:
                        expert_model.forward(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                        )

                bs = batch["input_ids"].size(0)
                bs_idx = torch.arange(bs, device=device)
                last_token_idx = batch["attention_mask"].sum(1) - 1
                hidden_states = self._retrieve_hidden_states(model)

                if self.config.compute_delta:
                    expert_hidden_states = self._retrieve_hidden_states(expert_model)
                    hidden_states = {
                        k: expert_hidden_states[k] - hidden_states[k]
                        for k in hidden_states.keys()
                    }

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

            # convert to regular dict
            centroids = {k: v for k, v in centroid.items()}

            if self.config.upload_to_hf:
                logger.info("Uploading centroids to HF")
                # add embeddings to the library
                with library.batched_commit():
                    library.add_embedding_dict(
                        dump_name=save_name,
                        expert_name=expert_name,
                        data=centroids,
                        force=True,  # make sure we overwrite
                    )

            return output


@dataclass
class SVDInputExtractorConfig:
    name: str = "svd_input_extractor"
    top_k: float = 1.0
    upload_to_hf: bool = True
    recompute: bool = False
    ab_only: bool = True
    scale: bool = False  # If True, scale by eigenvalue


class SVDInputExtractor(LibraryTransform):
    """
    Given a library of experts, extract the input direction most affected by the linear transforms
    """

    def __init__(self, config: SVDInputExtractorConfig = None):
        super().__init__(config or SVDInputExtractorConfig())

    def _maybe_scale(self, vectors, eigvals):
        """Post Processing of the retrieved outputs,
        scales the output by the eigenvalue if needed"""
        output = {}
        for expert_name, expert_data in vectors.items():
            output[expert_name] = {}
            for layer_name, vector in expert_data.items():
                if self.config.scale:
                    vector = vector * eigvals[expert_name][layer_name]
                output[expert_name][layer_name] = vector

        return output

    def _extract_from_hf_file(self, adidct, save_name):
        return {
            expert_name: {
                k: v for k, v in expert_data[self.config.name][save_name].items()
            }
            for expert_name, expert_data in adidct.items()
        }

    @torch.no_grad()
    def transform(self, library) -> Expert:
        if isinstance(library, str):
            library = HFExpertLibrary(library)

        args_in_name = [
            "name",
            "top_k",
            "ab_only",
        ]
        save_name = "-".join(
            [
                "" if x is None else str(x)
                for x in [getattr(self.config, k) for k in args_in_name]
            ]
        )

        # try to fetch auxiliary data
        vectors = library.get_auxiliary_data(data_type=save_name + "_vectors")
        eigvals = library.get_auxiliary_data(data_type=save_name + "_eigvals")

        if len(vectors) == len(eigvals) == len(library) and not self.config.recompute:
            logger.info("Found {} precomputed centroids".format(len(vectors)))

            # format the output to be dict[expert_name][layer_name] = embedding
            vectors = self._extract_from_hf_file(vectors, save_name + "_vectors")
            eigvals = self._extract_from_hf_file(eigvals, save_name + "_eigvals")
            return self._maybe_scale(vectors, eigvals)

        base_model = None
        vectors, eigvals = {}, {}
        for expert_name, expert in library.items():
            logger.info(f"Computing SVD for expert {expert_name}")
            vectors[expert_name] = {}
            eigvals[expert_name] = {}

            if not self.config.ab_only and base_model is None:
                training_config = expert.training_config
                training_config.model_modifier = None
                from projects.wiki_experts.src.expert_model import MultiExpertModel

                base_model = MultiExpertModel(**vars(training_config))

            state_dict_keys = sorted(
                list(
                    set(
                        ".".join(k.split(".")[:-1])
                        for k in expert.expert_weights.keys()
                    )
                )
            )

            for param_name in state_dict_keys:
                logger.info(f"\tComputing SVD for parameter {param_name}")
                A, B = (
                    expert.expert_weights[f"{param_name}.lora_a"],
                    expert.expert_weights[f"{param_name}.lora_b"],
                )
                W = (A @ B).T  # out_features, in_features

                if self.config.top_k < 1.0:
                    th = torch.quantile(torch.abs(W), 1.0 - self.config.top_k)
                    W = torch.where(torch.abs(W) > th, W, torch.zeros_like(W))

                if self.config.ab_only:
                    eig_input = W.T @ W  # in_features, in_features
                else:
                    base_W = base_model.model.state_dict()[f"{param_name}.weight"]
                    W_AB = base_W + W
                    eig_input = W_AB.T @ W_AB

                out = torch.linalg.eig(eig_input)
                eigvector = out.eigenvectors
                img_eig_vals = eigvector.imag.abs().mean().item()
                if img_eig_vals > 1e-2:
                    logger.warning(
                        f"Found {img_eig_vals} imaginary eigenvalues, this is likely due to numerical instability"
                    )

                largest, smallest = eigvector[:, 0], eigvector[:, -1]
                assert torch.norm(eig_input @ largest.real.unsqueeze(-1)) >= torch.norm(
                    eig_input @ smallest.real.unsqueeze(-1)
                ), breakpoint()

                empirical_eigvalue = (eig_input @ largest.real.unsqueeze(-1)).squeeze(
                    -1
                ) / largest.real
                max_, min_ = (
                    empirical_eigvalue.max(),
                    empirical_eigvalue.min(),
                )
                ratio_diff = (max_ - min_).abs().mean().item()
                if ratio_diff > 1e-1:
                    logger.warning(
                        f"Found {ratio_diff} max variation of A * v_eig / v, this is likely due to numerical instability"
                    )

                # Save eigenvector and eigvenvalue
                vectors[expert_name][param_name] = largest.real.cpu().numpy()
                eigvals[expert_name][param_name] = out.eigenvalues.real[0].item()

            # Upload to HF 1 expert at a time
            if self.config.upload_to_hf:
                logger.info(f"Uploading SVD centroids to HF for expert {expert_name}")
                # add embeddings to the library
                with library.batched_commit():
                    for name, data in [
                        ("vectors", vectors),
                        ("eigvals", eigvals),
                    ]:
                        library.add_embedding_dict(
                            dump_name=save_name + "_" + name,
                            expert_name=expert_name,
                            data=data[expert_name],
                            force=True,  # make sure we overwrite
                        )

        return self._maybe_scale(vectors, eigvals)


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
