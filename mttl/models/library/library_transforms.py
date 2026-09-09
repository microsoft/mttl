import abc
import copy
import dataclasses
import os
import re
from abc import abstractmethod
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import sklearn.decomposition
import torch
import torch.nn.functional as F
from pytorch_lightning import Trainer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm

from mttl.datamodule.base import get_datamodule
from mttl.logging import logger
from mttl.models.containers.lora_containers import ExpertContainer
from mttl.models.containers.selectors.phatgoose_selector import (
    PhatgooseTrainerSelectorConfig,
)
from mttl.models.expert_model import MultiExpertModel, MultiExpertModelConfig
from mttl.models.get_optimizer import get_optimizer_and_scheduler
from mttl.models.library.expert import Expert
from mttl.models.library.expert_library import ExpertLibrary
from mttl.models.lightning.callbacks import LiveCheckpointCallback
from mttl.models.lightning.loggers import get_pl_loggers
from mttl.models.modifiers.base import get_target_2_source_param_mapping
from mttl.models.modifiers.lora import LoRAConfig
from mttl.models.monitors import get_monitors
from mttl.models.train_utils import train_model
from mttl.models.utils import transfer_batch_to_device
from mttl.registrable import Registrable
from mttl.serializable import Serializable
from mttl.models.library.wudi_closed_form import AnalyticalSolver
from mttl.logging import TableLogger
import wandb


class LibraryTransform(abc.ABC, Registrable):
    """Defines a transformation of a library of experts."""

    def __init__(self, config):
        self.config = config

    @abstractmethod
    def transform(
        self, library: ExpertLibrary, persist: bool = False, recompute: bool = False
    ):
        pass


def _safe_quantile(tensor, q, dim=None):
    """``torch.quantile`` rejects inputs with more than 2**24 elements."""
    try:
        return tensor.quantile(q, dim=dim)
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "too large" not in msg and "quantile" not in msg:
            raise
        x = tensor.detach().float().cpu().numpy()
        q_np = float(q.item() if torch.is_tensor(q) else q)
        val = np.quantile(x, q_np) if dim is None else np.quantile(x, q_np, axis=dim)
        return torch.as_tensor(val, device=tensor.device, dtype=tensor.dtype)


def _hash_field(val):
    # from facebookresearch / ReAgent
    if val is None:
        return ""
    elif isinstance(val, list):
        return tuple(val)
    elif isinstance(val, dict):
        return tuple(sorted(val.items()))
    else:
        return val


def param_hash(p, exclude_fields=None):
    # from facebookresearch / ReAgent
    import hashlib

    m = hashlib.md5()
    m.update(
        str(
            tuple(
                _hash_field(getattr(p, f.name))
                for f in dataclasses.fields(p)
                if not exclude_fields or f.name not in exclude_fields
            )
        ).encode()
    )
    return m.hexdigest()


@dataclass
class LibraryTransformConfig(Serializable):
    name: str = None

    @property
    def save_name(self):
        """
        Returns name of the cached data to use when persisting the library.
        If not set, it will be automatically generated.
        """
        if self.name:
            return self.name
        else:
            # form auto name based on the arguments of the config
            save_name = self.__class__.__name__.lower() + f"-{self.param_hash()}"
            return save_name

    def param_hash(self):
        return param_hash(self)


@dataclass
class SVDEmbeddingTransformConfig(LibraryTransformConfig):
    n_components: int = 64
    sparsity_threshold: float = 0.8


@LibraryTransform.register("svd_embedding", SVDEmbeddingTransformConfig)
class SVDEmbeddingTransform(LibraryTransform):
    """Creates adapter embeddings by low-rank decomposition of a sparsified version
    of the adapter experts.
    """

    def __init__(self, config, random_state=None):
        super().__init__(config)
        self.random_state = random_state

    @classmethod
    @torch.no_grad()
    def fetch(cls, library: Union[str, ExpertLibrary], config_hash: str = None):
        if isinstance(library, str):
            library = ExpertLibrary.get_expert_library(library)

        config_hash = config_hash or SVDEmbeddingTransformConfig().save_name

        # try to fetch auxiliary data
        output = library.get_auxiliary_data(data_type=config_hash)

        if len(output) == len(library):
            logger.info("Found {} precomputed SVD Embeddings".format(len(output)))
            return output

        raise ValueError(
            "SVD embeddings are missing or corrupted, please recompute them."
        )

    def transform(self, library, persist=True, recompute=False):
        if type(library) == str:
            library = ExpertLibrary.get_expert_library(library)

        try:
            output = self.fetch(library, self.config.save_name)

            if not recompute:
                logger.info("Found {} precomputed SVD Embeddings".format(len(output)))
                return output
        except ValueError:
            pass

        logger.info("Computing SVD Embeddings for %s experts", len(library))
        logger.info("Saving to: %s", self.config.save_name)

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

        if persist:
            logger.info("Uploading SVD embeddings to the library.")

            # add embeddings to the library
            with library.batched_commit():
                for i, name in enumerate(names):
                    library.add_auxiliary_data(
                        data_type=self.config.save_name,
                        expert_name=name,
                        config=self.config.__dict__,
                        data=experts_embeddings[i],
                        force=True,  # make sure we overwrite
                    )
        return dict(zip(names, experts_embeddings))


@dataclass
class TiesMergeAfterConfig(LibraryTransformConfig):
    mask_rate: float = 0.8


@LibraryTransform.register("ties_merge_after", TiesMergeAfterConfig)
class TiesMergeAfter(LibraryTransform):
    """
    implement the ties merge after in the paper
    """

    def __init__(self, config: TiesMergeAfterConfig = None):
        super().__init__(config or TiesMergeAfterConfig())

    def mask_smallest_magnitude_param_values(
        self, param_tensor: torch.Tensor, param_value_mask_rate: float = 0.8
    ):
        """
        Mask the smallest-magnitude parameter values (set to zeros) based on parameter value mask rate.
        :param param_tensor: Tensor, parameter tensor to mask
        :param param_value_mask_rate: float, mask rate of the smallest-magnitude parameter values
        :return:
        """
        # Convert to float32 to support kthvalue operation
        original_dtype = param_tensor.dtype
        param_tensor = param_tensor.float()

        # Calculate the number of parameters to mask
        num_mask_params = int(param_tensor.numel() * param_value_mask_rate)
        # Flatten the parameter for kthvalue calculation
        flattened = param_tensor.reshape(-1)

        # Calculate the threshold
        kth_value = flattened.abs().kthvalue(k=num_mask_params).values

        # Create mask and apply
        mask = param_tensor.abs() >= kth_value

        # Apply mask and convert back to original dtype
        return (param_tensor * mask).to(original_dtype)

    def get_param_signs(self, param_tensors: list):
        """
        get the signs for each parameter, computed over individual models
        :param param_tensors: list of Tensor, parameters from different models
        :return:
        """
        # Calculate the sum of parameter signs
        param_sum = sum(param_tensors)
        param_signs = torch.sign(param_sum)

        # Handle the case where sign is zero
        if (param_signs == 0).any():
            # Calculate majority sign
            majority_sign = torch.sign(param_signs.sum())
            param_signs[param_signs == 0] = majority_sign

        return param_signs

    def disjoint_merge(self, param_tensors: list, param_signs: torch.Tensor):
        """
        disjoint merge for a single parameter across models
        :param param_tensors: list of Tensor, parameters from different models
        :param param_signs: Tensor, the signs of parameters
        :return:
        """
        preserved_params = []
        for param in param_tensors:
            # Create mask to preserve elements with the same sign as param_signs
            preserve_mask = ((param_signs > 0) & (param > 0)) | (
                (param_signs < 0) & (param < 0)
            )
            preserved_params.append(param * preserve_mask)

        # Calculate how many models preserve each position
        num_preserved = sum([(p != 0).float() for p in preserved_params])

        # Calculate the mean, avoid division by zero
        merged_param = sum(preserved_params) / torch.clamp(num_preserved, min=1.0)

        return merged_param

    def _get_task_vectors(self, expert):
        """
        get the incremental weights for each layer, LoRA A outproduct LoRA B
        """
        task_vectors = {}
        for key in expert.expert_weights.keys():
            base_layer_name = key.split(".lora_")[
                0
            ]  # Get base layer name by removing .lora_a or .lora_b
            if base_layer_name not in task_vectors:
                task_vectors[base_layer_name] = None

        for layer in task_vectors.keys():
            lora_a = expert.expert_weights[f"{layer}.lora_a"]
            lora_b = expert.expert_weights[f"{layer}.lora_b"]
            task_vectors[layer] = lora_a.data @ lora_b.data

        return task_vectors

    def transform(self, library, persist=True, recompute=False):
        if type(library) == str:
            library = ExpertLibrary.get_expert_library(library)

        expert_names = list(library.keys())
        experts = [library[name] for name in expert_names]

        layer_names = [
            name.split(".lora_")[0] for name in experts[0].expert_weights.keys()
        ]
        layer_names = sorted(list(set(layer_names)))

        task_vectors_experts = {}
        for expert in experts:
            task_vectors = self._get_task_vectors(expert)
            task_vectors_experts[expert.name] = task_vectors

        task_merged_vectors = {}
        for layer in layer_names:
            task_vectors = [
                task_vectors_experts[expert.name][layer] for expert in experts
            ]

            logger.info(f"Layer {layer} has {len(task_vectors)} task vectors")
            ## apply mask to each parameter
            logger.info(f"Applying mask to layer {layer}")
            masked_task_vectors = [
                self.mask_smallest_magnitude_param_values(
                    task_vector, self.config.mask_rate
                )
                for task_vector in task_vectors
            ]
            ## calculate signs
            logger.info(f"Calculating signs for layer {layer}")
            param_signs = self.get_param_signs(masked_task_vectors)
            ## Apply disjoin merge strategy
            logger.info(f"Applying disjoint merge strategy for layer {layer}")
            merged_delta = self.disjoint_merge(masked_task_vectors, param_signs)
            ## Combine merged delta with original model parameter
            task_merged_vectors[layer] = merged_delta

        return task_merged_vectors

@dataclass
class CPMergeConfig(LibraryTransformConfig):
    strategy: str = "cp"  # or tucker
    cp_rank: int = 1
    tucker_rank: int = "1,1,1"


@LibraryTransform.register("cp_merge", CPMergeConfig)
class CPMerge(LibraryTransform):
    """
    Compute the common direction of all the experts in the library

    """

    def __init__(self, config: CPMergeConfig = None):
        super().__init__(config or CPMergeConfig())

    @torch.no_grad()
    def transform(self, library) -> Expert:
        import tensorly as tl

        tl.set_backend("pytorch")
        from tensorly.decomposition import parafac
        from tensorly.decomposition import tucker

        expert_names = list(library.keys())
        experts = [library[name] for name in expert_names]

        logger.info("Computing common direction for {} experts".format(len(experts)))

        base_expert = copy.deepcopy(experts[0])
        base_expert.name = "td_expert"  # tensor decomposition expert

        layer_names = base_expert.expert_weights.keys()

        for layer_name in tqdm(layer_names):
            concat_weights = []
            for expert_name in expert_names:
                # get the weights for each expert
                expert_weights = library.get_expert(expert_name).expert_weights[
                    layer_name
                ]
                concat_weights.append(expert_weights)

            try:
                # concat
                third_order_tensor = torch.stack(concat_weights).to(
                    "cuda"
                )  # [lib_num, input_dim, output_dim]
                # conduct the tensor decomposition

                if self.config.strategy == "cp":
                    logger.info(
                        f"Computing tensor decomposition for layer {layer_name} using cuda with rank {self.config.cp_rank}"
                    )
                    factors_cp = parafac(
                        third_order_tensor,
                        rank=self.config.cp_rank,
                        init="svd",
                        random_state=42,
                    )
                    cp_third_order_tensor = tl.cp_tensor.cp_to_tensor(factors_cp)
                    expert_mean = cp_third_order_tensor.mean(0)
                elif self.config.strategy == "tucker":
                    tucker_rank = list(self.config.tucker_rank)
                    logger.info(
                        f"Computing tensor decomposition for layer {layer_name} using cuda with rank {self.config.tucker_rank}"
                    )
                    core, factors = tucker(third_order_tensor, rank=tucker_rank)
                    tucker_third_order_tensor = tl.tucker_to_tensor((core, factors))
                    expert_mean = tucker_third_order_tensor.mean(
                        0
                    )  # mean over the experts

            except Exception as e:
                logger.info(
                    e,
                    f"Computing tensor decomposition for layer {layer_name} using cpu with rank {self.config.cp_rank}",
                )

                third_order_tensor = torch.stack(concat_weights).to("cpu")
                if self.config.strategy == "cp":
                    factors_cp = parafac(
                        third_order_tensor,
                        rank=self.config.cp_rank,
                        init="svd",
                        random_state=42,
                    )
                    cp_third_order_tensor = tl.cp_tensor.cp_to_tensor(factors_cp)
                    expert_mean = cp_third_order_tensor.mean(0)
                elif self.config.strategy == "tucker":
                    tucker_rank = list(self.config.tucker_rank)
                    core, factors = tucker(third_order_tensor, rank=tucker_rank)
                    cp_third_order_tensor = tl.tucker_to_tensor((core, factors))
                    expert_mean = cp_third_order_tensor.mean(0)

            base_expert.expert_weights[layer_name] = expert_mean
        return base_expert

@dataclass
class CPMergeAfterConfig(LibraryTransformConfig):
    cp_rank: int = 4
    path: str = "cp_merge_after_ingredients.pt"
    plot_similarity: bool = False
    plot_output_dir: str = "cp_merge_after_plots"
    plot_cp_sti: bool = False
    plot_cpd_convergence: bool = False
    cpd_n_iter_max: int = 100
    cpd_tol: float = 1e-8

@LibraryTransform.register("cp_merge_after", CPMergeAfterConfig)
class CPMergeAfter(LibraryTransform):
    def __init__(self, config: CPMergeAfterConfig = None):
        super().__init__(config or CPMergeAfterConfig())

    def _save_similarity_heatmap(
        self, sim_matrix: np.ndarray, labels: List[str], title: str, save_path: str
    ):
        import matplotlib.pyplot as plt

        n = len(labels)
        fig_size = max(6.0, min(0.5 * n, 20.0))
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        im = ax.imshow(sim_matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0)
        ax.set_title(title)
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(labels, fontsize=22, rotation=45, ha="right")
        ax.set_yticklabels(labels, fontsize=22)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Cosine similarity")
        fig.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def _plot_factor_similarity(self, experts: List[Expert], layer_names: List[str]):
        os.makedirs(self.config.plot_output_dir, exist_ok=True)
        expert_names = [expert.name.split("_")[0] for expert in experts]
        factors_to_plot = {"U": "U (from SVD of TASK matrix)", "V": "V (from SVD of TASK matrix)"}
        summary_rows = []

        for factor_name, factor_title in factors_to_plot.items():
            layer_similarities = []
            for layer in tqdm(layer_names[:1]):
                key_a = f"{layer}.lora_a"
                key_b = f"{layer}.lora_b"
                if any(
                    key_a not in expert.expert_weights or key_b not in expert.expert_weights
                    for expert in experts
                ):
                    continue

                flattened = []
                for expert in experts:
                    lora_a = expert.expert_weights[key_a].detach().float()
                    lora_b = expert.expert_weights[key_b].detach().float()
                    task_matrix = lora_a @ lora_b
                    u, _, vh = torch.linalg.svd(task_matrix, full_matrices=False)
                    k = 4
                    if factor_name == "U":
                        factor = u[:, :k].cpu().numpy()
                    else:
                        factor = vh.transpose(-2, -1)[:, :k].cpu().numpy()
                    flattened.append(factor.reshape(-1))

                layer_matrix = cosine_similarity(np.stack(flattened, axis=0))
                layer_similarities.append(layer_matrix)

            if not layer_similarities:
                logger.warning(
                    f"Skipping {factor_name} similarity plot; no valid layers found."
                )
                continue

            mean_similarity = np.mean(np.stack(layer_similarities, axis=0), axis=0)
            output_path = os.path.join(
                self.config.plot_output_dir,
                f"cpmergeafter_svd_{factor_name.lower()}_cosine_similarity.png",
            )
            self._save_similarity_heatmap(
                sim_matrix=mean_similarity,
                labels=expert_names,
                title=f"Pairwise cosine similarity of {factor_title} across tasks",
                save_path=output_path,
            )
            matrix_csv_path = os.path.join(
                self.config.plot_output_dir,
                f"cpmergeafter_svd_{factor_name.lower()}_cosine_similarity_matrix.csv",
            )
            np.savetxt(
                matrix_csv_path,
                mean_similarity,
                delimiter=",",
                header=",".join(expert_names),
                comments="",
            )

            off_diag_mask = ~np.eye(mean_similarity.shape[0], dtype=bool)
            off_diag_vals = mean_similarity[off_diag_mask]
            if off_diag_vals.size > 0:
                summary_rows.append(
                    {
                        "factor": factor_name,
                        "mean_offdiag_cosine": float(np.mean(off_diag_vals)),
                        "std_offdiag_cosine": float(np.std(off_diag_vals)),
                        "min_offdiag_cosine": float(np.min(off_diag_vals)),
                        "max_offdiag_cosine": float(np.max(off_diag_vals)),
                    }
                )
            logger.info(
                f"Saved {factor_title} cosine-similarity plot to {output_path}"
            )
            logger.info(
                f"Saved {factor_title} cosine-similarity matrix CSV to {matrix_csv_path}"
            )

        if summary_rows:
            summary_csv_path = os.path.join(
                self.config.plot_output_dir,
                "cpmergeafter_svd_cosine_similarity_summary.csv",
            )
            with open(summary_csv_path, "w", encoding="utf-8") as f:
                f.write(
                    "factor,mean_offdiag_cosine,std_offdiag_cosine,min_offdiag_cosine,max_offdiag_cosine\n"
                )
                for row in summary_rows:
                    f.write(
                        f"{row['factor']},{row['mean_offdiag_cosine']:.8f},{row['std_offdiag_cosine']:.8f},{row['min_offdiag_cosine']:.8f},{row['max_offdiag_cosine']:.8f}\n"
                    )
            logger.info(f"Saved cosine-similarity summary CSV to {summary_csv_path}")

    def _save_cpd_convergence_artifacts(
        self,
        layer_name: str,
        rec_errors: List[float],
        input_tensor: torch.Tensor,
    ):
        import matplotlib.pyplot as plt

        os.makedirs(self.config.plot_output_dir, exist_ok=True)
        input_tensor = input_tensor.detach().float().cpu()
        numel = input_tensor.numel()
        fro_norm = torch.linalg.norm(input_tensor).item()
        rec_errors = [float(e) for e in rec_errors]
        mse_values = [((e * fro_norm) ** 2) / numel for e in rec_errors]
        iters = list(range(1, len(rec_errors) + 1))

        safe_layer_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", layer_name)
        csv_path = os.path.join(
            self.config.plot_output_dir,
            f"cpmergeafter_cpd_convergence_{safe_layer_name}.csv",
        )
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("iteration,rec_error,mse\n")
            for i, rec_e, mse in zip(iters, rec_errors, mse_values):
                f.write(f"{i},{rec_e:.12e},{mse:.12e}\n")

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.tick_params(axis="both", which="major", labelsize=22)
        ax.plot(iters, mse_values, marker="o", linewidth=1.5, markersize=3)
        ax.set_xlabel("CPD iteration", fontsize=22)
        ax.set_ylabel("MSE", fontsize=22)
        ax.set_title(f"CPD convergence (ALS) - {layer_name}", fontsize=22)
        ax.set_yscale("log")
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        png_path = os.path.join(
            self.config.plot_output_dir,
            f"cpmergeafter_cpd_convergence_{safe_layer_name}.png",
        )
        fig.savefig(png_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved CPD convergence CSV to {csv_path}")
        logger.info(f"Saved CPD convergence plot to {png_path}")
        return mse_values

    def _save_cpd_convergence_summary(self, layer_to_mse: Dict[str, List[float]]):
        import matplotlib.pyplot as plt

        if not layer_to_mse:
            return

        os.makedirs(self.config.plot_output_dir, exist_ok=True)
        max_len = max(len(v) for v in layer_to_mse.values())
        stacked = np.full((len(layer_to_mse), max_len), np.nan, dtype=np.float64)
        layer_names = list(layer_to_mse.keys())
        for row_idx, layer_name in enumerate(layer_names):
            vals = layer_to_mse[layer_name]
            stacked[row_idx, : len(vals)] = np.array(vals, dtype=np.float64)

        mean_mse = np.nanmean(stacked, axis=0)
        std_mse = np.nanstd(stacked, axis=0)
        iters = np.arange(1, max_len + 1)

        csv_path = os.path.join(
            self.config.plot_output_dir,
            "cpmergeafter_cpd_convergence_summary.csv",
        )
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("iteration,mean_mse,std_mse,num_layers\n")
            for idx in range(max_len):
                valid_count = int(np.sum(~np.isnan(stacked[:, idx])))
                f.write(
                    f"{iters[idx]},{mean_mse[idx]:.12e},{std_mse[idx]:.12e},{valid_count}\n"
                )

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(iters, mean_mse, linewidth=2.0, label="Mean MSE")
        ax.fill_between(
            iters,
            np.maximum(mean_mse - std_mse, 1e-20),
            mean_mse + std_mse,
            alpha=0.25,
            label="Mean +/- std",
        )
        ax.set_xlabel("CPD iteration", fontsize=22)
        ax.set_ylabel("Reconstruction MSE", fontsize=22)
        ax.set_title("CPD convergence summary (ALS, average over layers)", fontsize=22)
        ax.set_yscale("log")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        fig.tight_layout()
        png_path = os.path.join(
            self.config.plot_output_dir,
            "cpmergeafter_cpd_convergence_summary.png",
        )
        fig.savefig(png_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved CPD convergence summary CSV to {csv_path}")
        logger.info(f"Saved CPD convergence summary plot to {png_path}")

    def _normalize_factor_columns(self, x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        col_norms = torch.linalg.norm(x, dim=0, keepdim=True).clamp_min(eps)
        return x / col_norms

    def _compute_cp_sti_metrics(self, factors_cp) -> Dict[str, float]:
        a, b, c = factors_cp.factors
        a = a.detach().float()
        b = b.detach().float()
        c = c.detach().float()
        rank = a.shape[1]
        eye = torch.eye(rank, device=a.device, dtype=a.dtype)

        ga = a.T @ a
        gb = b.T @ b
        gc = c.T @ c
        ga_c = ga - eye
        gb_c = gb - eye
        gc_c = gc - eye

        diag_ga = torch.diag(torch.diag(ga_c))
        diag_gb = torch.diag(torch.diag(gb_c))
        diag_gc = torch.diag(torch.diag(gc_c))
        off_ga = ga_c - diag_ga
        off_gb = gb_c - diag_gb
        off_gc = gc_c - diag_gc

        a_n = self._normalize_factor_columns(a)
        b_n = self._normalize_factor_columns(b)
        c_n = self._normalize_factor_columns(c)
        ga_n = a_n.T @ a_n
        gb_n = b_n.T @ b_n
        gc_n = c_n.T @ c_n
        ga_n_c = ga_n - eye
        gb_n_c = gb_n - eye
        gc_n_c = gc_n - eye
        off_ga_n = ga_n_c - torch.diag(torch.diag(ga_n_c))
        off_gb_n = gb_n_c - torch.diag(torch.diag(gb_n_c))
        off_gc_n = gc_n_c - torch.diag(torch.diag(gc_n_c))

        metrics = {
            "cp_sti_raw": float(torch.norm(ga_c * gb_c * gc_c, p=1).item()),
            "cp_sti_diag_only": float(torch.norm(diag_ga * diag_gb * diag_gc, p=1).item()),
            "cp_sti_offdiag": float(torch.norm(off_ga * off_gb * off_gc, p=1).item()),
            "cp_sti_norm": float(torch.norm(ga_n_c * gb_n_c * gc_n_c, p=1).item()),
            "cp_sti_norm_offdiag": float(
                torch.norm(off_ga_n * off_gb_n * off_gc_n, p=1).item()
            ),
            "cp_sti_raw_fro": float(torch.norm(ga_c * gb_c * gc_c, p="fro").item()),
            "cp_sti_diag_only_fro": float(
                torch.norm(diag_ga * diag_gb * diag_gc, p="fro").item()
            ),
            "cp_sti_offdiag_fro": float(
                torch.norm(off_ga * off_gb * off_gc, p="fro").item()
            ),
            "cp_sti_norm_fro": float(
                torch.norm(ga_n_c * gb_n_c * gc_n_c, p="fro").item()
            ),
            "cp_sti_norm_offdiag_fro": float(
                torch.norm(off_ga_n * off_gb_n * off_gc_n, p="fro").item()
            ),
        }
        denom = metrics["cp_sti_raw"] + 1e-12
        metrics["cp_sti_diag_fraction"] = metrics["cp_sti_diag_only"] / denom
        metrics["cp_sti_offdiag_fraction"] = metrics["cp_sti_offdiag"] / denom
        denom_fro = metrics["cp_sti_raw_fro"] + 1e-12
        metrics["cp_sti_diag_fraction_fro"] = metrics["cp_sti_diag_only_fro"] / denom_fro
        metrics["cp_sti_offdiag_fraction_fro"] = (
            metrics["cp_sti_offdiag_fro"] / denom_fro
        )
        return metrics

    def _save_cp_sti_artifacts(self, layer_to_cp_sti: Dict[str, Dict[str, float]]):
        import matplotlib.pyplot as plt

        if not layer_to_cp_sti:
            return
        os.makedirs(self.config.plot_output_dir, exist_ok=True)
        layers = list(layer_to_cp_sti.keys())
        cp_sti_raw = [float(layer_to_cp_sti[layer]["cp_sti_raw"]) for layer in layers]
        cp_sti_diag_only = [
            float(layer_to_cp_sti[layer]["cp_sti_diag_only"]) for layer in layers
        ]
        cp_sti_offdiag = [
            float(layer_to_cp_sti[layer]["cp_sti_offdiag"]) for layer in layers
        ]
        cp_sti_norm = [float(layer_to_cp_sti[layer]["cp_sti_norm"]) for layer in layers]
        cp_sti_norm_offdiag = [
            float(layer_to_cp_sti[layer]["cp_sti_norm_offdiag"]) for layer in layers
        ]
        cp_sti_diag_fraction = [
            float(layer_to_cp_sti[layer]["cp_sti_diag_fraction"]) for layer in layers
        ]
        cp_sti_offdiag_fraction = [
            float(layer_to_cp_sti[layer]["cp_sti_offdiag_fraction"]) for layer in layers
        ]
        cp_sti_raw_fro = [
            float(layer_to_cp_sti[layer]["cp_sti_raw_fro"]) for layer in layers
        ]
        cp_sti_diag_only_fro = [
            float(layer_to_cp_sti[layer]["cp_sti_diag_only_fro"]) for layer in layers
        ]
        cp_sti_offdiag_fro = [
            float(layer_to_cp_sti[layer]["cp_sti_offdiag_fro"]) for layer in layers
        ]
        cp_sti_norm_fro = [
            float(layer_to_cp_sti[layer]["cp_sti_norm_fro"]) for layer in layers
        ]
        cp_sti_norm_offdiag_fro = [
            float(layer_to_cp_sti[layer]["cp_sti_norm_offdiag_fro"]) for layer in layers
        ]
        cp_sti_diag_fraction_fro = [
            float(layer_to_cp_sti[layer]["cp_sti_diag_fraction_fro"]) for layer in layers
        ]
        cp_sti_offdiag_fraction_fro = [
            float(layer_to_cp_sti[layer]["cp_sti_offdiag_fraction_fro"]) for layer in layers
        ]

        csv_path = os.path.join(
            self.config.plot_output_dir, "cpmergeafter_cp_sti_by_layer.csv"
        )
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("layer,cp_sti\n")
            for layer, value in zip(layers, cp_sti_raw):
                f.write(f"{layer},{value:.12e}\n")

        detailed_csv_path = os.path.join(
            self.config.plot_output_dir, "cpmergeafter_cp_sti_ablation_by_layer.csv"
        )
        with open(detailed_csv_path, "w", encoding="utf-8") as f:
            f.write(
                "layer,cp_sti_raw,cp_sti_diag_only,cp_sti_offdiag,cp_sti_norm,cp_sti_norm_offdiag,cp_sti_diag_fraction,cp_sti_offdiag_fraction,cp_sti_raw_fro,cp_sti_diag_only_fro,cp_sti_offdiag_fro,cp_sti_norm_fro,cp_sti_norm_offdiag_fro,cp_sti_diag_fraction_fro,cp_sti_offdiag_fraction_fro\n"
            )
            for i, layer in enumerate(layers):
                f.write(
                    f"{layer},{cp_sti_raw[i]:.12e},{cp_sti_diag_only[i]:.12e},{cp_sti_offdiag[i]:.12e},{cp_sti_norm[i]:.12e},{cp_sti_norm_offdiag[i]:.12e},{cp_sti_diag_fraction[i]:.12e},{cp_sti_offdiag_fraction[i]:.12e},{cp_sti_raw_fro[i]:.12e},{cp_sti_diag_only_fro[i]:.12e},{cp_sti_offdiag_fro[i]:.12e},{cp_sti_norm_fro[i]:.12e},{cp_sti_norm_offdiag_fro[i]:.12e},{cp_sti_diag_fraction_fro[i]:.12e},{cp_sti_offdiag_fraction_fro[i]:.12e}\n"
                )

        def _maybe_set_log_scale(ax, series_list, ratio_threshold: float = 50.0):
            positive = []
            for series in series_list:
                arr = np.array(series, dtype=np.float64)
                arr = arr[np.isfinite(arr)]
                arr = arr[arr > 0]
                if arr.size > 0:
                    positive.append(arr)
            if not positive:
                return
            all_pos = np.concatenate(positive)
            vmin = float(np.min(all_pos))
            vmax = float(np.max(all_pos))
            if vmin > 0 and (vmax / vmin) >= ratio_threshold:
                ax.set_yscale("log")
                ylabel = ax.get_ylabel() or "Interference"
                ax.set_ylabel(f"{ylabel} (log scale)", fontsize=12)

        fig, ax = plt.subplots(figsize=(12, 5))
        x = np.arange(len(layers))
        ax.bar(x, cp_sti_raw, color="#4CAF50", alpha=0.85, label="CP-STI (raw)")
        ax.plot(x, cp_sti_raw, color="#FF8C42", marker="o", linewidth=2, markersize=3)
        ax.set_xlabel("Layer", fontsize=16)
        ax.set_ylabel("Interference", fontsize=16)
        ax.set_title("CP-STI across layers", fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(layers, rotation=45, ha="right", fontsize=10)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        ax.legend()
        _maybe_set_log_scale(ax, [cp_sti_raw])
        fig.tight_layout()
        png_path = os.path.join(self.config.plot_output_dir, "cpmergeafter_cp_sti_across_layers.png")
        fig.savefig(png_path, dpi=250, bbox_inches="tight")
        plt.close(fig)

        fig2, ax2 = plt.subplots(figsize=(13, 5))
        ax2.plot(x, cp_sti_raw, marker="o", linewidth=1.8, label="raw")
        ax2.plot(x, cp_sti_diag_only, marker="o", linewidth=1.8, label="diag-only")
        ax2.plot(x, cp_sti_offdiag, marker="o", linewidth=1.8, label="offdiag-only")
        ax2.plot(x, cp_sti_norm, marker="o", linewidth=1.8, label="normalized")
        ax2.plot(
            x,
            cp_sti_norm_offdiag,
            marker="o",
            linewidth=1.8,
            label="normalized offdiag-only",
        )
        ax2.set_xlabel("Layer", fontsize=14)
        ax2.set_ylabel("Interference", fontsize=14)
        ax2.set_title("CP-STI ablation across layers", fontsize=14)
        ax2.set_xticks(x)
        ax2.set_xticklabels(layers, rotation=45, ha="right", fontsize=10)
        ax2.grid(True, axis="y", linestyle="--", alpha=0.4)
        ax2.legend()
        _maybe_set_log_scale(
            ax2,
            [
                cp_sti_raw,
                cp_sti_diag_only,
                cp_sti_offdiag,
                cp_sti_norm,
                cp_sti_norm_offdiag,
            ],
        )
        fig2.tight_layout()
        ablation_png_path = os.path.join(
            self.config.plot_output_dir, "cpmergeafter_cp_sti_ablation_across_layers.png"
        )
        fig2.savefig(ablation_png_path, dpi=250, bbox_inches="tight")
        plt.close(fig2)

        fig3, ax3 = plt.subplots(figsize=(13, 4.5))
        ax3.plot(x, cp_sti_diag_fraction, marker="o", linewidth=1.8, label="diag fraction")
        ax3.plot(
            x,
            cp_sti_offdiag_fraction,
            marker="o",
            linewidth=1.8,
            label="offdiag fraction",
        )
        ax3.set_ylim(0.0, 1.05)
        ax3.set_xlabel("Layer", fontsize=14)
        ax3.set_ylabel("Fraction of raw CP-STI", fontsize=14)
        ax3.set_title("CP-STI diagonal vs off-diagonal contribution", fontsize=14)
        ax3.set_xticks(x)
        ax3.set_xticklabels(layers, rotation=45, ha="right", fontsize=10)
        ax3.grid(True, axis="y", linestyle="--", alpha=0.4)
        ax3.legend()
        fig3.tight_layout()
        frac_png_path = os.path.join(
            self.config.plot_output_dir, "cpmergeafter_cp_sti_diag_offdiag_fraction.png"
        )
        fig3.savefig(frac_png_path, dpi=250, bbox_inches="tight")
        plt.close(fig3)

        fig3b, ax3b = plt.subplots(figsize=(13, 4.5))
        ax3b.plot(
            x, cp_sti_diag_fraction_fro, marker="o", linewidth=1.8, label="diag fraction (Fro)"
        )
        ax3b.plot(
            x,
            cp_sti_offdiag_fraction_fro,
            marker="o",
            linewidth=1.8,
            label="offdiag fraction (Fro)",
        )
        ax3b.set_ylim(0.0, 1.05)
        ax3b.set_xlabel("Layer", fontsize=14)
        ax3b.set_ylabel("Fraction of raw CP-STI (Fro)", fontsize=14)
        ax3b.set_title("CP-STI diagonal vs off-diagonal (Frobenius)", fontsize=14)
        ax3b.set_xticks(x)
        ax3b.set_xticklabels(layers, rotation=45, ha="right", fontsize=10)
        ax3b.grid(True, axis="y", linestyle="--", alpha=0.4)
        ax3b.legend()
        fig3b.tight_layout()
        frac_fro_png_path = os.path.join(
            self.config.plot_output_dir, "cpmergeafter_cp_sti_diag_offdiag_fraction_fro.png"
        )
        fig3b.savefig(frac_fro_png_path, dpi=250, bbox_inches="tight")
        plt.close(fig3b)

        fig6, axes6 = plt.subplots(1, 2, figsize=(16, 5), sharex=True)
        axes6[0].plot(x, cp_sti_raw, marker="o", linewidth=1.8, label="L1 raw")
        axes6[0].plot(x, cp_sti_raw_fro, marker="o", linewidth=1.8, label="Fro raw")
        axes6[0].set_title("Raw CP-STI: L1 vs Frobenius", fontsize=13)
        axes6[0].set_xlabel("Layer", fontsize=12)
        axes6[0].set_ylabel("Interference", fontsize=12)
        axes6[0].set_xticks(x)
        axes6[0].set_xticklabels(layers, rotation=45, ha="right", fontsize=9)
        axes6[0].grid(True, axis="y", linestyle="--", alpha=0.4)
        axes6[0].legend(fontsize=10)
        _maybe_set_log_scale(axes6[0], [cp_sti_raw, cp_sti_raw_fro], ratio_threshold=20.0)

        axes6[1].plot(x, cp_sti_offdiag, marker="o", linewidth=1.8, label="L1 offdiag")
        axes6[1].plot(
            x, cp_sti_offdiag_fro, marker="o", linewidth=1.8, label="Fro offdiag"
        )
        axes6[1].set_title("Offdiag CP-STI: L1 vs Frobenius", fontsize=13)
        axes6[1].set_xlabel("Layer", fontsize=12)
        axes6[1].set_ylabel("Interference", fontsize=12)
        axes6[1].set_xticks(x)
        axes6[1].set_xticklabels(layers, rotation=45, ha="right", fontsize=9)
        axes6[1].grid(True, axis="y", linestyle="--", alpha=0.4)
        axes6[1].legend(fontsize=10)
        _maybe_set_log_scale(
            axes6[1], [cp_sti_offdiag, cp_sti_offdiag_fro], ratio_threshold=20.0
        )
        fig6.tight_layout()
        l1_vs_fro_png_path = os.path.join(
            self.config.plot_output_dir, "cpmergeafter_cp_sti_l1_vs_fro_across_layers.png"
        )
        fig6.savefig(l1_vs_fro_png_path, dpi=250, bbox_inches="tight")
        plt.close(fig6)

        grouped = {"o_proj": [], "qkv_proj": []}
        for i, layer_name in enumerate(layers):
            m = re.search(r"layers\.(\d+)\.self_attn\.(o_proj|qkv_proj)", layer_name)
            if m is None:
                continue
            layer_idx = int(m.group(1))
            proj = m.group(2)
            grouped[proj].append(
                {
                    "layer_idx": layer_idx,
                    "raw": cp_sti_raw[i],
                    "diag_only": cp_sti_diag_only[i],
                    "offdiag": cp_sti_offdiag[i],
                    "norm": cp_sti_norm[i],
                    "norm_offdiag": cp_sti_norm_offdiag[i],
                }
            )

        if grouped["o_proj"] or grouped["qkv_proj"]:
            for proj in grouped:
                grouped[proj] = sorted(grouped[proj], key=lambda x: x["layer_idx"])

            fig4, axes4 = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
            proj_order = [("o_proj", "O projection"), ("qkv_proj", "QKV projection")]
            for ax4, (proj_key, proj_title) in zip(axes4, proj_order):
                rows = grouped[proj_key]
                if not rows:
                    ax4.set_title(f"{proj_title} (no data)", fontsize=13)
                    ax4.set_xlabel("Layer index", fontsize=12)
                    ax4.grid(True, axis="y", linestyle="--", alpha=0.4)
                    continue
                x4 = [r["layer_idx"] for r in rows]
                y4 = [r["raw"] for r in rows]
                ax4.plot(x4, y4, marker="o", linewidth=1.8, label=f"raw ({proj_key})")
                ax4.set_title(f"CP-STI raw: {proj_title}", fontsize=13)
                ax4.set_xlabel("Layer index", fontsize=12)
                ax4.grid(True, axis="y", linestyle="--", alpha=0.4)
                ax4.legend(fontsize=10)
                _maybe_set_log_scale(ax4, [y4], ratio_threshold=20.0)

            axes4[0].set_ylabel("Interference", fontsize=12)
            fig4.tight_layout()
            by_proj_raw_png_path = os.path.join(
                self.config.plot_output_dir, "cpmergeafter_cp_sti_raw_by_proj_across_layers.png"
            )
            fig4.savefig(by_proj_raw_png_path, dpi=250, bbox_inches="tight")
            plt.close(fig4)

            fig5, axes5 = plt.subplots(1, 2, figsize=(18, 5), sharey=True)
            for ax5, (proj_key, proj_title) in zip(axes5, proj_order):
                rows = grouped[proj_key]
                if not rows:
                    ax5.set_title(f"{proj_title} (no data)", fontsize=13)
                    ax5.set_xlabel("Layer index", fontsize=12)
                    ax5.grid(True, axis="y", linestyle="--", alpha=0.4)
                    continue
                x5 = [r["layer_idx"] for r in rows]
                series = {
                    "raw": [r["raw"] for r in rows],
                    "diag-only": [r["diag_only"] for r in rows],
                    "offdiag-only": [r["offdiag"] for r in rows],
                    "normalized": [r["norm"] for r in rows],
                    "norm offdiag": [r["norm_offdiag"] for r in rows],
                }
                for label, y5 in series.items():
                    ax5.plot(x5, y5, marker="o", linewidth=1.6, label=label)
                ax5.set_title(f"CP-STI ablation: {proj_title}", fontsize=13)
                ax5.set_xlabel("Layer index", fontsize=12)
                ax5.grid(True, axis="y", linestyle="--", alpha=0.4)
                ax5.legend(fontsize=9)
                _maybe_set_log_scale(ax5, list(series.values()), ratio_threshold=20.0)

            axes5[0].set_ylabel("Interference", fontsize=12)
            fig5.tight_layout()
            by_proj_ablation_png_path = os.path.join(
                self.config.plot_output_dir,
                "cpmergeafter_cp_sti_ablation_by_proj_across_layers.png",
            )
            fig5.savefig(by_proj_ablation_png_path, dpi=250, bbox_inches="tight")
            plt.close(fig5)

        logger.info(f"Saved CP-STI CSV to {csv_path}")
        logger.info(f"Saved CP-STI plot to {png_path}")
        logger.info(f"Saved CP-STI ablation CSV to {detailed_csv_path}")
        logger.info(f"Saved CP-STI ablation plot to {ablation_png_path}")
        logger.info(f"Saved CP-STI diag/offdiag fraction plot to {frac_png_path}")
        logger.info(f"Saved CP-STI diag/offdiag fraction (Fro) plot to {frac_fro_png_path}")
        logger.info(f"Saved CP-STI L1-vs-Fro plot to {l1_vs_fro_png_path}")
        if grouped["o_proj"] or grouped["qkv_proj"]:
            logger.info(f"Saved CP-STI raw by projection plot to {by_proj_raw_png_path}")
            logger.info(
                f"Saved CP-STI ablation by projection plot to {by_proj_ablation_png_path}"
            )

    def _get_task_vectors(self, expert):
        task_vectors = {}
        for key in expert.expert_weights.keys():
            base_layer_name = key.split(".lora_")[
                0
            ]  # Get base layer name by removing .lora_a or .lora_b
            if base_layer_name not in task_vectors:
                task_vectors[base_layer_name] = None    
        for layer in task_vectors.keys():
            lora_a = expert.expert_weights[f"{layer}.lora_a"]
            lora_b = expert.expert_weights[f"{layer}.lora_b"]
            task_vectors[layer] = lora_a.data @ lora_b.data

        return task_vectors
    @torch.no_grad()
    def transform(self, library, recompute=False) -> Expert:
        import tensorly as tl

        tl.set_backend("pytorch")
        from tensorly.decomposition import parafac

        expert_names = list(library.keys())
        experts = [library[name] for name in expert_names]
        logger.info("Merging {} experts using CPMergeAfter".format(len(experts)))
        one_expert = experts[0]
        layer_names = [
            name.split(".lora_")[0] for name in one_expert.expert_weights.keys()
        ]
        layer_names = sorted(list(set(layer_names)))
        if self.config.plot_similarity:
            self._plot_factor_similarity(experts=experts, layer_names=layer_names)
        if self.config.plot_cp_sti:
            cp_sti_by_layer = {}
        task_vectors_experts = {}
        if not os.path.exists(self.config.path) or recompute:
            for expert in experts:
                task_vectors = self._get_task_vectors(expert)
                task_vectors_experts[expert.name] = task_vectors
            task_merged_vectors = {}
            for layer in layer_names:
                task_vectors = [
                    task_vectors_experts[expert.name][layer] for expert in experts
                ]
                logger.info(f"Layer {layer} merged with CP decomposition")
                try:
                    task_vectors_stack = torch.stack(task_vectors, dim=0).to("cuda")
                    factors_cp, rec_errors = parafac(
                        task_vectors_stack,
                        rank=self.config.cp_rank,
                        init="random",
                        random_state=42,
                        n_iter_max=self.config.cpd_n_iter_max,
                        tol=self.config.cpd_tol,
                        return_errors=True,
                    )
                    if self.config.plot_cpd_convergence:
                        if "cpd_mse_by_layer" not in locals():
                            cpd_mse_by_layer = {}
                        cpd_mse_by_layer[layer] = self._save_cpd_convergence_artifacts(
                            layer_name=layer,
                            rec_errors=rec_errors,
                            input_tensor=task_vectors_stack,
                        )
                    ar = factors_cp.factors[0] #[n_experts, rank]
                    br = factors_cp.factors[1] #[input_dim, rank]
                    cr = factors_cp.factors[2] #[output_dim, rank]
                    if self.config.plot_cp_sti:
                        cp_sti_by_layer[layer] = self._compute_cp_sti_metrics(factors_cp)

                    ar_sum = torch.sum(ar, dim=0)
                    delta = (br * ar_sum.unsqueeze(0)) @ cr.T
                    task_merged_vectors[layer] = delta
                    
                except Exception as e:
                    logger.info(e)               
                    task_vectors_stack = torch.stack(task_vectors, dim=0).to("cpu")
                    factors_cp, rec_errors = parafac(
                        task_vectors_stack,
                        rank=self.config.cp_rank,
                        init="random",
                        random_state=42,
                        n_iter_max=self.config.cpd_n_iter_max,
                        tol=self.config.cpd_tol,
                        return_errors=True,
                    )
                    if self.config.plot_cpd_convergence:
                        if "cpd_mse_by_layer" not in locals():
                            cpd_mse_by_layer = {}
                        cpd_mse_by_layer[layer] = self._save_cpd_convergence_artifacts(
                            layer_name=layer,
                            rec_errors=rec_errors,
                            input_tensor=task_vectors_stack,
                        )
                    ar = factors_cp.factors[0]
                    br = factors_cp.factors[1]
                    cr = factors_cp.factors[2]
                    if self.config.plot_cp_sti:
                        cp_sti_by_layer[layer] = self._compute_cp_sti_metrics(factors_cp)
                    ar_sum = torch.sum(ar, dim=0)
                    delta = (br * ar_sum.unsqueeze(0)) @ cr.T
                    task_merged_vectors[layer] = delta
            if self.config.plot_cpd_convergence and "cpd_mse_by_layer" in locals():
                self._save_cpd_convergence_summary(cpd_mse_by_layer)
            if self.config.plot_cp_sti and cp_sti_by_layer:
                self._save_cp_sti_artifacts(cp_sti_by_layer)
            torch.save(task_merged_vectors, self.config.path)
        else:
            task_merged_vectors = torch.load(self.config.path)
        logger.info(f"Merged {len(task_merged_vectors)} layers")
        return task_merged_vectors

@dataclass
class WudiMergeConfig(LibraryTransformConfig):
    iter: int = 300
    lr: float = 1e-5


@LibraryTransform.register("wudi_merge_after", WudiMergeConfig)
class WudiMergeAfter(LibraryTransform):
    """
    implement the wudimerge in the paper https://arxiv.org/pdf/2503.08099v1

    we multiply the lora A and lora B and then merge the experts to the model(merge after).
    """

    def __init__(self, config: WudiMergeConfig = None):
        super().__init__(config or WudiMergeConfig())

    def _get_task_vectors(self, expert):
        """
        get the incremental weights for each layer, LoRA A outproduct LoRA B
        """
        task_vectors = {}
        for key in expert.expert_weights.keys():
            base_layer_name = key.split(".lora_")[
                0
            ]  # Get base layer name by removing .lora_a or .lora_b
            if base_layer_name not in task_vectors:
                task_vectors[base_layer_name] = None

        for layer in task_vectors.keys():
            lora_a = expert.expert_weights[f"{layer}.lora_a"]
            lora_b = expert.expert_weights[f"{layer}.lora_b"]
            task_vectors[layer] = lora_a.data @ lora_b.data

        return task_vectors

    def get_optimized_task_vector(
        self, layer_name, task_vectors, iter, lr
    ) -> torch.Tensor:
        """
        min Σᵢ (1/||τᵢ,ₗ||²F) ||(τₘ,ₗ - τᵢ,ₗ)(τᵢ,ₗ)ᵀ||²F

        return the optimized merged task vector for each layer
        """
        task_vectors = task_vectors.cuda()
        merging_vector = torch.nn.Parameter((torch.sum(task_vectors, dim=0)))
        optimizer = torch.optim.Adam([merging_vector], lr=lr, weight_decay=0)

        l2_norms = torch.square(
            torch.norm(task_vectors.reshape(task_vectors.shape[0], -1), p=2, dim=-1)
        )

        pbar = tqdm(range(iter), desc=f"Optimizing parameter {layer_name}")
        prev_loss = float("inf")
        patience = 5  # Number of steps to wait for improvement
        no_improve_count = 0
        min_delta = 1e-4  # Minimum change in loss to be considered improvement

        for step in pbar:
            disturbing_vectors = merging_vector.unsqueeze(0) - task_vectors
            inner_product = torch.matmul(
                disturbing_vectors, task_vectors.transpose(1, 2)
            )
            loss = torch.sum(torch.square(inner_product))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Check if loss improvement is significant
            if abs(prev_loss - loss.item()) < min_delta:
                no_improve_count += 1
            else:
                no_improve_count = 0

            # Early stopping if no significant improvement for patience steps
            if no_improve_count >= patience:
                logger.info(f"Early stopping at step {step} due to minimal loss change")
                break

            prev_loss = loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        return merging_vector

    def transform(self, library, persist=True, recompute=False) -> dict:
        """
        return the task merged vectors in each layer
        """
        rank_table = TableLogger()
        if type(library) == str:
            library = ExpertLibrary.get_expert_library(library)
        expert_names = list(library.keys())
        experts = [library[name] for name in expert_names]
        logger.info("Merging {} experts using WuDi merge after".format(len(experts)))
        one_expert = experts[0]
        # get the layer names from the model
        layer_names = [
            name.split(".lora_")[0] for name in one_expert.expert_weights.keys()
        ]
        layer_names = sorted(list(set(layer_names)))

        # get the task vectors for each expert
        task_vectors_experts = {}
        for expert in experts:
            task_vectors = self._get_task_vectors(expert)
            task_vectors_experts[expert.name] = task_vectors
        task_merged_vectors = {}
        # wudi merge the task vectors
        for layer in layer_names:

            # get the experts for this layer
            task_vectors = [
                task_vectors_experts[expert.name][layer] for expert in experts
            ]

            task_vectors = torch.stack(task_vectors, dim=0)
            # get the redundant task vector
            merged_task_vector = self.get_optimized_task_vector(
                layer_name=layer,
                task_vectors=task_vectors,
                iter=self.config.iter,
                lr=self.config.lr,
            )

            # save the merged task vector in each layer
            task_merged_vectors[layer] = merged_task_vector / len(experts)

            # get the rank of the merged task vector
            rank = torch.linalg.matrix_rank(merged_task_vector)
            logger.info(
                f"Rank of the merged task vector for {layer} is {rank}, original rank is {merged_task_vector.shape[0]}"
            )
            rank_table.log(
                {
                    "layer": layer,
                    "rank": rank.item(),
                    "original_rank": merged_task_vector.shape[0],
                }
            )
        rank_table.log_final_table()
        return task_merged_vectors


@LibraryTransform.register("wudi_merge", WudiMergeConfig)
class WudiMerge(LibraryTransform):
    """
    implement the wudimerge in the paper https://arxiv.org/pdf/2503.08099v1
    """

    def __init__(self, config: WudiMergeConfig = None):
        super().__init__(config or WudiMergeConfig())

    def transform(self, library) -> Expert:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if type(library) == str:
            library = ExpertLibrary.get_expert_library(library)

        expert_names = list(library.keys())
        experts = [library[name] for name in expert_names]

        logger.info("Merging {} experts using WuDi merge before".format(len(experts)))

        base_expert = copy.deepcopy(experts[0])

        # Get all parameter keys that we want to merge
        keys = [key for key in base_expert.expert_weights.keys()]
        task_merged_vectors = {}
        for key in keys:
            # Stack all expert weights for this parameter
            values = torch.stack([expert.expert_weights[key] for expert in experts])

            values = values.to(device)

            # Initialize merged vector as sum of all vectors
            merging_vector = torch.nn.Parameter(
                torch.sum(values, dim=0), requires_grad=True
            )
            optimizer = torch.optim.Adam(
                [merging_vector], lr=self.config.lr, weight_decay=0
            )

            # Compute L2 norms
            l2_norms = torch.square(
                torch.norm(values.reshape(values.shape[0], -1), p=2, dim=-1)
            )

            # Optimize merging vector
            pbar = tqdm(range(self.config.iter), desc=f"Optimizing parameter {key}")
            prev_loss = float("inf")
            patience = 5  # Number of steps to wait for improvement
            no_improve_count = 0
            min_delta = 1e-4  # Minimum change in loss to be considered improvement

            for step in pbar:
                disturbing_vectors = merging_vector.unsqueeze(0) - values
                inner_product = torch.matmul(disturbing_vectors, values.transpose(1, 2))

                loss = torch.sum(
                    torch.square(inner_product) / l2_norms.unsqueeze(-1).unsqueeze(-1)
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Check if loss improvement is significant
                if abs(prev_loss - loss.item()) < min_delta:
                    no_improve_count += 1
                else:
                    no_improve_count = 0

                # Early stopping if no significant improvement for patience steps
                if no_improve_count >= patience:
                    logger.info(
                        f"Early stopping at step {step} due to minimal loss change"
                    )
                    break

                prev_loss = loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            merging_vector = merging_vector / len(experts)
            task_merged_vectors[key] = merging_vector
        return task_merged_vectors


@dataclass
class WuDiMerge2Config(LibraryTransformConfig):
    iter: int = 300
    lr: float = 1e-4


@LibraryTransform.register("wudi_merge_2", WuDiMerge2Config)
class WuDiMerge2(WudiMergeAfter):
    """
    implement the wudimerge in the paper https://arxiv.org/pdf/2505.19892
    """

    def __init__(self, config: WuDiMerge2Config = None):
        super().__init__(config or WuDiMerge2Config())

    def get_optimized_task_vector(self, layer_name, task_vectors, iter=300, lr=1e-4):
        """
        get the optimized task vector for the layer
        min Σᵢ (1/||τᵢ,ₗ||²F) ||(τₘ,ₗ - τᵢ,ₗ)(τᵢ,ₗ)ᵀ||²F
        """
        original_dtype = task_vectors.dtype
        task_vectors = task_vectors.cuda()
        average_vector = task_vectors.mean(dim=0)
        low_rank_list = []
        taskvector_list = []
        for i in tqdm(
            range(task_vectors.shape[0]), desc=f"wudi merge 2 for {layer_name}"
        ):
            vector = task_vectors[i]
            u, s, v = torch.linalg.svd(vector, full_matrices=True)
            u2, s2, v2 = torch.linalg.svd(vector, full_matrices=False)
            reduced_index_s = int(s.shape[0] / task_vectors.shape[0])
            u2 = u2[:, :reduced_index_s]
            s2 = s2[:reduced_index_s]
            v2 = v2[:reduced_index_s, :]
            s_mask = torch.zeros_like(s)
            s_mask[:reduced_index_s] = 1
            s = s * s_mask
            v_mask = torch.zeros_like(v)
            v_mask[:reduced_index_s, :] = 1
            v = v * v_mask  # (n, n)
            S_matrix = torch.zeros(
                vector.shape[0], vector.shape[1], device=s.device
            )  # m x n
            min_dim = min(vector.shape)
            S_matrix[:min_dim, :min_dim] = torch.diag_embed(s)
            low_rank_list.append(S_matrix @ v)
            taskvector_list.append(u2 @ torch.diag_embed(s2) @ v2)
            # del u, s, v, u2, s2, v2, S_matrix, s_mask, v_mask
        low_rank = torch.stack(low_rank_list).to(original_dtype)
        taskvector = torch.stack(taskvector_list).to(original_dtype)

        merging_vector = torch.nn.Parameter(average_vector.to(original_dtype))
        # optimizer = torch.optim.SGD([merging_vector], lr=lr, momentum=0.9)
        optimizer = torch.optim.Adam([merging_vector], lr=lr, weight_decay=0)
        l2_norms = torch.square(
            torch.norm(taskvector.reshape(taskvector.shape[0], -1), p=2, dim=-1)
        ).to(original_dtype)

        pbar = tqdm(range(iter), desc=f"Optimizing {layer_name}", leave=False)
        prev_loss = float("inf")
        patience = 5  # Number of steps to wait for improvement
        no_improve_count = 0
        min_delta = 1e-4  # Minimum change in loss to be considered improvement

        for step in pbar:
            disturbing_vectors = merging_vector.unsqueeze(0) - taskvector
            inner_product = torch.matmul(disturbing_vectors, low_rank.transpose(1, 2))
            loss = torch.sum(
                torch.square(inner_product) / l2_norms.unsqueeze(-1).unsqueeze(-1)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if abs(prev_loss - loss.item()) < min_delta:
                no_improve_count += 1
            else:
                no_improve_count = 0

            if no_improve_count >= patience:
                logger.info(f"Early stopping at step {step} due to minimal loss change")
                break

            prev_loss = loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        return merging_vector

@dataclass
class ISOMergeConfig(LibraryTransformConfig):
    pass

@LibraryTransform.register("iso_merge", ISOMergeConfig)
class ISOMerge(LibraryTransform):
    def __init__(self, config: ISOMergeConfig = None):
        super().__init__(config or ISOMergeConfig())

    def _get_task_vectors(self, expert):
        task_vectors = {}
        for key in expert.expert_weights.keys():
            base_layer_name = key.split(".lora_")[
                0
            ]  # Get base layer name by removing .lora_a or .lora_b
            if base_layer_name not in task_vectors:
                task_vectors[base_layer_name] = None
        for layer in task_vectors.keys():
            lora_a = expert.expert_weights[f"{layer}.lora_a"]
            lora_b = expert.expert_weights[f"{layer}.lora_b"]
            task_vectors[layer] = lora_a.data @ lora_b.data
        return task_vectors

    def transform(self, library, recompute=False) -> Expert:
        if type(library) == str:
            library = ExpertLibrary.get_expert_library(library)
        expert_names = list(library.keys())
        experts = [library[name] for name in expert_names]
        logger.info("Merging {} experts using ISOMerge".format(len(experts)))
        one_expert = experts[0]
        layer_names = [
            name.split(".lora_")[0] for name in one_expert.expert_weights.keys()
        ]
        layer_names = sorted(list(set(layer_names)))

        task_vectors_experts = {}
        for expert in experts:
            task_vectors = self._get_task_vectors(expert)
            task_vectors_experts[expert.name] = task_vectors

        task_merged_vectors = {}
        for layer in layer_names:
            task_vectors = [
                task_vectors_experts[expert.name][layer] for expert in experts
            ]
            logger.info(f"layer {layer} has {len(task_vectors)} task vectors")
            original_dtype = task_vectors[0].dtype
            param_value = sum(task_vectors)
            u, s, v = torch.linalg.svd(param_value, full_matrices=False)
            # Compute the average of all singular values (a scalar)
            avg_singular_value = torch.mean(s)
            # Create a diagonal matrix where all diagonal elements are this average value
            avg_s = torch.diag(torch.full_like(s, avg_singular_value))

            merged_param = torch.linalg.multi_dot([
                u,avg_s, v
            ]).to(original_dtype)
            task_merged_vectors[layer] = merged_param

        return task_merged_vectors

@dataclass
class TSVMergeConfig(LibraryTransformConfig):
    path: str = "tsv_ingredients.pt"
    plot_sti: bool = False
    plot_output_dir: str = "tsv_merge_plots"


@LibraryTransform.register("tsv_merge", TSVMergeConfig)
class TSVMerge(LibraryTransform):
    """
    merge the task vectors using svd
    """

    def __init__(self, config: TSVMergeConfig = None):
        super().__init__(config or TSVMergeConfig())

    def _get_task_vectors(self, expert, layer):

        lora_a = expert.expert_weights[f"{layer}.lora_a"]
        lora_b = expert.expert_weights[f"{layer}.lora_b"]
        task_vector = lora_a.data @ lora_b.data

        return task_vector

    def _merge_task_vectors(
        self, task_vectors, layer, device, original_dtype, sv_reduction
    ):

        sum_u = None
        sum_s = None
        sum_v = None

        # Process each task vector
        for i, vec in tqdm(
            enumerate(task_vectors), desc=f"TSV merging compute for {layer}"
        ):
            # Move parameter to GPU for computation
            vec = vec.to(device).float()
            # Compute SVD
            u, s, v = torch.linalg.svd(vec, full_matrices=False)

            # Compute reduced index
            reduced_index_s = int(s.shape[0] * sv_reduction)

            # Initialize storage for the first vector
            if i == 0:
                sum_u = torch.zeros_like(u, device=device)
                sum_s = torch.zeros_like(s, device=device)
                sum_v = torch.zeros_like(v, device=device)

            # Store important components
            sum_u[:, i * reduced_index_s : (i + 1) * reduced_index_s] = u[
                :, :reduced_index_s
            ]
            sum_s[i * reduced_index_s : (i + 1) * reduced_index_s] = s[:reduced_index_s]
            sum_v[i * reduced_index_s : (i + 1) * reduced_index_s, :] = v[
                :reduced_index_s, :
            ]
        # Compute final merged parameter
        u_u, s_u, v_u = torch.linalg.svd(sum_u, full_matrices=False)
        u_v, s_v, v_v = torch.linalg.svd(sum_v, full_matrices=False)

        # Compute merged result and move back to CPU
        merged_param = (
            torch.linalg.multi_dot([u_u, v_u, torch.diag(sum_s), u_v, v_v])
            .to(original_dtype)
            .cpu()
        )

        return merged_param

    def _compute_sti(self, task_vectors, device, sv_reduction) -> Dict[str, float]:
        u_factors = []
        v_factors = []
        sigma_parts = []
        for vec in task_vectors:
            vec = vec.to(device).float()
            u, s, vh = torch.linalg.svd(vec, full_matrices=False)
            reduced_rank = max(1, int(s.shape[0] * sv_reduction))
            u_factors.append(u[:, :reduced_rank])
            v_factors.append(vh.transpose(-2, -1)[:, :reduced_rank])
            sigma_parts.append(s[:reduced_rank])

        u_cat = torch.cat(u_factors, dim=1)
        v_cat = torch.cat(v_factors, dim=1)
        sigma_vec = torch.cat(sigma_parts, dim=0)
        gram_u = u_cat.T @ u_cat
        gram_v = v_cat.T @ v_cat
        eye = torch.eye(gram_u.shape[0], device=device, dtype=gram_u.dtype)
        core_no_sigma = (gram_u - eye) * (gram_v - eye)
        sigma_mat = torch.diag(sigma_vec).to(gram_u.dtype)
        core_sigma = (gram_u - eye) @ sigma_mat @ (gram_v - eye)

        return {
            "sti": float(torch.norm(core_sigma, p=1).item()),
            "sti_fro": float(torch.norm(core_sigma, p="fro").item()),
            "sti_no_sigma": float(torch.norm(core_no_sigma, p=1).item()),
            "sti_no_sigma_fro": float(torch.norm(core_no_sigma, p="fro").item()),
        }

    def _save_sti_artifacts(self, layer_to_sti: Dict[str, Dict[str, float]]):
        import matplotlib.pyplot as plt

        if not layer_to_sti:
            return
        os.makedirs(self.config.plot_output_dir, exist_ok=True)
        layers = list(layer_to_sti.keys())
        values = [float(layer_to_sti[layer]["sti"]) for layer in layers]
        values_fro = [float(layer_to_sti[layer]["sti_fro"]) for layer in layers]
        values_no_sigma = [float(layer_to_sti[layer]["sti_no_sigma"]) for layer in layers]
        values_no_sigma_fro = [
            float(layer_to_sti[layer]["sti_no_sigma_fro"]) for layer in layers
        ]

        csv_path = os.path.join(self.config.plot_output_dir, "tsvmerge_sti_by_layer.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("layer,sti\n")
            for layer, value in zip(layers, values):
                f.write(f"{layer},{value:.12e}\n")

        detailed_csv_path = os.path.join(
            self.config.plot_output_dir, "tsvmerge_sti_ablation_by_layer.csv"
        )
        with open(detailed_csv_path, "w", encoding="utf-8") as f:
            f.write("layer,sti,sti_fro,sti_no_sigma,sti_no_sigma_fro\n")
            for i, layer in enumerate(layers):
                f.write(
                    f"{layer},{values[i]:.12e},{values_fro[i]:.12e},{values_no_sigma[i]:.12e},{values_no_sigma_fro[i]:.12e}\n"
                )

        fig, ax = plt.subplots(figsize=(12, 5))
        x = np.arange(len(layers))
        ax.bar(x, values, color="#4CAF50", alpha=0.85, label="STI")
        ax.plot(x, values, color="#FF8C42", marker="o", linewidth=2, markersize=3)
        ax.set_xlabel("Layer", fontsize=16)
        ax.set_ylabel("Interference", fontsize=16)
        ax.set_title("STI across layers (TSV merge)", fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(layers, rotation=45, ha="right", fontsize=10)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        ax.legend()
        fig.tight_layout()
        png_path = os.path.join(self.config.plot_output_dir, "tsvmerge_sti_across_layers.png")
        fig.savefig(png_path, dpi=250, bbox_inches="tight")
        plt.close(fig)

        fig2, axes2 = plt.subplots(1, 2, figsize=(16, 5), sharex=True)
        axes2[0].plot(x, values, marker="o", linewidth=1.8, label="L1 (with Sigma)")
        axes2[0].plot(
            x, values_no_sigma, marker="o", linewidth=1.8, label="L1 (no Sigma)"
        )
        axes2[0].set_title("TSV STI (L1)", fontsize=13)
        axes2[0].set_xlabel("Layer", fontsize=12)
        axes2[0].set_ylabel("Interference", fontsize=12)
        axes2[0].set_xticks(x)
        axes2[0].set_xticklabels(layers, rotation=45, ha="right", fontsize=9)
        axes2[0].grid(True, axis="y", linestyle="--", alpha=0.4)
        axes2[0].legend(fontsize=10)

        axes2[1].plot(x, values_fro, marker="o", linewidth=1.8, label="Fro (with Sigma)")
        axes2[1].plot(
            x, values_no_sigma_fro, marker="o", linewidth=1.8, label="Fro (no Sigma)"
        )
        axes2[1].set_title("TSV STI (Frobenius)", fontsize=13)
        axes2[1].set_xlabel("Layer", fontsize=12)
        axes2[1].set_ylabel("Interference", fontsize=12)
        axes2[1].set_xticks(x)
        axes2[1].set_xticklabels(layers, rotation=45, ha="right", fontsize=9)
        axes2[1].grid(True, axis="y", linestyle="--", alpha=0.4)
        axes2[1].legend(fontsize=10)

        fig2.tight_layout()
        ablation_png_path = os.path.join(
            self.config.plot_output_dir, "tsvmerge_sti_ablation_across_layers.png"
        )
        fig2.savefig(ablation_png_path, dpi=250, bbox_inches="tight")
        plt.close(fig2)
        logger.info(f"Saved STI CSV to {csv_path}")
        logger.info(f"Saved STI plot to {png_path}")
        logger.info(f"Saved STI ablation CSV to {detailed_csv_path}")
        logger.info(f"Saved STI ablation plot to {ablation_png_path}")

    @torch.no_grad()
    def transform(self, library, persist=True, recompute=False) -> dict:
        # empty the cache
        torch.cuda.empty_cache()
        if type(library) == str:
            library = ExpertLibrary.get_expert_library(library)
        expert_names = list(library.keys())
        experts = [library[name] for name in expert_names]
        logger.info("Merging {} experts using SVD merge".format(len(experts)))

        one_expert = experts[0]
        # get the layer names from the model
        layer_names = [
            name.split(".lora_")[0] for name in one_expert.expert_weights.keys()
        ]
        layer_names = sorted(list(set(layer_names)))
        task_merged_vectors = {}
        if os.path.exists(self.config.path) and not recompute:
            logger.info(f"load task vectors from {self.config.path}")
            task_merged_vectors = torch.load(self.config.path)
            return task_merged_vectors
        else:
            layer_to_sti = {}
            for layer in layer_names:
                logger.info(f"compute task vector for {layer}")
                # Get task vectors for this layer from all experts
                task_vectors = [
                    self._get_task_vectors(expert, layer) for expert in experts
                ]

                # Apply SVD merging for this layer
                sv_reduction = 1.0 / len(task_vectors)
                device = (
                    torch.device("cuda")
                    if torch.cuda.is_available()
                    else torch.device("cpu")
                )
                original_dtype = task_vectors[0].dtype
                if self.config.plot_sti:
                    layer_to_sti[layer] = self._compute_sti(task_vectors, device, sv_reduction)
                task_merged_vectors[layer] = self._merge_task_vectors(
                    task_vectors, layer, device, original_dtype, sv_reduction
                )
            if self.config.plot_sti and layer_to_sti:
                self._save_sti_artifacts(layer_to_sti)
            torch.save(task_merged_vectors, self.config.path)

        return task_merged_vectors


@dataclass
class AnalyticalWudiMergeConfig(LibraryTransformConfig):
    regularization: float = 1e-6
    pass


@LibraryTransform.register("analytical_wudi_merge", AnalyticalWudiMergeConfig)
class AnalyticalWudiMerge(LibraryTransform):
    """ """

    def __init__(self, config: AnalyticalWudiMergeConfig = None):
        super().__init__(config or AnalyticalWudiMergeConfig())
        self.merger = AnalyticalSolver(regularization_omega=self.config.regularization)

    def _get_task_vectors(self, expert):
        task_vectors = {}
        for key in expert.expert_weights.keys():
            base_layer_name = key.split(".lora_")[
                0
            ]  # Get base layer name by removing .lora_a or .lora_b
            if base_layer_name not in task_vectors:
                task_vectors[base_layer_name] = None

        for layer in task_vectors.keys():
            lora_a = expert.expert_weights[f"{layer}.lora_a"]
            lora_b = expert.expert_weights[f"{layer}.lora_b"]
            task_vectors[layer] = lora_a.data @ lora_b.data

        return task_vectors

    def transform(self, library) -> dict:
        if type(library) == str:
            library = ExpertLibrary.get_expert_library(library)
        expert_names = list(library.keys())
        experts = [library[name] for name in expert_names]
        logger.info(
            "Merging {} experts using WuDi analytical merge".format(len(experts))
        )

        # get the task vectors for each expert
        task_vectors_experts = {}
        for expert in experts:
            task_vectors = self._get_task_vectors(expert)
            task_vectors_experts[expert.name] = task_vectors
        task_merged_vectors = {}
        # merge the task vectors
        layer_names = list(task_vectors_experts[experts[0].name].keys())
        for layer in layer_names:
            task_vectors = [
                task_vectors_experts[expert.name][layer] for expert in experts
            ]

            # Get optimal merged matrix using analytical solution
            logger.info(f"Merging {layer} with {len(task_vectors)} task vectors")
            merged_matrix = self.merger.compute_analytical_solution(task_vectors)
            # Convert back to torch tensor with same device and dtype
            merged_task_vector = torch.tensor(
                merged_matrix,
                device=task_vectors[0].device,
                dtype=task_vectors[0].dtype,
            )
            # add the merged task vector to the model
            task_merged_vectors[layer] = merged_task_vector / len(experts)
        return task_merged_vectors


@dataclass
class WeightedLinearMergeConfig(LibraryTransformConfig):
    weights: dict = None


@LibraryTransform.register("weighted_linear_merge", WeightedLinearMergeConfig)
class WeightedLinearMerge(LibraryTransform):
    """
    Computes a uniform weight mixture across experts of a given library
    """

    def __init__(self, config: WeightedLinearMergeConfig = None):
        super().__init__(config or WeightedLinearMergeConfig())

    @torch.no_grad()
    def transform(self, library) -> Expert:
        if type(library) == str:
            library = ExpertLibrary.get_expert_library(library)

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
                logger.warning(
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

        # manually change the config of the expert to remove the tie_params
        base_expert.expert_config.tie_params = None

        return base_expert

@dataclass
class UniformMergeAfterConfig(WeightedLinearMergeConfig):
    pass

@LibraryTransform.register("uniform_merge_after", UniformMergeAfterConfig)
class UniformMergeAfter(LibraryTransform):
    def __init__(self, config: UniformMergeAfterConfig = None):
        super().__init__(config or UniformMergeAfterConfig())

    def _get_task_vectors(self, expert):
        task_vectors = {}
        for key in expert.expert_weights.keys():
            base_layer_name = key.split(".lora_")[
                0
            ]  # Get base layer name by removing .lora_a or .lora_b
            if base_layer_name not in task_vectors:
                task_vectors[base_layer_name] = None

        for layer in task_vectors.keys():
            lora_a = expert.expert_weights[f"{layer}.lora_a"]
            lora_b = expert.expert_weights[f"{layer}.lora_b"]
            task_vectors[layer] = lora_a.data @ lora_b.data

        return task_vectors

    def transform(self, library) -> Expert:
        

        if type(library) == str:
            library = ExpertLibrary.get_expert_library(library)

        expert_names = list(library.keys())
        experts = [library[name] for name in expert_names]

        layer_names = [
            name.split(".lora_")[0] for name in experts[0].expert_weights.keys()
        ]
        layer_names = sorted(list(set(layer_names)))

        task_vectors_experts = {}
        for expert in experts:
            task_vectors = self._get_task_vectors(expert)
            task_vectors_experts[expert.name] = task_vectors

        task_merged_vectors = {}
        for layer in layer_names:
            task_vectors = [
                task_vectors_experts[expert.name][layer] for expert in experts
            ]
            task_merged_vectors[layer] = sum(task_vectors) / len(experts)
        return task_merged_vectors


def _clone_expert_with_weights(base_expert: Expert, new_weights: dict, name: str) -> Expert:
    expert = copy.deepcopy(base_expert)
    expert.name = name
    for key, value in new_weights.items():
        expert.expert_weights[key] = value
    if expert.expert_config is not None:
        expert.expert_config.tie_params = None
    return expert


def _apply_dare_to_weights(weights: dict, drop_rate: float, generator: torch.Generator):
    """Drop delta parameters with probability `drop_rate` and rescale the rest by 1/(1-p)."""
    if drop_rate <= 0.0:
        return {k: v.clone() for k, v in weights.items()}
    if drop_rate >= 1.0:
        raise ValueError("dare_drop_rate must be in [0, 1)")
    scale = 1.0 / (1.0 - drop_rate)
    dared = {}
    for key, value in weights.items():
        mask = (
            torch.rand(value.shape, generator=generator, dtype=torch.float32)
            > drop_rate
        ).to(device=value.device, dtype=value.dtype)
        dared[key] = value * mask * scale
    return dared


def _slerp_tensors(t: float, v0: torch.Tensor, v1: torch.Tensor, eps: float = 1e-8):
    """Spherical linear interpolation between two tensors of equal shape."""
    v0_flat = v0.reshape(-1).float()
    v1_flat = v1.reshape(-1).float()
    v0_norm = torch.linalg.norm(v0_flat)
    v1_norm = torch.linalg.norm(v1_flat)
    if v0_norm < eps or v1_norm < eps:
        return ((1.0 - t) * v0 + t * v1).to(v0.dtype)
    v0_n = v0_flat / v0_norm
    v1_n = v1_flat / v1_norm
    dot = torch.clamp((v0_n * v1_n).sum(), -1.0, 1.0)
    omega = torch.acos(dot)
    so = torch.sin(omega)
    if so.abs() < eps:
        return ((1.0 - t) * v0 + t * v1).to(v0.dtype)
    out = (
        torch.sin((1.0 - t) * omega) / so * v0_flat
        + torch.sin(t * omega) / so * v1_flat
    )
    return out.reshape(v0.shape).to(v0.dtype)


@dataclass
class TaskArithmeticConfig(LibraryTransformConfig):
    """Task Arithmetic (Ilharco et al.): θ' = θ_0 + λ Σ_t τ_t.

    For LoRA experts, τ_t is the adapter itself. Unlike uniform averaging this
    does **not** divide by the number of tasks; `ta_scaling` is λ.
    """

    ta_scaling: float = 1.0


@LibraryTransform.register("task_arithmetic", TaskArithmeticConfig)
class TaskArithmeticMerge(LibraryTransform):
    def __init__(self, config: TaskArithmeticConfig = None):
        super().__init__(config or TaskArithmeticConfig())

    @torch.no_grad()
    def transform(self, library) -> Expert:
        if type(library) == str:
            library = ExpertLibrary.get_expert_library(library)

        experts = [library[name] for name in library.keys()]
        logger.info("Task arithmetic over {} experts, λ={}".format(len(experts), self.config.ta_scaling))

        merged_weights = {
            k: v.clone() * self.config.ta_scaling for k, v in experts[0].expert_weights.items()
        }
        for expert in experts[1:]:
            for key, value in expert.expert_weights.items():
                merged_weights[key] = merged_weights[key] + value * self.config.ta_scaling
        return _clone_expert_with_weights(experts[0], merged_weights, "task_arithmetic_expert")


@dataclass
class DareMergeConfig(LibraryTransformConfig):
    """DARE (Yu et al., 2024): randomly drop delta weights then rescale.

    `dare_merge_method` is applied after sparsification:
      - "task_arithmetic" (DARE-TA)
      - "ties" (DARE-TIES)
      - "uniform" (DARE + weight average)
    """

    dare_drop_rate: float = 0.7
    dare_merge_method: str = "ties"
    ta_scaling: float = 1.0
    ties_top_k: float = 0.2
    dare_seed: int = 42


@LibraryTransform.register("dare_merge", DareMergeConfig)
class DareMerge(LibraryTransform):
    def __init__(self, config: DareMergeConfig = None):
        super().__init__(config or DareMergeConfig())

    @torch.no_grad()
    def transform(self, library) -> Expert:
        if type(library) == str:
            library = ExpertLibrary.get_expert_library(library)

        expert_names = list(library.keys())
        experts = [library[name] for name in expert_names]
        generator = torch.Generator().manual_seed(int(self.config.dare_seed))
        logger.info(
            "DARE drop_rate={} merge={} over {} experts".format(
                self.config.dare_drop_rate, self.config.dare_merge_method, len(experts)
            )
        )

        dared_experts = []
        for expert in experts:
            dared_weights = _apply_dare_to_weights(
                expert.expert_weights, float(self.config.dare_drop_rate), generator
            )
            dared_experts.append(
                _clone_expert_with_weights(expert, dared_weights, expert.name)
            )

        # stash dared experts in a virtual library-like dict wrapper
        class _ListLib:
            def __init__(self, items):
                self._items = {e.name: e for e in items}

            def keys(self):
                return self._items.keys()

            def __iter__(self):
                return iter(self._items)

            def __getitem__(self, name):
                return self._items[name]

        dared_lib = _ListLib(dared_experts)
        method = str(self.config.dare_merge_method).lower()
        if method in ("ties", "dare_ties"):
            return TiesMerge(TiesMergeConfig(top_k=float(self.config.ties_top_k))).transform(dared_lib)
        if method in ("uniform", "average"):
            return WeightedLinearMerge(WeightedLinearMergeConfig()).transform(dared_lib)
        if method in ("task_arithmetic", "ta", "dare_ta"):
            return TaskArithmeticMerge(
                TaskArithmeticConfig(ta_scaling=float(self.config.ta_scaling))
            ).transform(dared_lib)
        raise ValueError(f"Unknown dare_merge_method {self.config.dare_merge_method}")


@dataclass
class SlerpMergeConfig(LibraryTransformConfig):
    """Iterative spherical linear interpolation over LoRA experts (Goddard et al.)."""

    slerp_t: Optional[float] = None  # if None, use equal-weight sequential SLERP t=1/(i+1)


@LibraryTransform.register("slerp_merge", SlerpMergeConfig)
class SlerpMerge(LibraryTransform):
    def __init__(self, config: SlerpMergeConfig = None):
        super().__init__(config or SlerpMergeConfig())

    @torch.no_grad()
    def transform(self, library) -> Expert:
        if type(library) == str:
            library = ExpertLibrary.get_expert_library(library)

        experts = [library[name] for name in library.keys()]
        if len(experts) == 1:
            return copy.deepcopy(experts[0])

        logger.info("SLERP over {} experts".format(len(experts)))
        merged_weights = {k: v.clone() for k, v in experts[0].expert_weights.items()}
        for i, expert in enumerate(experts[1:], start=1):
            t = (
                float(self.config.slerp_t)
                if self.config.slerp_t is not None
                else 1.0 / (i + 1)
            )
            for key in merged_weights:
                merged_weights[key] = _slerp_tensors(t, merged_weights[key], expert.expert_weights[key])
        return _clone_expert_with_weights(experts[0], merged_weights, "slerp_expert")


@dataclass
class RegMeanMergeConfig(LibraryTransformConfig):
    """RegMean (Jin et al., 2023) on linearized LoRA deltas.

    If calibration activations are unavailable, Gram matrices default to I and
    the operator reduces to uniform averaging of task vectors.
    """

    regmean_lambda: float = 1.0
    regmean_max_samples: int = 1000


@LibraryTransform.register("regmean_merge", RegMeanMergeConfig)
class RegMeanMerge(LibraryTransform):
    def __init__(self, config: RegMeanMergeConfig = None):
        super().__init__(config or RegMeanMergeConfig())

    def _get_task_vectors(self, expert: Expert) -> dict:
        layers = sorted({k.split(".lora_")[0] for k in expert.expert_weights.keys()})
        vectors = {}
        for layer in layers:
            lora_a = expert.expert_weights[f"{layer}.lora_a"]
            lora_b = expert.expert_weights[f"{layer}.lora_b"]
            vectors[layer] = lora_a.data @ lora_b.data
        return vectors

    @torch.no_grad()
    def transform(self, library, grams: dict = None) -> dict:
        """Return {layer: merged_delta} in (in, out) orientation like UniformMergeAfter.

        `grams` is an optional mapping {expert_name: {layer: G}} where G is
        (in_features, in_features). Missing grams default to identity.
        """
        if type(library) == str:
            library = ExpertLibrary.get_expert_library(library)

        experts = [library[name] for name in library.keys()]
        logger.info("RegMean over {} experts, λ={}".format(len(experts), self.config.regmean_lambda))

        layer_names = sorted(
            {name.split(".lora_")[0] for name in experts[0].expert_weights.keys()}
        )
        task_vectors = {expert.name: self._get_task_vectors(expert) for expert in experts}

        merged = {}
        ridge = float(self.config.regmean_lambda)
        for layer in layer_names:
            num = None
            den = None
            for expert in experts:
                tau = task_vectors[expert.name][layer].float()
                gram = None
                if grams is not None:
                    gram = grams.get(expert.name, {}).get(layer)
                if gram is None:
                    gram = torch.eye(tau.shape[0], dtype=tau.dtype, device=tau.device)
                else:
                    gram = gram.float().to(tau.device)
                    if gram.shape[0] != tau.shape[0]:
                        gram = torch.eye(tau.shape[0], dtype=tau.dtype, device=tau.device)
                contrib = gram @ tau
                num = contrib if num is None else num + contrib
                den = gram.clone() if den is None else den + gram
            den = den + ridge * torch.eye(den.shape[0], dtype=den.dtype, device=den.device)
            try:
                merged[layer] = torch.linalg.solve(den, num).to(task_vectors[experts[0].name][layer].dtype)
            except Exception as exc:
                logger.warning("RegMean solve failed for %s (%s); falling back to mean", layer, exc)
                stacked = torch.stack(
                    [task_vectors[e.name][layer].float() for e in experts], dim=0
                )
                merged[layer] = stacked.mean(0).to(task_vectors[experts[0].name][layer].dtype)
        return merged


@dataclass
class OSRMMergeConfig(LibraryTransformConfig):
    """Config for OSRM (Orthogonal Subspaces for Robust model Merging).

    Zhang & Zhou, ACL 2025, https://arxiv.org/abs/2505.22934
    """

    max_samples_per_task: int = 100
    pool: str = "mean"  # mean over tokens, then over samples (Eq. 4)
    use_base_model_only: bool = True  # collect H from the pretrained backbone
    average_features: bool = True  # one mean vector per task (official impl.)
    merge_method: str = "uniform"  # uniform (task arithmetic) or ties
    ties_mask_rate: float = 0.8
    path: str = "osrm_hidden_states.pt"

    def __post_init__(self):
        # EvaluationConfig unions all LibraryTransform configs; conflicting
        # defaults (e.g. vs HiddenStateComputerConfig) stay as MultiDefaultValue.
        if not isinstance(self.max_samples_per_task, (int, float)):
            self.max_samples_per_task = 100
        else:
            self.max_samples_per_task = int(self.max_samples_per_task)


@LibraryTransform.register("osrm_merge", OSRMMergeConfig)
class OSRMMerge(LibraryTransform):
    """Post-hoc OSRM merge for already-trained LoRA experts.

    Section 5.4 of Zhang & Zhou, ACL 2025 (https://arxiv.org/abs/2505.22934):
    1. Collect pretrained hidden states H_t for every task / layer.
    2. For expert t, take the r smallest right-singular directions of H_{¬t}
       as the constrained LoRA-A subspace (Eq. 3).
    3. Re-decompose ΔW_t ≈ B̃_t Ã_t in that subspace and merge the recovered
       task vectors with task arithmetic (or TIES).

    Returns ``{layer: delta_W}`` in ``(in, out)`` orientation, same as the
    other ``*_after`` merges.
    """

    def __init__(self, config: OSRMMergeConfig = None):
        super().__init__(config or OSRMMergeConfig())

    def _update_args(self, args, default_args):
        if default_args is None:
            return
        for k, v in vars(default_args).items():
            if not hasattr(args, k):
                setattr(args, k, v)
        if hasattr(default_args, "updated_kwargs"):
            for k, v in default_args.updated_kwargs.items():
                setattr(args, k, v)

    def _get_task_vectors(self, expert):
        task_vectors = {}
        for key in expert.expert_weights.keys():
            base_layer_name = key.split(".lora_")[0]
            if base_layer_name not in task_vectors:
                task_vectors[base_layer_name] = None
        for layer in task_vectors.keys():
            lora_a = expert.expert_weights[f"{layer}.lora_a"]
            lora_b = expert.expert_weights[f"{layer}.lora_b"]
            task_vectors[layer] = lora_a.data @ lora_b.data
        return task_vectors

    def _get_lora_rank(self, expert, layer):
        return expert.expert_weights[f"{layer}.lora_a"].shape[-1]

    @staticmethod
    def smallest_right_singular_vectors(H: torch.Tensor, rank: int) -> torch.Tensor:
        """Ã^T with shape ``(n, r)``: last ``r`` right singular vectors of H.

        Paper Eq. (3) / official ``peft_analytical_init``: SVD of H, then
        ``V[:, n-r:n]``. MTTL stores LoRA-A as ``(in, r)``, i.e. Ã^T.
        """
        if H.ndim != 2:
            raise ValueError(f"Expected H of shape (k, n), got {tuple(H.shape)}")
        n = H.shape[-1]
        rank = min(int(rank), n)
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        orig_dtype = H.dtype
        device = H.device
        H32 = H.detach().to(device=device, dtype=torch.float32)
        # full_matrices=True so Vt is (n, n) even when k < n
        _, _, Vt = torch.linalg.svd(H32, full_matrices=True)
        return Vt[-rank:, :].T.to(dtype=orig_dtype)

    @staticmethod
    def reproject_delta(delta_W: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """Least-squares recovery ΔW̃ = A (A^+ ΔW), paper Section 5.4.

        ``delta_W`` is MTTL ``lora_a @ lora_b`` with shape ``(in, out)``.
        ``A`` has orthonormal columns of shape ``(in, r)``.
        """
        A = A.to(device=delta_W.device, dtype=delta_W.dtype)
        B = A.T @ delta_W
        return A @ B

    def _pool_hidden(self, hidden_state, attention_mask, device):
        if hidden_state.ndim == 2:
            return hidden_state
        if hidden_state.ndim != 3:
            raise ValueError(
                f"Unexpected hidden state rank {hidden_state.ndim}, expected 2 or 3"
            )
        bs = hidden_state.size(0)
        if self.config.pool == "last":
            if attention_mask is None:
                return hidden_state[:, -1]
            last_token_idx = attention_mask.sum(1).to(hidden_state.device) - 1
            bs_idx = torch.arange(bs, device=hidden_state.device)
            return hidden_state[bs_idx, last_token_idx]
        if self.config.pool == "mean":
            if attention_mask is None:
                return hidden_state.mean(dim=1)
            mask = attention_mask.to(hidden_state.device).unsqueeze(-1)
            denom = mask.sum(1).clamp(min=1.0)
            return (hidden_state * mask).sum(1) / denom
        raise ValueError(f"Unknown pool={self.config.pool}")

    def _track_hidden_states(self, model, device="cpu"):
        model.container = {}

        def build_hook(name):
            def retrieve_input(module, input, output):
                model.container[name] = input[0].detach().to(device)

            return retrieve_input

        handles = []
        for container in model.experts_containers:
            handles.append(
                container.register_forward_hook(build_hook(container.layer_name))
            )
        return handles

    def _retrieve_hidden_states(self, model):
        keys = list(model.container.keys())
        values = [model.container[k] for k in keys]
        for key in keys:
            del model.container[key]
        return {k: v for k, v in zip(keys, values)}

    @torch.no_grad()
    def _encode_expert_hidden_states(self, model, expert, default_args, device):
        from mttl.arguments import ExpertConfig

        training_config = ExpertConfig.from_dict(expert.training_config)
        self._update_args(training_config, default_args)
        training_config.dataset = expert.expert_info.dataset
        n_tasks = 1
        if expert.expert_info.expert_task_name:
            train_tasks = expert.expert_info.expert_task_name.split(",")
            training_config.finetune_task_name = ",".join(train_tasks)
            n_tasks = len(train_tasks)
        training_config.subsample_train = int(self.config.max_samples_per_task) * n_tasks
        training_config.train_batch_size = (
            default_args.predict_batch_size if default_args is not None else 4
        )

        dm = get_datamodule(training_config)
        dataloader = dm.train_dataloader()
        device_model = next(model.parameters()).device

        summed = defaultdict(lambda: 0.0)
        stacked = defaultdict(list)
        count = 0

        pbar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"OSRM hidden states [{expert.name}]",
        )
        for _, batch in pbar:
            batch = transfer_batch_to_device(batch, device_model)
            model.forward(**batch)
            hidden_states = self._retrieve_hidden_states(model)
            attention_mask = batch.get("attention_mask", None)
            for layer, hidden_state in hidden_states.items():
                pooled = self._pool_hidden(hidden_state, attention_mask, device)
                if self.config.average_features:
                    summed[layer] += pooled.sum(0)
                else:
                    stacked[layer].append(pooled.detach().cpu())
            if "input_ids" in batch:
                count += batch["input_ids"].size(0)
            elif hidden_states:
                count += next(iter(hidden_states.values())).size(0)

        features = {}
        if self.config.average_features:
            if count == 0:
                raise ValueError(f"No samples encoded for expert {expert.name}")
            for layer, value in summed.items():
                features[layer] = (value / count).detach().cpu().unsqueeze(0)
        else:
            for layer, chunks in stacked.items():
                features[layer] = torch.cat(chunks, dim=0)
        return features

    @torch.no_grad()
    def collect_hidden_states(self, library, experts, default_args=None):
        """Collect pretrained per-layer features H_t for every expert."""
        from mttl.arguments import ExpertConfig

        first = experts[0]
        training_config = ExpertConfig.from_dict(first.training_config)
        self._update_args(training_config, default_args)

        device_map = getattr(training_config, "device_map", None) or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        model = MultiExpertModel(
            MultiExpertModelConfig(base_model=training_config.model),
            device_map=device_map,
        )
        # Containers give us layer-aligned hooks; disable adapters so H is
        # computed on the pretrained backbone (Algorithm 1 / official code).
        model.add_expert_instance(first, is_default=True)
        if self.config.use_base_model_only:
            for container in model.experts_containers:
                container.disable()

        handles = self._track_hidden_states(model, device="cpu")
        output = {}
        try:
            for expert in experts:
                output[expert.name] = self._encode_expert_hidden_states(
                    model, expert, default_args, device="cpu"
                )
        finally:
            for handle in handles:
                handle.remove()
            del model
        return output

    def _other_task_features(self, hidden_states, expert_name, layer):
        rows = []
        for name, per_layer in hidden_states.items():
            if name == expert_name:
                continue
            if layer not in per_layer:
                continue
            feat = per_layer[layer]
            if feat.ndim == 1:
                feat = feat.unsqueeze(0)
            rows.append(feat)
        if not rows:
            return None
        return torch.cat(rows, dim=0)

    def _merge_layer_vectors(self, task_vectors):
        if self.config.merge_method == "uniform":
            return sum(task_vectors) / len(task_vectors)
        if self.config.merge_method == "ties":
            ties = TiesMergeAfter(
                TiesMergeAfterConfig(mask_rate=self.config.ties_mask_rate)
            )
            masked = [
                ties.mask_smallest_magnitude_param_values(
                    tv, self.config.ties_mask_rate
                )
                for tv in task_vectors
            ]
            signs = ties.get_param_signs(masked)
            return ties.disjoint_merge(masked, signs)
        raise ValueError(
            f"Unknown OSRM merge_method={self.config.merge_method}. "
            "Use 'uniform' or 'ties'."
        )

    @torch.no_grad()
    def transform(
        self,
        library,
        persist=False,
        recompute=False,
        default_args=None,
        hidden_states=None,
    ):
        if type(library) == str:
            library = ExpertLibrary.get_expert_library(library)

        expert_names = list(library.keys())
        experts = [library[name] for name in expert_names]
        logger.info(
            "Merging {} experts using OSRM ({})".format(
                len(experts), self.config.merge_method
            )
        )

        layer_names = [
            name.split(".lora_")[0] for name in experts[0].expert_weights.keys()
        ]
        layer_names = sorted(list(set(layer_names)))

        if hidden_states is None:
            if (
                self.config.path
                and os.path.exists(self.config.path)
                and not recompute
            ):
                logger.info(f"Loading cached OSRM hidden states from {self.config.path}")
                hidden_states = torch.load(self.config.path, map_location="cpu")
            else:
                hidden_states = self.collect_hidden_states(
                    library, experts, default_args=default_args
                )
                if self.config.path:
                    os.makedirs(
                        os.path.dirname(os.path.abspath(self.config.path)) or ".",
                        exist_ok=True,
                    )
                    torch.save(hidden_states, self.config.path)
                    logger.info(f"Saved OSRM hidden states to {self.config.path}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        recovered_experts = {}
        for expert in experts:
            deltas = self._get_task_vectors(expert)
            recovered = {}
            for layer in layer_names:
                H_neg = self._other_task_features(hidden_states, expert.name, layer)
                delta = deltas[layer]
                if H_neg is None:
                    logger.warning(
                        f"No out-of-task features for {expert.name}/{layer}; "
                        "keeping the original task vector"
                    )
                    recovered[layer] = delta
                    continue
                rank = self._get_lora_rank(expert, layer)
                H_neg = H_neg.to(device=device)
                A = self.smallest_right_singular_vectors(H_neg, rank)
                recovered[layer] = self.reproject_delta(delta.to(device), A).cpu()
            recovered_experts[expert.name] = recovered

        task_merged_vectors = {}
        for layer in layer_names:
            task_vectors = [
                recovered_experts[expert.name][layer] for expert in experts
            ]
            logger.info(
                f"OSRM merging layer {layer} with {len(task_vectors)} recovered "
                f"task vectors ({self.config.merge_method})"
            )
            task_merged_vectors[layer] = self._merge_layer_vectors(task_vectors)
        return task_merged_vectors


@dataclass
class IterISMergeConfig(LibraryTransformConfig):
    """Config for IterIS (Iterative Inference-Solving Alignment).

    Chen et al., CVPR 2025, https://arxiv.org/abs/2411.15231
    Official: https://github.com/HKUST-LongGroup/IterIS-merging
    """

    max_iter: int = 10
    max_samples_per_task: int = 50
    max_tokens_per_task: int = 1024
    alpha_1: float = 1e-7
    alpha_2: float = 1e-7
    reg_coef: float = 0.0
    with_pretrain_matrix: int = 0
    include_lora_scaling: bool = True
    manual_coef: list = None
    path: str = "iteris_hidden_states.pt"

    def __post_init__(self):
        if not isinstance(self.max_samples_per_task, (int, float)):
            self.max_samples_per_task = 50
        else:
            self.max_samples_per_task = int(self.max_samples_per_task)
        if not isinstance(self.max_iter, (int, float)):
            self.max_iter = 10
        else:
            self.max_iter = int(self.max_iter)
        if not isinstance(self.max_tokens_per_task, (int, float)):
            self.max_tokens_per_task = 1024
        else:
            self.max_tokens_per_task = int(self.max_tokens_per_task)
        if not isinstance(self.path, str):
            self.path = "iteris_hidden_states.pt"


@LibraryTransform.register("iteris_merge", IterISMergeConfig)
class IterISMerge(LibraryTransform):
    """Post-hoc IterIS merge for already-trained LoRA experts.

    Chen et al., CVPR 2025 (https://arxiv.org/abs/2411.15231):
    1. Collect layer inputs X_t on each task from that task's LoRA-merged model.
    2. Closed-form solve for a merged delta W* that aligns LoRA outputs
       W_t X_t with the current merged inference activations W* X̃_t.
    3. Apply W* to the backbone, re-collect X̃, and iterate.

    Returns ``{layer: delta_W}`` in ``(in, out)`` orientation, same as the
    other ``*_after`` merges. Official PEFT uses ``(out, in)``; we transpose
    at the boundary.
    """

    def __init__(self, config: IterISMergeConfig = None):
        super().__init__(config or IterISMergeConfig())

    def _update_args(self, args, default_args):
        if default_args is None:
            return
        for k, v in vars(default_args).items():
            if not hasattr(args, k):
                setattr(args, k, v)
        if hasattr(default_args, "updated_kwargs"):
            for k, v in default_args.updated_kwargs.items():
                setattr(args, k, v)

    def _lora_scaling(self, expert):
        if not self.config.include_lora_scaling:
            return 1.0
        cfg = getattr(expert, "expert_config", None)
        rank = getattr(cfg, "lora_rank", None) if cfg is not None else None
        alpha = getattr(cfg, "lora_alpha", None) if cfg is not None else None
        if rank is None or alpha is None or rank == 0:
            return 1.0
        return float(alpha) / float(rank)

    def _get_task_vectors(self, expert):
        """LoRA deltas in official ``(out, in)`` orientation."""
        scaling = self._lora_scaling(expert)
        task_vectors = {}
        for key in expert.expert_weights.keys():
            base_layer_name = key.split(".lora_")[0]
            if base_layer_name not in task_vectors:
                task_vectors[base_layer_name] = None
        for layer in task_vectors.keys():
            lora_a = expert.expert_weights[f"{layer}.lora_a"]
            lora_b = expert.expert_weights[f"{layer}.lora_b"]
            # MTTL: input @ A @ B, Linear.weight is (out, in)
            task_vectors[layer] = scaling * (lora_a.data @ lora_b.data).T
        return task_vectors

    def _track_hidden_states(self, model, device="cpu"):
        model.container = {}

        def build_hook(name):
            def retrieve_input(module, input, output):
                model.container[name] = input[0].detach().to(device)

            return retrieve_input

        handles = []
        for container in model.experts_containers:
            handles.append(
                container.register_forward_hook(build_hook(container.layer_name))
            )
        return handles

    def _retrieve_hidden_states(self, model):
        keys = list(model.container.keys())
        values = [model.container[k] for k in keys]
        for key in keys:
            del model.container[key]
        return {k: v for k, v in zip(keys, values)}

    def _flatten_hidden(self, hidden_state, attention_mask):
        if hidden_state.ndim == 2:
            return hidden_state
        if hidden_state.ndim != 3:
            raise ValueError(
                f"Unexpected hidden state rank {hidden_state.ndim}, expected 2 or 3"
            )
        if attention_mask is not None and attention_mask.shape[:2] == hidden_state.shape[:2]:
            mask = attention_mask.bool().to(hidden_state.device)
            return hidden_state[mask]
        return hidden_state.reshape(-1, hidden_state.shape[-1])

    def _snapshot_base_weights(self, model):
        w0 = {}
        for container in model.experts_containers:
            w0[container.layer_name] = container.layer.weight.detach().clone().cpu()
        return w0

    def _restore_base_weights(self, model, w0):
        for container in model.experts_containers:
            name = container.layer_name
            if name not in w0:
                continue
            container.layer.weight.data.copy_(
                w0[name].to(
                    device=container.layer.weight.device,
                    dtype=container.layer.weight.dtype,
                )
            )

    def _apply_deltas(self, model, deltas_out_in, w0, replace=False):
        for container in model.experts_containers:
            name = container.layer_name
            if name not in deltas_out_in:
                continue
            delta = deltas_out_in[name].to(
                device=container.layer.weight.device,
                dtype=container.layer.weight.dtype,
            )
            if replace:
                container.layer.weight.data.copy_(delta)
            else:
                base = w0[name].to(
                    device=container.layer.weight.device,
                    dtype=container.layer.weight.dtype,
                )
                container.layer.weight.data.copy_(base + delta)

    def _stack_layer_features(self, hidden_states, expert_names, layer):
        feats = []
        for name in expert_names:
            if name not in hidden_states or layer not in hidden_states[name]:
                raise ValueError(f"Missing IterIS features for {name}/{layer}")
            feat = hidden_states[name][layer]
            if feat.ndim == 1:
                feat = feat.unsqueeze(0)
            feats.append(feat)
        min_t = min(f.shape[0] for f in feats)
        if min_t == 0:
            raise ValueError(f"No tokens collected for layer {layer}")
        return torch.stack([f[:min_t] for f in feats], dim=0)

    @staticmethod
    def _reg_math(term, alpha):
        eye = torch.eye(term.size(-1), dtype=term.dtype, device=term.device)
        return term + alpha.view(-1, 1, 1) * eye

    @staticmethod
    def solution_matrix(
        W_list,
        X_list,
        X_tilde_list,
        coef_list,
        manual_coef,
        alpha_1=1e-7,
        alpha_2=1e-7,
        reg_coef=0.0,
    ):
        """Closed-form IterIS step (official ``solution_matrix``).

        ``W_list`` is ``(N, out, in)``, ``X_*`` are ``(N, tokens, in)``.
        Returns ``W*`` with shape ``(out, in)``.
        """
        n_tasks = W_list.shape[0]
        weights = (coef_list * manual_coef).view(n_tasks, 1, 1)
        x_tilde = (1.0 - reg_coef) * X_tilde_list + reg_coef * X_list

        x_x_tilde = torch.matmul(X_list.transpose(-1, -2), x_tilde)
        x_x_tilde = IterISMerge._reg_math(
            x_x_tilde, torch.norm(x_x_tilde, p="fro", dim=[-2, -1]) * alpha_1
        )
        x_tilde_x_tilde = torch.matmul(x_tilde.transpose(-1, -2), x_tilde)
        x_tilde_x_tilde = IterISMerge._reg_math(
            x_tilde_x_tilde,
            torch.norm(x_tilde_x_tilde, p="fro", dim=[-2, -1]) * alpha_2,
        )

        term1 = torch.sum(torch.matmul(W_list, x_x_tilde) * weights, dim=0).double()
        term2 = torch.sum(x_tilde_x_tilde * weights, dim=0).double()
        try:
            result = torch.linalg.solve(term2.T, term1.T).T
        except RuntimeError:
            ridge = 1e-5 * torch.eye(
                term2.size(0), dtype=term2.dtype, device=term2.device
            )
            result = torch.linalg.solve(term2.T + ridge, term1.T).T
        return result.to(dtype=W_list.dtype)

    def _task_coefficients(self, W_list, X_list, w0_layer):
        merge_w = W_list
        if w0_layer is not None:
            merge_w = W_list + w0_layer.to(device=W_list.device, dtype=W_list.dtype)
        # official: ||W||_F^2 / sum_i ||X_i W_i^T||_F^2  (shared denominator)
        numer = torch.norm(merge_w, p="fro", dim=[-2, -1]) ** 2
        outputs = torch.matmul(X_list, merge_w.transpose(1, 2))
        denom = torch.sum(torch.norm(outputs, p="fro", dim=[-2, -1]) ** 2)
        return numer / denom.clamp_min(1e-12)

    def _manual_coef(self, n_tasks, device, dtype):
        manual = getattr(self.config, "manual_coef", None)
        if manual is None:
            return torch.ones(n_tasks, device=device, dtype=dtype)
        values = torch.as_tensor(manual, device=device, dtype=dtype).flatten()
        if values.numel() == 1:
            return values.repeat(n_tasks)
        if values.numel() != n_tasks:
            logger.warning(
                "IterIS manual_coef length %s != %s tasks; using ones",
                values.numel(),
                n_tasks,
            )
            return torch.ones(n_tasks, device=device, dtype=dtype)
        return values

    @torch.no_grad()
    def _encode_batches(self, model, batches, device, desc):
        device_model = next(model.parameters()).device
        stacked = defaultdict(list)
        pbar = tqdm(enumerate(batches), total=len(batches), desc=desc)
        for _, batch in pbar:
            batch = {
                k: v.to(device_model) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }
            model.forward(**batch)
            hidden_states = self._retrieve_hidden_states(model)
            attention_mask = batch.get("attention_mask", None)
            for layer, hidden_state in hidden_states.items():
                stacked[layer].append(
                    self._flatten_hidden(hidden_state, attention_mask).detach().cpu()
                )
        features = {}
        max_tokens = int(self.config.max_tokens_per_task)
        for layer, chunks in stacked.items():
            feat = torch.cat(chunks, dim=0)
            if feat.size(0) > max_tokens:
                feat = feat[:max_tokens]
            features[layer] = feat
        return features

    @torch.no_grad()
    def _collect_expert_batches(self, expert, default_args):
        from mttl.arguments import ExpertConfig

        training_config = ExpertConfig.from_dict(expert.training_config)
        self._update_args(training_config, default_args)
        training_config.dataset = expert.expert_info.dataset
        n_tasks = 1
        if expert.expert_info.expert_task_name:
            train_tasks = expert.expert_info.expert_task_name.split(",")
            training_config.finetune_task_name = ",".join(train_tasks)
            n_tasks = len(train_tasks)
        training_config.subsample_train = int(self.config.max_samples_per_task) * n_tasks
        training_config.train_batch_size = (
            default_args.predict_batch_size if default_args is not None else 4
        )
        dm = get_datamodule(training_config)
        batches = []
        for batch in dm.train_dataloader():
            cpu_batch = {}
            for key, value in batch.items():
                if torch.is_tensor(value):
                    cpu_batch[key] = value.detach().cpu()
                else:
                    cpu_batch[key] = value
            batches.append(cpu_batch)
        return batches

    @torch.no_grad()
    def collect_hidden_states(self, library, experts, default_args=None):
        """Collect per-task layer inputs from each LoRA-merged backbone."""
        from mttl.arguments import ExpertConfig

        first = experts[0]
        training_config = ExpertConfig.from_dict(first.training_config)
        self._update_args(training_config, default_args)
        device_map = getattr(training_config, "device_map", None) or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        model = MultiExpertModel(
            MultiExpertModelConfig(base_model=training_config.model),
            device_map=device_map,
        )
        model.add_expert_instance(first, is_default=True)
        model.disable_adapters()
        handles = self._track_hidden_states(model, device="cpu")
        w0 = self._snapshot_base_weights(model)
        hidden_states = {}
        batches = {}
        try:
            for expert in experts:
                logger.info("IterIS: collecting mid-features for %s", expert.name)
                batches[expert.name] = self._collect_expert_batches(
                    expert, default_args
                )
                deltas = self._get_task_vectors(expert)
                self._apply_deltas(model, deltas, w0, replace=False)
                hidden_states[expert.name] = self._encode_batches(
                    model,
                    batches[expert.name],
                    device="cpu",
                    desc=f"IterIS hidden states [{expert.name}]",
                )
                self._restore_base_weights(model, w0)
        finally:
            for handle in handles:
                handle.remove()
        return hidden_states, batches, model, w0

    def _solve_once(self, experts, layer_names, hidden_states, hidden_tilde, w0, device):
        expert_names = [expert.name for expert in experts]
        n_tasks = len(experts)
        merged = {}
        for layer in tqdm(layer_names, desc="IterIS closed-form"):
            X = self._stack_layer_features(hidden_states, expert_names, layer).to(
                device=device, dtype=torch.float32
            )
            X_tilde = self._stack_layer_features(hidden_tilde, expert_names, layer).to(
                device=device, dtype=torch.float32
            )
            W = torch.stack(
                [self._get_task_vectors(expert)[layer] for expert in experts], dim=0
            ).to(device=device, dtype=torch.float32)
            w0_layer = None
            if w0 is not None and layer in w0:
                w0_layer = w0[layer].to(device=device, dtype=torch.float32)
            if int(self.config.with_pretrain_matrix) == 1:
                if w0_layer is None:
                    raise ValueError(
                        "with_pretrain_matrix=1 requires base weights; "
                        "cannot use the hidden_states-only path."
                    )
                W_solve = W + w0_layer
            else:
                W_solve = W
            coef = self._task_coefficients(W, X, w0_layer)
            manual = self._manual_coef(n_tasks, device=device, dtype=W.dtype)
            W_star = self.solution_matrix(
                W_solve,
                X,
                X_tilde,
                coef,
                manual,
                alpha_1=float(self.config.alpha_1),
                alpha_2=float(self.config.alpha_2),
                reg_coef=float(self.config.reg_coef),
            )
            if int(self.config.with_pretrain_matrix) == 1:
                W_star = W_star - w0_layer.to(device=W_star.device, dtype=W_star.dtype)
            merged[layer] = W_star.cpu()
        return merged

    @torch.no_grad()
    def transform(
        self,
        library,
        persist=False,
        recompute=False,
        default_args=None,
        hidden_states=None,
        batches=None,
    ):
        if type(library) == str:
            library = ExpertLibrary.get_expert_library(library)

        expert_names = list(library.keys())
        experts = [library[name] for name in expert_names]
        logger.info("Merging {} experts using IterIS".format(len(experts)))

        layer_names = [
            name.split(".lora_")[0] for name in experts[0].expert_weights.keys()
        ]
        layer_names = sorted(list(set(layer_names)))

        model = None
        w0 = None
        handles = []
        cache = None
        if hidden_states is None and self.config.path and os.path.exists(self.config.path) and not recompute:
            logger.info("Loading cached IterIS hidden states from %s", self.config.path)
            cache = torch.load(self.config.path, map_location="cpu")
            if isinstance(cache, dict) and "hidden_states" in cache:
                hidden_states = cache["hidden_states"]
                batches = cache.get("batches")
            else:
                hidden_states = cache

        if hidden_states is None:
            hidden_states, batches, model, w0 = self.collect_hidden_states(
                library, experts, default_args=default_args
            )
            if self.config.path:
                os.makedirs(
                    os.path.dirname(os.path.abspath(self.config.path)) or ".",
                    exist_ok=True,
                )
                torch.save(
                    {"hidden_states": hidden_states, "batches": batches},
                    self.config.path,
                )
                logger.info("Saved IterIS hidden states to %s", self.config.path)

        available = None
        if hidden_states:
            available = set.intersection(
                *(set(per_layer.keys()) for per_layer in hidden_states.values())
            )
            layer_names = [layer for layer in layer_names if layer in available]

        if model is not None and int(self.config.max_iter) > 1 and not handles:
            # collect_hidden_states removes its hooks; re-register for X̃ passes.
            handles = self._track_hidden_states(model, device="cpu")
        elif model is None and batches and int(self.config.max_iter) > 1:
            from mttl.arguments import ExpertConfig

            first = experts[0]
            training_config = ExpertConfig.from_dict(first.training_config)
            self._update_args(training_config, default_args)
            device_map = getattr(training_config, "device_map", None) or (
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            model = MultiExpertModel(
                MultiExpertModelConfig(base_model=training_config.model),
                device_map=device_map,
            )
            model.add_expert_instance(first, is_default=True)
            model.disable_adapters()
            handles = self._track_hidden_states(model, device="cpu")
            w0 = self._snapshot_base_weights(model)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        max_iter = int(self.config.max_iter)
        if model is None or not batches:
            max_iter = 1

        hidden_tilde = hidden_states
        merged_out_in = None
        try:
            for step in range(max_iter):
                logger.info("IterIS iteration %s / %s", step + 1, max_iter)
                merged_out_in = self._solve_once(
                    experts, layer_names, hidden_states, hidden_tilde, w0, device
                )
                if step == max_iter - 1 or model is None:
                    break
                self._apply_deltas(
                    model,
                    merged_out_in,
                    w0,
                    replace=False,
                )
                hidden_tilde = {}
                for expert in experts:
                    hidden_tilde[expert.name] = self._encode_batches(
                        model,
                        batches[expert.name],
                        device="cpu",
                        desc=f"IterIS X̃ [{expert.name}] iter={step + 1}",
                    )
                self._restore_base_weights(model, w0)
        finally:
            for handle in handles:
                handle.remove()
            if model is not None:
                del model

        # Return (in, out) so task_vector_apply matches Linear.weight.
        return {layer: delta.T.contiguous() for layer, delta in merged_out_in.items()}


@dataclass
class KnotMergeConfig(WeightedLinearMergeConfig):
    path: str = "knot_ingredients.pt"  # path to store SVD components
    # Coordinate merge in the shared right-subspace: "ties" (KnOTS / SOATA-TIES)
    # or "linear" (average the aligned V_k, SOATA-Linear).
    merge_method: str = "ties"
    # Truncate the joint SVD to the top-R components. -1 keeps every singular
    # value above 1e-5 (full numerical rank of the concatenated adapter).
    retained_rank: int = -1


@LibraryTransform.register("weighted_knot_merge", KnotMergeConfig)
class KnotMerge(LibraryTransform):
    """
    Computes a weighted KnoT merge for LoRA experts as in https://arxiv.org/pdf/2410.19735

    Joint SVD aligns task adapters onto a shared left basis U; merging then
    operates on the aligned right coordinates. ``merge_method="ties"`` is the
    original KnOTS operator; ``merge_method="linear"`` averages coordinates.
    """

    def __init__(self, config: KnotMergeConfig = None):
        super().__init__(config or KnotMergeConfig())
        self.ingredients = None

    def transform(self, library, recompute=False):
        if type(library) == str:
            library = ExpertLibrary.get_expert_library(library)
        # TODO: this should probably be stored in the library instead of the local path.
        # Current libary.add_auxiliary_data requires that aux data is associated with an expert, this is not associated with any expert.
        if not os.path.exists(self.config.path) or recompute:
            U, task_Ss, task_sVs, UsV_dict, expert_names = self.apply_svd(library)
            self.ingredients = {
                "U": U,
                "task_Ss": task_Ss,
                "task_sVs": task_sVs,  # premultiplied with s, cause Vs alone do not have the scale info.
                "UsV_dict": UsV_dict,
                "expert_names": expert_names,
            }

            parent = os.path.dirname(self.config.path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            torch.save(self.ingredients, self.config.path)
        self.ingredients = torch.load(
            self.config.path, map_location="cpu", weights_only=False
        )
        task_sVs = self.ingredients["task_sVs"]
        U = self.ingredients["U"]
        expert_names = self.ingredients.get("expert_names")
        retained = int(getattr(self.config, "retained_rank", -1) or -1)
        merge_method = str(getattr(self.config, "merge_method", "ties") or "ties").lower()

        if retained > 0:
            U = {k: v[:, : min(retained, v.shape[1])] for k, v in U.items()}
            task_sVs = [
                {k: v[: min(retained, v.shape[0]), :] for k, v in params.items()}
                for params in task_sVs
            ]

        usv = self.ingredients.get("UsV_dict") or {}
        if usv:
            ratios = []
            for pack in usv.values():
                s = pack["s"].detach().float().cpu()
                keep = s.numel() if retained <= 0 else min(retained, s.numel())
                denom = s.pow(2).sum().clamp_min(1e-12)
                ratios.append((s[:keep].pow(2).sum() / denom).item())
            logger.info(
                "KnoT mean Frobenius energy ratio at R=%s: %.4f",
                "full" if retained <= 0 else retained,
                sum(ratios) / max(len(ratios), 1),
            )

        param_names = list(task_sVs[0].keys())
        state_dict = {}
        if merge_method in ("ties", "knots"):
            ties_mergert = TiesMerge()
            expert_vectors = torch.stack(
                [
                    torch.nn.utils.parameters_to_vector(
                        list(params[k] for k in param_names)
                    )
                    for params in task_sVs
                ],
                dim=0,
            )
            per_exp_th = _safe_quantile(
                expert_vectors.abs(), 1.0 - ties_mergert.config.top_k, dim=1
            )
            for p_name in param_names:
                expert_weights = torch.stack(
                    [expert[p_name] for expert in task_sVs], dim=0
                )
                TH = per_exp_th.view(-1, *((1,) * (expert_weights.ndim - 1)))
                final_param, _, _ = ties_mergert.merge_param(TH, expert_weights)
                state_dict[p_name] = U[p_name] @ final_param  # out_features, in_features
        elif merge_method in ("linear", "uniform", "average"):
            coord_w = None
            if self.config.weights is not None and expert_names is not None:
                coord_w = torch.tensor(
                    [self.config.weights[n] for n in expert_names], dtype=torch.float32
                )
                coord_w = coord_w / coord_w.sum().clamp_min(1e-12)
            for p_name in param_names:
                expert_weights = torch.stack(
                    [expert[p_name] for expert in task_sVs], dim=0
                )
                if coord_w is None:
                    final_param = expert_weights.mean(0)
                else:
                    w = coord_w.to(dtype=expert_weights.dtype).view(
                        -1, *((1,) * (expert_weights.ndim - 1))
                    )
                    final_param = (expert_weights * w).sum(0)
                state_dict[p_name] = U[p_name] @ final_param
        else:
            raise ValueError(f"Unknown KnotMerge merge_method={merge_method}")

        return state_dict

    def apply_svd(self, library):
        """
        Reused from https://github.com/gstoica27/KnOTS/blob/main/task_merger.py
        """
        expert_names = list(library.keys())
        experts = [library[name] for name in expert_names]

        logger.info("Knotting {} experts".format(len(experts)))

        base_expert = copy.deepcopy(experts[0])
        base_expert.name = "weighted_expert"

        if self.config.weights is not None:
            assert set(self.config.weights.keys()) == set(
                expert_names
            ), "Weights must have the same keys as the experts"
            if not (1 - 1e-6) <= sum(self.config.weights.values()) <= (1 + 1e-6):
                logger.warning(
                    "Weights do not sum to 1.0, please make sure this is intended"
                )

        layers = set(
            [
                k.split(".lora")[0]
                for k in base_expert.expert_weights.keys()
                if ".lora" in k
            ]
        )
        d_in, d_out = (
            base_expert.expert_weights[f"{list(layers)[0]}.lora_a"].shape[0],
            base_expert.expert_weights[f"{list(layers)[0]}.lora_b"].shape[1],
        )

        UsV_dict = {}
        basis_dict = {}  # basis for reconstruction
        s_compositions_dict = [
            dict() for _ in range(len(experts))
        ]  # singular values composition information per task
        V_compositions_dict = [
            dict() for _ in range(len(experts))
        ]  # basis composition information per task

        for layer in layers:
            Ws = []
            logger.info(f"Computing KnoT merge for layer {layer}")
            # retreieve lora A and B from all experts
            # create W
            for _, expert in zip(expert_names, experts):
                # Validate that the expert is compatible
                assert (
                    type(expert.expert_info.expert_config) == LoRAConfig
                ), "Expert configs must be the same type"
                assert set(expert.expert_weights.keys()) == set(
                    base_expert.expert_weights.keys()
                ), "Expert weights must have the same keys"
                lora_a = expert.expert_weights[f"{layer}.lora_a"]
                lora_b = expert.expert_weights[f"{layer}.lora_b"]
                rank = expert.expert_config.lora_rank
                assert (
                    lora_b.shape[0] == lora_a.shape[1] == rank
                ), "lora_a and lora_a must have the same rank as the expert"
                W = (lora_a @ lora_b).T  # out_features, in_features
                Ws.append(W)

            # SVD
            device = "cuda" if torch.cuda.is_available() else "cpu"
            W_l = torch.cat(Ws, dim=1).to(device)
            U, s, Vt = torch.linalg.svd(W_l, full_matrices=False)
            U = U[:, s > 1e-5]
            Vt = Vt[s > 1e-5]
            s = s[s > 1e-5]
            UsV_dict[layer] = {
                "U": deepcopy(U.cpu()),
                "s": deepcopy(s.cpu()),
                "V": deepcopy(Vt.cpu()),
            }
            # Set all s to be the same scale
            s[s <= 1e-5] = 0
            cat_hidden_dim = Vt.shape[1] // len(experts)

            basis_dict[layer] = U.cpu()
            sV_concat = Vt
            Vs = list(torch.split(sV_concat, cat_hidden_dim, dim=1))
            for idx, V in enumerate(Vs):
                V = (
                    torch.diag(s) @ V
                )  # WE use Ties merging hat relies on magnitde info, which is not present in Vs only. Comment from original code base: Simple and safe for all merging methods we use.
                s_model = torch.ones_like(s)

                s_compositions_dict[idx][layer] = s_model.cpu()
                V_compositions_dict[idx][layer] = V.cpu()
        return basis_dict, s_compositions_dict, V_compositions_dict, UsV_dict, expert_names


@dataclass
class SoataMergeConfig(KnotMergeConfig):
    # Optional SOATA-specific post-processing: match each merged layer norm to the
    # weighted mean norm of source task deltas (computed in out,in coordinates).
    preserve_energy: bool = True
    eps: float = 1e-12


@LibraryTransform.register("soata_merge", SoataMergeConfig)
class SoataMerge(KnotMerge):
    """
    SOATA wrapper over KnotMerge with an explicit method identity and optional
    energy-preserving post-process.

    This class intentionally differs from KnotMerge: after coordinate merging it
    can re-scale each layer to preserve the weighted mean adapter energy of the
    source experts.
    """

    def __init__(self, config: SoataMergeConfig = None):
        super().__init__(config or SoataMergeConfig())

    @torch.no_grad()
    def transform(self, library, recompute=False):
        merged = super().transform(library, recompute=recompute)
        if not bool(getattr(self.config, "preserve_energy", True)):
            return merged

        if type(library) == str:
            library = ExpertLibrary.get_expert_library(library)

        expert_names = list(library.keys())
        coord_w = None
        if self.config.weights is not None:
            coord_w = torch.tensor(
                [float(self.config.weights[n]) for n in expert_names], dtype=torch.float32
            )
            coord_w = coord_w / coord_w.sum().clamp_min(float(self.config.eps))

        # Match ||ΔW_merged||_F to weighted mean ||ΔW_k||_F per layer.
        layer_to_target = {}
        for layer in merged.keys():
            per_exp = []
            for name in expert_names:
                lora_a = library[name].expert_weights[f"{layer}.lora_a"]
                lora_b = library[name].expert_weights[f"{layer}.lora_b"]
                # Knot/Soata merged deltas are in (out, in), so use the same layout.
                per_exp.append((lora_a @ lora_b).T.float().norm())
            norms = torch.stack(per_exp)
            target = norms.mean() if coord_w is None else (coord_w * norms).sum()
            layer_to_target[layer] = float(target.item())

        out = {}
        for layer, delta in merged.items():
            cur = float(delta.float().norm().item())
            target = layer_to_target[layer]
            if cur <= float(self.config.eps) or target <= float(self.config.eps):
                out[layer] = delta
                continue
            scale = target / cur
            out[layer] = delta * scale
        return out


@dataclass
class TiesMergeConfig(LibraryTransformConfig):
    top_k: float = 0.2
    only_sparsify: bool = False


@LibraryTransform.register("ties_merge", TiesMergeConfig)
class TiesMerge(LibraryTransform):
    """
    Computes a uniform weight mixture across experts of a given library
    """

    def __init__(self, config: TiesMergeConfig = None):
        super().__init__(config or TiesMergeConfig())

        assert self.config.top_k > 0.0 and self.config.top_k <= 1.0

    @torch.no_grad()
    def merge_param(self, TH, expert_weights):
        # keep weights over the threshold
        keep_mask = expert_weights.abs() >= TH
        expert_weights = expert_weights * keep_mask
        used = 0

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
        return final_param, used, expert_weights

    @torch.no_grad()
    def transform(self, library) -> Expert:
        if type(library) == str:
            library = ExpertLibrary.get_expert_library(library)

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
        per_exp_th = _safe_quantile(
            expert_vectors.abs(), 1.0 - self.config.top_k, dim=1
        )
        keep_param = expert_vectors.abs() >= per_exp_th.view(-1, 1)

        mean_valid_per_task = keep_param.float().mean(1)
        assert torch.all((mean_valid_per_task - self.config.top_k).abs() < 1e-4)

        used, kept, total = 0, 0, 0

        for param_name in state_dict_keys:
            # stack the expert weights
            expert_weights = torch.stack(
                [expert.expert_weights[param_name] for expert in experts], dim=0
            )
            TH = per_exp_th.view(-1, *((1,) * (expert_weights.ndim - 1)))
            final_param, used_per_pa, expert_weights = self.merge_param(
                TH, expert_weights
            )

            used += used_per_pa
            kept += (expert_weights.abs() > TH).sum()
            total += expert_weights.numel()

            base_expert.expert_weights[param_name].data.copy_(final_param)

        logger.info(
            "Params not reset to 0 in TIES merge: {:.10f}%".format(100.0 * kept / total)
        )
        logger.info(
            "Params used to compute TIES mean: {:.10f}%".format(100.0 * used / total)
        )

        # manually change the config of the expert to remove the tie_params
        base_expert.expert_config.tie_params = None

        return base_expert


@dataclass
class HiddenStateComputerConfig(LibraryTransformConfig):
    use_base_model_only: bool = (
        False  # This computes sentence embeddings without the adapter
    )
    model: str = (
        None  # If `use_base_model_only`, can pass a specific model to compute embeddings with
    )
    max_samples_per_task: int = 10
    track: str = "each_layer"  # last layer, or each layer
    pool: str = "last"  # last, or mean


@LibraryTransform.register("hidden_state_computer", HiddenStateComputerConfig)
class HiddenStateComputer(LibraryTransform):
    """
    Encodes a dataset and computes the average embedding
    """

    def __init__(self, config: HiddenStateComputerConfig = None):
        super().__init__(config or HiddenStateComputerConfig())

    def _update_args(self, args, default_args):
        for k, v in vars(default_args).items():
            if not hasattr(args, k):
                setattr(args, k, v)

        # Also, overwrite the updated args even if already present
        for k, v in default_args.updated_kwargs.items():
            setattr(args, k, v)

        for arg_name in [
            "include_task_source",
        ]:
            value = getattr(default_args, arg_name, None)
            setattr(args, arg_name, value)

        for arg_name in [
            "include_task_source",
        ]:
            value = getattr(default_args, arg_name, None)
            setattr(args, arg_name, value)

    def _track_hidden_states(self, model, keys=None, device="cpu"):
        model.container = {}

        if model.model is None:
            raise ValueError("Model must have a model attribute")

        if self.config.track == "last_layer":
            # Add a hook to the last layer
            def fetch_input(module, input, output):
                model.container["last_layer"] = input[0].detach().to(device)

            model.model.get_output_embeddings().register_forward_hook(fetch_input)
        elif self.config.track == "each_layer":
            # add a hook for all the layers that an expert modifies
            def build_hook(name):
                def retrieve_input(module, input, output):
                    model.container[name] = input[0].detach().to(device)

                return retrieve_input

            for container in model.experts_containers:
                container.register_forward_hook(build_hook(container.layer_name))
        else:
            raise NotImplementedError()

    def _retrieve_hidden_states(self, model):
        keys = list(model.container.keys())
        values = [model.container[k] for k in keys]
        for key in keys:
            del model.container[key]

        return {k: v for k, v in zip(keys, values)}

    @classmethod
    @torch.no_grad()
    def fetch(cls, library: Union[str, ExpertLibrary], config_hash: str = None):
        if isinstance(library, str):
            library = ExpertLibrary.get_expert_library(library)

        config_hash = config_hash or HiddenStateComputerConfig().save_name

        # try to fetch auxiliary data
        output = library.get_auxiliary_data(data_type=config_hash)

        if len(output) > 0:
            logger.info("Found {} precomputed centroids".format(len(output)))
            return output

        raise ValueError(
            "Hidden states are missing or corrupted, please recompute them."
        )

    @torch.no_grad()
    def transform(
        self,
        library: ExpertLibrary,
        persist=False,
        recompute=False,
        default_args=None,
        device="cpu",
    ) -> Expert:
        from mttl.arguments import ExpertConfig
        from mttl.models.lightning.expert_module import ExpertModule, MultiExpertModule

        if isinstance(library, str):
            library = ExpertLibrary.get_expert_library(library)

        try:
            protos = self.fetch(library, self.config.save_name)

            if not recompute:
                logger.info("Found {} precomputed centroids".format(len(protos)))
                return protos
        except ValueError:
            pass

        logger.info("Computing centroids for {} experts".format(len(library)))
        output = {}

        for _, (expert_name, expert) in enumerate(library.items()):
            training_config = ExpertConfig.from_dict(expert.training_config)

            if default_args is not None:
                self._update_args(training_config, default_args)

            if self.config.use_base_model_only and self.config.model is not None:
                training_config.model = self.config.model

            model = MultiExpertModel(
                MultiExpertModelConfig(
                    base_model=training_config.model,
                ),
                device_map=training_config.device_map,
            )
            if not self.config.use_base_model_only:
                model.add_expert_instance(expert, is_default=True)

            self._track_hidden_states(model, device=device)

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
            device_model = next(model.parameters()).device

            for _, batch in pbar:
                batch = transfer_batch_to_device(batch, device_model)
                model.forward(**batch)

                bs = batch["input_ids"].size(0)
                last_token_idx = batch["attention_mask"].sum(1).to(device) - 1
                hidden_states = self._retrieve_hidden_states(model)
                bs_idx = torch.arange(
                    bs, device=hidden_states[list(hidden_states.keys())[0]].device
                )

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
            output[expert_name] = centroids

            del model

        if persist:
            # add embeddings to the library
            with library.batched_commit():
                for expert_name, data in output.items():
                    library.add_auxiliary_data(
                        data_type=self.config.save_name,
                        expert_name=expert_name,
                        config=self.config.__dict__,
                        data=data,
                        force=True,  # make sure we overwrite
                    )
        return output


@dataclass
class PhatgooseTransformConfig(LibraryTransformConfig):
    n_steps: int = 100
    learning_rate: float = 1e-3
    warmup_ratio: float = 0.1
    micro_batch_size: int = 4
    batch_size: int = 4
    seed: int = 42


@LibraryTransform.register("phatgoose", PhatgooseTransformConfig)
class PhatgooseTransform(HiddenStateComputer):
    def __init__(self, config: PhatgooseTransformConfig = None):
        super().__init__(config or PhatgooseTransformConfig())

    @classmethod
    @torch.no_grad()
    def fetch(cls, library: Union[str, ExpertLibrary], config_hash: str):
        if isinstance(library, str):
            library = ExpertLibrary.get_expert_library(library)

        config_hash = config_hash or PhatgooseTransformConfig().save_name

        # try to fetch auxiliary data
        output = library.get_auxiliary_data(data_type=config_hash)

        if len(output) != len(library):
            logger.warning(
                "Found {} precomputed Phatgoose prototypes. Some experts might not have prototypes.".format(
                    len(output)
                )
            )

        return output

    def transform(
        self,
        library,
        persist: bool = True,
        recompute: bool = False,
        expert_names: list = None,
        default_args=None,
    ):
        from mttl.arguments import ExpertConfig
        from mttl.models.lightning.expert_module import MultiExpertModule

        if type(library) == str:
            library = ExpertLibrary.get_expert_library(library)

        outputs = {}
        expert_names = expert_names or list(library.keys())
        loaded_output = library.get_auxiliary_data(data_type=self.config.save_name)

        for expert_name in expert_names:
            logger.info(f"Computing PHATGOOSE gates for expert {expert_name}")
            expert: Expert = library[expert_name]
            logger.info("Phatgoose save name : {}".format(self.config.save_name))

            if not recompute and expert_name in loaded_output:
                logger.info("Loading precomputed gates for {}".format(expert_name))

                # format is dict[layer_name] = embedding, layer_name ends with selector.{task_name}.v
                outputs[expert_name] = loaded_output[expert_name]
                continue

            training_config: ExpertConfig = ExpertConfig.from_dict(
                expert.training_config
            )

            if default_args is not None:
                self._update_args(training_config, default_args)

            training_config.trainable_param_names = ".*selector.*"
            training_config.weight_decay = 0.0
            training_config.total_steps = self.config.n_steps
            training_config.learning_rate = self.config.learning_rate
            training_config.warmup_proportion = self.config.warmup_ratio
            training_config.train_batch_size = self.config.batch_size
            training_config.micro_batch_size = self.config.micro_batch_size
            training_config.dataset = expert.expert_info.dataset

            if expert.expert_info.expert_task_name:
                train_tasks = expert.expert_info.expert_task_name.split(",")
                training_config.finetune_task_name = ",".join(train_tasks)
            else:
                train_tasks = None

            dm = get_datamodule(training_config)

            logger.info("Training config: {}".format(vars(training_config)))

            model = MultiExpertModel(
                MultiExpertModelConfig(
                    base_model=training_config.model,
                    selector_config=PhatgooseTrainerSelectorConfig(
                        lora_merge_after=True,
                    ),
                ),
                precision=training_config.precision,
                device_map="cuda" if torch.cuda.is_available() else "cpu",
            )
            model.add_expert_instance(expert, is_default=True)

            # for checksum
            frozen_sum, unfrozen_sum = 0, 0
            for key, value in model.state_dict().items():
                if re.match(".*selector.gates.*.v", key):
                    assert torch.allclose(
                        value, torch.zeros_like(value)
                    ), "gate should be 0 init"
                    unfrozen_sum += value.sum()
                else:
                    frozen_sum += value.sum()
                    value.requires_grad = False

            train_model(training_config, model, dm)

            # for checksum
            frozen_sum_after, unfrozen_sum_after = 0, 0
            for key, value in model.state_dict().items():
                if re.match(".*selector.gates.*.v", key):
                    unfrozen_sum_after += value.sum()
                else:
                    frozen_sum_after += value.sum()

            assert (
                frozen_sum == frozen_sum_after
            ), "Frozen params changed during training"
            assert (
                unfrozen_sum != unfrozen_sum_after
            ), "Unfrozen params did not change during training"

            # extract prototypes
            prototypes = {}
            for name, module in model.model.named_modules():
                if isinstance(module, ExpertContainer) and hasattr(
                    module.selector, "get_prototypes"
                ):
                    # expand dict
                    prototypes_module = {}
                    for k, v in module.selector.get_prototypes().items():
                        prototypes_module[f"{name}.selector.{k}.v"] = v
                    prototypes = {**prototypes, **prototypes_module}

            outputs[expert_name] = prototypes

            if persist:
                with library.batched_commit():
                    for expert_name, data in outputs.items():
                        library.add_auxiliary_data(
                            data_type=self.config.save_name,
                            expert_name=expert_name,
                            config=self.config.__dict__,
                            data=data,
                            force=True,  # make sure we overwrite
                        )
            del model
        return outputs


@dataclass
class ArrowTransformConfig(LibraryTransformConfig):
    ab_only: bool = True
    scale: bool = False  # If True, scale by eigenvalue
    tie_params: str = (
        "default"  # If default, ties the same params as during training. If a regex, processed the same way as during training
    )
    tie_op: str = "concat"  # or "sum"


@LibraryTransform.register("arrow", ArrowTransformConfig)
class ArrowTransform(LibraryTransform):
    """
    Given a library of experts, extract the input direction most affected by the linear transforms
    """

    def __init__(self, config: ArrowTransformConfig = None):
        super().__init__(config or ArrowTransformConfig())

    def _maybe_scale(self, vectors, eigvals):
        """
        Post Processing of the retrieved outputs,
        scales the output by the eigenvalue if needed.
        """
        output = {}
        for expert_name, expert_data in vectors.items():
            output[expert_name] = {}
            for layer_name, vector in expert_data.items():
                if self.config.scale:
                    vector = vector * eigvals[expert_name][layer_name]
                output[expert_name][layer_name] = torch.from_numpy(vector)
        return output

    def _low_rank_svd(self, A, B):
        """Faster SVD computation for low rank matrices"""

        # Compute SVD of A
        U_A, Sigma_A, V_A = torch.svd(A)

        # Compute SVD of B.T (transpose of B)
        U_B, Sigma_B, V_B = torch.svd(B.T)

        # Compute product matrix C = Sigma_A * (V_A.T @ V_B) * Sigma_B
        # Since V_A and V_B are orthogonal, their product is also an orthogonal matrix
        C = Sigma_A.diag_embed() @ V_A.t() @ V_B @ Sigma_B.diag_embed()

        # Compute SVD of the product matrix C
        U_C, Sigma_C, V_C = torch.svd(C)

        # Construct the final SVD components of W
        U_W = U_A @ U_C
        V_W_T = V_C.t() @ U_B.t()

        diff_AB = (U_W.T @ U_A).abs().diag()
        if diff_AB[0] < 0.9:
            logger.debug("The first singular vector of U_A and U_AB are not aligned")

        return U_W, Sigma_C, V_W_T

    def _get_unique_parent_names(self, alist):
        """
        if adict.keys() = ['model.layer1.lora_a', 'model.layer.lora_b', 'model.layer2.lora_a']
        output will be {'model.layer1', 'model.layer2'}
        """
        dict_keys = sorted(list(set(".".join(k.split(".")[:-1]) for k in alist)))
        return dict_keys

    @classmethod
    @torch.no_grad()
    def fetch(cls, library: Union[str, ExpertLibrary], config_hash: str):
        """Fetch arrow prototypes from the library, raises ValueError if they are not computed.

        Args:
            library (Union[str, ExpertLibrary]): ExpertLibrary object or its name
            scale (bool): If True, scale the output by the eigenvalue
        """
        if not isinstance(library, ExpertLibrary):
            library = ExpertLibrary.get_expert_library(library)

        config_hash = config_hash or ArrowTransformConfig().save_name

        # try to fetch auxiliary data
        protos = library.get_auxiliary_data(data_type=config_hash + "_protos")
        return protos

    @torch.no_grad()
    def transform(
        self,
        library,
        persist=True,
        recompute=False,
    ) -> Expert:
        logger.info("Arrow save name : {}".format(self.config.save_name))

        if isinstance(library, str):
            library = ExpertLibrary.get_expert_library(library)

        base_model = None

        # Try to fetch the precomputed Arrow prototypes
        protos = self.fetch(library, self.config.save_name)
        already_computed = []

        vectors = {}
        eigvals = {}
        for expert_name, expert in library.items():
            if expert_name in protos and not recompute:
                logger.info(
                    "Found precomputed Arrow prototypes for expert {}".format(
                        expert_name
                    )
                )
                already_computed.append(expert_name)
                continue

            logger.info(f"Computing SVD for expert {expert_name}")
            vectors[expert_name] = {}
            eigvals[expert_name] = {}

            if base_model is None and not self.config.ab_only:
                training_config = expert.training_config
                training_config.model_modifier = None
                from mttl.models.lightning.expert_module import MultiExpertModule

                base_model = MultiExpertModule(**vars(training_config))

            # get parameters tied during training
            param_map = get_target_2_source_param_mapping(
                expert.expert_weights.items(),
                expert.expert_info.expert_config.tie_params,
            )
            if self.config.tie_params != "default":
                # get parameters we wish to tie for Arrow
                _tied_params = get_target_2_source_param_mapping(
                    expert.expert_weights.items(), self.config.tie_params
                )
                # Make sure that params tied during training are also tied for Arrow
                if any(key not in _tied_params for key in param_map):
                    logger.warning(
                        "Some parameters that are tied during training are not tied during Arrow computation."
                    )
                param_map = _tied_params

            tied_params = list(param_map.keys()) + list(param_map.values())
            assert all(
                "lora_b" not in param_name for param_name in tied_params
            ), "Support for tied B not available"
            assert all(
                "lora_a" in param_name for param_name in tied_params
            ), "Only support tied As for now"

            # Now that we know only A's are tied, we can proceed using only the parent names
            # e.g. 'model.layers.30.self_attn.q_proj' instead of 'model.layers.30.self_attn.q_proj.lora_a'
            tied_parents = self._get_unique_parent_names(tied_params)

            untied_parents = [
                parent
                for parent in self._get_unique_parent_names(
                    expert.expert_weights.keys()
                )
                if parent not in tied_parents
            ]

            # Build a mapping from source to target parameters
            # e.g. <name_of_parent_of_param> : [<list of all other params tied to it>]
            # NOTE: list will be empty if the param is not tied to anything
            tied_param_bins = defaultdict(list)

            for tgt_name, src_name in param_map.items():
                parent_src = ".".join(src_name.split(".")[:-1])
                parent_tgt = ".".join(tgt_name.split(".")[:-1])
                tied_param_bins[parent_src].append(parent_tgt)
            for parent in untied_parents:
                tied_param_bins[parent] = []

            for parent_name, dependents in tied_param_bins.items():
                logger.info(f"\tComputing SVD for parameter {parent_name}")

                parent_names = [parent_name]
                A_name, B_name = f"{parent_name}.lora_a", f"{parent_name}.lora_b"
                As = [expert.expert_weights[A_name]]
                Bs = [expert.expert_weights[B_name]]
                base_W = []

                for tied_module in dependents:
                    logger.info(f"\t\t\tTying Arrow with {tied_module}")
                    As += [expert.expert_weights[f"{tied_module}.lora_a"]]
                    Bs += [expert.expert_weights[f"{tied_module}.lora_b"]]
                    parent_names += [tied_module]

                    if not self.config.ab_only:
                        base_W += [
                            base_model.model.state_dict()[f"{tied_module}.weight"]
                        ]

                if len(As) > 1:
                    if self.config.tie_op == "concat":
                        # Mimicking phi-2 behavior
                        assert self.config.ab_only
                        assert all(
                            torch.allclose(A, As[0]) for A in As
                        ), "A should be the same for all tied parameters"
                        A = As[0]
                        B = torch.cat(Bs, dim=1)
                    elif self.config.tie_op == "sum":
                        # A1B1 + A2B2 == [A1 A2] [B1; B2].
                        # We do it this way to leverage the low-rank SVD
                        A = torch.cat(As, dim=1)
                        B = torch.cat(Bs, dim=0)
                    else:
                        raise NotImplementedError()
                else:
                    A, B = As[0], Bs[0]

                # Reshape As and Bs (needed for Poly / MHR weights)
                rank = expert.expert_config.lora_rank
                A = A.reshape(-1, rank).float()
                B = B.reshape(rank, -1).float()

                W = (A @ B).T  # out_features, in_features

                if self.config.ab_only:
                    U_W, Sigma_W, _ = self._low_rank_svd(A, B)
                    top_value = Sigma_W[0] ** 2
                    bottom_vector = U_W[:, -1]
                    top_vector = U_W[:, 0]
                else:
                    base_W += [
                        base_model.model.state_dict()[f"{parent_name}.weight"]
                    ].float()
                    base_W = torch.stack(base_W).sum(0)
                    W += base_W
                    U, E, Vt = torch.linalg.svd(W)
                    top_vector = Vt[0]
                    bottom_vector = Vt[-1]
                    top_value = E[0]

                # Check that top vector is indeed an eigenvector
                WTW = W.T @ W
                ratio = WTW @ top_vector / (top_vector * top_value)
                torch.allclose(ratio, torch.ones_like(ratio), atol=1e-3)

                # Check that top vector is indeed the top eigenvector
                assert (WTW @ top_vector).pow(2).sum() >= (WTW @ bottom_vector).pow(
                    2
                ).sum()

                # Save eigenvector and eigvenvalue
                for parent in parent_names:
                    assert parent not in vectors[expert_name]
                    vectors[expert_name][parent] = top_vector.real.cpu().numpy()
                    eigvals[expert_name][parent] = top_value.item()

        to_upload = [x for x in library.keys() if x not in already_computed]
        new_protos = self._maybe_scale(vectors, eigvals)

        if persist and len(to_upload) > 0:
            # add embeddings to the library
            with library.batched_commit():
                for expert_name in to_upload:
                    logger.info(
                        f"Uploading centroids to the library for expert {expert_name}"
                    )
                    for data_name, data in [
                        ("vectors", vectors),
                        ("eigvals", eigvals),
                        ("protos", new_protos),
                    ]:
                        library.add_auxiliary_data(
                            data_type=self.config.save_name + "_" + data_name,
                            expert_name=expert_name,
                            config=self.config.__dict__,
                            data=data[expert_name],
                            force=True,  # make sure we overwrite
                        )

        protos.update(new_protos)
        return protos


@dataclass
class ExpertProjectorConfig:
    granularity: str = (
        "finegrained"  # whether to use the same coefficients for all parameters or per `nn.Parameter` instance
    )
    project_over_all_experts: bool = (
        False  # whether to project over all experts or just the ones in the cluster
    )


@LibraryTransform.register("expert_projector", ExpertProjectorConfig)
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
        )

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
            expert_library = ExpertLibrary.get_expert_library(expert_library)

        if isinstance(cluster_library, str):
            cluster_library = ExpertLibrary.get_expert_library(cluster_library)

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
class CrossExpertNormComputerConfig:
    pass


@LibraryTransform.register("cross_expert_norm_computer", CrossExpertNormComputerConfig)
class CrossExpertNormComputer(HiddenStateComputer):
    """
    Given a library of experts, compute the norm of ABx for both in-dist and ood experts
    """

    def __init__(self, config: CrossExpertNormComputerConfig = None):
        super().__init__(config or CrossExpertNormComputerConfig())

    @torch.no_grad()
    def transform(self, library, default_args=None) -> Expert:
        if isinstance(library, str):
            library = ExpertLibrary.get_expert_library(library)

        expert_names = list(library.keys())
        an_expert = library[expert_names[0]]
        training_config = an_expert.training_config

        # overwrite required args
        training_config.library_id = library.repo_id
        training_config.router_selector = "task_selector"

        if default_args is not None:
            self._update_args(training_config, default_args)

        training_config.train_batch_size = (
            default_args.predict_batch_size if default_args is not None else 4
        )
        training_config.finetune_task_name = ",".join(
            [
                library[exp_name].training_config.finetune_task_name
                for exp_name in library.keys()
            ]
        )

        from mttl.models.containers import ExpertContainer
        from mttl.models.lightning.expert_module import ExpertModule, MoEModel

        model = MoEModel(**vars(training_config))

        # build a hook to forward across other (ood) experts
        def build_hook(layer_name, container, task_id_container):
            def retrieve_input(module, input, output):
                task_names = task_id_container["routing_infos"].task_names
                attn_mask = task_id_container["routing_infos"].attention_mask
                container[layer_name] = input[0].detach()

                # output (bs, seq_len, D) is the correctly routed outpu
                # let's generate the outputs for random task routing

                not_picked = np.array(
                    list(set(module.selector.expert_names) - set(task_names))
                )
                random_tasks = np.random.choice(
                    not_picked,
                    size=len(task_names),
                    replace=not_picked.size < len(task_names),
                )

                # Redo ExpertContainer forward
                selector_out = module.selector(input[0])
                selector_out.experts = random_tasks.tolist()
                random_out = module.route(input[0], selector_out)

                norm_correct = (output * attn_mask.unsqueeze(-1)).pow(2).sum(
                    -1
                ).sqrt().sum() / attn_mask.sum()
                norm_wrong = (random_out * attn_mask.unsqueeze(-1)).pow(2).sum(
                    -1
                ).sqrt().sum() / attn_mask.sum()

                container[layer_name] = (norm_correct, norm_wrong)

                return output

            return retrieve_input

        hooks = []
        container = {}
        for module_name, module in model.named_experts():
            if isinstance(module, ExpertContainer):
                hook = build_hook(module_name, container, model.model.task_id_container)
                module.register_forward_hook(hook)
                hooks += [hook]

        logger.info(f"set {len(hooks)} hooks")
        training_config.subsample_train = 2_000
        dm = get_datamodule(training_config)
        dataloader = dm.train_dataloader()

        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        device = next(model.parameters()).device

        total_avg_diff, total_rel_diff = [], []
        for num_batch, batch in pbar:
            batch = transfer_batch_to_device(batch, device)

            if isinstance(model, ExpertModule):
                model.forward(batch, reduction="none")
            else:
                model.forward(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )

            avg_diff, rel_diff = 0, 0
            for layer, (correct, wrong) in container.items():
                avg_diff += (correct - wrong).item()
                rel_diff += (correct / wrong).item()

            avg_diff /= len(container)
            rel_diff /= len(container)

            total_avg_diff += [avg_diff]
            total_rel_diff += [rel_diff]

            print(
                f"avg_diff: {avg_diff / len(container)}, rel_diff: {rel_diff / len(container)}"
            )


@dataclass
class MBClusteringTransformConfig(SVDEmbeddingTransformConfig):
    random_state: int = 42
    k: int = 10


@LibraryTransform.register("mbc_with_cos_sim", MBClusteringTransformConfig)
class MBCWithCosSimTransform(LibraryTransform):
    """
    Computes clusters based on the embedding similarity of the experts.
    The input to KMeans is the cosine similarity matrix between the experts' embeddings.
    """

    def __init__(self, config: MBClusteringTransformConfig = None):
        super().__init__(config or MBClusteringTransformConfig())

    def transform(
        self,
        library: ExpertLibrary,
        persist: bool = False,
        recompute: bool = False,
    ) -> Dict[str, List[str]]:
        svd_config = SVDEmbeddingTransformConfig(
            name=self.config.name,
            n_components=self.config.n_components,
            sparsity_threshold=self.config.sparsity_threshold,
        )

        def create_embeddings():
            svd_embedder = SVDEmbeddingTransform(
                svd_config,
                random_state=self.config.random_state,
            )
            embeddings = svd_embedder.transform(library, persist=persist)
            del svd_embedder
            return embeddings

        embeddings = library.get_auxiliary_data(svd_config.save_name)

        if len(embeddings) != len(library) or recompute:
            logger.info("Recomputing embeddings for clustering.")
            embeddings = create_embeddings()

        # Extract the embeddings as a numpy array
        expert_names, embeddings = zip(*sorted(embeddings.items()))

        embeddings_array = np.stack(embeddings)
        cosine_sim_matrix = cosine_similarity(embeddings_array, embeddings_array)

        kmeans = KMeans(
            n_clusters=self.config.k,
            init="k-means++",
            n_init=10,
            random_state=self.config.random_state,
        )
        kmeans.fit(cosine_sim_matrix)
        cluster_labels = kmeans.labels_
        clusters = defaultdict(list)

        for key, label in zip(expert_names, cluster_labels):
            clusters[f"cluster_{label}"].append(key)
        return clusters
