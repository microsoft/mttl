import abc
import copy
import dataclasses
import re
from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Union

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
from mttl.models.monitors import get_monitors
from mttl.models.train_utils import train_model
from mttl.models.utils import transfer_batch_to_device
from mttl.registrable import Registrable
from mttl.serializable import Serializable


class LibraryTransform(abc.ABC, Registrable):
    """Defines a transformation of a library of experts."""

    def __init__(self, config):
        self.config = config

    @abstractmethod
    def transform(
        self, library: ExpertLibrary, persist: bool = False, recompute: bool = False
    ):
        pass


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
                sign_per_dim = expert_weights.sum(0, keepdim=True).sign()
                # resolve zero signs: https://github.com/rezazzr/breadcrumbs/blob/main/src/task_vectors.py#L334
                majority_sign = torch.sign(sign_per_dim.sum())
                sign_per_dim[sign_per_dim == 0] = majority_sign

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

# --------------------------
# LoRA AB linear merge LoRA
# -------------------------

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
        weight_names = [n for n in model.state_dict().keys() if ('Wqkv.weight' in n or 'out_proj.weight' in n)]

        # iterate over the library
        import collections
        store_W = collections.defaultdict(dict)
        for expert_name, expert in library.items():
            # iterate over the expert weights           
            for l in weight_names:
                common_name = '.'.join(l.split('.')[1:-1])
                A, B = expert.expert_weights[f'{common_name}.lora_a'], expert.expert_weights[f'{common_name}.lora_b']
                W = A @ B
                store_W[l][expert_name] = W

        store_average_W = collections.defaultdict(dict)
        # iterate over all the layers of W
        for l in weight_names:
            average_W = 0
            for k, v in store_W[l].items():
                average_W+=v
            average_W /=len(store_W[l])
            store_average_W[l] = average_W
                # average the Ws for each layer

        new_state_dict = {}
        # add the averaged Ws to the model
        for key, value in model.state_dict().items():
            if key in weight_names:
                print(f'added {key}')
                new_state_dict[key] = value + store_average_W[key].T
            else:
                new_state_dict[key] = value
 
        # load state_dict into model
        model.load_state_dict(new_state_dict)
        return model

# --------------------------
# Sparse merge
# -------------------------
@dataclass
class SparseWeightLinearMergeConfig(LibraryTransformConfig):
    weights: dict = None


class SparseWeightLinearMerge(LibraryTransform):
    """
    Computes a uniform weight mixture across experts of a given library
    """
    def __init__(self, config: SparseWeightLinearMergeConfig = None):
        super().__init__(config or SparseWeightLinearMergeConfig())
    def load_mask(self, expert):
        try:
            print('trying to load mask from hf')
            library_id = expert.training_config.library_id
            from huggingface_hub import hf_hub_download
            destination_type, f_name = library_id.split('://')
            repo_id=('/').join(f_name.split('/')[:2])
            filename = f'{expert.expert_info.expert_name}_mask.npz'
            f_path=hf_hub_download(repo_id=repo_id, filename=filename)
            Mask = np.load(f_path, allow_pickle=True)['arr'].item()
        except:
            print('trying to load mask from local dir')
            m_loc = f'experiment/{expert.training_config.exp_name}/mask.npz'
            Mask = np.load(m_loc, allow_pickle=True)['arr'].item()
        return Mask

    def convert_idx_2_mask(self, weight_idx, mat_dim):
        m = np.zeros(mat_dim)
        m[tuple(zip(*weight_idx))] = 1
        return torch.FloatTensor(m)
    

    def update_module_mask(self, module, expert):
        Mask = self.load_mask(expert)
        for m_name, m in dict(module.named_modules()).items():
            if 'sparse_layer' in m_name:
                keep_mask = self.convert_idx_2_mask(weight_idx=Mask[m_name],
                                            mat_dim=m.weight_mask.shape)
            
                m.weight_mask=keep_mask.data.clone()

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
        print('Trying to load base model from:', training_config.model)
        model = ExpertModel(**vars(training_config))
        sparse_layer_names = [n for n in model.state_dict().keys() if ('sparse_layer' in n)] # allow to add weights and bias
        assert sparse_layer_names!=[], print('could not find sparse-layer modules')

        # iterate over the library
        import collections
        store_W = collections.defaultdict(dict)
        store_m = collections.defaultdict(dict)
        expert_weight_hist = collections.defaultdict(dict)    # weight stats

        for expert_name, expert in library.items():
            print(f'Merging sparse weight for task: {expert_name}')
            Mask = self.load_mask(expert)
            expert_weight_hist[expert_name] = collections.defaultdict(dict) # weight stats
            # for each layer compute the "average of the overlapped weight" 
            for l in sparse_layer_names:
                common_name = '.'.join(l.split('.')[1:-1])
                param_type = l.split('.')[-1]

                if param_type == 'weight':
                    # get mask-m for layer-l: convert the weight_indx to convert sparse-mask
                    m = self.convert_idx_2_mask(weight_idx=Mask[f'model.{common_name}'], 
                                       mat_dim=expert.expert_weights[f'{common_name}.weight'].shape)
                else:
                    m = 1.0
                # Check if entry exists
                if l in store_W:
                    # store weight
                    store_W[l]+=expert.expert_weights[f'{common_name}.{param_type}'] * m
                    # store mask
                    store_m[l]+=m
                # new entry for expert 1
                else:
                    store_W[l]=expert.expert_weights[f'{common_name}.{param_type}'] * m
                    store_m[l]=m
                expert_weight_hist[expert_name][common_name]=(float(expert.expert_weights[f'{common_name}.weight'].mean().data.numpy()), float(expert.expert_weights[f'{common_name}.weight'].std().data.numpy())) # weight stats
        
        # we sum the total per-layer weight overlap and devide the count as an alternate to calculate accurate weight average
        for l in sparse_layer_names:
            param_type = l.split('.')[-1]
            if param_type =='weight':
                store_m[l][store_m[l]==0]=1 # assigning 1 to the zero-masked weights positions to avoid numerical error in the next step
                store_W[l] /= store_m[l]
            else:
                store_W[l] /= len(library)
        new_state_dict = {}
        # add the averaged Ws to the model
        for key, value in model.state_dict().items():
            if key in sparse_layer_names:
                print(f'added {key}')
                new_state_dict[key] = value + store_W[key]
            else:
                new_state_dict[key] = value
 
        # load state_dict into model
        model.load_state_dict(new_state_dict)
        # save weights stats
        import os
        import json
        exp_temp = training_config.library_id.split('/')[-1]
        file_loc = f'Weight_Stats/{exp_temp}'
        os.makedirs(file_loc, exist_ok=True)
        with open(f'{file_loc}/weight_stats.json','w') as json_file:
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
        weight_names = [n for n in model.state_dict().keys() if ('Wqkv.sparse_layer.weight' in n or 'out_proj.sparse_layer.weight' in n)]

        # iterate over the library
        import collections
        store_W = collections.defaultdict(dict)
        store_m = collections.defaultdict(dict)
        for expert_name, expert in library.items():
            # TODO: only consider the weights that matches the given `get_expert` input
            if expert_name == get_expert:
                print(f'Merging sparse weight for task: {expert_name}')
                Mask = self.load_mask(expert)
                # for each layer compute the "average of the overlapped weight" 
                for l in weight_names:
                    common_name = '.'.join(l.split('.')[1:-1])
                    # get mask-m for layer-l: convert the weight_indx to convert sparse-mask
                    m = self.convert_idx_2_mask(weight_idx=Mask[f'model.{common_name}'], 
                                       mat_dim=expert.expert_weights[f'{common_name}.weight'].shape)
                    if l in store_W.keys():
                        # store weight
                        store_W[l]+=expert.expert_weights[f'{common_name}.weight'] * m
                        # store mask
                        store_m[l]+=m
                    else:
                        store_W[l]=expert.expert_weights[f'{common_name}.weight'] * m
                        store_m[l]=m
        # we sum the total per-layer weight overlap and devide the count as an alternate to calculate accurate weight average
        for l in weight_names:
            store_m[l][store_m[l]==0]=1 # assigning 1 to the zero-masked weights positions to avoid numerical error in the next step
            store_W[l] /= store_m[l]

        new_state_dict = {}
        # add the averaged Ws to the model
        for key, value in model.state_dict().items():
            if key in weight_names:
                print(f'added {key}')
                new_state_dict[key] = value + store_W[key]
            else:
                new_state_dict[key] = value
 
        # load state_dict into model
        model.load_state_dict(new_state_dict)
        return model




# -----------------------------------
# SLERP and LERP implementation
# -----------------------------------
def lerp(t, v0, v1, origin_data_type=None):
    v2 = (1 - t) * v0 + t * v1
    return torch.from_numpy(v2).to(origin_data_type)
# SLERP
def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    '''
    Spherical linear interpolation
    Args:
        t (float/np.ndarray): Float value between 0.0 and 1.0
        v0 (np.ndarray): Starting vector
        v1 (np.ndarray): Final vector
        DOT_THRESHOLD (float): Threshold for considering the two vectors as
                               colineal. Not recommended to alter this.
    Returns:
        v2 (np.ndarray): Interpolation vector between v0 and v1
    '''
    origin_data_type = v0.dtype
    v0 = v0.detach().cpu().float().numpy()
    v1 = v1.detach().cpu().float().numpy()
    
    # Copy the vectors to reuse them later
    v0_copy = np.copy(v0)
    v1_copy = np.copy(v1)
    # Normalize the vectors to get the directions and angles
    v0 = v0 / np.linalg.norm(v0)
    v1 = v1 / np.linalg.norm(v1)
    # Dot product with the normalized vectors (can't use np.dot in W)
    dot = np.sum(v0 * v1)
    # If absolute value of dot product is almost 1, vectors are ~colineal, so use lerp
    if np.abs(dot) > DOT_THRESHOLD:
        return lerp(t, v0_copy, v1_copy, origin_data_type=origin_data_type)
    # Calculate initial angle between v0 and v1
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    # Angle at timestep t
    theta_t = theta_0 * t
    sin_theta_t = np.sin(theta_t)
    # Finish the slerp algorithm
    s0 = np.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0
    v2 = s0 * v0_copy + s1 * v1_copy

    return torch.from_numpy(v2).to(origin_data_type)

@dataclass
class SLERPMergeConfig(LibraryTransformConfig):
    weights: dict = None


class SLERPMerge(LibraryTransform):
    """
    Computes a uniform weight mixture across experts of a given library
    """

    def __init__(self, config: SLERPMergeConfig = None):
        super().__init__(config or SLERPMergeConfig())


    def load_mask(self, expert):
        try:
            print('trying to load mask from hf')
            library_id = expert.training_config.library_id
            from huggingface_hub import hf_hub_download
            destination_type, f_name = library_id.split('://')
            repo_id=('/').join(f_name.split('/')[:2])
            filename = f'{expert.expert_info.expert_name}_mask.npz'
            f_path=hf_hub_download(repo_id=repo_id, filename=filename)
            Mask = np.load(f_path, allow_pickle=True)['arr'].item()
        except:
            print('trying to load mask from local dir')
            m_loc = f'experiment/{expert.training_config.exp_name}/mask.npz'
            Mask = np.load(m_loc, allow_pickle=True)['arr'].item()
        return Mask

    def convert_idx_2_mask(self, weight_idx, mat_dim):
        m = np.zeros(mat_dim)
        m[tuple(zip(*weight_idx))] = 1
        return torch.FloatTensor(m)

    def sparse_SLERP(self, model, experts, base_expert):
        base_expert_mask = self.load_mask(base_expert)
        for layer, _ in  base_expert_mask.items():
            mod_layer = '.'.join(layer.split('.')[1:])
            base_expert_mask[layer] = self.convert_idx_2_mask(weight_idx=base_expert_mask[layer], 
                                    mat_dim=base_expert.expert_weights[f'{mod_layer}.weight'].shape)
        weight_names = [n for n in model.state_dict().keys() if 'sparse_layer' in n]

        for expert in experts:
            mask = self.load_mask(expert)
            # for expert_name, expert in library.items():
            for layer, weight in model.state_dict().items():
                if layer in weight_names:
                    common_name = '.'.join(layer.split('.')[1:-1])
                    param_type = layer.split('.')[-1]

                    if param_type == 'weight':
                        # get mask-m for layer-l: convert the weight_indx to convert sparse-mask
                        m = self.convert_idx_2_mask(weight_idx=mask[f'model.{common_name}'], 
                                        mat_dim=expert.expert_weights[f'{common_name}.weight'].shape)
                        bm = base_expert_mask[f'model.{common_name}']
                    else:
                        m = 1.0
                        bm = 1.0
                    base_expert._expert_weights[f'{common_name}.{param_type}'] = slerp(float(1.0)-0.5, 
                                                                                       v0=base_expert._expert_weights[f'{common_name}.{param_type}']*bm, 
                                                                                       v1=expert._expert_weights[f'{common_name}.{param_type}']*m
                                                                                       )
                    if param_type == 'weight':
                        base_expert_mask[f'model.{common_name}'] = torch.logical_or(m, bm).float()
        
        updated_state_dict = {}
        for layer, weight in model.state_dict().items():
            if layer in weight_names:
                mod_layer = '.'.join(layer.split('.')[1:])
                updated_state_dict[layer] = base_expert._expert_weights[mod_layer]
            else:
                updated_state_dict[layer] = weight
        # load state_dict into model
        model.load_state_dict(updated_state_dict)
        return model
    
    def FFT_SLERP(self, model, experts, base_expert):
        for expert in experts:
            for layer, _ in model.state_dict().items():
                common_name = '.'.join(layer.split('.')[1:-1])
                param_type = layer.split('.')[-1]
                base_expert._expert_weights[f'{common_name}.{param_type}'] = slerp(float(1.0)-0.5, 
                                                                                    v0=base_expert._expert_weights[f'{common_name}.{param_type}'], 
                                                                                    v1=expert._expert_weights[f'{common_name}.{param_type}']
                                                                                    )
        updated_state_dict = {}
        for layer, _ in model.state_dict().items():
            mod_layer = '.'.join(layer.split('.')[1:])
            updated_state_dict[layer] = base_expert._expert_weights[mod_layer]
        # load state_dict into model
        model.load_state_dict(updated_state_dict)
        return model

    @torch.no_grad()
    def transform(self, library) -> Expert:
        if type(library) == str:
            library = ExpertLibrary.get_expert_library(library)

        expert_names = list(library.keys())
        experts = [library[name] for name in expert_names]
        base_expert = copy.deepcopy(experts[0])
        training_config = copy.deepcopy(base_expert.training_config)
        from mttl.models.expert_model import ExpertModel
        model = ExpertModel(**vars(training_config))
        if training_config.model_modifier == None:
            # skip the first expert as it's now acting as base_expert
            model = self.FFT_SLERP(model, experts[1:], base_expert)
        elif training_config.model_modifier == 'sparse_mask_adapter':
            model = self.sparse_SLERP(model, experts[1:], base_expert)
        return model
    

def topk_multiple_experts(expert_vectors, topk, TH_type=None):
    assert TH_type!=None
    n_tasks = expert_vectors.shape[0]
    values = []
    for t in range(n_tasks):
        print('topk expert', t)
        v, _ = torch.topk(expert_vectors[t,:], topk)
        if TH_type == 'lower': values.append(v[-1])
        elif TH_type == 'upper': values.append(v[0])
        del v
    values = torch.stack(values, dim=0)  # Shape will be (n_tasks,)
    return values

@dataclass
class BaseMergeConfig(LibraryTransformConfig):
    merging_method: str = 'BaseMerge'

class BaseMerge(LibraryTransform):
    """
    Base class for TIES-Merge and Model-Breadcrumbs: Computes a merged weight across experts of a given library
    """
    def __init__(self, config: BaseMergeConfig = None):
        super().__init__(config or BaseMergeConfig())

    
    @torch.no_grad()
    def pre_configure(self, library):
        if type(library) == str:
            library = ExpertLibrary.get_expert_library(library)
        expert_names = list(library.keys())
        experts = [library[name] for name in expert_names]
        logger.info("Averaging {} experts".format(len(experts)))
        expert_type = experts[0].training_config.model_modifier
        if expert_type is None:
            expert_type = 'FFT' 

        # transform experts. NOTE: MUST
        self.transform_experts(experts, expert_type)

        # get base expert    
        base_expert = copy.deepcopy(experts[0])
        base_expert.name = self.config.merging_method
        train_cfg = copy.deepcopy(base_expert.training_config)
        train_cfg.device_map = "cpu"
        trainable_params = list(base_expert.expert_weights.keys()) # 'model.layers.0.self_attn.o_proj.weight'

        # get base model
        from mttl.models.utils import model_loader_helper
        base_model = model_loader_helper(
                train_cfg.model,
                load_in_8bit=train_cfg.load_in_8bit,
                load_in_4bit=train_cfg.load_in_4bit,
                device_map=getattr(train_cfg, "device_map", "cpu"),
            )
        
        return experts, expert_type, base_expert, base_model, trainable_params

    @torch.no_grad()
    def extract_expert_vector(self,experts, expert_type, base_model_state_dict, trainable_params):
        # Build n_tasks x D experts
        expert_vectors = []
        for expert in experts:
            if expert_type == 'FFT':
                # W_t = W_base - W'
                expert_vectors += [
                    torch.nn.utils.parameters_to_vector(
                        list(expert.expert_weights[k] - base_model_state_dict[k] for k in trainable_params)
                        )
                    ]
                # W_t = (lora_a*lora_b).T
                # NOTE: it's already done when we call self.transform_experts()
            elif expert_type == 'lora':
                weights_list = [param.contiguous() for param in (expert.expert_weights[k] for k in trainable_params)]
                expert_vectors +=[torch.nn.utils.parameters_to_vector(weights_list)]

        return expert_vectors
    
    @torch.no_grad()
    def extract_expert_weight(self, base_model_state_dict, experts, param_name, expert_type):
        # for given "param_name", iterates over all expert and gets the trained expert-weights
        if expert_type == 'FFT':
            expert_weights = torch.stack(
                    [expert.expert_weights[param_name]-base_model_state_dict[param_name] for expert in experts], dim=0
                )
        elif expert_type == 'lora':
            expert_weights = torch.stack(
                    [expert.expert_weights[param_name] for expert in experts], dim=0
                )
        elif expert_type == 'sparse_mask_adapter':
            expert_weights = torch.stack(
                    [expert.expert_weights[param_name] for expert in experts], dim=0
                )
        return expert_weights

    
    @torch.no_grad()
    def transform_experts(self, experts, expert_type):
        assert expert_type in ['FFT','lora', 'sparse_mask_adapter'], print(f'{expert_type} is not implemented')
        if expert_type == 'FFT':
            pass
        elif expert_type == 'lora':
            # Lora
            base_expert = copy.deepcopy(experts[0])
            trainable_layers = ['.'.join(l.split('.')[:-1]) for l in list(base_expert.expert_weights.keys()) if 'qkv_proj' in l] ## 'layers.0.self_attn.o_proj'
            trainable_layers = list(dict.fromkeys(trainable_layers)) # removes duplicate layers (loraA,loraB), w/o changing layer order
            for expert in experts:
                    for l in trainable_layers:
                        expert.expert_weights[f'{l}.weight'] = (expert.expert_weights[f'{l}.lora_a'] @ expert.expert_weights[f'{l}.lora_b']).T
                        del expert.expert_weights[f'{l}.lora_a'], expert.expert_weights[f'{l}.lora_b']

            # NOTE: sanity check, please don't remove this block
            for expert in experts:
                for l in list(expert.expert_weights.keys()):
                    if ('lora_a' in l or 'lora_b' in l):
                        del expert.expert_weights[l]

        elif expert_type == 'sparse_mask_adapter':
            base_expert = copy.deepcopy(experts[0])
            trainable_layers = ['.'.join(l.split('.')[:-1]) for l in list(base_expert.expert_weights.keys()) if 'qkv_proj' in l] ## 'layers.0.self_attn.o_proj'
            trainable_layers = list(dict.fromkeys(trainable_layers)) # removes duplicate layers (loraA,loraB), w/o changing layer order

            for expert in experts:
                Mask = self.load_mask(expert)
                # for each layer compute the "average of the overlapped weight" 
                for l in trainable_layers:
                    # weight
                    m = self.convert_idx_2_mask(weight_idx=Mask[f'model.{l}'], 
                                        mat_dim=expert.expert_weights[f'{l}.weight'].shape)  
                    expert.expert_weights[f'{l}.weight'] = expert.expert_weights[f'{l}.weight'] * m
                    expert.expert_weights[f'{l}.bias'] = expert.expert_weights[f'{l}.bias']
               
    @torch.no_grad()
    def compute_per_task_threhold(self, expert_vectors):
        # Given expert vector, W_task compute TH score to prune parameters
        # NOTE: W_task = W_base - W'
        pass
    
    @torch.no_grad()
    def merge_expert(self, experts, expert_vectors, trainable_params, base_expert, base_model_state_dict, expert_type):
        pass

    @torch.no_grad()
    def transform(self, library):
        experts, expert_type, base_expert, base_model, trainable_params = self.pre_configure(library)
        base_model_state_dict = dict(base_model.state_dict())
        # ----------------------------------------------------------------------
        # Collect Expert-vector
        # for FFT:    expert_vectors, delta_W = W-W'
        # for LoRA:   expert_vectors, delta_W = A.B
        # for sparse: expert_vectors, delta_W = delta_W*mask (NOTE: not implemented yet)
        expert_vectors= self.extract_expert_vector(experts, expert_type, base_model_state_dict, trainable_params)
        base_expert = self.merge_expert(experts, expert_vectors, trainable_params, base_expert, base_model_state_dict, expert_type)
        # load to base model:
        from mttl.models.expert_model import ExpertModel
        config = base_expert.training_config
        config.model_modifier = None  # load only the base model
        config.device_map='cpu'
        config.trainable_param_names='.*' # allows to train all linear layers
        base_model = ExpertModel(**vars(config))
        # load state_dict into model
        assert set(base_model.model.state_dict().keys()) == set(base_expert.expert_weights.keys()), "Expert weights must have the same keys"
        base_model.model.load_state_dict(base_expert._expert_weights)
        return base_model


@dataclass
class UniformMergeConfig(BaseMergeConfig):
    merging_method: str = 'uniform_merge_expert'
    alpha: float = 1.0 

class UniformMerge(BaseMerge):
    """
    Computes a uniform weight mixture across experts of a given library
    """

    def __init__(self, config: UniformMergeConfig = None):
        super().__init__(config or UniformMergeConfig())

    def merge_expert(self, experts, expert_vectors, trainable_params, base_expert, base_model_state_dict, expert_type):
        used, total = 0, 0
        for param_name in base_model_state_dict.keys():
            if param_name in trainable_params:
                # stack the expert weights
                expert_weights = self.extract_expert_weight(base_model_state_dict, experts, param_name, expert_type)

                # collect mask
                keep_mask = torch.ones_like(expert_weights)
                # keep weights over the threshold
                expert_weights = expert_weights * keep_mask
                # uniform
                final_param = expert_weights.mean(0)

                # -----------------------------------------------------
                # base_weight + sum of the "filtered" task-vector
                # W = W + delta_W
                # source : (a) https://openreview.net/pdf?id=6t0Kwf8-jrj (b) https://arxiv.org/pdf/2306.01708
                final_param = base_model_state_dict[param_name] + self.config.alpha * final_param

                
                used += keep_mask.sum().item()
                total += expert_weights.numel()

                base_expert.expert_weights[param_name].data.copy_(final_param)
            else:
                base_expert.expert_weights[param_name]=copy.deepcopy(base_model_state_dict[param_name])

        logger.info(
            "Params used to compute Ties mean: {:.10f}%".format(100.0 * used / total)
        )
        return base_expert
    

@dataclass
class TaskArithmeticConfig(BaseMergeConfig):
    merging_method: str = 'task_arithmetic_merge_expert'
    alpha: float = 0.4 # scaling

class TaskArithmetic(BaseMerge):
    """
    Computes a uniform weight mixture across experts of a given library
    """
    def __init__(self, config: TaskArithmeticConfig = None):
        super().__init__(config or TaskArithmeticConfig())

    def merge_expert(self, experts, expert_vectors, trainable_params, base_expert, base_model_state_dict, expert_type):
        used, total = 0, 0
        for param_name in base_model_state_dict.keys():
            if param_name in trainable_params:
                # stack the expert weights
                expert_weights = self.extract_expert_weight(base_model_state_dict, experts, param_name, expert_type)

                # collect mask
                keep_mask = torch.ones_like(expert_weights)
                # keep weights over the threshold
                expert_weights = expert_weights * keep_mask
                # NOTE: sum for Task-Arithmetic
                final_param = expert_weights.sum(0)

                # -----------------------------------------------------
                # base_weight + sum of the "filtered" task-vector
                # W = W + delta_W
                # source : (a) https://openreview.net/pdf?id=6t0Kwf8-jrj (b) https://arxiv.org/pdf/2306.01708
                final_param = base_model_state_dict[param_name] + self.config.alpha * final_param

                
                used += keep_mask.sum().item()
                total += expert_weights.numel()

                base_expert.expert_weights[param_name].data.copy_(final_param)
            else:
                base_expert.expert_weights[param_name]=copy.deepcopy(base_model_state_dict[param_name])

        logger.info(
            "Params used to compute Ties mean: {:.10f}%".format(100.0 * used / total)
        )
        return base_expert

@dataclass
class TiesMergeSimpleConfig(BaseMergeConfig):
    top_k: float = 0.2
    merging_method: str = 'ties_merge_expert'
    alpha: float  = 0.4 # scaling factor # source : (a) https://openreview.net/pdf?id=6t0Kwf8-jrj (b) https://arxiv.org/pdf/2306.01708
    beta: float = 0.8  # 80% beta=sparsity, keep-ratio=1-beta, fig 3 https://arxiv.org/pdf/2306.01708 suggest to keep top20% params

class TiesMergeSimple(BaseMerge):
    """
    Computes a uniform weight mixture across experts of a given library
    """

    def __init__(self, config: TiesMergeSimpleConfig = None):
        super().__init__(config or TiesMergeSimpleConfig())

        assert self.config.top_k > 0.0 and self.config.top_k <= 1.0

    def compute_per_task_threhold(self, expert_vectors):
        # take the absolute value:
        expert_vectors = torch.stack(expert_vectors, dim=0).abs()
        topk = int(expert_vectors.size(1)* self.config.beta)
        per_exp_lth = topk_multiple_experts(expert_vectors, topk, TH_type='lower')

        return per_exp_lth

    @torch.no_grad()
    def merge_expert(self, experts, expert_vectors, trainable_params, base_expert, base_model_state_dict, expert_type):
        # ----------------------------------------------------------------------
        # Compute Threshold score, TH
        per_exp_lth = self.compute_per_task_threhold(expert_vectors)

        used, total = 0, 0
        for param_name in base_model_state_dict.keys():
            if param_name in trainable_params:
                # stack the expert weights
                expert_weights = self.extract_expert_weight(base_model_state_dict, experts, param_name, expert_type)

                # keep weights over the threshold
                TH = per_exp_lth.view(-1, *((1,) * (expert_weights.ndim - 1))) # reshape
                keep_mask = expert_weights.abs() > TH 
                expert_weights = expert_weights * keep_mask

                # sign majority vote
                #sign_per_dim = expert_weights.sign().sum(0, keepdim=True).sign()
                sign_per_dim = expert_weights.sum(0, keepdim=True).sign()
                # resolve zero signs: https://github.com/rezazzr/breadcrumbs/blob/main/src/task_vectors.py#L334
                majority_sign = torch.sign(sign_per_dim.sum())
                sign_per_dim[sign_per_dim == 0] = majority_sign

                # keep only weights whose sign agree with the majority
                use_for_avg = expert_weights.sign() == sign_per_dim

                deno = (use_for_avg!=0).sum(0).clamp(min=1.0)
                sum_param = (expert_weights * use_for_avg).sum(0)
                final_param = sum_param / deno
                used += (use_for_avg & (sign_per_dim != 0.0)).sum().item()

                # -----------------------------------------------------
                # base_weight + sum of the "filtered" task-vector
                # W = W + delta_W
                # source : (a) https://openreview.net/pdf?id=6t0Kwf8-jrj (b) https://arxiv.org/pdf/2306.01708
                final_param = base_model_state_dict[param_name] + self.config.alpha * final_param
                used += keep_mask.sum().item()
                total += expert_weights.numel()
                base_expert.expert_weights[param_name].data.copy_(final_param)
            else:
                base_expert.expert_weights[param_name]=copy.deepcopy(base_model_state_dict[param_name])

        logger.info(
            "Params used to compute Ties mean: {:.10f}%".format(100.0 * used / total)
        )
        return base_expert


@dataclass
class SparseSignFixConfig(BaseMergeConfig):
    top_k: float = 0.2
    merging_method: str = 'sparse_signfix_merge_expert'
    alpha: float  = 1 # scaling factor # source : (a) https://openreview.net/pdf?id=6t0Kwf8-jrj (b) https://arxiv.org/pdf/2306.01708
 

class SparseSignFix(BaseMerge):
    """
    Computes a uniform weight mixture across experts of a given library
    """

    def __init__(self, config: SparseSignFixConfig = None):
        super().__init__(config or SparseSignFixConfig())

        assert self.config.top_k > 0.0 and self.config.top_k <= 1.0

    @torch.no_grad()
    def merge_expert(self, experts, trainable_params, base_expert, base_model_state_dict, expert_type):
        param_dict= {}
        for param_name, base_w in base_expert.model.state_dict().items():
            if param_name in trainable_params:
                if 'weight' in param_name:
                    # stack the expert weights
                    expert_weights = self.extract_expert_weight(base_model_state_dict, experts, param_name, expert_type)
                    
                    # sign majority vote
                    sign_per_dim = expert_weights.sum(0, keepdim=True).sign()  # sum over N experts
                    use_for_avg = expert_weights.sign() == sign_per_dim
                    
                    sum_param = (expert_weights * use_for_avg).sum(0)
                    mask_overlaps = torch.stack([(e!=0).float() for e in (expert_weights * use_for_avg)],dim=0).sum(0)

                    # sum_param = expert_weights.sum(0)
                    # mask_overlaps = torch.stack([(e!=0).float() for e in expert_weights],dim=0).sum(0)
                    
                    mask_overlaps[mask_overlaps==0]=1
                    final_param = sum_param/mask_overlaps

                else:
                    expert_weights = self.extract_expert_weight(base_model_state_dict, experts, param_name, expert_type)
                    final_param = expert_weights.mean(0)

                param_dict[param_name] = final_param
            else:
                param_dict[param_name] = base_w
            

        return param_dict

    def load_mask(self, expert):
        try:
            print('trying to load mask from hf')
            library_id = expert.training_config.library_id
            from huggingface_hub import hf_hub_download
            destination_type, f_name = library_id.split('://')
            repo_id=('/').join(f_name.split('/')[:2])
            filename = f'{expert.expert_info.expert_name}_mask.npz'
            f_path=hf_hub_download(repo_id=repo_id, filename=filename)
            Mask = np.load(f_path, allow_pickle=True)['arr'].item()
        except:
            print('trying to load mask from local dir')
            m_loc = f'experiment/{expert.training_config.exp_name}/mask.npz'
            Mask = np.load(m_loc, allow_pickle=True)['arr'].item()
        return Mask

    def convert_idx_2_mask(self, weight_idx, mat_dim):
        m = np.zeros(mat_dim)
        m[tuple(zip(*weight_idx))] = 1
        return torch.FloatTensor(m)
    

    @torch.no_grad()
    def transform(self, library):
        experts, expert_type, base_expert, base_model, trainable_params = self.pre_configure(library)
        base_model_state_dict = dict(base_model.state_dict())


        from mttl.models.expert_model import ExpertModel
        base_expert.training_config.device_map='cpu'
        base_expert = ExpertModel(**vars(base_expert.training_config))
        trainable_params = [n for n in base_expert.model.state_dict().keys() if ('sparse_layer' in n)] # allow to add weights and bias
        assert trainable_params!=[], print('could not find sparse-layer modules')
        base_model_state_dict = base_expert.model.state_dict()

        param_dict = self.merge_expert(experts, trainable_params, base_expert, base_model_state_dict, expert_type)
        base_expert.model.load_state_dict(param_dict)

        return base_expert



class UniformSparse(SparseSignFix):
    """
    Computes a uniform weight mixture across experts of a given library
    """

    def __init__(self, config: SparseSignFixConfig = None):
        super().__init__(config or SparseSignFixConfig())
        assert self.config.top_k > 0.0 and self.config.top_k <= 1.0

    @torch.no_grad()
    def merge_expert(self, experts, trainable_params, base_expert, base_model_state_dict, expert_type):
        param_dict= {}
        for param_name, base_w in base_expert.model.state_dict().items():
            if param_name in trainable_params:
                # ignore bias
                if 'weight' in param_name:
                    # stack the expert weights
                    expert_weights = self.extract_expert_weight(base_model_state_dict, experts, param_name, expert_type)

                    sum_param = expert_weights.sum(0)
                    mask_overlaps = torch.stack([(e!=0).float() for e in expert_weights],dim=0).sum(0)
                    
                    mask_overlaps[mask_overlaps==0]=1
                    final_param = sum_param/mask_overlaps

                    layer_name = '.'.join(param_name.split('.')[:-2])
                    updated_param_name = f'{layer_name}.weight'
                    param_dict[updated_param_name] = final_param

        return param_dict


    @torch.no_grad()
    def transform(self, library):
        experts, expert_type, base_expert, base_model, trainable_params = self.pre_configure(library)
        base_model_state_dict = dict(base_model.state_dict())


        from mttl.models.expert_model import ExpertModel
        base_expert.training_config.device_map='cpu'
        base_expert = ExpertModel(**vars(base_expert.training_config))
        trainable_params = [n for n in base_expert.model.state_dict().keys() if ('sparse_layer' in n)] # allow to add weights and bias
        assert trainable_params!=[], print('could not find sparse-layer modules')
        base_model_state_dict = base_expert.model.state_dict()

        param_dict = self.merge_expert(experts, trainable_params, base_expert, base_model_state_dict, expert_type)
        
        config = base_expert.training_config
        config.model_modifier = None  # load only the base model
        config.device_map='cpu'
        config.trainable_param_names='.*' # allows to train all linear layers
        base_model = ExpertModel(**vars(config))

        for param_name, base_w in base_model.model.state_dict().items():
            if param_name in param_dict:
                param_dict[param_name] = base_w + param_dict[param_name].to(base_w.dtype)
            else:
                param_dict[param_name] = base_w

        assert set(base_model.model.state_dict().keys()) == set(param_dict.keys()), "Expert weights must have the same keys"
        base_model.model.load_state_dict(param_dict)

        return base_model


@dataclass
class ModelBreadcrumbsConfig(BaseMergeConfig):
    merging_method: str = 'model_breadcrumbs_expert'
    alpha: float = 0.4 # scaling factor # source : (a) https://openreview.net/pdf?id=6t0Kwf8-jrj (b) https://arxiv.org/pdf/2306.01708
    beta: float = 0.9  # 90% beta=sparsity, keep-ratio=1-beta
    gamma: float = 0.99 # mask out top 1%


class ModelBreadcrumbs(BaseMerge):
    """
    Computes a uniform weight mixture across experts of a given library
    """

    def __init__(self, config: ModelBreadcrumbsConfig = None):
        super().__init__(config or ModelBreadcrumbsConfig())

    @torch.no_grad()
    def compute_per_task_threhold(self, expert_vectors):

        # take the absolute value:
        expert_vectors = torch.stack(expert_vectors, dim=0).abs()
        lower_topk = int(expert_vectors.size(1)* self.config.beta)
        upper_topk = int(expert_vectors.size(1)* self.config.gamma)

        per_exp_lth = topk_multiple_experts(expert_vectors, lower_topk, TH_type='lower')
        per_exp_uth = topk_multiple_experts(expert_vectors, upper_topk, TH_type='upper')

        return per_exp_lth, per_exp_uth
    
    @torch.no_grad()
    def merge_expert(self, experts, expert_vectors, trainable_params, base_expert, base_model_state_dict, expert_type):
        # Compute Threshold score, TH
        per_exp_lth, per_exp_uth = self.compute_per_task_threhold(expert_vectors)
        used, total = 0, 0
        for param_name in base_model_state_dict.keys():
            if param_name in trainable_params:
                # stack the expert weights
                expert_weights = self.extract_expert_weight(base_model_state_dict, experts, param_name, expert_type)

                # keep weights over the threshold
                Lower_TH = per_exp_lth.view(-1, *((1,) * (expert_weights.ndim - 1)))
                Upper_TH = per_exp_uth.view(-1, *((1,) * (expert_weights.ndim - 1)))
                
                keep_mask = torch.logical_and(expert_weights.abs() > Lower_TH, expert_weights.abs() < Upper_TH) 
                #keep_mask = (expert_weights.abs() > Lower_TH and expert_weights.abs() < Upper_TH) 
                expert_weights = expert_weights * keep_mask

                # base_weight + sum of the "filtered" task-vector
                final_param = base_model_state_dict[param_name] + self.config.alpha * expert_weights.sum(0)
                
                
                used += keep_mask.sum().item()
                total += expert_weights.numel()

                base_expert.expert_weights[param_name].data.copy_(final_param)
            else:
                base_expert.expert_weights[param_name]=copy.deepcopy(base_model_state_dict[param_name])
        logger.info(
            "Params used to compute Model-breadcrumb mean: {:.10f}%".format(100.0 * used / total)
        )

        return base_expert