from abc import abstractmethod
import abc
from dataclasses import dataclass
import dataclasses
import copy
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import sklearn.decomposition
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from collections import defaultdict

from mttl.models.modifiers.expert_containers.expert import Expert
from mttl.models.modifiers.expert_containers.expert_containers import ExpertContainer
from mttl.models.modifiers.expert_containers.expert_library import ExpertLibrary
from mttl.utils import logger
from mttl.models.utils import EfficientCheckpointModule, transfer_batch_to_device
from mttl.datamodule.base import get_datamodule
from mttl.models.expert_config import ExpertConfig


class LibraryTransform(abc.ABC):
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


def param_hash(p):
    # from facebookresearch / ReAgent
    import hashlib

    m = hashlib.md5()
    m.update(
        str(
            tuple(_hash_field(getattr(p, f.name)) for f in dataclasses.fields(p))
        ).encode()
    )
    return m.hexdigest()


@dataclass
class LibraryTransformConfig:
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
            save_name = self.__class__.__name__.lower() + f"-{param_hash(self)}"
            return save_name


@dataclass
class SVDEmbeddingTransformConfig(LibraryTransformConfig):
    n_components: int = 64
    sparsity_threshold: float = 0.8


class SVDEmbeddingTransform(LibraryTransform):
    """Creates adapter embeddings by low-rank decomposition of a sparsified version
    of the adapter modules.
    """

    def __init__(self, config, random_state=None):
        super().__init__(config)
        self.random_state = random_state

    def transform(self, library, persist=True, recompute=False):
        if type(library) == str:
            library = ExpertLibrary.get_expert_library(library)

        # try to fetch auxiliary data
        output = library.get_auxiliary_data(data_type=self.config.save_name)

        if len(output) > 0 and not recompute:
            logger.info("Found {} precomputed SVD Embeddings".format(len(output)))

            return (
                np.stack([output[expert_name] for expert_name in library.keys()]),
                None,
            )

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
        return dict(zip(names, experts_embeddings)), svd


@dataclass
class WeightedLinearMergeConfig(LibraryTransformConfig):
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
class TiesMergeConfig(LibraryTransformConfig):
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
class HiddenStateComputerConfig(LibraryTransformConfig):
    use_base_model_only: bool = (
        False  # This computes sentence embeddings without the adapter
    )
    model: str = None  # If `use_base_model_only`, can pass a specific model to compute embeddings with
    max_samples_per_task: int = 10
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
        model.container = {}
        if model.model is None:
            raise ValueError("Model must have a model attribute")
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
    def transform(
        self, library: ExpertLibrary, persist=False, recompute=False, default_args=None
    ) -> Expert:
        from mttl.models.expert_model import MultiExpertModel

        if type(library) == str:
            library = ExpertLibrary.get_expert_library(library)

        logger.info(f"Hidden state computer dumps to: {self.config.save_name}")

        output = library.get_auxiliary_data(data_type=self.config.save_name)

        if len(output) == len(library) and not recompute:
            logger.info("Found {} precomputed centroids".format(len(output)))
            return output

        logger.info("Computing centroids for {} experts".format(len(library)))
        output = {}

        for _, (expert_name, expert) in enumerate(library.items()):
            training_config = expert.training_config
            if default_args is not None:
                self._fill_missing_args(training_config, default_args)

            if self.config.use_base_model_only and self.config.model is not None:
                training_config.model = self.config.model

            model = MultiExpertModel(**vars(training_config)).to("cuda")

            if not self.config.use_base_model_only:
                model.add_expert_instance(expert, is_default=True)

            self._track_hidden_states(model, keys=expert.expert_weights.keys())
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

            for _, batch in pbar:
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
                hidden_states = self._retrieve_hidden_states(model)

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
class PhatgooseConfig(LibraryTransformConfig):
    n_steps: int = 1000
    learning_rate: float = 3e-3


class PhatgooseTransform(HiddenStateComputer):
    def __init__(self, config: PhatgooseConfig = None):
        super().__init__(config or PhatgooseConfig())

    @torch.no_grad()
    def transform(
        self,
        library,
        persist: bool = True,
        recompute: bool = False,
        expert_names: list = None,
        default_args=None,
    ):
        from mttl.models.expert_model import MultiExpertModel
        from mttl.models.modifiers.expert_containers.utils import train_module

        outputs = {}
        expert_names = expert_names or list(library.keys())
        for expert_name in expert_names:
            logger.info(f"Computing centroids for expert {expert_name}")
            expert: Expert = library[expert_name]

            if type(library) == str:
                library = ExpertLibrary.get_expert_library(library)
            loaded_output = library.get_auxiliary_data(data_type=self.config.save_name)

            if (
                not recompute
                and len(loaded_output) > 0
                and expert_name
                in loaded_output  # cause loaded_output loads all experts
            ):
                logger.info("Found {} precomputed centroids".format(len(loaded_output)))
                # format is dict[layer_name] = embedding, layer_name ends with selector.{task_name}.v
                outputs[expert_name] = (
                    loaded_output
                    if not expert_name in loaded_output
                    else loaded_output[expert_name]
                )
                continue

            training_config: ExpertConfig = expert.training_config
            training_config.router_selector = "phatgoose_selector"
            training_config.trainable_param_names = ".*selector.*"
            training_config.logging_prefix = expert_name + "/"
            training_config.weight_decay = 0.0
            training_config.lora_merge_after = True
            model = MultiExpertModel(**vars(training_config)).to("cuda")
            model.add_expert_instance(expert, is_default=True)

            # for checksum
            p_sum_before = sum(
                p.sum()
                for n, p in model.named_parameters()
                if "selector" not in n
                and "norm" not in n
                and "ln" not in n
                and "layer_norm" not in n
            )
            p_sum_sel_before = sum(
                p.sum() for n, p in model.named_parameters() if "selector" in n
            )

            training_config.dataset = expert.expert_info.dataset
            if expert.expert_info.expert_task_name:
                train_tasks = expert.expert_info.expert_task_name.split(",")
                training_config.finetune_task_name = ",".join(train_tasks)
            else:
                train_tasks = None

            if default_args is not None:
                self._fill_missing_args(training_config, default_args)
                training_config.include_task_source = default_args.include_task_source
                training_config.output_dir = default_args.output_dir
                training_config.wandb_project = default_args.wandb_project
                # we set also train_batch_size, micro_batch_size, and gradient_accumulation_steps from default_args
                # TODO: correct this in the future
                # background: gradient_accumulation_steps is set in post_init of the mttl/config.py and hence not saved correctly in expert's config
                training_config.gradient_accumulation_steps = (
                    default_args.gradient_accumulation_steps
                )
                training_config.train_batch_size = default_args.train_batch_size
                training_config.micro_batch_size = default_args.micro_batch_size

            # get datamodule
            dm = get_datamodule(training_config)
            training_config.eval_every = -1
            training_config.total_steps = self.config.n_steps
            training_config.learning_rate = self.config.learning_rate
            training_config.warmup_steps = 0
            training_config.warmup_proportion = 0.0
            checkpoint = train_module(training_config, model, dm)

            model_after = MultiExpertModel(**vars(training_config)).to("cuda")
            model_after.add_expert_instance(expert, is_default=True)
            model_after.load_state_dict(torch.load(checkpoint)["state_dict"])

            p_sum_after = sum(
                p.sum()
                for n, p in model_after.named_parameters()
                if ".selector" not in n
                and ".norm" not in n
                and ".ln." not in n
                and "layer_norm" not in n
            )
            assert p_sum_before == p_sum_after

            model_before = MultiExpertModel(**vars(training_config)).to("cuda")
            model_before.add_expert_instance(expert, is_default=True)

            p_sum_sel_after = sum(
                p.sum() for n, p in model.named_parameters() if "selector" in n
            )
            assert (
                p_sum_sel_before != p_sum_sel_after
            ), "Selector parameters have not changed after training"

            # extract prototypes
            prototypes = {}
            for name, module in model.named_modules():
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
        return outputs


@dataclass
class ArrowConfig(LibraryTransformConfig):
    ab_only: bool = True
    scale: bool = False  # If True, scale by eigenvalue


class ArrowTransform(LibraryTransform):
    """
    Given a library of experts, extract the input direction most affected by the linear transforms
    """

    def __init__(self, config: ArrowConfig = None):
        super().__init__(config or ArrowConfig())

    def _maybe_scale(self, vectors, eigvals):
        """Post Processing of the retrieved outputs,
        scales the output by the eigenvalue if needed"""
        output = {}
        for expert_name, expert_data in vectors.items():
            output[expert_name] = {}
            for layer_name, vector in expert_data.items():
                if self.config.scale:
                    vector = vector * eigvals[expert_name][layer_name]
                output[expert_name][layer_name] = torch.from_numpy(vector)

        return output

    @torch.no_grad()
    def transform(self, library, persist=True, recompute=False) -> Expert:
        if isinstance(library, str):
            library = ExpertLibrary.get_expert_library(library)

        # try to fetch auxiliary data
        vectors = library.get_auxiliary_data(
            data_type=self.config.save_name + "_vectors"
        )
        eigvals = library.get_auxiliary_data(
            data_type=self.config.save_name + "_eigvals"
        )

        if len(vectors) == len(eigvals) == len(library) and not recompute:
            logger.info("Found {} precomputed centroids".format(len(vectors)))

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
                from mttl.models.expert_model import MultiExpertModel

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

                if self.config.ab_only:
                    # Now, the efficient way
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
                    top_vector = U_W[:, 0]
                    top_value = Sigma_C[0] ** 2
                    bottom_vector = U_W[:, -1]

                else:
                    base_W = base_model.model.state_dict()[f"{param_name}.weight"]
                    W_AB = base_W + W
                    eig_input = W_AB.T @ W_AB
                    out = torch.linalg.eig(eig_input)
                    eigvector = out.eigenvectors
                    largest, smallest = eigvector[:, 0], eigvector[:, -1]

                    img_eig_vals = eigvector.imag.abs().mean().item()
                    if img_eig_vals > 1e-2:
                        logger.warning(
                            f"Found {img_eig_vals} imaginary eigenvalues, this is likely due to numerical instability"
                        )

                    top_vector = largest.real
                    top_value = out.eigenvalues.real[0]
                    bottom_vector = smallest.real

                # Check that top vector is indeed an eigenvector
                WTW = W.T @ W
                ratio = WTW @ top_vector / (top_vector * top_value)
                torch.allclose(ratio, torch.ones_like(ratio), atol=1e-3)

                # Check that top vector is indeed the top eigenvector
                assert (WTW @ top_vector).pow(2).sum() > (WTW @ bottom_vector).pow(
                    2
                ).sum()

                # Save eigenvector and eigvenvalue
                vectors[expert_name][param_name] = top_vector.real.cpu().numpy()
                eigvals[expert_name][param_name] = top_value.item()

        if persist:
            # add embeddings to the library
            with library.batched_commit():
                for expert_name in library.keys():
                    logger.info(
                        f"Uploading centroids to the library for expert {expert_name}"
                    )
                    for data_name, data in [
                        ("vectors", vectors),
                        ("eigvals", eigvals),
                    ]:
                        library.add_auxiliary_data(
                            data_type=self.config.save_name + "_" + data_name,
                            expert_name=expert_name,
                            config=self.config.__dict__,
                            data=data[expert_name],
                            force=True,  # make sure we overwrite
                        )
        return self._maybe_scale(vectors, eigvals)


@dataclass
class ExpertProjectorConfig:
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
            self._fill_missing_args(training_config, default_args)

        training_config.train_batch_size = (
            default_args.predict_batch_size if default_args is not None else 4
        )
        training_config.finetune_task_name = ",".join(
            [
                library[exp_name].training_config.finetune_task_name
                for exp_name in library.keys()
            ]
        )

        from projects.wiki_experts.src.expert_model import MoETrainer
        from mttl.models.modifiers.expert_containers import ExpertContainer

        model = MoETrainer(**vars(training_config)).to("cuda")

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
                selector_out.modules = random_tasks.tolist()
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
        for module_name, module in model.named_modules():
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

            if isinstance(model, EfficientCheckpointModule):
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
    ) -> dict[str, list[str]]:
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
            embeddings, svd = svd_embedder.transform(library, persist=persist)
            del svd_embedder
            return embeddings, svd

        embeddings = library.get_auxiliary_data(svd_config.save_name)

        if len(embeddings) != len(library) or recompute:
            logger.info("Recomputing embeddings for clustering.")
            embeddings, _ = create_embeddings()

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
            clusters[label].append(key)
        return clusters
