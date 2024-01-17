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
class PrototypeComputerConfig:
    use_base_model_only: bool = (
        False  # This computes sentence embeddings without the adapter
    )
    model: str = None  # If `use_base_model_only`, can pass a specific model to compute embeddings with
    max_samples_per_task: int = 100
    upload_to_hf: bool = False
    name: str = "dataset_centroids"
    recompute: bool = False


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
