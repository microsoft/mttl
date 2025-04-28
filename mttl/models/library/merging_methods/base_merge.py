import copy
import torch
from dataclasses import dataclass
from mttl.logging import logger
from mttl.models.library.expert_library import ExpertLibrary
from mttl.models.library.library_transforms import (
    LibraryTransform,
    LibraryTransformConfig,
)
from mttl.models.expert_model import ExpertModel, ExpertModelConfig
from mttl.models.utils import model_loader_helper
from mttl.models.library.merging_methods.utils import (
    load_mask,
    convert_idx_2_mask,
    dict_to_config,
)


@dataclass
class BaseMergeConfig(LibraryTransformConfig):
    merging_method: str = "BaseMerge"


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
        expert_type = experts[0].training_config["model_modifier"]
        if expert_type is None:
            expert_type = "FFT"

        # transform experts. NOTE: MUST
        self.transform_experts(experts, expert_type)

        # get base expert
        base_expert = copy.deepcopy(experts[0])
        base_expert.name = self.config.merging_method
        train_cfg = copy.deepcopy(base_expert.training_config)
        train_cfg["device_map"] = "cpu"
        trainable_params = list(
            base_expert.expert_weights.keys()
        )  # 'model.layers.0.self_attn.o_proj.weight'

        return experts, expert_type, base_expert, trainable_params

    @torch.no_grad()
    def extract_expert_vector(
        self, experts, expert_type, base_model_state_dict, trainable_params
    ):
        # Build n_tasks x D experts
        expert_vectors = []
        for expert in experts:
            if expert_type == "FFT":
                # W_t = W_base - W'
                expert_vectors += [
                    torch.nn.utils.parameters_to_vector(
                        list(
                            expert.expert_weights[k] - base_model_state_dict[k]
                            for k in trainable_params
                        )
                    )
                ]
                # W_t = (lora_a*lora_b).T
                # NOTE: it's already done when we call self.transform_experts()
            elif expert_type == "lora":
                weights_list = [
                    param.contiguous()
                    for param in (expert.expert_weights[k] for k in trainable_params)
                ]
                expert_vectors += [torch.nn.utils.parameters_to_vector(weights_list)]

        return expert_vectors

    @torch.no_grad()
    def extract_expert_weight(
        self, base_model_state_dict, experts, param_name, expert_type
    ):
        # for given "param_name", iterates over all expert and gets the trained expert-weights
        if expert_type == "FFT":
            expert_weights = torch.stack(
                [
                    expert.expert_weights[param_name]
                    - base_model_state_dict[param_name]
                    for expert in experts
                ],
                dim=0,
            )
        elif expert_type == "lora":
            expert_weights = torch.stack(
                [expert.expert_weights[param_name] for expert in experts], dim=0
            )
        elif expert_type == "sparse_mask_adapter":
            expert_weights = torch.stack(
                [expert.expert_weights[param_name] for expert in experts], dim=0
            )
        return expert_weights

    @torch.no_grad()
    def transform_experts(self, experts, expert_type):
        assert expert_type in ["FFT", "lora", "sparse_mask_adapter"], print(
            f"{expert_type} is not implemented"
        )
        if expert_type == "FFT":
            pass
        elif expert_type == "lora":
            # Lora
            base_expert = copy.deepcopy(experts[0])
            trainable_layers = [
                ".".join(l.split(".")[:-1])
                for l in list(base_expert.expert_weights.keys())
            ]  ## 'layers.0.self_attn.o_proj'
            trainable_layers = list(
                dict.fromkeys(trainable_layers)
            )  # removes duplicate layers (loraA,loraB), w/o changing layer order
            for expert in experts:
                for l in trainable_layers:
                    expert.expert_weights[f"{l}.weight"] = (
                        expert.expert_weights[f"{l}.lora_a"]
                        @ expert.expert_weights[f"{l}.lora_b"]
                    ).T
                    del (
                        expert.expert_weights[f"{l}.lora_a"],
                        expert.expert_weights[f"{l}.lora_b"],
                    )

            # NOTE: sanity check, please don't remove this block
            for expert in experts:
                for l in list(expert.expert_weights.keys()):
                    if "lora_a" in l or "lora_b" in l:
                        del expert.expert_weights[l]

        elif expert_type == "sparse_mask_adapter":
            base_expert = copy.deepcopy(experts[0])
            trainable_layers = [
                ".".join(l.split(".")[:-2])
                for l in list(base_expert.expert_weights.keys())
            ]  ## 'layers.0.self_attn.o_proj'
            trainable_layers = list(
                dict.fromkeys(trainable_layers)
            )  # removes duplicate layers (loraA,loraB), w/o changing layer order

            for expert in experts:
                Mask = load_mask(expert)
                # for each layer compute the "average of the overlapped weight"
                for l in trainable_layers:
                    # weight
                    mask = convert_idx_2_mask(
                        weight_idx=Mask["mask"][f"model.{l}.sparse_layer"],
                        mat_dim=Mask["mask_shape"][f"model.{l}.sparse_layer"],
                    )
                    mask_idx = torch.where(mask.flatten() == 1)[0]
                    expert_dtype = expert.expert_weights[
                        f"{l}.sparse_layer.weight"
                    ].dtype

                    dense_weight = torch.zeros(
                        Mask["mask_shape"][f"model.{l}.sparse_layer"],
                        dtype=expert_dtype,
                    )
                    dense_weight.flatten().scatter_add_(
                        0, mask_idx, expert.expert_weights[f"{l}.sparse_layer.weight"]
                    )
                    expert.expert_weights[f"{l}.weight"] = dense_weight
                    del expert.expert_weights[f"{l}.sparse_layer.weight"]

            # NOTE: sanity check, please don't remove this block
            for expert in experts:
                for l in list(expert.expert_weights.keys()):
                    if "sparse_layer" in l:
                        del expert.expert_weights[l]

    @torch.no_grad()
    def compute_per_task_threhold(self, expert_vectors):
        # Given expert vector, W_task compute TH score to prune parameters
        # NOTE: W_task = W_base - W'
        pass

    @torch.no_grad()
    def merge_expert(
        self,
        experts,
        expert_vectors,
        trainable_params,
        base_expert,
        base_model_state_dict,
        expert_type,
    ):
        pass

    @torch.no_grad()
    def transform(self, library):
        experts, expert_type, base_expert, trainable_params = self.pre_configure(
            library
        )

        train_cfg = base_expert.training_config
        if isinstance(train_cfg, dict):
            train_cfg = dict_to_config(train_cfg)
        # change the config to load the base model
        train_cfg.model_modifier = None  # load only the base model
        train_cfg.device_map = "cpu"
        train_cfg.trainable_param_names = ".*"  # change trainable param to all
        base_model = model_loader_helper(
            train_cfg.model,
            load_in_8bit=train_cfg.load_in_8bit,
            load_in_4bit=train_cfg.load_in_4bit,
            device_map=getattr(train_cfg, "device_map", "cpu"),
        )
        # wrap base-model with `ExpertModel` class
        base_model = ExpertModel(
            ExpertModelConfig(base_model=base_model), **vars(train_cfg)
        )
        base_model_state_dict = dict(base_model.state_dict())
        # ----------------------------------------------------------------------
        # Collect Expert-vector
        # for FFT:    expert_vectors, delta_W = W-W'
        # for LoRA:   expert_vectors, delta_W = A.B
        # for sparse: expert_vectors, delta_W = delta_W*mask (NOTE: not implemented yet)
        expert_vectors = self.extract_expert_vector(
            experts, expert_type, base_model_state_dict, trainable_params
        )
        # merge weights of all the experts
        base_expert = self.merge_expert(
            experts,
            expert_vectors,
            trainable_params,
            base_expert,
            base_model_state_dict,
            expert_type,
        )
        # load state_dict into model
        assert set(base_model.state_dict().keys()) == set(
            base_expert.expert_weights.keys()
        ), "Expert weights must have the same keys"
        base_model.load_state_dict(base_expert._expert_weights)
        return base_model
