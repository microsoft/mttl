"""
Arrow transform algorithm for Expert objects.

This module implements Arrow transform that extracts input directions most affected
by the linear transforms in expert weights.
"""

import copy
from collections import defaultdict
from typing import Dict, List

import torch

from mttl.logging import logger
from mttl.models.library.expert import Expert
from mttl.models.library.library_transforms import ArrowTransformConfig
from mttl.models.modifiers.base import get_target_2_source_param_mapping


def arrow_transform(
    experts: List[Expert], config: ArrowTransformConfig
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Apply Arrow transform to experts to extract input directions most affected by linear transforms.

    Args:
        experts: List of Expert objects to transform
        config: ArrowTransformConfig containing transform parameters

    Returns:
        Dict mapping expert names to their Arrow prototypes (layer_name -> prototype vector)
    """
    if not experts:
        raise ValueError("Cannot transform empty list of experts")

    logger.info(f"Computing Arrow prototypes for {len(experts)} experts")

    vectors = {}
    eigvals = {}

    for expert in experts:
        expert_name = expert.name
        logger.info(f"Computing SVD for expert {expert_name}")
        vectors[expert_name] = {}
        eigvals[expert_name] = {}

        # get parameters tied during training
        param_map = get_target_2_source_param_mapping(
            expert.expert_weights.items(),
            expert.expert_info.expert_config.tie_params,
        )
        if config.tie_params != "default":
            # get parameters we wish to tie for Arrow
            _tied_params = get_target_2_source_param_mapping(
                expert.expert_weights.items(), config.tie_params
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
        tied_parents = _get_unique_parent_names(tied_params)
        untied_parents = [
            parent
            for parent in _get_unique_parent_names(expert.expert_weights.keys())
            if parent not in tied_parents
        ]

        # Build a mapping from source to target parameters
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

            for tied_module in dependents:
                logger.info(f"\t\t\tTying Arrow with {tied_module}")
                As += [expert.expert_weights[f"{tied_module}.lora_a"]]
                Bs += [expert.expert_weights[f"{tied_module}.lora_b"]]
                parent_names += [tied_module]

            if len(As) > 1:
                if config.tie_op == "concat":
                    # Mimicking phi-2 behavior
                    assert config.ab_only
                    assert all(
                        torch.allclose(A, As[0]) for A in As
                    ), "A should be the same for all tied parameters"
                    A = As[0]
                    B = torch.cat(Bs, dim=1)
                elif config.tie_op == "sum":
                    # A1B1 + A2B2 == [A1 A2] [B1; B2].
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

            if config.ab_only:
                U_W, Sigma_W, _ = _low_rank_svd(A, B)
                top_value = Sigma_W[0] ** 2
                top_vector = U_W[:, 0]
            else:
                raise NotImplementedError(
                    "Base model weights not supported in standalone function"
                )

            # Save eigenvector and eigenvalue
            for parent in parent_names:
                assert parent not in vectors[expert_name]
                vectors[expert_name][parent] = top_vector.real.cpu()
                eigvals[expert_name][parent] = top_value.item()

    # Apply scaling if requested
    if config.scale:
        output = {}
        for expert_name, expert_data in vectors.items():
            output[expert_name] = {}
            for layer_name, vector in expert_data.items():
                vector = vector * eigvals[expert_name][layer_name]
                output[expert_name][layer_name] = vector
        return output
    else:
        return vectors


def _get_unique_parent_names(alist):
    """
    if adict.keys() = ['model.layer1.lora_a', 'model.layer.lora_b', 'model.layer2.lora_a']
    output will be {'model.layer1', 'model.layer2'}
    """
    dict_keys = sorted(list(set(".".join(k.split(".")[:-1]) for k in alist)))
    return dict_keys


def _low_rank_svd(A, B):
    """Faster SVD computation for low rank matrices"""
    # Compute SVD of A
    U_A, Sigma_A, V_A = torch.svd(A)

    # Compute SVD of B.T (transpose of B)
    U_B, Sigma_B, V_B = torch.svd(B.T)

    # Compute product matrix C = Sigma_A * (V_A.T @ V_B) * Sigma_B
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
