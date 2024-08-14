# unit test for adapter_ranker
import copy
import logging
from collections import OrderedDict

import numpy as np
import pytest
import torch
from pytorch_lightning import seed_everything

from mttl.config import ExpertConfig
from mttl.logging import logger
from mttl.models.library.expert_library import HFExpertLibrary, LocalExpertLibrary
from mttl.models.library.library_transforms import (
    ArrowConfig,
    ArrowTransform,
    MBClusteringTransformConfig,
    MBCWithCosSimTransform,
    TiesMerge,
    TiesMergeConfig,
    WeightedLinearMerge,
    WeightedLinearMergeConfig,
)


def test_config():
    cfg = ArrowConfig(ab_only=True, scale=False)
    assert cfg.save_name == "arrowconfig-a8327e21d374166ceeb94c40d2e7676f"

    cfg2 = ArrowConfig(ab_only=True, scale=True)
    assert cfg2.save_name != cfg.save_name

    cfg3 = ArrowConfig(ab_only=True, scale=False)
    assert cfg3.save_name == cfg.save_name


def test_arrow():
    logger.setLevel(logging.DEBUG)

    library = HFExpertLibrary("sordonia/new-test-library")

    cfg = ArrowConfig(ab_only=True, scale=False)
    transform = ArrowTransform(cfg)

    protos = transform.transform(library, persist=False, recompute=True)
    sums = []
    for task_name in sorted(protos.keys()):
        task_sum = 0.0
        for key in protos[task_name].keys():
            task_sum += protos[task_name][key].sum().item()
        sums.append(task_sum)

    assert np.allclose(sums, [2728.4163, 2284.9968])


def test_arrow_with_tiedlora(tmp_path, create_dummy_expert):
    seed_everything(0)
    logger.setLevel(logging.DEBUG)

    def patch_expert_weights(expert, offset=0):
        keys = sorted(expert.expert_weights.keys())
        for idx, k in enumerate(keys):
            v = expert.expert_weights[k]
            if "q_proj" in k or "k_proj" in k or "v_proj" in k or "o_proj" in k:
                parent = ".".join(k.split(".")[:-1])
                assert parent + ".lora_a" in expert.expert_weights
                assert parent + ".lora_b" in expert.expert_weights
            gen = torch.Generator()
            if "lora_b" in k:
                gen.manual_seed(idx + offset)
            elif "lora_a" in k:
                # map q_proj, k_proj, v_proj or o_proj to q_proj
                base_name = parent = ".".join(k.split(".")[:-2] + ["k_proj.lora_a"])
                logger.debug(f"from {k} to {base_name}")
                gen.manual_seed(keys.index(base_name) + offset)

            expert.expert_weights[k] = torch.randn(
                size=v.size(), dtype=v.dtype, generator=gen
            )

        return expert

    config = ExpertConfig(
        **{
            "tie_params": "q_proj.*\\.lora_a|k_proj.*\\.lora_a|v_proj.*\\.lora_a",
            "model_modifier": "lora",
            "lora_rank": 16,
            "lora_alpha": 1.0,
            "modify_layers": "k_proj|v_proj|q_proj|o_proj",
            "modify_modules": ".*self_attn.*",
            "trainable_param_names": ".*lora_[ab].*",
            "output_dir": tmp_path,
        }
    )
    # create random Lora
    expert1 = patch_expert_weights(create_dummy_expert(config, "module1"), offset=0)
    expert2 = patch_expert_weights(create_dummy_expert(config, "module2"), offset=1_000)

    library = LocalExpertLibrary(tmp_path)
    library.add_expert(expert1, expert1.name)
    library.add_expert(expert2, expert2.name)

    cfg = ArrowConfig(ab_only=True, scale=False)
    transform = ArrowTransform(cfg)

    protos = transform.transform(library, persist=False, recompute=True)
    sums = []
    for task_name in sorted(protos.keys()):
        task_sum = 0.0
        for key in protos[task_name].keys():
            task_sum += protos[task_name][key].sum().item()
        sums.append(task_sum)

    assert np.allclose(sums, [-13.642, -7.734], atol=1e-3)


def test_compute_svd_embeddings():
    from mttl.models.library.library_transforms import (
        SVDEmbeddingTransform,
        SVDEmbeddingTransformConfig,
    )

    library = HFExpertLibrary("sordonia/new-test-library")
    embeddings = SVDEmbeddingTransform(
        SVDEmbeddingTransformConfig(n_components=2)
    ).transform(library=library, persist=False)

    assert len(embeddings) == 2
    assert embeddings["abstract_algebra"].shape[0] == 2


def test_mbc_clustering(tmp_path):
    library = HFExpertLibrary("sordonia/new-test-library")
    k = 2
    cfg = MBClusteringTransformConfig(
        k=k,
        random_state=42,
        sparsity_threshold=0.1,
    )
    transform = MBCWithCosSimTransform(cfg)
    clusters = transform.transform(library, recompute=True)
    assert len(clusters) == k


def test_weighted_merge():
    library = HFExpertLibrary("sordonia/new-test-library")

    transform = WeightedLinearMerge()
    exp = transform.transform(library)

    weights = torch.zeros(10).uniform_(0, 10)
    weights /= weights.sum()
    exp_names = list(library.keys())
    weights = {exp_names[i]: weights[i] for i in range(len(exp_names))}

    cfg = WeightedLinearMergeConfig(weights=weights)
    transform = WeightedLinearMerge(cfg)
    weighted_exp = transform.transform(library)

    assert set(weighted_exp.expert_weights.keys()) == set(exp.expert_weights.keys())

    for key in weighted_exp.expert_weights.keys():
        weighted_param = torch.stack(
            [
                exp.expert_weights[key] * weights[exp_name]
                for (exp_name, exp) in library.items()
            ]
        ).sum(0)

        avg_param = torch.stack(
            [exp.expert_weights[key] for (exp_name, exp) in library.items()]
        ).mean(0)

        assert torch.allclose(weighted_param, weighted_exp.expert_weights[key])
        assert torch.allclose(avg_param, exp.expert_weights[key])


def test_ties_merge():
    logger.setLevel(logging.DEBUG)

    top_k = 0.2
    library = HFExpertLibrary("sordonia/new-test-library")
    names = list(library.keys())
    experts = list([library[name] for name in names])

    """ Copy Pasta of the original implementation
    https://github.com/prateeky2806/ties-merging/blob/main/src/ties_minimal.ipynb
    """

    def vector_to_state_dict(vector, state_dict, remove_keys=[]):
        # create a reference dict to define the order of the vector
        reference_dict = copy.deepcopy(state_dict)
        for key in remove_keys:
            if key in reference_dict:
                del reference_dict[key]
        sorted_reference_dict = OrderedDict(sorted(reference_dict.items()))

        # create a shared state dict using the refence dict
        torch.nn.utils.vector_to_parameters(vector, sorted_reference_dict.values())

        return sorted_reference_dict

    def state_dict_to_vector(state_dict, remove_keys=[]):
        shared_state_dict = copy.deepcopy(state_dict)
        for key in remove_keys:
            if key in shared_state_dict:
                del shared_state_dict[key]
        sorted_shared_state_dict = OrderedDict(sorted(shared_state_dict.items()))
        return torch.nn.utils.parameters_to_vector(
            [value.reshape(-1) for key, value in sorted_shared_state_dict.items()]
        )

    tv_flat_checks = torch.vstack(
        [state_dict_to_vector(expert.expert_weights) for expert in experts]
    )

    ## TIES MERGING UTILS
    def topk_values_mask(M, K=0.7, return_th=False):
        if K > 1:
            K /= 100

        original_shape = M.shape
        if M.dim() == 1:
            M = M.unsqueeze(0)

        n, d = M.shape
        k = int(d * K)
        k = d - k  # Keep top k elements instead of bottom k elements

        # Find the k-th smallest element by magnitude for each row
        kth_values, _ = M.abs().kthvalue(k, dim=1, keepdim=True)
        # Create a mask tensor with True for the top k elements in each row
        mask_ = M.abs() >= kth_values
        # my implementation
        TH = M.abs().quantile(1 - K, dim=1, keepdim=True)
        mask = M.abs() >= TH
        print(f"mask diff : {(mask_ != mask).sum()} over {mask.numel()}")

        final_mask = mask.squeeze() if original_shape == M.squeeze().shape else mask

        if return_th:
            return M * final_mask, final_mask.float().mean(dim=1), TH
        return M * final_mask, final_mask.float().mean(dim=1)

    def resolve_zero_signs(sign_to_mult, method="majority"):
        majority_sign = torch.sign(sign_to_mult.sum())

        if method == "majority":
            sign_to_mult[sign_to_mult == 0] = majority_sign
        elif method == "minority":
            sign_to_mult[sign_to_mult == 0] = -1 * majority_sign
        return sign_to_mult

    def resolve_sign(Tensor):
        sign_to_mult = torch.sign(Tensor.sum(dim=0))
        sign_to_mult = resolve_zero_signs(sign_to_mult, "majority")
        return sign_to_mult

    def disjoint_merge(Tensor, merge_func, sign_to_mult):
        merge_func = merge_func.split("-")[-1]

        # If sign is provided then we select the corresponding entries and aggregate.
        if sign_to_mult is not None:
            rows_to_keep = torch.where(
                sign_to_mult.unsqueeze(0) > 0, Tensor > 0, Tensor < 0
            )
            selected_entries = Tensor * rows_to_keep
        # Else we select all non-zero entries and aggregate.
        else:
            rows_to_keep = Tensor != 0
            selected_entries = Tensor * rows_to_keep

        if merge_func == "mean":
            non_zero_counts = (selected_entries != 0).sum(dim=0).float()
            disjoint_aggs = torch.sum(selected_entries, dim=0) / torch.clamp(
                non_zero_counts, min=1
            )
        elif merge_func == "sum":
            disjoint_aggs = torch.sum(selected_entries, dim=0)
        elif merge_func == "max":
            disjoint_aggs = selected_entries.abs().max(dim=0)[0]
            disjoint_aggs *= sign_to_mult
        else:
            raise ValueError(f"Merge method {merge_func} is not defined.")

        return disjoint_aggs

    def ties_merging(
        flat_task_checks,
        reset_thresh=None,
        merge_func="",
    ):
        all_checks = flat_task_checks.clone()
        updated_checks, _, TH = topk_values_mask(
            all_checks, K=reset_thresh, return_th=True
        )
        final_signs = resolve_sign(updated_checks)
        assert final_signs is not None

        merged_tv = disjoint_merge(updated_checks, merge_func, final_signs)

        return merged_tv, TH

    # return merged flat task vector
    merged_tv, TH = ties_merging(
        tv_flat_checks,
        reset_thresh=top_k,
        merge_func="dis-mean",
    )

    ref_ties_ckpt = vector_to_state_dict(
        merged_tv, experts[0].expert_weights, remove_keys=[]
    )

    # Compare ref implementation to ours
    cfg = TiesMergeConfig(top_k=top_k)
    transform = TiesMerge(cfg)
    ties_exp = transform.transform(library)

    assert set(ties_exp.expert_weights.keys()) == set(ref_ties_ckpt.keys())

    for param_name, expected_param in ref_ties_ckpt.items():
        value = ties_exp.expert_weights[param_name]
        assert torch.allclose(expected_param, value)


if __name__ == "__main__":
    pytest.main([__file__])
