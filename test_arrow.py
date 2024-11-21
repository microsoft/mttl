# load the library

from mttl.models.library.expert_library import ExpertLibrary
import torch
from collections import defaultdict


def _low_rank_svd(A, B):
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
    # if diff_AB[0] < 0.9:
    #     print("The first singular vector of U_A and U_AB are not aligned")

    return U_W, Sigma_C, V_W_T


library = ExpertLibrary.get_expert_library("zhan1993/trained_gpt125m_mbc_experts_colab")

## compute arrow embeddings

vectors = {}
eigvals = {}

for expert_name, expert in library.items():
    vectors[expert_name] = {}
    eigvals[expert_name] = {}

    # get the parent layer of the LoRA adapter
    layer_keys = list(expert.expert_weights.keys())
    layer_parent_keys = list(map(lambda x: ".".join(x.split(".")[:-1]), layer_keys))

    tied_param_bins = defaultdict(list)

    for parent in layer_parent_keys:
        tied_param_bins[parent] = []

    for parent_name, dependents in tied_param_bins.items():
        print(f"\tComputing SVD for parameter {parent_name}")

        parent_names = [parent_name]
        A_name, B_name = f"{parent_name}.lora_a", f"{parent_name}.lora_b"
        As = [expert.expert_weights[A_name]]
        Bs = [expert.expert_weights[B_name]]
        base_W = []

        A, B = As[0], Bs[0]

        # Reshape As and Bs (needed for Poly / MHR weights)
        rank = expert.expert_config.lora_rank
        A = A.reshape(-1, rank).float()
        B = B.reshape(rank, -1).float()

        W = (A @ B).T  # out_features, in_features

        # Compute SVD of expert
        U_W, Sigma_W, _ = _low_rank_svd(A, B)
        top_value = Sigma_W[0] ** 2
        bottom_vector = U_W[:, -1]
        top_vector = U_W[:, 0]

        # Compute the SVD of the W
        U, E, Vt = torch.linalg.svd(W)
        top_vector_ = Vt[0]
        bottom_vector_ = Vt[-1]
        top_value_ = E[0] ** 2

        # Check that top vector is indeed an eigenvector
        WTW = W.T @ W
        ratio = WTW @ top_vector / (top_vector * top_value)

        ratio_ = WTW @ top_vector_ / (top_vector_ * top_value_)

        torch.allclose(ratio, torch.ones_like(ratio), atol=1e-3)
        torch.allclose(ratio_, torch.ones_like(ratio_), atol=1e-3)

        # Check that top vector is indeed the top eigenvector
        assert (WTW @ top_vector).pow(2).sum() > (WTW @ bottom_vector).pow(2).sum()

        vectors[expert_name][parent] = top_vector.real.cpu().numpy()
        eigvals[expert_name][parent] = top_value.item()
