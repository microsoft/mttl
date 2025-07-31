import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional


class AdaptiveLoRAMerger:
    def __init__(self, math_lora_A, math_lora_B, code_lora_A, code_lora_B):
        """
        adaptive merge the lora: based on the input, we merge the lora to get the weights

        Args:
            math_lora_A, math_lora_B: math lora matrix
            code_lora_A, code_lora_B: code lora matrix
        """
        self.math_lora_A = math_lora_A
        self.math_lora_B = math_lora_B
        self.code_lora_A = code_lora_A
        self.code_lora_B = code_lora_B

        # we precompute the eigenvectors and eigenvalues of the lora matrix
        self._precompute_eigenvectors()

    def _precompute_eigenvectors(self):
        """
        use svd to precompute the eigenvectors and eigenvalues of the lora matrix
        """
        # math_lora_A: [rank, d_in] -> U[rank, rank], S[rank], V[d_in, rank]
        math_U, math_S, math_V = torch.svd(self.math_lora_A)
        code_U, code_S, code_V = torch.svd(self.code_lora_A)

        # V matrix is the main direction vector in the input space [d_in, rank]
        self.math_eigenvecs = math_V  # [d_in, rank]
        self.code_eigenvecs = code_V  # [d_in, rank]

        # singular values are the square roots of the eigenvalues
        self.math_eigenvals = math_S  # [rank]
        self.code_eigenvals = code_S  # [rank]

    def compute_projection_based_coefficients(
        self, X: torch.Tensor, top_k: int = 5
    ) -> Tuple[float, float]:
        """
        based on the input X, we compute the coefficients of the lora

        Args:
            X: input vector, shape: [batch_size, d_in] or [d_in]
            top_k: use the top k eigenvectors

        Returns:
            (code_coeff, math_coeff): code and math adapter coefficients
        """
        if X.dim() == 1:
            X = X.unsqueeze(0)
        X = X.to(self.math_eigenvecs.dtype)

        batch_size = X.shape[0]

        # compute the projection strength of X on each eigenvector
        math_projections = []
        code_projections = []

        for i in range(min(top_k, self.math_eigenvecs.shape[1])):
            # project X to the main direction vector: X @ v_i, where v_i is the i-th main direction vector
            math_proj = torch.abs(X @ self.math_eigenvecs[:, i])  # [batch_size]
            code_proj = torch.abs(X @ self.code_eigenvecs[:, i])  # [batch_size]

            #  weight the projection strength by the corresponding singular value
            math_weighted = math_proj * self.math_eigenvals[i]
            code_weighted = code_proj * self.code_eigenvals[i]

            math_projections.append(math_weighted)
            code_projections.append(code_weighted)

        # aggregate the projection strength
        math_strength = (
            torch.stack(math_projections).sum(dim=0).mean()
        )  # average over batch
        code_strength = torch.stack(code_projections).sum(dim=0).mean()

        # normalize the coefficients
        total_strength = math_strength + code_strength + 1e-8  # avoid division by zero
        code_coeff = code_strength / total_strength
        math_coeff = math_strength / total_strength

        return code_coeff.item(), math_coeff.item()

    def compute_fast_svd_coefficients(
        self, X: torch.Tensor, top_k: int = 5
    ) -> Tuple[float, float]:
        """
        the fastest svd method: directly use the singular value decomposition result of LoRA_A
        avoid any matrix multiplication and covariance calculation

        Args:
            X: input vector, shape: [batch_size, d_in] or [d_in]
            top_k: use the top k eigenvectors

        Returns:
            (code_coeff, math_coeff): code and math adapter coefficients
        """
        if X.dim() == 1:
            X = X.unsqueeze(0)

        # directly use the svd result of LoRA_A, without storing, for real-time calculation
        with torch.no_grad():  # no gradient, accelerate the calculation
            _, math_S, math_V = torch.svd(self.math_lora_A)  # math_V: [d_in, rank]
            _, code_S, code_V = torch.svd(self.code_lora_A)  # code_V: [d_in, rank]

            # compute the projection strength of X on each eigenvector
            actual_k = min(top_k, math_V.shape[1], code_V.shape[1])
            math_V_topk = math_V[:, :actual_k]  # [d_in, actual_k]
            code_V_topk = code_V[:, :actual_k]  # [d_in, actual_k]
            math_S_topk = math_S[:actual_k]  # [actual_k]
            code_S_topk = code_S[:actual_k]  # [actual_k]

            # compute the projection strength: X @ V, then weight and sum
            math_proj = (X @ math_V_topk).abs()  # [batch_size, actual_k]
            code_proj = (X @ code_V_topk).abs()  # [batch_size, actual_k]

            # weight and sum the projection strength
            math_strength = (math_proj * math_S_topk).sum(dim=1).mean()
            code_strength = (code_proj * code_S_topk).sum(dim=1).mean()

            # normalize the coefficients
            total_strength = math_strength + code_strength + 1e-8
            code_coeff = code_strength / total_strength
            math_coeff = math_strength / total_strength

            return code_coeff.item(), math_coeff.item()

    def compute_ultra_fast_coefficients(self, X: torch.Tensor) -> Tuple[float, float]:
        """
        the fastest method: directly compute the inner product strength of X and LoRA_A
        suitable for real-time inference
        """
        if X.dim() == 1:
            X = X.unsqueeze(0)

        with torch.no_grad():
            # compute the correlation between X and LoRA_A
            math_corr = (
                X.unsqueeze(1) @ self.math_lora_A.unsqueeze(0).transpose(-1, -2)
            ).abs()  # [batch, 1, rank]
            code_corr = (
                X.unsqueeze(1) @ self.code_lora_A.unsqueeze(0).transpose(-1, -2)
            ).abs()  # [batch, 1, rank]

            # sum the correlation strength
            math_strength = math_corr.sum(dim=-1).mean()
            code_strength = code_corr.sum(dim=-1).mean()

            # normalize the coefficients
            total_strength = math_strength + code_strength + 1e-8
            code_coeff = code_strength / total_strength
            math_coeff = math_strength / total_strength

            return code_coeff.item(), math_coeff.item()

    def compute_directional_alignment_coefficients(
        self, X: torch.Tensor
    ) -> Tuple[float, float]:
        """
        based on the alignment between X and LoRA, we compute the coefficients
        theory: the input is more aligned with the main direction of the LoRA, the LoRA should get higher weight
        """
        if X.dim() == 1:
            X = X.unsqueeze(0)

        # compute the alignment between X and LoRA A matrix
        X_norm = X / (X.norm(dim=1, keepdim=True) + 1e-8)

        # Math LoRA alignment
        math_A_norm = self.math_lora_A / (
            self.math_lora_A.norm(dim=1, keepdim=True) + 1e-8
        )
        math_alignment = torch.abs(X_norm @ math_A_norm.T).max(dim=1)[0].mean()

        # Code LoRA alignment
        code_A_norm = self.code_lora_A / (
            self.code_lora_A.norm(dim=1, keepdim=True) + 1e-8
        )
        code_alignment = torch.abs(X_norm @ code_A_norm.T).max(dim=1)[0].mean()

        # normalize the coefficients
        total_alignment = math_alignment + code_alignment + 1e-8
        math_coeff = math_alignment / total_alignment
        code_coeff = code_alignment / total_alignment

        return code_coeff.item(), math_coeff.item()

    def compute_activation_based_coefficients(
        self, X: torch.Tensor
    ) -> Tuple[float, float]:
        """
        based on the activation strength of X after LoRA, we compute the coefficients
        theory: the LoRA with higher activation strength is more important for the current input
        """
        if X.dim() == 1:
            X = X.unsqueeze(0)

        # compute the activation strength of X after LoRA
        math_activation = X @ self.math_lora_A.T  # [batch_size, rank]
        code_activation = X @ self.code_lora_A.T  # [batch_size, rank]

        # compute the activation strength (using L2 norm)
        math_strength = math_activation.norm(dim=1).mean()
        code_strength = code_activation.norm(dim=1).mean()

        # normalize the coefficients
        total_strength = math_strength + code_strength + 1e-8
        math_coeff = math_strength / total_strength
        code_coeff = code_strength / total_strength

        return code_coeff.item(), math_coeff.item()

    def compute_gradient_based_coefficients(
        self, X: torch.Tensor, y_target: torch.Tensor
    ) -> Tuple[float, float]:
        """
        based on the gradient information, we compute the coefficients
        theory: the LoRA with higher gradient should get higher weight
        """
        X = X.clone().detach().requires_grad_(True)
        if X.dim() == 1:
            X = X.unsqueeze(0)

        # compute the output of two LoRA
        math_output = X @ self.math_adapter.T
        code_output = X @ self.code_adapter.T

        # compute the loss between the output and the target
        math_loss = ((math_output - y_target) ** 2).mean()
        code_loss = ((code_output - y_target) ** 2).mean()

        # compute the gradient of the input
        math_grad = torch.autograd.grad(math_loss, X, retain_graph=True)[0]
        code_grad = torch.autograd.grad(code_loss, X, retain_graph=True)[0]

        # gradient strength
        math_grad_strength = math_grad.norm()
        code_grad_strength = code_grad.norm()

        # we use the inverse of the gradient strength as the weight
        math_weight = 1.0 / (math_grad_strength + 1e-8)
        code_weight = 1.0 / (code_grad_strength + 1e-8)

        # normalize the coefficients
        total_weight = math_weight + code_weight
        math_coeff = math_weight / total_weight
        code_coeff = code_weight / total_weight

        return code_coeff.item(), math_coeff.item()


class InputAwareCoefficientsLearner:
    """
    learn the mapping function from input to coefficients
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # a simple neural network to learn the mapping from input to coefficients
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim // 2, 2),  # 输出两个系数
            torch.nn.Softmax(dim=-1),  # 确保系数和为1
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        input X, output the corresponding coefficients
        Returns: [code_coeff, math_coeff]
        """
        if X.dim() == 1:
            X = X.unsqueeze(0)
        return self.net(X)

    def train(
        self,
        X_samples: List[torch.Tensor],
        optimal_coeffs: List[Tuple[float, float]],
        epochs: int = 1000,
    ):
        """
        train the coefficient prediction network

        Args:
            X_samples: input sample list
            optimal_coeffs: corresponding optimal coefficient list [(code_coeff, math_coeff), ...]
        """
        optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001)

        X_tensor = torch.stack(X_samples)
        coeffs_tensor = torch.tensor(optimal_coeffs, dtype=torch.float32)

        for epoch in range(epochs):
            optimizer.zero_grad()

            predicted_coeffs = self.forward(X_tensor)
            loss = torch.nn.functional.mse_loss(predicted_coeffs, coeffs_tensor)

            loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.6f}")


def analyze_input_clusters(
    X_samples: List[torch.Tensor], merger: AdaptiveLoRAMerger, n_clusters: int = 5
) -> Dict:
    """
    analyze the coefficients of different input clusters
    """
    # flatten the input and cluster
    X_flat = torch.stack([x.flatten() for x in X_samples]).numpy()
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_flat)

    # analyze the coefficients of each cluster
    cluster_coeffs = {}
    for cluster_id in range(n_clusters):
        cluster_mask = clusters == cluster_id
        cluster_samples = [
            X_samples[i] for i in range(len(X_samples)) if cluster_mask[i]
        ]

        if cluster_samples:
            # compute the average coefficients of the cluster
            coeffs_list = []
            for sample in cluster_samples:
                code_coeff, math_coeff = merger.compute_projection_based_coefficients(
                    sample
                )
                coeffs_list.append((code_coeff, math_coeff))

            avg_code_coeff = np.mean([c[0] for c in coeffs_list])
            avg_math_coeff = np.mean([c[1] for c in coeffs_list])

            cluster_coeffs[cluster_id] = {
                "code_coeff": avg_code_coeff,
                "math_coeff": avg_math_coeff,
                "sample_count": len(cluster_samples),
                "centroid": kmeans.cluster_centers_[cluster_id],
            }

    return cluster_coeffs


# example usage
def example_usage():
    """
    complete usage example
    """

    # simulate the LoRA weights
    torch.manual_seed(42)
    d_in, d_out, rank = 512, 768, 16

    math_lora_A = torch.randn(rank, d_in) * 0.1
    math_lora_B = torch.randn(d_out, rank) * 0.1
    code_lora_A = torch.randn(rank, d_in) * 0.1
    code_lora_B = torch.randn(d_out, rank) * 0.1

    # create the adaptive merger
    merger = AdaptiveLoRAMerger(math_lora_A, math_lora_B, code_lora_A, code_lora_B)

    # simulate different types of input
    math_input = torch.randn(d_in) + torch.tensor([1.0] * d_in) * 0.1  # 偏向某个方向
    code_input = torch.randn(d_in) + torch.tensor([-1.0] * d_in) * 0.1  # 偏向另一个方向
    mixed_input = torch.randn(d_in)  # 中性输入

    print("different input adaptive coefficient selection results:")
    print("=" * 60)

    inputs = {
        "math style input": math_input,
        "code style input": code_input,
        "mixed input": mixed_input,
    }

    for input_name, X in inputs.items():
        print(f"\n{input_name}:")

        # method 1: fast svd method
        code_coeff, math_coeff = merger.compute_fast_svd_coefficients(X)
        print(f"  fast svd: code={code_coeff:.3f}, math={math_coeff:.3f}")

        # method 2: ultra fast method
        code_coeff, math_coeff = merger.compute_ultra_fast_coefficients(X)
        print(f"  ultra fast: code={code_coeff:.3f}, math={math_coeff:.3f}")

        # method 3: projection method
        code_coeff, math_coeff = merger.compute_projection_based_coefficients(X)
        print(f"  projection: code={code_coeff:.3f}, math={math_coeff:.3f}")

        # method 4: directional alignment method
        code_coeff, math_coeff = merger.compute_directional_alignment_coefficients(X)
        print(f"  directional alignment: code={code_coeff:.3f}, math={math_coeff:.3f}")

        # method 5: activation strength method
        code_coeff, math_coeff = merger.compute_activation_based_coefficients(X)
        print(f"  activation: code={code_coeff:.3f}, math={math_coeff:.3f}")

    # show the cluster analysis
    print(f"\ninput cluster analysis:")
    print("=" * 30)

    # generate more samples
    sample_inputs = [torch.randn(d_in) for _ in range(100)]
    cluster_analysis = analyze_input_clusters(sample_inputs, merger, n_clusters=3)

    for cluster_id, info in cluster_analysis.items():
        print(
            f"cluster {cluster_id}: code={info['code_coeff']:.3f}, "
            f"math={info['math_coeff']:.3f}, sample count={info['sample_count']}"
        )


if __name__ == "__main__":
    example_usage()
