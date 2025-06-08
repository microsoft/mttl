import torch
from typing import List, Optional, Union


class AnalyticalLoRAMerger:
    """
    Analytical solver for LoRA merging optimization problem using PyTorch.

    Solves: min Σᵢ (1/||τᵢ,ₗ||²F) ||(τₘ,ₗ - τᵢ,ₗ)(τᵢ,ₗ)ᵀ||²F
    """

    def __init__(
        self, regularization: float = 1e-8, device: Optional[torch.device] = None
    ):
        """
        Initialize the solver.

        Args:
            regularization: Small value added to diagonal for numerical stability
            device: Device to perform computations on (GPU/CPU)
        """
        self.regularization = regularization
        self.device = device if device is not None else torch.device("cpu")

    def merge_loras(
        self, lora_matrices: List[torch.Tensor], use_pseudoinverse: bool = False
    ) -> torch.Tensor:
        """
        Analytically merge multiple LoRA matrices.

        Args:
            lora_matrices: List of LoRA tensors τᵢ,ₗ to merge
            use_pseudoinverse: Whether to use pseudoinverse for numerical stability

        Returns:
            Optimal merged LoRA tensor τₘ,ₗ*
        """
        if len(lora_matrices) == 0:
            raise ValueError("At least one LoRA matrix is required")

        # Move tensors to the specified device
        lora_matrices = [mat.to(self.device) for mat in lora_matrices]

        # Get dimensions
        n_matrices = len(lora_matrices)
        rows, cols = lora_matrices[0].shape

        # Validate all matrices have same shape
        for i, matrix in enumerate(lora_matrices):
            if matrix.shape != (rows, cols):
                raise ValueError(
                    f"Matrix {i} has shape {matrix.shape}, expected {(rows, cols)}"
                )

        # Compute weights wᵢ = 1/||τᵢ,ₗ||²F
        weights = []
        for matrix in lora_matrices:
            frobenius_norm_sq = torch.sum(matrix**2)
            if frobenius_norm_sq < 1e-12:
                print(
                    f"Warning: Matrix has very small Frobenius norm: {frobenius_norm_sq.item()}"
                )
                weights.append(torch.tensor(1.0, device=self.device))  # Fallback weight
            else:
                weights.append(1.0 / frobenius_norm_sq)

        # Build the system Gm = y
        G, y = self._build_system(lora_matrices, weights)

        # Solve the system
        if use_pseudoinverse:
            m_optimal = torch.linalg.pinv(G) @ y
        else:
            # Use normal equation: m* = (G^T G)^(-1) G^T y
            GtG = G.T @ G

            # Add regularization for numerical stability
            GtG += self.regularization * torch.eye(
                GtG.shape[0], device=self.device, dtype=G.dtype
            )

            try:
                # Use torch.linalg.solve for better numerical stability
                m_optimal = torch.linalg.solve(GtG, G.T @ y)
            except torch.linalg.LinAlgError:
                print("Warning: Normal equation failed, using pseudoinverse")
                m_optimal = torch.linalg.pinv(G) @ y

        # Reshape back to matrix form
        merged_matrix = m_optimal.reshape(cols, rows).T

        return merged_matrix

    def _build_system(
        self, lora_matrices: List[torch.Tensor], weights: List[torch.Tensor]
    ):
        """
        Build the weighted least squares system Gm = y.

        Args:
            lora_matrices: List of LoRA tensors
            weights: Corresponding weights

        Returns:
            G: System matrix
            y: Target vector
        """
        rows, cols = lora_matrices[0].shape
        n_matrices = len(lora_matrices)
        dtype = lora_matrices[0].dtype

        # Each matrix contributes (rows * cols) equations
        total_equations = n_matrices * rows * cols
        total_variables = rows * cols

        G = torch.zeros(
            total_equations, total_variables, device=self.device, dtype=dtype
        )
        y = torch.zeros(total_equations, device=self.device, dtype=dtype)

        I = torch.eye(
            rows, device=self.device, dtype=dtype
        )  # Identity matrix for Kronecker product

        for i, (matrix, weight) in enumerate(zip(lora_matrices, weights)):
            sqrt_weight = torch.sqrt(weight)

            # Compute Kronecker product: C_i ⊗ I
            # Using torch.kron for efficient Kronecker product
            kron_product = torch.kron(matrix, I)

            # Add to system matrix with weighting
            start_row = i * rows * cols
            end_row = (i + 1) * rows * cols
            G[start_row:end_row, :] = sqrt_weight * kron_product

            # Compute target: vec(C_i^2)
            target_matrix = matrix @ matrix  # C_i^2 = C_i * C_i
            target_vec = target_matrix.T.flatten()  # vec(M^T) for consistency

            # Add to target vector with weighting
            y[start_row:end_row] = sqrt_weight * target_vec

        return G, y

    def compute_objective_value(
        self, merged_matrix: torch.Tensor, lora_matrices: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute the objective function value for verification.

        Args:
            merged_matrix: The merged LoRA tensor
            lora_matrices: List of individual LoRA tensors

        Returns:
            Objective function value
        """
        total_loss = torch.tensor(0.0, device=self.device, dtype=merged_matrix.dtype)

        # Ensure all tensors are on the same device
        merged_matrix = merged_matrix.to(self.device)
        lora_matrices = [mat.to(self.device) for mat in lora_matrices]

        for matrix in lora_matrices:
            # Compute weight
            weight = 1.0 / torch.sum(matrix**2)

            # Compute the term: (τₘ,ₗ - τᵢ,ₗ)(τᵢ,ₗ)ᵀ
            diff = merged_matrix - matrix
            term = diff @ matrix.T

            # Add weighted Frobenius norm squared
            total_loss += weight * torch.sum(term**2)

        return total_loss


def merge_loras(
    lora_tensors: List[torch.Tensor],
    regularization: float = 1e-8,
    use_pseudoinverse: bool = False,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Convenient function to merge LoRA tensors analytically.

    Args:
        lora_tensors: List of PyTorch LoRA tensors
        regularization: Regularization parameter
        use_pseudoinverse: Whether to use pseudoinverse for stability
        device: Device to perform computations on

    Returns:
        Merged LoRA tensor
    """
    if device is None:
        device = lora_tensors[0].device if lora_tensors else torch.device("cpu")

    merger = AnalyticalLoRAMerger(regularization=regularization, device=device)
    return merger.merge_loras(lora_tensors, use_pseudoinverse=use_pseudoinverse)


def merge_loras_batched(
    lora_tensors: List[torch.Tensor],
    batch_size: int = 10,
    regularization: float = 1e-8,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Merge LoRA tensors in batches for memory efficiency.

    Args:
        lora_tensors: List of PyTorch LoRA tensors
        batch_size: Number of tensors to process at once
        regularization: Regularization parameter
        device: Device to perform computations on

    Returns:
        Merged LoRA tensor
    """
    if len(lora_tensors) <= batch_size:
        return merge_loras(lora_tensors, regularization, device=device)

    # Process in batches and then merge the results
    merged_results = []
    for i in range(0, len(lora_tensors), batch_size):
        batch = lora_tensors[i : i + batch_size]
        merged_batch = merge_loras(batch, regularization, device=device)
        merged_results.append(merged_batch)

    # Recursively merge the batch results
    return merge_loras_batched(merged_results, batch_size, regularization, device)


# Example usage and benchmarking
if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create some example LoRA matrices
    torch.manual_seed(42)

    # Generate some random LoRA matrices
    n_matrices = 5
    matrix_size = (64, 64)  # Larger size to showcase GPU benefits

    lora_matrices = []
    for i in range(n_matrices):
        # Create structured matrices (low-rank-like)
        U = torch.randn(matrix_size[0], 4, device=device)
        V = torch.randn(4, matrix_size[1], device=device)
        matrix = U @ V + 0.1 * torch.randn(*matrix_size, device=device)
        lora_matrices.append(matrix)

    print(f"Created {n_matrices} LoRA matrices of size {matrix_size}")

    # Create merger and solve
    merger = AnalyticalLoRAMerger(regularization=1e-6, device=device)

    print("Merging LoRA matrices...")

    # Time the operation
    if device.type == "cuda":
        torch.cuda.synchronize()
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        start_time.record()

    merged_matrix = merger.merge_loras(lora_matrices)

    if device.type == "cuda":
        end_time.record()
        torch.cuda.synchronize()
        elapsed_time = start_time.elapsed_time(end_time)
        print(f"GPU time: {elapsed_time:.2f} ms")

    # Compute objective value
    obj_value = merger.compute_objective_value(merged_matrix, lora_matrices)
    print(f"Objective function value: {obj_value.item():.6f}")

    # Compare with average (baseline)
    avg_matrix = torch.stack(lora_matrices).mean(dim=0)
    avg_obj_value = merger.compute_objective_value(avg_matrix, lora_matrices)
    print(f"Average baseline objective: {avg_obj_value.item():.6f}")
    print(f"Improvement: {(avg_obj_value - obj_value).item():.6f}")

    # Test convenient function
    print("\nTesting convenient merge_loras function...")
    merged_matrix_2 = merge_loras(lora_matrices, device=device)
    diff = torch.norm(merged_matrix - merged_matrix_2)
    print(f"Difference between methods: {diff.item():.8f}")

    # Test batched merging for memory efficiency
    print("\nTesting batched merging...")
    merged_batched = merge_loras_batched(lora_matrices, batch_size=3, device=device)
    batch_diff = torch.norm(merged_matrix - merged_batched)
    print(f"Difference from batched method: {batch_diff.item():.8f}")

    # Memory usage info
    if device.type == "cuda":
        print(
            f"GPU memory allocated: {torch.cuda.memory_allocated(device) / 1024**2:.1f} MB"
        )
        print(
            f"GPU memory cached: {torch.cuda.memory_reserved(device) / 1024**2:.1f} MB"
        )
