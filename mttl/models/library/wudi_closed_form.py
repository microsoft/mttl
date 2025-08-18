import numpy as np
from torch.linalg import inv, pinv, solve
import warnings
import torch


class AnalyticalSolver:
    """
    Implementation of the analytical solution:
    τ_{m,l} = Matmul(Σ_i (1/||τ_{i,l}||_F^2) τ_{i,l}(τ_{i,l}^T τ_{i,l} + ωI),
                     (Σ_i (1/||τ_{i,l}||_F^2) (τ_{i,l}^T τ_{i,l} + ωI))^{-1})

    Where:
    - τ_{i,l} are the input matrices (corresponding to your low-rank factors)
    - ω is a regularization parameter
    - The formula computes a weighted combination with Frobenius norm weighting
    """

    def __init__(self, regularization_omega=1e-6):
        """
        Initialize the analytical solver

        Args:
            regularization_omega: Regularization parameter ω for numerical stability
        """
        self.omega = regularization_omega

    def compute_analytical_solution(self, tau_matrices, omega=None):
        """
        Compute the analytical solution using the given formula

        Args:
            tau_matrices: List of matrices τ_{i,l}, each of shape (n, k)
            omega: Regularization parameter (if None, uses self.omega)

        Returns:
            tau_ml: The computed solution matrix
        """
        if omega is None:
            omega = self.omega

        n_matrices = len(tau_matrices)
        if n_matrices == 0:
            raise ValueError("Need at least one input matrix")

        # Get dimensions from first matrix
        n_rows, n_cols = tau_matrices[0].shape

        # Verify all matrices have same shape
        for i, tau_i in enumerate(tau_matrices):
            if tau_i.shape != (n_rows, n_cols):
                raise ValueError(
                    f"Matrix {i} has shape {tau_i.shape}, expected {(n_rows, n_cols)}"
                )

        # Compute Frobenius norms and weights
        frobenius_norms = []
        weights = []

        for tau_i in tau_matrices:
            frob_norm = torch.linalg.norm(tau_i, "fro")
            if frob_norm < 1e-12:  # Handle near-zero matrices
                warnings.warn(
                    "Found matrix with very small Frobenius norm, using small regularization"
                )
                frob_norm = 1e-12

            frobenius_norms.append(frob_norm)
            weights.append(1.0 / (frob_norm**2))

        # print(f"Frobenius norms: {frobenius_norms}")
        # print(f"Weights: {weights}")

        # Compute the first sum: Σ_i (1/||τ_{i,l}||_F^2) τ_{i,l}(τ_{i,l}^T τ_{i,l} + ωI)
        first_sum = torch.zeros((n_rows, n_cols))

        for tau_i, weight in zip(tau_matrices, weights):
            # Compute τ_{i,l}^T τ_{i,l} + ωI
            gram_matrix = tau_i.T @ tau_i + omega * torch.eye(n_cols)

            # Add weighted contribution: weight * τ_{i,l} * gram_matrix
            first_sum += weight * tau_i @ gram_matrix

        # Compute the second sum: Σ_i (1/||τ_{i,l}||_F^2) (τ_{i,l}^T τ_{i,l} + ωI)
        second_sum = torch.zeros((n_cols, n_cols))

        for tau_i, weight in zip(tau_matrices, weights):
            # Compute τ_{i,l}^T τ_{i,l} + ωI
            gram_matrix = tau_i.T @ tau_i + omega * np.eye(n_cols)

            # Add weighted contribution
            second_sum += weight * gram_matrix

        # Compute the final result: first_sum @ inv(second_sum)
        try:
            # Try regular inverse first
            second_sum_inv = inv(second_sum)
        except torch.linalg.LinAlgError:
            # Fall back to pseudo-inverse if singular
            warnings.warn("Second sum matrix is singular, using pseudo-inverse")
            second_sum_inv = pinv(second_sum)

        tau_ml = first_sum @ second_sum_inv

        return tau_ml

    def compute_analytical_solution_stable(self, tau_matrices, omega=None):
        """
        Numerically stable version using solve instead of explicit inversion

        Args:
            tau_matrices: List of matrices τ_{i,l}
            omega: Regularization parameter

        Returns:
            tau_ml: The computed solution matrix
        """
        if omega is None:
            omega = self.omega

        n_matrices = len(tau_matrices)
        if n_matrices == 0:
            raise ValueError("Need at least one input matrix")

        n_rows, n_cols = tau_matrices[0].shape

        # Compute weights
        weights = []
        for tau_i in tau_matrices:
            frob_norm = torch.linalg.norm(tau_i, "fro")
            if frob_norm < 1e-12:
                frob_norm = 1e-12
            weights.append(1.0 / (frob_norm**2))

        # Compute sums
        first_sum = torch.zeros((n_rows, n_cols))
        second_sum = torch.zeros((n_cols, n_cols))

        for tau_i, weight in zip(tau_matrices, weights):
            gram_matrix = tau_i.T @ tau_i + omega * torch.eye(n_cols)
            first_sum += weight * tau_i @ gram_matrix
            second_sum += weight * gram_matrix

        # Solve instead of inverting: tau_ml @ second_sum = first_sum
        # This is equivalent to: tau_ml = first_sum @ inv(second_sum)
        try:
            tau_ml = solve(second_sum.T, first_sum.T).T
        except torch.linalg.LinAlgError:
            # Fall back to least squares if singular
            warnings.warn("Using least squares solve due to singular matrix")
            tau_ml = np.linalg.lstsq(second_sum.T, first_sum.T, rcond=None)[0].T

        return tau_ml

    def verify_solution_properties(self, tau_ml, tau_matrices):
        """
        Verify properties of the computed solution

        Args:
            tau_ml: Computed solution
            tau_matrices: Original input matrices

        Returns:
            dict: Dictionary of verification metrics
        """
        metrics = {}

        # Compute residuals for each input matrix
        residuals = []
        for i, tau_i in enumerate(tau_matrices):
            residual = torch.linalg.norm(tau_ml - tau_i, "fro")
            residuals.append(residual)

        residuals = torch.stack(residuals)

        metrics["residuals"] = residuals
        metrics["mean_residual"] = torch.mean(residuals)
        metrics["solution_norm"] = torch.linalg.norm(tau_ml, "fro")

        # Check rank
        try:
            metrics["solution_rank"] = torch.linalg.matrix_rank(tau_ml)
        except:
            metrics["solution_rank"] = None

        return metrics


def demo_analytical_solution():
    """
    Demonstrate the analytical solution with example data
    """
    print("Demo: Analytical Solution Implementation")
    print("=" * 50)

    # Create test data
    torch.manual_seed(42)
    n_rows, n_cols = 3047, 3047  # Small example
    n_matrices = 5

    # Generate test matrices with some shared structure
    base_matrix = torch.randn(n_rows, n_cols)
    tau_matrices = []

    for i in range(n_matrices):
        # Add noise to base matrix to create related matrices
        noise = 0.3 * torch.randn(n_rows, n_cols)
        tau_i = base_matrix + noise
        tau_matrices.append(tau_i)

    print(f"Generated {n_matrices} test matrices of shape {(n_rows, n_cols)}")

    # Initialize solver
    solver = AnalyticalSolver(regularization_omega=1e-4)

    # Compute analytical solution
    print("\nComputing analytical solution...")
    tau_ml_regular = solver.compute_analytical_solution(tau_matrices)

    print(f"Solution shape: {tau_ml_regular.shape}")
    print(f"Solution Frobenius norm: {torch.linalg.norm(tau_ml_regular, 'fro'):.6f}")

    # Compute stable version
    print("\nComputing numerically stable version...")
    tau_ml_stable = solver.compute_analytical_solution_stable(tau_matrices)

    # Compare solutions
    difference = torch.linalg.norm(tau_ml_regular - tau_ml_stable, "fro")
    print(f"Difference between regular and stable solutions: {difference:.8f}")

    # Verify solution properties
    print("\nVerifying solution properties...")
    metrics = solver.verify_solution_properties(tau_ml_stable, tau_matrices)

    print(f"Mean residual: {metrics['mean_residual']:.6f}")
    print(f"Solution rank: {metrics['solution_rank']}")
    print(f"Individual residuals: {[f'{r:.4f}' for r in metrics['residuals']]}")

    return tau_ml_stable, tau_matrices, metrics


def adapt_for_lora_problem(J_matrices, K_matrices, omega=1e-6):
    """
    Adapt the analytical solution for LoRA-style problems where you have
    pairs of matrices (J_i, K_i) and want to find optimal (J*, K*)

    Args:
        J_matrices: List of J matrices (left factors)
        K_matrices: List of K matrices (right factors)
        omega: Regularization parameter

    Returns:
        J_optimal, K_optimal: Optimal low-rank factors
    """
    print("Adapting analytical solution for LoRA problem...")

    solver = AnalyticalSolver(regularization_omega=omega)

    # Apply analytical solution separately to J and K matrices
    J_optimal = solver.compute_analytical_solution_stable(J_matrices, omega)
    K_optimal = solver.compute_analytical_solution_stable(K_matrices, omega)

    return J_optimal, K_optimal


if __name__ == "__main__":
    # Run demonstration
    tau_ml, tau_matrices, metrics = demo_analytical_solution()

    print("\n" + "=" * 50)
    print("Example usage for LoRA problem:")
    print("=" * 50)

    # Example for LoRA-style problem
    torch.manual_seed(123)
    n_dim, rank = 20, 4
    n_problems = 3

    # Generate example J and K matrices
    J_matrices = [torch.randn(n_dim, rank) for _ in range(n_problems)]
    K_matrices = [torch.randn(n_dim, rank) for _ in range(n_problems)]

    print(f"LoRA problem: {n_problems} matrix pairs of size ({n_dim}, {rank})")

    # Compute optimal factors
    J_opt, K_opt = adapt_for_lora_problem(J_matrices, K_matrices, omega=1e-4)

    print(f"Optimal J shape: {J_opt.shape}")
    print(f"Optimal K shape: {K_opt.shape}")
    print(f"Optimal J Frobenius norm: {torch.linalg.norm(J_opt, 'fro'):.4f}")
    print(f"Optimal K Frobenius norm: {torch.linalg.norm(K_opt, 'fro'):.4f}")

    # Verify the solution makes sense
    print("\nVerification:")
    for i, (J_i, K_i) in enumerate(zip(J_matrices, K_matrices)):
        residual_J = torch.linalg.norm(J_opt - J_i, "fro")
        residual_K = torch.linalg.norm(K_opt - K_i, "fro")
        print(
            f"Problem {i}: J residual = {residual_J:.4f}, K residual = {residual_K:.4f}"
        )
