"""
Diagnostic Metrics for Latent Representations and Gaussianity Health.

Monitors representation richness, effective dimensionality, and Gaussian goodness-of-fit
to verify that the teacher-free world model does not suffer from dimensional or complete collapse.
"""

from __future__ import annotations

from typing import Any

import torch


def compute_effective_rank(z: torch.Tensor, eps: float = 1e-8) -> dict[str, float]:
    r"""
    Compute Roy's Effective Rank / Participation Ratio of representations.

    Participation Ratio (PR):
        PR = \frac{(\sum_{i=1}^D \sigma_i)^2}{\sum_{i=1}^D \sigma_i^2}

    Where \sigma_i are the singular values of the centered latent batch.
    For an isotropic standard Gaussian N(0, I_D), PR \approx D.
    If the representations collapse onto a k-dimensional subspace, PR \to k.

    Args:
        z: Latent representation tensor [N, D] or [B, T, D].
        eps: Small numerical stabilizer.

    Returns:
        Dictionary with:
            - 'effective_rank': Participation ratio value in [1, D].
            - 'rank_ratio': Normalized ratio effective_rank / D in (0, 1].
            - 'condition_number': Ratio of largest to smallest singular value \sigma_max / \sigma_min.
    """
    z_flat = z.reshape(-1, z.shape[-1]).float()
    N, D = z_flat.shape

    if N < 2:
        return {
            "effective_rank": 1.0,
            "rank_ratio": 1.0 / D,
            "condition_number": 1.0,
        }

    # Center latents
    z_centered = z_flat - z_flat.mean(dim=0, keepdim=True)

    # Compute singular values via SVD
    try:
        # SVD of [N, D] matrix
        singular_values = torch.linalg.svdvals(z_centered)  # [min(N, D)]
    except RuntimeError:
        # Fallback to covariance eigvals if SVD encounters numerical issue
        cov = (z_centered.T @ z_centered) / (N - 1)
        eigvals = torch.linalg.eigvalsh(cov).clamp(min=eps)
        singular_values = torch.sqrt(eigvals)

    sv = singular_values.clamp(min=eps)
    sum_sv = sv.sum()
    sum_sq_sv = (sv.square()).sum()

    pr = float((sum_sv.square() / sum_sq_sv).item())
    rank_ratio = pr / float(D)

    sigma_max = float(sv.max().item())
    sigma_min = float(sv.min().item())
    cond = sigma_max / max(sigma_min, eps)

    return {
        "effective_rank": pr,
        "rank_ratio": rank_ratio,
        "condition_number": cond,
    }


def compute_gaussianity_diagnostics(z: torch.Tensor, eps: float = 1e-6) -> dict[str, float]:
    r"""
    Compute statistical moments and goodness-of-fit diagnostics against N(0, I_D).

    Evaluates:
    - Mean absolute error from zero mean: ||\mu||_1 / D
    - Mean variance deviation from unit variance: \frac{1}{D} \sum |\sigma_d^2 - 1|
    - Off-diagonal covariance correlation energy: mean of |Cov(i, j)| for i \ne j
    - Mean excess kurtosis (should be ~0 for Gaussian)

    Args:
        z: Latent tensor [N, D].
        eps: Small constant for numerical stability.

    Returns:
        Dictionary of diagnostic scalar metrics.
    """
    z_flat = z.reshape(-1, z.shape[-1]).float()
    N, D = z_flat.shape

    if N < 4:
        return {
            "mean_norm": 0.0,
            "var_deviation": 0.0,
            "off_diag_corr": 0.0,
            "excess_kurtosis": 0.0,
        }

    mean = z_flat.mean(dim=0)
    mean_norm = float(mean.abs().mean().item())

    # Sample variance per dimension
    var = z_flat.var(dim=0, unbiased=True)
    var_deviation = float((var - 1.0).abs().mean().item())

    # Correlation matrix and off-diagonal magnitude
    std = var.clamp(min=eps).sqrt()
    z_normed = (z_flat - mean) / std
    corr_matrix = (z_normed.T @ z_normed) / (N - 1)
    # Mask out diagonal
    off_diag_mask = ~torch.eye(D, dtype=torch.bool, device=z_flat.device)
    off_diag_corr = float(corr_matrix[off_diag_mask].abs().mean().item())

    # Excess kurtosis along dimensions: E[(z-\mu)^4] / \sigma^4 - 3
    centered = z_flat - mean
    m4 = (centered.pow(4)).mean(dim=0)
    kurtosis = m4 / (var.pow(2).clamp(min=eps)) - 3.0
    mean_excess_kurtosis = float(kurtosis.abs().mean().item())

    return {
        "mean_norm": mean_norm,
        "var_deviation": var_deviation,
        "off_diag_corr": off_diag_corr,
        "excess_kurtosis": mean_excess_kurtosis,
    }


def check_representation_health(
    z: torch.Tensor,
    min_rank_ratio: float = 0.5,
    max_var_dev: float = 0.8,
) -> dict[str, Any]:
    r"""
    Comprehensive representation health check for monitoring training stability.

    Args:
        z: Latent embeddings [N, D].
        min_rank_ratio: Minimum acceptable effective rank ratio (default: 0.5 = 50% of D).
        max_var_dev: Maximum allowable deviation from unit variance.

    Returns:
        Dictionary with boolean 'is_healthy', warning message if unhealthy, and all metrics.
    """
    rank_metrics = compute_effective_rank(z)
    gauss_metrics = compute_gaussianity_diagnostics(z)

    is_healthy = True
    issues = []

    if rank_metrics["rank_ratio"] < min_rank_ratio:
        is_healthy = False
        issues.append(
            f"Low effective rank ratio: {rank_metrics['rank_ratio']:.3f} < {min_rank_ratio} (dimensional collapse risk)"
        )

    if gauss_metrics["var_deviation"] > max_var_dev:
        is_healthy = False
        issues.append(
            f"High variance deviation from N(0, 1): {gauss_metrics['var_deviation']:.3f} > {max_var_dev}"
        )

    return {
        "is_healthy": is_healthy,
        "issues": issues,
        **rank_metrics,
        **gauss_metrics,
    }


class RidgeRegressionProbe:
    """Closed-form Ridge Regression Probe with analytical L2 regularization and feature standardization."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = float(alpha)
        self.weights: torch.Tensor | None = None
        self.x_mean: torch.Tensor | None = None
        self.x_std: torch.Tensor | None = None
        self.y_mean: torch.Tensor | None = None
        self.y_std: torch.Tensor | None = None

    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> RidgeRegressionProbe:
        """
        Fit linear probe W on standardized features: (X_norm @ W) ~= Y_norm.
        """
        self.x_mean = X.mean(dim=0, keepdim=True)
        self.x_std = X.std(dim=0, keepdim=True).clamp(min=1e-6)
        self.y_mean = Y.mean(dim=0, keepdim=True)
        self.y_std = Y.std(dim=0, keepdim=True).clamp(min=1e-6)

        X_norm = (X - self.x_mean) / self.x_std
        Y_norm = (Y - self.y_mean) / self.y_std

        # Append bias: [N, D + 1]
        ones = torch.ones((X_norm.shape[0], 1), device=X.device, dtype=X.dtype)
        X_aug = torch.cat([X_norm, ones], dim=1)

        # (X^T X + alpha * I)^{-1} X^T Y
        D_aug = X_aug.shape[1]
        eye = torch.eye(D_aug, device=X.device, dtype=X.dtype)
        eye[-1, -1] = 0.0  # Do not regularize bias

        A = X_aug.T @ X_aug + self.alpha * eye
        b = X_aug.T @ Y_norm

        try:
            self.weights = torch.linalg.solve(A, b)
        except RuntimeError:
            self.weights = torch.linalg.pinv(A) @ b

        return self

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        if (
            self.weights is None
            or self.x_mean is None
            or self.x_std is None
            or self.y_mean is None
            or self.y_std is None
        ):
            raise RuntimeError("Ridge probe must be fitted before predict.")
        X_norm = (X - self.x_mean) / self.x_std
        ones = torch.ones((X_norm.shape[0], 1), device=X.device, dtype=X.dtype)
        X_aug = torch.cat([X_norm, ones], dim=1)
        pred_norm = X_aug @ self.weights
        return pred_norm * self.y_std + self.y_mean

    def score_r2(self, X: torch.Tensor, Y: torch.Tensor, labels: list[str] | None = None) -> dict[str, Any]:
        """Compute R^2 score for each target dimension and overall mean R^2."""
        preds = self.predict(X)
        ss_res = ((Y - preds) ** 2).sum(dim=0)
        y_mean = Y.mean(dim=0, keepdim=True)
        ss_tot = ((Y - y_mean) ** 2).sum(dim=0).clamp(min=1e-8)

        r2_per_dim = (1.0 - ss_res / ss_tot).cpu().numpy().tolist()
        mean_r2 = float(torch.tensor(r2_per_dim).mean().item())
        res: dict[str, Any] = {
            "mean_r2": mean_r2,
            "r2_per_dim": r2_per_dim,
        }
        if labels is not None:
            res["per_dimension_r2"] = {labels[i]: r2_per_dim[i] for i in range(min(len(labels), len(r2_per_dim)))}
        return res

