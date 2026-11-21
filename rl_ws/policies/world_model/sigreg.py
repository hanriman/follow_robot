"""
Sketched Isotropic Gaussian Regularization (SIGReg) & Transition Losses for AckBot.

References:
-----------
- Epps, T. W., & Pulley, L. B. (1983). "A test for normality based on the empirical
  characteristic function." Biometrika, 70(3), 723-726.
- Cramér, H., & Wold, H. (1936). "Some theorems on distribution functions."
  Journal of the London Mathematical Society, 1(4), 290-294.
- Balestriero, R., & LeCun, Y. (2025). "LeJEPA: Provable and Scalable Self-Supervised Learning
  Without the Heuristics." arXiv:2511.08544.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class EppsPulleyGaussianityTest(nn.Module):
    r"""
    Epps-Pulley Goodness-of-Fit Test Statistic for 1D Gaussianity.

    Mathematical Formulation:
    -------------------------
    1. Cramér-Wold Theorem (1936):
       A d-dimensional multivariate probability distribution P on R^d is uniquely determined
       by the family of its 1D marginal distributions under all 1D linear projections
       u \in S^{d-1}. Testing multivariate standard normal N(0, I_d) is equivalent to testing
       that projected scalars u^T z ~ N(0, 1) for all u \in S^{d-1}.

    2. Epps-Pulley Test (1983):
       Compares the 1D Empirical Characteristic Function (ECF):
           \hat{\phi}_N(t) = \frac{1}{N} \sum_{n=1}^N \exp(i t p_n)
       against the analytical standard Gaussian characteristic function:
           \phi_0(t) = \exp(-t^2 / 2)
       under the Gaussian-weighted L2 metric:
           T_{EP} = N \int_{-\infty}^{\infty} |\hat{\phi}_N(t) - \phi_0(t)|^2 d\mu(t)
       where d\mu(t) = \frac{1}{\sqrt{2\pi}} \exp(-t^2 / 2) dt is the standard Gaussian measure.

    3. Gradient Scaling (Why multiply by N?):
       The derivative of the empirical characteristic function with respect to projection p_n is:
           \frac{\partial \hat{\phi}_N(t)}{\partial p_n} = \frac{i t}{N} \exp(i t p_n)
       Without multiplying the test statistic by sample count N, \frac{\partial \mathcal{L}}{\partial p_n}
       would carry an extraneous 1/N factor, leading to vanishing gradients on representations as batch size
       or token length scales up. Multiplying by N cancels this factor, producing an O(1) per-sample gradient
       that balances naturally with the transition prediction loss.

    Args:
        t_max: Upper integration limit for frequency t (default: 3.0, tail mass beyond 3 is negligible).
        n_knots: Number of quadrature grid points in [0, t_max] (default: 17).
        normalize_measure: Whether to include 1 / sqrt(2 * pi) normalizing factor in Gaussian measure.
    """

    def __init__(
        self,
        t_max: float = 3.0,
        n_knots: int = 17,
        normalize_measure: bool = True,
    ) -> None:
        super().__init__()
        self.t_max = float(t_max)
        self.n_knots = int(n_knots)
        self.normalize_measure = bool(normalize_measure)

        # Trapezoidal quadrature rule over [0, t_max] folded from [-t_max, t_max]
        t = torch.linspace(0.0, self.t_max, self.n_knots, dtype=torch.float32)
        dt = self.t_max / (self.n_knots - 1)
        weights = torch.full((self.n_knots,), 2.0 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt  # Boundary weights for trapezoidal integration
        phi = torch.exp(-0.5 * t.square())
        norm_const = 1.0 / math.sqrt(2.0 * math.pi) if self.normalize_measure else 1.0

        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("weights", weights * phi * norm_const)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        r"""
        Evaluate Epps-Pulley test statistic across K random 1D projections.

        Args:
            proj: [N, K] where N is number of sample tokens/vectors, K is number of random projections.

        Returns:
            Scalar tensor representing mean Epps-Pulley statistic across all K projections.
        """
        orig_dtype = proj.dtype
        # Compute in float32 to prevent underflow under AMP / half precision
        proj_f32 = proj.float()
        t = self.t.to(device=proj.device, dtype=torch.float32)
        phi = self.phi.to(device=proj.device, dtype=torch.float32)
        weights = self.weights.to(device=proj.device, dtype=torch.float32)

        # Outer product: [N, K, 1] * [1, 1, Q] -> [N, K, Q]
        x_t = proj_f32.unsqueeze(-1) * t
        ecf_real = x_t.cos().mean(dim=0)  # [K, Q]
        ecf_imag = x_t.sin().mean(dim=0)  # [K, Q]

        err = (ecf_real - phi).square() + ecf_imag.square()  # [K, Q]

        # Multiply by sample count N = proj.size(0) to cancel the 1/N factor in d(ecf)/dz
        statistic = (err @ weights) * proj.shape[0]  # [K]
        return statistic.mean().to(dtype=orig_dtype)


class SigRegLoss(nn.Module):
    r"""
    Sketched Isotropic Gaussian Regularization (SIGReg) Loss.

    Projects latent representations along random 1D unit directions on the hypersphere
    S^{D-1} and computes the Epps-Pulley Gaussianity test on each projection.

    Prevents representation collapse in teacher-free world models without requiring
    exponential moving average (EMA) target encoders, stop-gradients, or negative pairs.

    Args:
        sigreg_weight: Multiplier weight \lambda for the SIGReg penalty.
        num_projections: Number of random 1D direction vectors M sampled on S^{D-1}.
        t_max: Upper integration limit for Epps-Pulley quadrature.
        n_knots: Number of quadrature integration knots.
        normalize_measure: Whether to normalize the Gaussian measure in EP test.
    """

    def __init__(
        self,
        sigreg_weight: float = 1.0,
        num_projections: int = 1024,
        t_max: float = 3.0,
        n_knots: int = 17,
        normalize_measure: bool = True,
    ) -> None:
        super().__init__()
        self.sigreg_weight = float(sigreg_weight)
        self.num_projections = int(num_projections)
        self.ep_test = EppsPulleyGaussianityTest(
            t_max=t_max,
            n_knots=n_knots,
            normalize_measure=normalize_measure,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        r"""
        Args:
            tokens: Latent tensor of arbitrary shape whose last dimension is D:
                    e.g., [B, D], [B, T, D], or [B, N, D].
            generator: Optional PyTorch random generator for bitwise reproducible projection sampling.

        Returns:
            Scalar tensor representing the SIGReg loss penalty.
        """
        # Flatten leading dimensions into a single sample dimension N: [N, D]
        z = tokens.reshape(-1, tokens.shape[-1])
        if z.dim() != 2 or z.shape[0] < 1 or z.shape[1] < 1:
            raise ValueError(
                f"SigRegLoss expects non-empty [N, D] tokens after flatten, got {tuple(z.shape)}."
            )

        D = z.shape[-1]

        # Sample M random projection directions on unit hypersphere S^{D-1} in float32.
        if generator is None:
            A = torch.randn(D, self.num_projections, device=z.device, dtype=torch.float32)
        else:
            A = torch.randn(D, self.num_projections, generator=generator, dtype=torch.float32).to(
                z.device
            )
        A = F.normalize(A, p=2, dim=0)  # [D, M]

        # 1D linear projections: [N, M] in float32
        proj = z.float() @ A

        sigreg_stat = self.ep_test(proj)
        return sigreg_stat


class ActionConditionedTransitionLoss(nn.Module):
    r"""
    Action-Conditioned Transition and World Model Composite Objective.

    Combines transition prediction discrepancy in representation space:
        \mathcal{L}_{\text{trans}} = \mathcal{D}(\hat{z}_{t+1}, z_{t+1})
    with the anti-collapse Sketched Isotropic Gaussian Regularization:
        \mathcal{L}_{\text{SIGReg}} = \text{EP-Test}(z)

    Total Loss:
        \mathcal{L} = \mathcal{L}_{\text{trans}} + \lambda_{\text{sigreg}} \cdot \mathcal{L}_{\text{SIGReg}}

    Args:
        loss_type: Distance metric for transition loss ('smooth_l1', 'mse', 'l1').
        beta: Huber transition parameter for 'smooth_l1'.
        sigreg_weight: Multiplier weight for SIGReg anti-collapse penalty.
        num_projections: Number of random hypersphere projections for SIGReg.
        normalize_target: Whether to apply LayerNorm to target latents before computing distance.
    """

    def __init__(
        self,
        loss_type: str = "smooth_l1",
        beta: float = 1.0,
        sigreg_weight: float = 1.0,
        num_projections: int = 1024,
        t_max: float = 3.0,
        n_knots: int = 17,
        normalize_target: bool = False,
    ) -> None:
        super().__init__()
        self.loss_type = loss_type
        self.beta = float(beta)
        self.sigreg_weight = float(sigreg_weight)
        self.normalize_target = bool(normalize_target)

        self.sigreg = SigRegLoss(
            sigreg_weight=sigreg_weight,
            num_projections=num_projections,
            t_max=t_max,
            n_knots=n_knots,
        )

    def compute_prediction_error(
        self,
        pred_next_z: torch.Tensor,
        target_next_z: torch.Tensor,
    ) -> torch.Tensor:
        r"""Compute transition prediction error in latent space."""
        target = target_next_z
        if self.normalize_target:
            target = F.layer_norm(target, (target.shape[-1],))

        pred = pred_next_z.float()
        target = target.float()

        if self.loss_type == "smooth_l1":
            return F.smooth_l1_loss(pred, target, beta=self.beta)
        elif self.loss_type == "mse":
            return F.mse_loss(pred, target)
        elif self.loss_type == "l1":
            return F.l1_loss(pred, target)
        else:
            raise ValueError(f"Unsupported transition loss_type: {self.loss_type}")

    def forward(
        self,
        pred_next_z: torch.Tensor,
        target_next_z: torch.Tensor,
        regularize_tokens: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        reg_scale: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        r"""
        Args:
            pred_next_z: Predicted future latent representations [B, D] or [B, T, D].
            target_next_z: True future latent representations [B, D] or [B, T, D].
            regularize_tokens: Latent tensor to regularize under SIGReg. If None,
                               defaults to concatenated [pred_next_z, target_next_z].
            generator: Optional random generator for projection sampling.
            reg_scale: Optional dynamic multiplier for scheduling or warm-up.

        Returns:
            Dictionary with keys:
                - 'loss': Total combined scalar loss.
                - 'transition_loss': Latent prediction error.
                - 'sigreg_loss': Raw Epps-Pulley Gaussianity test statistic.
                - 'reg_scale': Active regularization scaling factor.
        """
        trans_loss = self.compute_prediction_error(pred_next_z, target_next_z)

        if regularize_tokens is None:
            regularize_tokens = torch.cat([pred_next_z, target_next_z], dim=0)

        sigreg_val = self.sigreg(regularize_tokens, generator=generator)

        effective_sigreg_weight = float(reg_scale) * self.sigreg_weight
        total_loss = trans_loss + effective_sigreg_weight * sigreg_val.to(dtype=trans_loss.dtype)

        return {
            "loss": total_loss,
            "transition_loss": trans_loss,
            "sigreg_loss": sigreg_val,
            "reg_scale": torch.tensor(reg_scale, device=trans_loss.device),
        }
