"""
Loss Objectives and Evaluation Metrics for Spatial VAE.

Components:
- Pixel Reconstruction Loss: L1 + MSE + Gradient Edge Loss (Sobel filters)
- Regularization: Scaled Kullback-Leibler (KL) divergence to standard Gaussian prior
- Image Quality Metrics: Peak Signal-to-Noise Ratio (PSNR) & Structural Similarity Index (SSIM)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_psnr(
    recon: torch.Tensor,
    target: torch.Tensor,
    max_val: float = 1.0,
    eps: float = 1e-10,
) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) in decibels (dB).
    Higher is better (> 25 dB denotes high fidelity).
    """
    mse = F.mse_loss(recon.float(), target.float()).clamp(min=eps)
    return float(10.0 * torch.log10(torch.tensor(max_val**2, device=recon.device) / mse).item())


def _gaussian_window(window_size: int = 11, sigma: float = 1.5, channels: int = 3) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords**2) / (2.0 * sigma**2))
    g = g / g.sum()
    w1d = g.unsqueeze(1)
    w2d = (w1d @ w1d.t()).unsqueeze(0).unsqueeze(0)
    return w2d.expand(channels, 1, window_size, window_size).contiguous()


def compute_ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
) -> float:
    """
    Compute Structural Similarity Index (SSIM) between [B, C, H, W] images in [0, 1].
    Values range from -1 to 1; higher is better (> 0.85 denotes sharp structural preservation).
    """
    channels = img1.size(1)
    window = _gaussian_window(window_size, sigma, channels).to(device=img1.device, dtype=img1.dtype)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channels)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channels)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channels) - mu1_mu2

    c1 = 0.01**2
    c2 = 0.03**2

    ssim_map = ((2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return float(ssim_map.mean().item())


class EdgeGradientLoss(nn.Module):
    """Sobel edge gradient loss for preserving sharp road and vehicle contours."""

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32
        ).view(1, 1, 3, 3)

        kernel_x = sobel_x.repeat(in_channels, 1, 1, 1)
        kernel_y = sobel_y.repeat(in_channels, 1, 1, 1)

        self.register_buffer("kernel_x", kernel_x)
        self.register_buffer("kernel_y", kernel_y)
        self.in_channels = in_channels

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        kx = self.kernel_x.to(pred.device, dtype=pred.dtype)
        ky = self.kernel_y.to(pred.device, dtype=pred.dtype)

        grad_x_pred = F.conv2d(pred, kx, padding=1, groups=self.in_channels)
        grad_y_pred = F.conv2d(pred, ky, padding=1, groups=self.in_channels)

        grad_x_tgt = F.conv2d(target, kx, padding=1, groups=self.in_channels)
        grad_y_tgt = F.conv2d(target, ky, padding=1, groups=self.in_channels)

        return F.l1_loss(grad_x_pred, grad_x_tgt) + F.l1_loss(grad_y_pred, grad_y_tgt)


class VAELoss(nn.Module):
    """
    Composite Spatial VAE loss:
    L = L_rec + beta_kl * L_kl

    Where L_rec = L1 + lambda_mse * MSE + lambda_edge * EdgeLoss.
    """

    def __init__(
        self,
        beta_kl: float = 1e-4,
        mse_weight: float = 0.5,
        edge_weight: float = 0.2,
    ) -> None:
        super().__init__()
        self.beta_kl = float(beta_kl)
        self.mse_weight = float(mse_weight)
        self.edge_weight = float(edge_weight)
        self.edge_loss = EdgeGradientLoss(in_channels=3)

    def forward(
        self,
        recon: torch.Tensor,
        target: torch.Tensor,
        mean: torch.Tensor,
        logvar: torch.Tensor,
        beta_scale: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        # 1. Pixel reconstruction losses
        loss_l1 = F.l1_loss(recon, target)
        loss_mse = F.mse_loss(recon, target)
        loss_edge = self.edge_loss(recon, target)

        rec_loss = loss_l1 + self.mse_weight * loss_mse + self.edge_weight * loss_edge

        # 2. KL Divergence normalized by total latent dimensions per sample
        # D_KL = -0.5 * sum(1 + logvar - mean^2 - exp(logvar)) / (C * H * W)
        kl_per_element = -0.5 * (1.0 + logvar - mean.pow(2) - logvar.exp())
        kl_loss = kl_per_element.mean()

        effective_beta = self.beta_kl * float(beta_scale)
        total_loss = rec_loss + effective_beta * kl_loss

        return {
            "loss": total_loss,
            "rec_loss": rec_loss,
            "loss_l1": loss_l1,
            "loss_mse": loss_mse,
            "loss_edge": loss_edge,
            "kl_loss": kl_loss,
            "effective_beta": torch.tensor(effective_beta, device=recon.device),
        }
