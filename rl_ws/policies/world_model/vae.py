"""
Multi-Modal Sensor VAE for AckBot Target Following.

Compresses the 426-D multi-modal sensor observation ([64 depth, 360 LiDAR, 2 state])
into a continuous structured latent representation [B, 4, 32] (128-D total)
with reparameterization and decoder reconstruction.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiModalSensorVAE(nn.Module):
    """
    Variational Autoencoder for AckBot 426-D sensor vectors.

    Encodes:
        obs [B, 426] -> latent map [B, latent_channels, latent_length] (e.g. [B, 4, 32])
    Decodes:
        latent map [B, 4, 32] -> reconstructed obs [B, 426] in [0, 1]
    """

    def __init__(
        self,
        in_dim: int = 426,
        latent_channels: int = 4,
        latent_length: int = 32,
        hidden_dim: int = 256,
        kl_weight: float = 1e-4,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.latent_channels = latent_channels
        self.latent_length = latent_length
        self.latent_dim = latent_channels * latent_length  # 4 * 32 = 128
        self.kl_weight = float(kl_weight)

        # Encoder MLP
        self.encoder_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.fc_mu = nn.Linear(hidden_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, self.latent_dim)

        # Decoder MLP
        self.decoder_net = nn.Sequential(
            nn.Linear(self.latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, in_dim),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode observation into latent mean and log-variance shaped [B, C_z, L_z].
        """
        h = self.encoder_net(x)
        mu = self.fc_mu(h).view(-1, self.latent_channels, self.latent_length)
        logvar = self.fc_logvar(h).view(-1, self.latent_channels, self.latent_length)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample z = mu + eps * std."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent tensor [B, C_z, L_z] or [B, D] back to [B, 426].
        First 424 features (camera + LiDAR) are bound to [0, 1] via Sigmoid.
        Last 2 features (vehicle state v, steer) are bound to [-1, 1] via Tanh.
        """
        z_flat = z.reshape(z.shape[0], -1)
        raw_out = self.decoder_net(z_flat)
        sensor_part = torch.sigmoid(raw_out[..., :424])
        state_part = torch.tanh(raw_out[..., 424:])
        return torch.cat([sensor_part, state_part], dim=-1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward pass returning reconstruction, mu, logvar, sampled latent, and losses.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)

        # 1. Reconstruction loss (Smooth L1)
        recon_loss = F.smooth_l1_loss(recon_x, x, beta=0.1)

        # 2. Analytical KL divergence: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kl_loss = -0.5 * torch.mean(
            torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=(1, 2))
        )

        total_loss = recon_loss + self.kl_weight * kl_loss

        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "recon_x": recon_x,
            "mu": mu,
            "logvar": logvar,
            "z": z,
        }

    @torch.no_grad()
    def compute_metrics(self, x: torch.Tensor, recon_x: torch.Tensor) -> dict[str, float]:
        """Compute MSE and Peak Signal-to-Noise Ratio (PSNR) for reconstructed observations."""
        mse = F.mse_loss(recon_x, x).item()
        psnr = 10.0 * math.log10(1.0 / max(mse, 1e-8))
        return {
            "mse": float(mse),
            "psnr": float(psnr),
        }
