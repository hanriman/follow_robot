"""
Spatial Variational Autoencoder (Spatial VAE) for Continuous Visual Control.

Architecture:
- Downsamples 64x64x3 observations into a compact 4x16x16 spatial continuous latent tensor.
- Employs residual convolutional blocks with Group Normalization and SiLU activations.
- Preserves 2D spatial topology for downstream Action-Conditioned Latent Diffusion (AC-LDM).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """Residual convolutional block with Group Normalization and SiLU."""

    def __init__(self, channels: int, num_groups: int = 8) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(num_groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class SpatialEncoder(nn.Module):
    """
    Residual Convolutional Encoder mapping [B, 3, 64, 64] -> (mean, logvar) of shape [B, C_z, 16, 16].
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 4,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        self.latent_channels = latent_channels

        self.net = nn.Sequential(
            # Stage 0: 64x64
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            ResBlock(base_channels),
            # Stage 1: 64x64 -> 32x32
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            ResBlock(base_channels * 2),
            # Stage 2: 32x32 -> 16x16
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1),
            ResBlock(base_channels * 4),
            # Output head: map to 2 * latent_channels (mean, logvar)
            nn.GroupNorm(8, base_channels * 4),
            nn.SiLU(),
            nn.Conv2d(base_channels * 4, latent_channels * 2, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        mean, logvar = torch.chunk(h, 2, dim=1)
        return mean, logvar


class SpatialDecoder(nn.Module):
    """
    Residual Convolutional Decoder mapping [B, C_z, 16, 16] -> [B, 3, 64, 64].
    """

    def __init__(
        self,
        latent_channels: int = 4,
        out_channels: int = 3,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            # Stage 0: 16x16
            nn.Conv2d(latent_channels, base_channels * 4, kernel_size=3, padding=1),
            ResBlock(base_channels * 4),
            # Stage 1: 16x16 -> 32x32
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1),
            ResBlock(base_channels * 2),
            # Stage 2: 32x32 -> 64x64
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1),
            ResBlock(base_channels),
            # Output reconstruction
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class SpatialVAE(nn.Module):
    """
    Complete Continuous Spatial Autoencoder.

    Args:
        in_channels: Input image channels (default: 3).
        latent_channels: Number of spatial latent feature maps (default: 4).
        base_channels: Base channel width for convolutions (default: 32).
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 4,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.encoder = SpatialEncoder(
            in_channels=in_channels,
            latent_channels=latent_channels,
            base_channels=base_channels,
        )
        self.decoder = SpatialDecoder(
            latent_channels=latent_channels,
            out_channels=in_channels,
            base_channels=base_channels,
        )

    def reparameterize(
        self,
        mean: torch.Tensor,
        logvar: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Reparameterization trick: z = mean + std * eps."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn(std.shape, device=std.device, dtype=std.dtype, generator=generator)
            return mean + std * eps
        return mean

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode image to distribution parameters (mean, logvar)."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent tensor back to reconstructed image space."""
        return self.decoder(z)

    def forward(
        self,
        x: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass computing reconstruction and distribution parameters.

        Args:
            x: Input observations [B, 3, 64, 64] normalized in [0, 1].

        Returns:
            Dictionary containing:
                - 'recon': Reconstructed observation [B, 3, 64, 64].
                - 'mean': Latent distribution mean [B, C_z, 16, 16].
                - 'logvar': Latent distribution log-variance [B, C_z, 16, 16].
                - 'z': Sampled continuous latent tensor [B, C_z, 16, 16].
        """
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar, generator=generator)
        recon = self.decode(z)
        return {
            "recon": recon,
            "mean": mean,
            "logvar": logvar,
            "z": z,
        }
