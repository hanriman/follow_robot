"""
Latent Reward and Progress Predictors for Model Predictive Control (MPC).

Maps learned latent representations (from LeWM or Spatial VAE) to predicted
task rewards, forward velocities, and track centering metrics to evaluate
candidate action sequences during latent trajectory optimization.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LatentRewardHead(nn.Module):
    """
    Symmetric 2-layer MLP mapping latent state representations to control reward signals.

    Outputs a 3-dimensional vector:
        1. Environment step reward r_env
        2. Forward velocity v_fwd (progress metric)
        3. Centerline offset distance d_center (stability/stay-on-track metric)

    Composite Latent Control Value:
        V(z) = r_env + w_v * v_fwd - w_d * (d_center)^2
    """

    def __init__(
        self,
        in_dim: int = 128,
        hidden_dim: int = 256,
        is_spatial: bool = False,
        spatial_channels: int = 4,
    ) -> None:
        super().__init__()
        self.is_spatial = is_spatial
        self.in_dim = in_dim

        if is_spatial:
            # For Spatial VAE: pool [B, 4, 16, 16] -> [B, 64]
            self.pool = nn.AdaptiveAvgPool2d((4, 4))
            effective_in = spatial_channels * 16  # 64
        else:
            self.pool = nn.Identity()
            effective_in = in_dim

        self.net = nn.Sequential(
            nn.Linear(effective_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 3),  # [reward, forward_vel, dist_to_center]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent tensor. Either [B, D] or [B, C, H, W] or with sequence dim [B, H, ...].

        Returns:
            Predictions [..., 3]: [r_env, v_forward, d_center]
        """
        orig_shape = z.shape
        if self.is_spatial:
            if z.dim() == 5:
                # [B, H, C, H_z, W_z]
                b, h, c, hz, wz = orig_shape
                z = z.view(b * h, c, hz, wz)
                pooled = self.pool(z).flatten(start_dim=1)
                out = self.net(pooled)
                return out.view(b, h, 3)
            else:
                pooled = self.pool(z).flatten(start_dim=1)
                return self.net(pooled)
        else:
            if z.dim() == 3:
                # [B, H, D]
                b, h, d = orig_shape
                out = self.net(z.view(b * h, d))
                return out.view(b, h, 3)
            else:
                return self.net(z)

    def compute_control_score(
        self,
        z: torch.Tensor,
        weight_vel: float = 0.5,
        weight_center: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute scalar trajectory planning score:
            score = r_env + weight_vel * v_forward - weight_center * (dist_to_center)^2
        """
        preds = self.forward(z)  # [..., 3]
        r_env = preds[..., 0]
        v_fwd = preds[..., 1]
        d_center = preds[..., 2]

        score = r_env + weight_vel * v_fwd - weight_center * (d_center**2)
        return score
