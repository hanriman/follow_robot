"""
Latent Reward and Progress Predictors for AckBot Target Following.

Maps learned latent representations (from LeWM or MultiModalSensorVAE) to predicted
task rewards, target distance errors, and visual lock metrics to evaluate
candidate action sequences during latent trajectory optimization (CEM-MPC).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LatentRewardHead(nn.Module):
    """
    MLP mapping latent state representations to control reward signals.

    Outputs a 3-dimensional vector:
        1. Environment step reward r_env
        2. Target distance error |d - d_desired| (where d_desired = 0.80m)
        3. Visual lock / target heading alignment cos(rel_heading)

    Composite Latent Control Value:
        V(z) = r_env - w_d * (dist_error)^2 + w_align * visual_lock
    """

    def __init__(
        self,
        in_dim: int = 128,
        hidden_dim: int = 256,
        is_spatial: bool = False,
        spatial_channels: int = 4,
        spatial_length: int = 32,
    ) -> None:
        super().__init__()
        self.is_spatial = is_spatial
        self.in_dim = in_dim

        if is_spatial:
            effective_in = spatial_channels * spatial_length  # 4 * 32 = 128
        else:
            effective_in = in_dim

        self.net = nn.Sequential(
            nn.Linear(effective_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 3),  # [reward, dist_error, visual_lock]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent tensor. [B, D] or [B, C, L] or sequence [B, H, ...].

        Returns:
            Predictions [..., 3]: [r_env, dist_error, visual_lock]
        """
        orig_shape = z.shape
        if self.is_spatial:
            if z.dim() == 4:
                # [B, H, C_z, L_z]
                b, h, c, l = orig_shape
                z_flat = z.view(b * h, c * l)
                out = self.net(z_flat)
                return out.view(b, h, 3)
            elif z.dim() == 3:
                # [B, C_z, L_z]
                b, c, l = orig_shape
                z_flat = z.view(b, c * l)
                return self.net(z_flat)
            else:
                return self.net(z.flatten(start_dim=1))
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
        weight_dist: float = 1.0,
        weight_align: float = 0.5,
    ) -> torch.Tensor:
        """
        Compute scalar trajectory planning score:
            score = r_env - weight_dist * (dist_error)^2 + weight_align * visual_lock
        """
        preds = self.forward(z)  # [..., 3]
        r_env = preds[..., 0]
        dist_err = preds[..., 1]
        visual_lock = preds[..., 2]

        score = r_env - weight_dist * (dist_err.pow(2)) + weight_align * visual_lock
        return score
