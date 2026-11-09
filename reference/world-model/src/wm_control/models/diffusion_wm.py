"""
Action-Conditioned Latent Diffusion World Model (AC-LDM).

Architecture:
- Spatial U-Net operating on continuous latent feature maps [B, 4, 16, 16].
- Injects continuous 3D action conditioning [steering, gas, brake] and diffusion timestep embeddings
  via Adaptive Group Normalization (AdaGN) modulation layers.
- Fast multi-step latent unrolling using Denoising Diffusion Implicit Models (DDIM).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from wm_control.models.vae import SpatialVAE


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for discrete diffusion timesteps."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb_scale = math.log(10000.0) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -emb_scale)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class AdaGNResBlock(nn.Module):
    """
    Residual block with Adaptive Group Normalization (AdaGN).
    Modulates normalized features via linear projection of joint condition embedding:
        norm(h) * (1 + scale) + shift
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        cond_dim: int,
        num_groups: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.cond_proj = nn.Linear(cond_dim, out_ch * 2)
        self.norm2 = nn.GroupNorm(num_groups, out_ch)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))

        # Condition modulation: scale and shift
        scale_shift = self.cond_proj(F.silu(cond))
        scale, shift = torch.chunk(scale_shift, 2, dim=-1)
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)

        h = self.norm2(h) * (1.0 + scale) + shift
        h = self.conv2(self.dropout(F.silu(h)))
        return self.skip(x) + h


class ActionConditionedUNet(nn.Module):
    """
    Spatial U-Net predicting diffusion noise epsilon conditioned on context state and action.

    Inputs:
        noisy_target: [B, C_z, 16, 16] noisy future latent z_{t+1}^{(k)}.
        timestep: [B] discrete diffusion step k in [0, K-1].
        context_latent: [B, C_z, 16, 16] current state latent z_t.
        action: [B, A] continuous action vector a_t.

    Output:
        Predicted noise epsilon: [B, C_z, 16, 16].
    """

    def __init__(
        self,
        latent_channels: int = 4,
        action_dim: int = 3,
        base_channels: int = 64,
        cond_dim: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.action_dim = action_dim

        # Condition encoders
        self.time_emb = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels),
            nn.Linear(base_channels, cond_dim // 2),
            nn.SiLU(),
        )
        self.action_emb = nn.Sequential(
            nn.Linear(action_dim, cond_dim // 2),
            nn.SiLU(),
            nn.Linear(cond_dim // 2, cond_dim // 2),
            nn.SiLU(),
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # U-Net Encoder: inputs concat [noisy_target, context_latent] -> 2 * latent_channels
        in_ch = latent_channels * 2
        self.conv_in = nn.Conv2d(in_ch, base_channels, kernel_size=3, padding=1)

        # Level 1: 16x16
        self.down1 = AdaGNResBlock(base_channels, base_channels, cond_dim, dropout=dropout)
        self.downsample = nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1)

        # Level 2: 8x8
        self.down2 = AdaGNResBlock(base_channels * 2, base_channels * 2, cond_dim, dropout=dropout)

        # Mid / Bottleneck: 8x8
        self.mid = AdaGNResBlock(base_channels * 2, base_channels * 2, cond_dim, dropout=dropout)

        # U-Net Decoder
        self.upsample = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1)
        self.up1 = AdaGNResBlock(base_channels * 2, base_channels, cond_dim, dropout=dropout)

        # Output head
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, latent_channels, kernel_size=3, padding=1),
        )

    def forward(
        self,
        noisy_target: torch.Tensor,
        timestep: torch.Tensor,
        context_latent: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        # 1. Joint condition embedding
        t_emb = self.time_emb(timestep)
        a_emb = self.action_emb(action)
        cond = self.cond_mlp(torch.cat([t_emb, a_emb], dim=-1))

        # 2. Channel concatenation: [B, 2 * C_z, 16, 16]
        x = torch.cat([noisy_target, context_latent], dim=1)

        # 3. U-Net forward pass
        h0 = self.conv_in(x)
        h1 = self.down1(h0, cond)
        h2 = self.downsample(h1)
        h3 = self.down2(h2, cond)

        h_mid = self.mid(h3, cond)

        h_up = self.upsample(h_mid)
        # Skip connection concatenation
        h_cat = torch.cat([h_up, h1], dim=1)
        h_out = self.up1(h_cat, cond)

        return self.conv_out(h_out)


class GaussianDiffusion(nn.Module):
    """
    Gaussian Diffusion process and accelerated DDIM sampler.

    Args:
        num_timesteps: Total discrete diffusion steps (default: 100).
        beta_start: Initial noise variance (default: 1e-4).
        beta_end: Final noise variance (default: 0.02).
    """

    def __init__(
        self,
        num_timesteps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ) -> None:
        super().__init__()
        self.num_timesteps = num_timesteps

        betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion noising step:
            z_t = sqrt(alpha_bar_t) * z_0 + sqrt(1 - alpha_bar_t) * eps
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        noisy = sqrt_alpha * x_start + sqrt_one_minus_alpha * noise
        return noisy, noise

    @torch.no_grad()
    def ddim_sample(
        self,
        model: nn.Module,
        context_latent: torch.Tensor,
        action: torch.Tensor,
        ddim_steps: int = 10,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """
        Accelerated deterministic DDIM sampling (Song et al., 2020) to predict next latent z_{t+1}.

        Args:
            model: ActionConditionedUNet instance.
            context_latent: Current state latent z_t [B, C_z, H_z, W_z].
            action: Continuous action a_t [B, A].
            ddim_steps: Number of accelerated reverse sampling steps (e.g. 5 or 10).
            eta: Stochasticity factor (eta=0.0 corresponds to fully deterministic DDIM).

        Returns:
            Predicted future latent z_{t+1} [B, C_z, H_z, W_z].
        """
        device = context_latent.device
        b, c, h, w = context_latent.shape

        # Start from pure Gaussian noise z_K ~ N(0, I)
        z = torch.randn((b, c, h, w), device=device, dtype=torch.float32)

        # Uniformly subsample timesteps: [K-1, ..., 0]
        step_indices = torch.linspace(self.num_timesteps - 1, 0, ddim_steps, dtype=torch.long, device=device)

        for i, step_idx in enumerate(step_indices):
            t = torch.full((b,), step_idx.item(), device=device, dtype=torch.long)
            pred_noise = model(z, t, context_latent, action)

            sqrt_alpha_bar = self.sqrt_alphas_cumprod[step_idx]
            sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[step_idx]

            # 1. Estimate z_0
            pred_z0 = (z - sqrt_one_minus_alpha_bar * pred_noise) / sqrt_alpha_bar

            # 2. Determine previous step
            if i < len(step_indices) - 1:
                prev_step_idx = step_indices[i + 1]
                sqrt_alpha_bar_prev = self.sqrt_alphas_cumprod[prev_step_idx]
                sqrt_one_minus_alpha_bar_prev = self.sqrt_one_minus_alphas_cumprod[prev_step_idx]

                # Direction pointing to z_t
                dir_xt = sqrt_one_minus_alpha_bar_prev * pred_noise
                z = sqrt_alpha_bar_prev * pred_z0 + dir_xt
            else:
                z = pred_z0

        return z


class ActionConditionedDiffusionWorldModel(nn.Module):
    """
    Complete Action-Conditioned Latent Diffusion World Model (AC-LDM).

    Args:
        latent_channels: Latent channel depth (default: 4).
        action_dim: Continuous action space dimension (default: 3).
        num_timesteps: Discrete diffusion schedule timesteps (default: 100).
        vae: Optional pretrained and frozen SpatialVAE instance.
    """

    def __init__(
        self,
        latent_channels: int = 4,
        action_dim: int = 3,
        num_timesteps: int = 100,
        base_channels: int = 64,
        vae: SpatialVAE | None = None,
    ) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.action_dim = action_dim
        self.unet = ActionConditionedUNet(
            latent_channels=latent_channels,
            action_dim=action_dim,
            base_channels=base_channels,
        )
        self.diffusion = GaussianDiffusion(num_timesteps=num_timesteps)
        self.vae = vae

    def compute_loss(
        self,
        target_latent: torch.Tensor,
        context_latent: torch.Tensor,
        action: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Compute mean-squared epsilon noise prediction loss.
        """
        b = target_latent.shape[0]
        device = target_latent.device
        t = torch.randint(0, self.diffusion.num_timesteps, (b,), device=device, dtype=torch.long)

        noisy_target, noise = self.diffusion.q_sample(target_latent, t)
        pred_noise = self.unet(noisy_target, t, context_latent, action)

        loss = F.mse_loss(pred_noise, noise)
        return {
            "loss": loss,
            "pred_noise": pred_noise,
            "target_noise": noise,
        }

    @torch.no_grad()
    def predict_next(
        self,
        context_latent: torch.Tensor,
        action: torch.Tensor,
        ddim_steps: int = 10,
    ) -> torch.Tensor:
        """Predict next latent transition z_{t+1} using fast DDIM sampler."""
        return self.diffusion.ddim_sample(
            model=self.unet,
            context_latent=context_latent,
            action=action,
            ddim_steps=ddim_steps,
        )

    @torch.no_grad()
    def rollout_latents(
        self,
        initial_latent: torch.Tensor,
        action_sequence: torch.Tensor,
        ddim_steps: int = 10,
    ) -> torch.Tensor:
        """
        Autoregressively unroll future latents over an extended action horizon.

        Args:
            initial_latent: Initial state latent z_0 [B, C_z, H_z, W_z].
            action_sequence: Candidate actions [B, H, A].
            ddim_steps: DDIM inference steps per rollout step.

        Returns:
            Unrolled trajectory of predicted future latents [B, H, C_z, H_z, W_z].
        """
        _b, h, _a = action_sequence.shape
        trajectory: list[torch.Tensor] = []
        current_z = initial_latent

        for t in range(h):
            action_t = action_sequence[:, t, :]
            current_z = self.predict_next(current_z, action_t, ddim_steps=ddim_steps)
            trajectory.append(current_z)

        return torch.stack(trajectory, dim=1)
