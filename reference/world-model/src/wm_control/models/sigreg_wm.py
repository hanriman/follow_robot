"""
Action-Conditioned Teacher-Free World Model Architecture (LeWM).

Implements single-encoder visual world model dynamics with continuous action conditioning,
eliminating EMA target encoders and stop-gradient operations via Sketched Isotropic
Gaussian Regularization (SIGReg).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ActionConditionedPredictor(nn.Module):
    r"""
    Residual Latent Transition Predictor T_\psi(z_t, a_t) \to \hat{z}_{t+1}.

    Maps the current state latent representation z_t \in R^D and continuous action
    vector a_t \in R^A to predicted future latent \hat{z}_{t+1} \in R^D using a
    deep residual MLP architecture with layer normalization.

    Args:
        latent_dim: Dimensionality D of state representation z.
        action_dim: Dimensionality A of continuous action vector a.
        hidden_dim: Width of intermediate MLP layers.
        num_layers: Number of residual MLP blocks.
        dropout: Optional dropout rate for regularizing transition dynamics.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        action_dim: int = 3,
        hidden_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # Input fusion: project [z, a] into hidden space
        self.input_proj = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Residual transition blocks
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
            )

        # Output projection for residual delta \Delta z
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.delta_head = nn.Linear(hidden_dim, latent_dim)

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        r"""
        Predict next latent representation \hat{z}_{t+1} = z_t + \Delta(z_t, a_t).

        Args:
            z: Current latent state [B, D] or [B, T, D].
            action: Continuous action vector [B, A] or [B, T, A].

        Returns:
            Predicted next latent state [B, D] or [B, T, D].
        """
        if z.shape[:-1] != action.shape[:-1]:
            raise ValueError(
                f"Batch shape mismatch between z {z.shape[:-1]} and action {action.shape[:-1]}."
            )

        x = torch.cat([z, action], dim=-1)
        h = self.input_proj(x)

        for block in self.blocks:
            h = h + block(h)

        h = self.out_norm(h)
        delta = self.delta_head(h)

        # Residual formulation preserves identity transition when delta -> 0
        return z + delta

    def rollout(
        self,
        initial_z: torch.Tensor,
        action_sequence: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Autoregressively unroll latent transitions over an extended action horizon.

        Args:
            initial_z: Initial latent state z_0 [B, D].
            action_sequence: Sequence of actions [B, H, A] over horizon H.

        Returns:
            Unrolled latent trajectory [B, H, D] corresponding to \hat{z}_1, ..., \hat{z}_H.
        """
        _, H, _ = action_sequence.shape
        trajectory: list[torch.Tensor] = []
        current_z = initial_z

        for t in range(H):
            action_t = action_sequence[:, t, :]
            current_z = self.forward(current_z, action_t)
            trajectory.append(current_z)

        return torch.stack(trajectory, dim=1)


class SimpleVisualEncoder(nn.Module):
    r"""
    Convolutional Visual Encoder mapping image observations to latent representations.

    Suitable for visual control observations (e.g. [B, C, H, W] -> [B, D]).
    Can be replaced with any pretrained encoder or spatial VAE representation.

    Args:
        in_channels: Observation color channels (default: 3).
        latent_dim: Dimension D of output representation space.
        base_channels: Channel width of the first conv stage.
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 128,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        self.conv_net = nn.Sequential(
            # Stage 1: e.g. 64x64 -> 32x32
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),
            # Stage 2: e.g. 32x32 -> 16x16
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.GELU(),
            # Stage 3: e.g. 16x16 -> 8x8
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.GELU(),
            # Stage 4: e.g. 8x8 -> 4x4
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.GELU(),
            # Global adaptive pooling
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        self.head = nn.Sequential(
            nn.Linear(base_channels * 8, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            x: Input image tensor [B, C, H, W].

        Returns:
            Latent representation [B, D].
        """
        features = self.conv_net(x)
        return self.head(features)


class TeacherFreeLeWorldModel(nn.Module):
    r"""
    Teacher-Free Action-Conditioned Le World Model (LeWM).

    Architecture Principles:
    ------------------------
    1. Single Visual Encoder:
       Both context frames x_t and future target frames x_{t+1} are mapped through
       a single shared visual encoder E_\theta.
    2. Elimination of Dual-Network Heuristics:
       No exponential moving average (EMA) teacher network, no momentum parameters,
       and no stop-gradient operations on target latents.
    3. Joint End-to-End Training:
       Both encoder E_\theta and dynamics predictor T_\psi receive gradients jointly
       from the transition error \mathcal{L}_{\text{trans}}(\hat{z}_{t+1}, z_{t+1}).
    4. Collapse Prevention via SIGReg:
       Without negative pairs or EMA drift, representation collapse to a trivial constant
       is prevented by regularizing the empirical distribution of representations
       toward an isotropic standard normal N(0, I_D) using the analytical Epps-Pulley test.

    Args:
        encoder: Visual encoder network mapping observation [B, C, H, W] -> [B, D].
                 If None, initializes a default SimpleVisualEncoder.
        latent_dim: Latent representation dimensionality D.
        action_dim: Continuous action space dimensionality A.
        predictor_hidden_dim: Hidden dimension of transition predictor MLP.
        predictor_layers: Number of residual blocks in predictor.
    """

    def __init__(
        self,
        encoder: nn.Module | None = None,
        latent_dim: int = 128,
        action_dim: int = 3,
        predictor_hidden_dim: int = 512,
        predictor_layers: int = 3,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim

        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = SimpleVisualEncoder(
                in_channels=in_channels,
                latent_dim=latent_dim,
            )

        self.predictor = ActionConditionedPredictor(
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=predictor_hidden_dim,
            num_layers=predictor_layers,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        r"""Encode visual observation into latent representation z."""
        return self.encoder(x)

    def predict_next(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        r"""Predict next latent state \hat{z}_{t+1} = T(z_t, a_t)."""
        return self.predictor(z, action)

    def forward(
        self,
        obs_t: torch.Tensor,
        action_t: torch.Tensor,
        obs_next: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        r"""
        Full single-pass training forward pass.

        Both obs_t and obs_next pass through the same shared encoder E_\theta.
        Notice that obs_next maintains active autograd graph (no torch.no_grad, no .detach()),
        allowing true end-to-end joint optimization.

        Args:
            obs_t: Current visual observation [B, C, H, W].
            action_t: Continuous action vector [B, A].
            obs_next: Next visual observation [B, C, H, W].

        Returns:
            Dictionary containing:
                - 'z_t': Encoded current state [B, D].
                - 'pred_next_z': Predicted next state [B, D].
                - 'target_next_z': Ground truth encoded next state [B, D].
                - 'all_tokens': Concatenated representations for SIGReg regularization.
        """
        # Encode current and next observations using single shared encoder
        z_t = self.encoder(obs_t)
        target_next_z = self.encoder(obs_next)

        # Predict future representation conditioned on action
        pred_next_z = self.predictor(z_t, action_t)

        # Collect representations to be regularized by SIGReg
        all_tokens = torch.cat([z_t, target_next_z], dim=0)

        return {
            "z_t": z_t,
            "pred_next_z": pred_next_z,
            "target_next_z": target_next_z,
            "all_tokens": all_tokens,
        }

    def rollout(
        self,
        initial_obs: torch.Tensor,
        action_sequence: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Encode initial observation and unroll latent trajectory under action sequence.

        Args:
            initial_obs: Initial visual observation [B, C, H, W].
            action_sequence: Candidate action sequence [B, H, A].

        Returns:
            Unrolled latent trajectory [B, H, D].
        """
        z_0 = self.encode(initial_obs)
        return self.predictor.rollout(z_0, action_sequence)
