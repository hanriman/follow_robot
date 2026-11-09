"""
Action-Conditioned Teacher-Free World Model (LeWM) for AckBot.

Implements single-encoder multi-modal world model dynamics with continuous Ackermann
action conditioning, eliminating EMA target encoders and stop-gradient operations
via Sketched Isotropic Gaussian Regularization (SIGReg).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AckBotSensorEncoder(nn.Module):
    r"""
    Multi-Modal Sensor Encoder mapping 426-D AckBot observation vector to latent state z.

    Observation Structure:
    ----------------------
    - [0:64]    : Front RGB-D Camera Depth (64 normalized horizontal rays)
    - [64:424]  : Elevated Rear 2D LiDAR (360 planar distance measurements)
    - [424:426] : Robot Proprioception (normalized linear velocity v, steering angle delta)
    """

    def __init__(
        self,
        cam_dim: int = 64,
        lidar_dim: int = 360,
        proprio_dim: int = 2,
        latent_dim: int = 128,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.cam_dim = cam_dim
        self.lidar_dim = lidar_dim
        self.proprio_dim = proprio_dim
        self.latent_dim = latent_dim

        # 1. Front Camera Depth Branch (64 -> 64)
        self.cam_net = nn.Sequential(
            nn.Linear(cam_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
        )

        # 2. Elevated Rear LiDAR Branch (360 -> 128 -> 64)
        self.lidar_net = nn.Sequential(
            nn.Linear(lidar_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )

        # 3. Proprioception Branch (2 -> 16)
        self.proprio_net = nn.Sequential(
            nn.Linear(proprio_dim, 16),
            nn.LayerNorm(16),
            nn.GELU(),
        )

        # 4. Multi-Modal Fusion Head (64 + 64 + 16 = 144 -> 256 -> latent_dim)
        fused_dim = 64 + 64 + 16
        self.fusion_head = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            obs: [..., 426] sensor observation vector.

        Returns:
            Latent representation z [..., latent_dim].
        """
        # Split observation into constituent sensors
        cam_depth = obs[..., : self.cam_dim]
        lidar = obs[..., self.cam_dim : self.cam_dim + self.lidar_dim]
        proprio = obs[..., self.cam_dim + self.lidar_dim :]

        h_cam = self.cam_net(cam_depth)
        h_lidar = self.lidar_net(lidar)
        h_prop = self.proprio_net(proprio)

        fused = torch.cat([h_cam, h_lidar, h_prop], dim=-1)
        z = self.fusion_head(fused)
        return z


class ActionConditionedPredictor(nn.Module):
    r"""
    Residual Latent Transition Predictor T_\psi(z_t, a_t) \to \hat{z}_{t+1}.

    Maps the current state latent representation z_t \in R^D and continuous Ackermann
    action vector a_t \in R^2 ([v, \delta]) to predicted future latent \hat{z}_{t+1} \in R^D
    using a deep residual MLP architecture with layer normalization.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        action_dim: int = 2,
        hidden_dim: int = 256,
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


class TeacherFreeLeWorldModel(nn.Module):
    r"""
    Teacher-Free Action-Conditioned Le World Model (LeWM) for AckBot.

    Architecture Principles:
    ------------------------
    1. Single Sensor Encoder:
       Both context observation o_t and future observation o_{t+1} pass through
       a single shared encoder E_\theta.
    2. Elimination of Dual-Network Heuristics:
       No exponential moving average (EMA) teacher network, no momentum parameters,
       and no stop-gradient operations on target latents.
    3. Joint End-to-End Training:
       Both encoder E_\theta and dynamics predictor T_\psi receive gradients jointly
       from the transition error \mathcal{L}_{\text{trans}}(\hat{z}_{t+1}, z_{t+1}).
    4. Collapse Prevention via SIGReg:
       Regularizes the empirical distribution of representations toward an isotropic
       standard normal N(0, I_D) using the analytical Epps-Pulley test.
    """

    def __init__(
        self,
        encoder: nn.Module | None = None,
        latent_dim: int = 128,
        action_dim: int = 2,
        predictor_hidden_dim: int = 256,
        predictor_layers: int = 3,
        cam_dim: int = 64,
        lidar_dim: int = 360,
        proprio_dim: int = 2,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim

        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = AckBotSensorEncoder(
                cam_dim=cam_dim,
                lidar_dim=lidar_dim,
                proprio_dim=proprio_dim,
                latent_dim=latent_dim,
            )

        self.predictor = ActionConditionedPredictor(
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=predictor_hidden_dim,
            num_layers=predictor_layers,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        r"""Encode sensor observation into latent representation z."""
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
        """
        z_t = self.encoder(obs_t)
        target_next_z = self.encoder(obs_next)

        pred_next_z = self.predictor(z_t, action_t)

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
            initial_obs: Initial sensor observation [B, 426].
            action_sequence: Candidate action sequence [B, H, 2].

        Returns:
            Unrolled latent trajectory [B, H, D].
        """
        z_0 = self.encode(initial_obs)
        return self.predictor.rollout(z_0, action_sequence)
