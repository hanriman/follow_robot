"""
Actor-Critic neural network architecture for AckBot target following with PPO.
Dual-branch encoder processing 360-ray 2D LiDAR and vehicle/target states,
optimized for PyTorch on Apple Silicon (Metal/MPS).
"""

import math

import torch
from torch import nn
from torch.distributions.normal import Normal


def layer_init(layer, std=math.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, obs_dim=426, act_dim=2, cam_dim=64, lidar_dim=360):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.cam_dim = cam_dim
        self.lidar_dim = lidar_dim
        self.state_dim = obs_dim - cam_dim - lidar_dim  # 2 features (v, steer)

        # Action physical bounds (from spec.txt)
        # v in [-0.40, 0.80] m/s, steer in [-0.553953, +0.553953] rad
        self.register_buffer("v_min", torch.tensor(-0.40, dtype=torch.float32))
        self.register_buffer("v_max", torch.tensor(0.80, dtype=torch.float32))
        self.register_buffer("steer_max", torch.tensor(0.553953, dtype=torch.float32))

        # Action scaling constants: action in [-1, 1] mapped to physical range
        self.v_scale = (self.v_max - self.v_min) / 2.0  # 0.60
        self.v_bias = (self.v_max + self.v_min) / 2.0  # 0.20
        self.steer_scale = self.steer_max  # 0.553953

        # 1. Front Camera Depth Branch (64 rays)
        self.cam_net = nn.Sequential(
            layer_init(nn.Linear(self.cam_dim, 64)),
            nn.LayerNorm(64),
            nn.Tanh(),
        )

        # 2. LiDAR Feature Extractor Branch (360 rays)
        self.lidar_net = nn.Sequential(
            layer_init(nn.Linear(self.lidar_dim, 128)),
            nn.LayerNorm(128),
            nn.Tanh(),
            layer_init(nn.Linear(128, 64)),
            nn.Tanh(),
        )

        # 3. Vehicle Proprioceptive State Branch (2 features: v, steer)
        self.state_net = nn.Sequential(
            layer_init(nn.Linear(self.state_dim, 16)),
            nn.LayerNorm(16),
            nn.Tanh(),
        )

        # 4. Fused Multi-Modal Representation
        fused_dim = 64 + 64 + 16  # 144
        self.fused_net = nn.Sequential(
            layer_init(nn.Linear(fused_dim, 128)),
            nn.Tanh(),
        )

        # 5. Critic Head (Value function V(s))
        self.critic = nn.Sequential(
            layer_init(nn.Linear(128, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

        # 6. Actor Head (Policy mean mu(s))
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(128, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, self.act_dim), std=0.01),
        )

        # Learnable log standard deviation for continuous Gaussian policy
        self.actor_logstd = nn.Parameter(torch.zeros(1, self.act_dim) - 0.5)

    def _extract_features(self, x):
        cam_feats = x[:, : self.cam_dim]
        lidar_feats = x[:, self.cam_dim : self.cam_dim + self.lidar_dim]
        state_feats = x[:, self.cam_dim + self.lidar_dim :]

        h_cam = self.cam_net(cam_feats)
        h_lidar = self.lidar_net(lidar_feats)
        h_state = self.state_net(state_feats)

        fused = torch.cat([h_cam, h_lidar, h_state], dim=-1)
        return self.fused_net(fused)

    def get_value(self, x):
        features = self._extract_features(x)
        return self.critic(features)

    def scale_action(self, norm_action):
        """Map normalized action [-1, 1] to physical vehicle commands."""
        v = norm_action[:, 0] * self.v_scale + self.v_bias
        steer = norm_action[:, 1] * self.steer_scale
        return torch.stack([v, steer], dim=-1)

    def unscale_action(self, phys_action):
        """Map physical commands back to normalized [-1, 1] range."""
        norm_v = (phys_action[:, 0] - self.v_bias) / self.v_scale
        norm_steer = phys_action[:, 1] / self.steer_scale
        return torch.stack([norm_v, norm_steer], dim=-1)

    def get_action_and_value(self, x, action=None):
        features = self._extract_features(x)
        # Tanh ensures action_mean stays strictly within normalized [-1, 1] bounds
        action_mean = torch.tanh(self.actor_mean(features))
        # Clamp logstd so standard deviation never collapses or explodes
        action_logstd = torch.clamp(self.actor_logstd.expand_as(action_mean), -2.0, 0.5)
        action_std = torch.exp(action_logstd)

        dist = Normal(action_mean, action_std)

        if action is None:
            # Sample normalized action in unconstrained space
            raw_action = dist.sample()
            clipped_raw = torch.clamp(raw_action, -1.0, 1.0)
            phys_action = self.scale_action(clipped_raw)
            action = phys_action

        # If action is provided, compute its log_prob in normalized space
        if len(action.shape) == 1:
            action = action.unsqueeze(0)
        norm_action = self.unscale_action(action)
        norm_action = torch.clamp(norm_action, -0.99999, 0.99999)

        log_prob = dist.log_prob(norm_action).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.critic(features)

        return action, log_prob, entropy, value

    def get_deterministic_action(self, x):
        """Greedy inference for evaluation (no exploration noise)."""
        with torch.no_grad():
            features = self._extract_features(x)
            action_mean = torch.tanh(self.actor_mean(features))
            clipped_mean = torch.clamp(action_mean, -1.0, 1.0)
            return self.scale_action(clipped_mean)
