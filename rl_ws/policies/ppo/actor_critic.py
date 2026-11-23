"""
Recurrent Spatial Actor-Critic neural network architecture for AckBot target following with PPO.

Features:
- Pure 426-D sensor input contract (64 camera depth + 360 LiDAR + 2 state)
- Proximity inversion (1.0 - range_norm) so sparse targets stand out over empty background
- Angular CoordConv 1D encoders (with circular padding for 360-deg LiDAR)
- Differentiable polar soft-argmin & angular sector summary features
- GRU(128, 128) temporal memory with current-frame residual skip for velocity & FOV-exit memory
- Decoupled Actor and Critic trunks with non-saturating scaled-tanh action mean
"""

import math

import torch
from torch import nn
from torch.distributions.normal import Normal


def layer_init(layer, std=math.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class _MPSAdaptiveAvgPool1d(nn.Module):
    """AdaptiveAvgPool1d compatible with Apple Silicon MPS for non-divisible lengths."""

    def __init__(self, output_size: int):
        super().__init__()
        self.output_size = int(output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[-1]
        out = self.output_size
        if x.device.type != "mps" or length % out == 0:
            return nn.functional.adaptive_avg_pool1d(x, out)
        # Match PyTorch adaptive_avg_pool1d bin boundaries without hitting MPS divisibility error
        return torch.cat(
            [
                x[..., (i * length) // out : ((i + 1) * length + out - 1) // out].mean(
                    dim=-1, keepdim=True
                )
                for i in range(out)
            ],
            dim=-1,
        )


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim=426,
        act_dim=2,
        cam_dim=64,
        lidar_dim=360,
        gru_hidden_dim=128,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.cam_dim = cam_dim
        self.lidar_dim = lidar_dim
        self.state_dim = obs_dim - cam_dim - lidar_dim  # 2 features (v, steer)
        self.gru_hidden_dim = gru_hidden_dim

        # Action physical bounds (from spec.txt)
        # v in [-0.40, 0.80] m/s, steer in [-0.553953, +0.553953] rad
        self.register_buffer("v_min", torch.tensor(-0.40, dtype=torch.float32))
        self.register_buffer("v_max", torch.tensor(0.80, dtype=torch.float32))
        self.register_buffer("steer_max", torch.tensor(0.553953, dtype=torch.float32))

        # Action scaling constants: action in [-1, 1] mapped to physical range
        self.v_scale = (self.v_max - self.v_min) / 2.0  # 0.60
        self.v_bias = (self.v_max + self.v_min) / 2.0  # 0.20
        self.steer_scale = self.steer_max  # 0.553953

        # Registered angular coordinate buffers for CoordConv and polar soft-argmin
        self.register_buffer(
            "cam_angles", torch.linspace(-1.0, 1.0, self.cam_dim, dtype=torch.float32)
        )
        lidar_angles = torch.linspace(0.0, 2.0 * math.pi, self.lidar_dim + 1, dtype=torch.float32)[
            :-1
        ]
        self.register_buffer("lidar_cos", torch.cos(lidar_angles))
        self.register_buffer("lidar_sin", torch.sin(lidar_angles))

        # 1. Front Camera Depth Branch (64 rays -> 1D CoordConv + 6 polar summary features)
        self.cam_conv = nn.Sequential(
            layer_init(nn.Conv1d(2, 16, kernel_size=5, stride=2, padding=2)),
            nn.SiLU(),
            layer_init(nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1)),
            nn.SiLU(),
            _MPSAdaptiveAvgPool1d(4),
        )
        self.cam_proj = nn.Sequential(
            layer_init(nn.Linear(32 * 4 + 6, 64)),
            nn.LayerNorm(64),
            nn.Tanh(),
        )

        # 2. 360-deg LiDAR Branch (360 rays -> circular 1D CoordConv + 11 polar/sector features)
        self.lidar_conv = nn.Sequential(
            layer_init(
                nn.Conv1d(
                    3, 16, kernel_size=7, stride=3, padding=3, padding_mode="circular"
                )
            ),
            nn.SiLU(),
            layer_init(
                nn.Conv1d(
                    16, 32, kernel_size=5, stride=3, padding=2, padding_mode="circular"
                )
            ),
            nn.SiLU(),
            _MPSAdaptiveAvgPool1d(6),
        )
        self.lidar_proj = nn.Sequential(
            layer_init(nn.Linear(32 * 6 + 11, 64)),
            nn.LayerNorm(64),
            nn.Tanh(),
        )

        # 3. Vehicle Proprioceptive State Branch (2 features: v, steer)
        self.state_net = nn.Sequential(
            layer_init(nn.Linear(self.state_dim, 16)),
            nn.LayerNorm(16),
            nn.Tanh(),
        )

        # 4. Fused Multi-Modal Spatial Representation (64 + 64 + 16 = 144 -> 128)
        self.fused_net = nn.Sequential(
            layer_init(nn.Linear(144, 128)),
            nn.Tanh(),
        )

        # 5. Temporal Memory (GRU over fused spatial features)
        self.gru = nn.GRU(128, self.gru_hidden_dim)
        for name, param in self.gru.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)

        # 6. Decoupled Critic & Actor Heads (input = 128 current spatial + 128 GRU temporal = 256)
        rep_dim = 128 + self.gru_hidden_dim
        self.critic = nn.Sequential(
            layer_init(nn.Linear(rep_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(rep_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, self.act_dim), std=0.01),
        )

        # Learnable log standard deviation for continuous Gaussian policy
        self.actor_logstd = nn.Parameter(torch.zeros(1, self.act_dim) - 0.5)

    def init_hidden(self, batch_size=1, device=None):
        """Return zero-initialized GRU hidden state (1, batch_size, gru_hidden_dim)."""
        if device is None:
            device = self.v_min.device
        return torch.zeros((1, batch_size, self.gru_hidden_dim), dtype=torch.float32, device=device)

    def _extract_spatial_features(self, x):
        """Extract CoordConv + polar soft-argmin features for a flattened batch [N, obs_dim]."""
        cam_feats = x[:, : self.cam_dim]
        lidar_feats = x[:, self.cam_dim : self.cam_dim + self.lidar_dim]
        state_feats = x[:, self.cam_dim + self.lidar_dim :]

        # Invert normalized ranges -> proximity (0.0 = empty max-range, 1.0 = close hit)
        cam_prox = 1.0 - cam_feats
        lidar_prox = 1.0 - lidar_feats

        # Differentiable polar soft-argmin & sector summaries for camera
        cam_w = torch.softmax(cam_prox * 12.0, dim=-1)
        cam_soft_bearing = (cam_w * self.cam_angles).sum(dim=-1, keepdim=True)
        cam_max_prox = cam_prox.max(dim=-1, keepdim=True).values
        cam_mean_prox = cam_prox.mean(dim=-1, keepdim=True)
        third = self.cam_dim // 3
        cam_right_prox = cam_prox[:, :third].max(dim=-1, keepdim=True).values
        cam_center_prox = cam_prox[:, third : 2 * third].max(dim=-1, keepdim=True).values
        cam_left_prox = cam_prox[:, 2 * third :].max(dim=-1, keepdim=True).values
        cam_polar = torch.cat(
            [
                cam_soft_bearing,
                cam_max_prox,
                cam_mean_prox,
                cam_right_prox,
                cam_center_prox,
                cam_left_prox,
            ],
            dim=-1,
        )

        cam_coord = self.cam_angles.unsqueeze(0).expand_as(cam_prox)
        cam_in = torch.stack([cam_prox, cam_coord], dim=1)  # [N, 2, 64]
        cam_conv_out = self.cam_conv(cam_in).flatten(start_dim=1)
        h_cam = self.cam_proj(torch.cat([cam_conv_out, cam_polar], dim=-1))

        # Differentiable circular polar soft-argmin & 8-sector summaries for 360-deg LiDAR
        lidar_w = torch.softmax(lidar_prox * 12.0, dim=-1)
        lidar_soft_cos = (lidar_w * self.lidar_cos).sum(dim=-1, keepdim=True)
        lidar_soft_sin = (lidar_w * self.lidar_sin).sum(dim=-1, keepdim=True)
        lidar_max_prox = lidar_prox.max(dim=-1, keepdim=True).values
        sector_size = self.lidar_dim // 8
        lidar_sectors = lidar_prox[:, : sector_size * 8].view(-1, 8, sector_size).max(dim=-1).values
        lidar_polar = torch.cat(
            [lidar_soft_cos, lidar_soft_sin, lidar_max_prox, lidar_sectors], dim=-1
        )

        lidar_cos_exp = self.lidar_cos.unsqueeze(0).expand_as(lidar_prox)
        lidar_sin_exp = self.lidar_sin.unsqueeze(0).expand_as(lidar_prox)
        lidar_in = torch.stack([lidar_prox, lidar_cos_exp, lidar_sin_exp], dim=1)  # [N, 3, 360]
        lidar_conv_out = self.lidar_conv(lidar_in).flatten(start_dim=1)
        h_lidar = self.lidar_proj(torch.cat([lidar_conv_out, lidar_polar], dim=-1))

        h_state = self.state_net(state_feats)

        fused = torch.cat([h_cam, h_lidar, h_state], dim=-1)
        return self.fused_net(fused)

    def _extract_features(self, x, hx=None, episode_dones=None):
        """
        Encode observations into combined spatial + GRU temporal representation.

        Supports:
        - Step input `x`: [B, obs_dim] with optional `hx`: [1, B, H] -> returns ([B, 256], next_hx [1, B, H])
        - Sequence input `x`: [T, B, obs_dim] with `hx`: [1, B, H] and optional `episode_dones`: [T, B]
          -> returns ([T, B, 256], next_hx [1, B, H])
        """
        is_seq = x.dim() == 3
        if not is_seq:
            spatial = self._extract_spatial_features(x)  # [B, 128]
            bsz = spatial.shape[0]
            if hx is None:
                hx = self.init_hidden(bsz, device=spatial.device)
            gru_out, next_hx = self.gru(spatial.unsqueeze(0), hx)
            gru_out = gru_out.squeeze(0)
            if episode_dones is not None:
                mask = (1.0 - episode_dones).view(1, bsz, 1)
                next_hx = next_hx * mask
            return torch.cat([spatial, gru_out], dim=-1), next_hx

        seq_len, bsz, _ = x.shape
        spatial_flat = self._extract_spatial_features(x.reshape(seq_len * bsz, self.obs_dim))
        spatial_seq = spatial_flat.view(seq_len, bsz, 128)
        if hx is None:
            hx = self.init_hidden(bsz, device=x.device)

        if episode_dones is None or not torch.any(episode_dones[:-1] > 0.5):
            gru_seq, next_hx = self.gru(spatial_seq, hx)
            if episode_dones is not None:
                next_hx = next_hx * (1.0 - episode_dones[-1]).view(1, bsz, 1)
        else:
            outputs = []
            cur_hx = hx
            for t in range(seq_len):
                out_t, cur_hx = self.gru(spatial_seq[t : t + 1], cur_hx)
                outputs.append(out_t)
                cur_hx = cur_hx * (1.0 - episode_dones[t]).view(1, bsz, 1)
            gru_seq = torch.cat(outputs, dim=0)
            next_hx = cur_hx

        return torch.cat([spatial_seq, gru_seq], dim=-1), next_hx

    def get_value(self, x, hx=None, episode_dones=None):
        features, _ = self._extract_features(x, hx=hx, episode_dones=episode_dones)
        return self.critic(features)

    def scale_action(self, norm_action):
        """Map normalized action [-1, 1] to physical vehicle commands."""
        v = norm_action[..., 0] * self.v_scale + self.v_bias
        steer = norm_action[..., 1] * self.steer_scale
        return torch.stack([v, steer], dim=-1)

    def unscale_action(self, phys_action):
        """Map physical commands back to normalized [-1, 1] range."""
        norm_v = (phys_action[..., 0] - self.v_bias) / self.v_scale
        norm_steer = phys_action[..., 1] / self.steer_scale
        return torch.stack([norm_v, norm_steer], dim=-1)

    def _compute_action_mean(self, features):
        # Bound pre-tanh Gaussian mean to [-3.0, 3.0] so tanh reaches +/-0.995 without inf atanh
        return 3.0 * torch.tanh(self.actor_mean(features))

    def get_action_and_value(
        self, x, action=None, hx=None, episode_dones=None, return_hidden=False
    ):
        features, next_hx = self._extract_features(x, hx=hx, episode_dones=episode_dones)
        raw_mean = self._compute_action_mean(features)

        # Clamp logstd so exploration std stays in [0.08, 0.70] and never inflates uncontrollably
        action_logstd = torch.clamp(self.actor_logstd.expand_as(raw_mean), -2.5, -0.35)
        action_std = torch.exp(action_logstd)

        dist = Normal(raw_mean, action_std)

        if action is None:
            u = dist.sample()
            norm_action = torch.tanh(u)
            action = self.scale_action(norm_action)
        else:
            if action.dim() == 1:
                action = action.unsqueeze(0)
            norm_action = torch.clamp(self.unscale_action(action), -0.9999, 0.9999)
            u = torch.atanh(norm_action)

        # Exact squashed-Gaussian log-prob with tanh Jacobian correction (no clamping bias)
        log_prob = (dist.log_prob(u) - torch.log(1.0 - norm_action.pow(2) + 1e-6)).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.critic(features)

        if return_hidden:
            return action, log_prob, entropy, value, next_hx
        return action, log_prob, entropy, value

    def get_deterministic_action(self, x, hx=None, return_hidden=False):
        """Greedy inference for evaluation (no exploration noise)."""
        with torch.no_grad():
            features, next_hx = self._extract_features(x, hx=hx)
            norm_mean = torch.tanh(self._compute_action_mean(features))
            phys_action = self.scale_action(norm_mean)
            if return_hidden:
                return phys_action, next_hx
            return phys_action
