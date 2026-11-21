"""
Cross-Entropy Method (CEM) Latent Action Planner for AckBot Target Following.

Executes population-based trajectory optimization over learned latent state dynamics
to generate continuous Ackermann steering and throttle commands.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch


class CEMPlanner:
    """
    Cross-Entropy Method (CEM) Action Planner in Latent Space for AckBot.

    Args:
        horizon: Planning horizon H (number of future steps, default: 5).
        pop_size: Number of candidate action sequences sampled per iteration (default: 50).
        num_elites: Number of highest-scoring candidate sequences used to update distribution (default: 5).
        num_iters: Number of CEM refinement iterations per control step (default: 3).
        gamma: Temporal discount factor for future rewards (default: 0.95).
        momentum: Smoothing factor for distribution parameter updates (default: 0.1).
        action_dim: Continuous action dimensionality (default: 2 for AckBot: [v, delta]).
        v_min: Minimum linear speed (-0.40 m/s).
        v_max: Maximum linear speed (+0.80 m/s).
        steer_max: Maximum steering angle (+0.553953 rad / 31.74 deg).
        device: Device to run rollouts on.
    """

    def __init__(
        self,
        horizon: int = 5,
        pop_size: int = 50,
        num_elites: int = 5,
        num_iters: int = 3,
        gamma: float = 0.95,
        momentum: float = 0.1,
        action_dim: int = 2,
        v_min: float = -0.40,
        v_max: float = 0.80,
        steer_max: float = 0.553953,
        device: torch.device | None = None,
    ) -> None:
        self.horizon = horizon
        self.pop_size = pop_size
        self.num_elites = num_elites
        self.num_iters = num_iters
        self.gamma = gamma
        self.momentum = momentum
        self.action_dim = action_dim
        self.device = device or torch.device("cpu")

        # Canonical physical Ackermann action bounds
        self.action_low = torch.tensor([v_min, -steer_max], device=self.device, dtype=torch.float32)
        self.action_high = torch.tensor([v_max, steer_max], device=self.device, dtype=torch.float32)

        # Temporal discount weights: [1, gamma, gamma^2, ..., gamma^{H-1}]
        self.discounts = torch.pow(
            self.gamma,
            torch.arange(horizon, device=self.device, dtype=torch.float32),
        )

        # Previous mean plan for warm-starting receding horizon
        self.prev_mean: torch.Tensor | None = None

    def reset(self) -> None:
        """Reset warm-started trajectory memory."""
        self.prev_mean = None

    def plan(
        self,
        initial_z: torch.Tensor,
        rollout_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        score_fn: Callable[[torch.Tensor], torch.Tensor],
        obs_t: np.ndarray | torch.Tensor | None = None,
        tracker_status: dict | None = None,
    ) -> torch.Tensor:
        """
        Execute CEM optimization to find optimal action sequence and return first action.

        Args:
            initial_z: Current latent state z_0 [1, ...].
            rollout_fn: Function mapping (z_0 [N, ...], actions [N, H, A]) -> unrolled_z [N, H, ...].
            score_fn: Function mapping unrolled_z [N, H, ...] -> scalar scores per step [N, H].
            obs_t: Optional current 426-D observation for LiDAR safety shielding.
            tracker_status: Optional Kalman tracker status dict for target-pursuit guidance.

        Returns:
            Optimal continuous action vector for current timestep [A] ([v, delta]).
        """
        device = self.device

        # Initialize distribution parameters
        if self.prev_mean is not None:
            # Shift previous plan by 1 step, append neutral follow action
            mean = torch.cat([self.prev_mean[1:], self.prev_mean[-1:]], dim=0).to(device)
        else:
            # Default initialization: gentle forward pursuit [v=0.30 m/s, delta=0.0 rad]
            mean = torch.tensor([[0.30, 0.0]], device=device, dtype=torch.float32).repeat(
                self.horizon, 1
            )

        std = torch.tensor([[0.20, 0.15]], device=device, dtype=torch.float32).repeat(
            self.horizon, 1
        )

        # Parse observation for LiDAR clearance if provided
        lidar_front_min = 3.0
        if obs_t is not None:
            if isinstance(obs_t, torch.Tensor):
                obs_np = obs_t.detach().cpu().numpy().flatten()
            else:
                obs_np = np.asarray(obs_t).flatten()
            if len(obs_np) >= 424:
                # Denormalize LiDAR (360 samples at index 64:424)
                lidar_m = obs_np[64:424] * 2.90 + 0.10
                front_rays = np.concatenate([lidar_m[:30], lidar_m[330:]])
                lidar_front_min = float(np.min(front_rays))

        # Replicate initial state across population: [N, ...]
        batch_z0 = initial_z.repeat(self.pop_size, *([1] * (initial_z.dim() - 1))).to(device)

        best_actions = mean

        for _ in range(self.num_iters):
            # Sample candidate actions: [N, H, A]
            noise = torch.randn(self.pop_size, self.horizon, self.action_dim, device=device)
            candidates = mean.unsqueeze(0) + noise * std.unsqueeze(0)

            # Clamp to physical Ackermann limits
            candidates = torch.max(torch.min(candidates, self.action_high), self.action_low)

            # Roll out dynamics in latent space: [N, H, ...]
            unrolled_z = rollout_fn(batch_z0, candidates)

            # Compute step scores and discounted return: [N, H] -> [N]
            step_scores = score_fn(unrolled_z)  # [N, H]
            returns = (step_scores * self.discounts.unsqueeze(0)).sum(dim=-1)  # [N]

            # 1. Safety penalty if front obstacle is close and candidate moves forward
            if lidar_front_min < 0.50:
                forward_speed = torch.clamp(candidates[:, 0, 0], min=0.0)
                safety_penalty = 5.0 * (0.50 - lidar_front_min) * forward_speed
                returns = returns - safety_penalty

            # 2. Kalman tracker guidance during occlusion
            if tracker_status is not None and tracker_status.get("is_occluded", False):
                bearing = float(tracker_status.get("estimated_bearing", 0.0))
                dist = float(tracker_status.get("estimated_dist", 0.80))
                target_steer = float(np.clip(1.2 * bearing, -0.553953, 0.553953))
                # Encourage steering toward predicted target and matching distance
                steer_err = torch.abs(candidates[:, 0, 1] - target_steer)
                dist_err = abs(dist - 0.80)
                v_target = float(np.clip(0.30 + 0.50 * dist_err, 0.0, 0.80))
                speed_err = torch.abs(candidates[:, 0, 0] - v_target)
                returns = returns - 2.0 * steer_err - 1.5 * speed_err

            # Select top K elite trajectories
            _, elite_indices = torch.topk(returns, k=self.num_elites, largest=True)
            elites = candidates[elite_indices]  # [K, H, A]

            # Update distribution moments
            elite_mean = elites.mean(dim=0)
            elite_std = elites.std(dim=0).clamp(min=1e-3)

            mean = self.momentum * mean + (1.0 - self.momentum) * elite_mean
            std = self.momentum * std + (1.0 - self.momentum) * elite_std
            best_actions = elites[0]

        # Store for warm-starting next control step
        self.prev_mean = mean.detach().clone()
        return best_actions[0].detach()
