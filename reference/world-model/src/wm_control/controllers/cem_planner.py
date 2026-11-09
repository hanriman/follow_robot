"""
Cross-Entropy Method (CEM) Latent Action Planner.

Implements accelerated population-based trajectory optimization over learned
latent state dynamics for continuous visual control.
"""

from __future__ import annotations

from collections.abc import Callable

import torch


class CEMPlanner:
    """
    Cross-Entropy Method (CEM) Action Planner in Latent Space.

    Args:
        horizon: Planning horizon H (number of future steps).
        pop_size: Number of candidate action sequences sampled per iteration.
        num_elites: Number of highest-scoring candidate sequences used to update distribution.
        num_iters: Number of CEM refinement iterations per control step.
        gamma: Temporal discount factor for future rewards.
        momentum: Smoothing factor for distribution parameter updates.
        action_dim: Continuous action dimensionality (default: 3 for CarRacing: [steering, gas, brake]).
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
        action_dim: int = 3,
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

        # Action bounds for CarRacing-v3: steering in [-1, 1], gas in [0, 1], brake in [0, 1]
        self.action_low = torch.tensor([-1.0, 0.0, 0.0], device=self.device)
        self.action_high = torch.tensor([1.0, 1.0, 1.0], device=self.device)

        # Discount weights: [1, gamma, gamma^2, ..., gamma^{H-1}]
        self.discounts = torch.pow(self.gamma, torch.arange(horizon, device=self.device, dtype=torch.float32))

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
    ) -> torch.Tensor:
        """
        Execute CEM optimization to find optimal action sequence and return first action.

        Args:
            initial_z: Current latent state z_0 [1, ...].
            rollout_fn: Function mapping (z_0 [N, ...], actions [N, H, A]) -> unrolled_z [N, H, ...].
            score_fn: Function mapping unrolled_z [N, H, ...] -> scalar scores per step [N, H].

        Returns:
            Optimal continuous action vector for current timestep [A].
        """
        device = self.device

        # Initialize distribution parameters
        if self.prev_mean is not None:
            # Shift previous plan by 1 step, append default action
            mean = torch.cat([self.prev_mean[1:], self.prev_mean[-1:]], dim=0).to(device)
        else:
            # Default initialization: slight forward throttle [steering=0, gas=0.3, brake=0]
            mean = torch.tensor([[0.0, 0.3, 0.0]], device=device).repeat(self.horizon, 1)

        std = torch.full((self.horizon, self.action_dim), 0.5, device=device)

        # Replicate initial state across population: [N, ...]
        batch_z0 = initial_z.repeat(self.pop_size, *([1] * (initial_z.dim() - 1))).to(device)

        best_actions = mean

        for _ in range(self.num_iters):
            # Sample candidate actions: [N, H, A]
            noise = torch.randn(self.pop_size, self.horizon, self.action_dim, device=device)
            candidates = mean.unsqueeze(0) + noise * std.unsqueeze(0)

            # Clamp to physical action boundaries
            candidates = torch.max(torch.min(candidates, self.action_high), self.action_low)

            # Roll out dynamics in latent space: [N, H, ...]
            unrolled_z = rollout_fn(batch_z0, candidates)

            # Compute step scores and discounted return: [N, H] -> [N]
            step_scores = score_fn(unrolled_z)  # [N, H]
            returns = (step_scores * self.discounts.unsqueeze(0)).sum(dim=-1)  # [N]

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
