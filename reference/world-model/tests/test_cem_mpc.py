"""
Unit tests for Latent Reward Heads and Cross-Entropy Method (CEM) Action Planner.

Enforces:
- Section 6.3 of agents.md: Test output isolation in tests/outputs/
"""

import torch

from wm_control.controllers.cem_planner import CEMPlanner
from wm_control.models.reward_head import LatentRewardHead


def test_latent_reward_head_shapes():
    """Verify reward head processes both 1D vectors and 4D spatial tensors."""
    # 1D vector (LeWM)
    head_lewm = LatentRewardHead(in_dim=128, is_spatial=False)
    z_lewm = torch.randn(4, 128)
    out_lewm = head_lewm(z_lewm)
    assert out_lewm.shape == (4, 3)

    # 1D vector trajectory: [B=2, H=5, D=128]
    z_lewm_seq = torch.randn(2, 5, 128)
    out_lewm_seq = head_lewm(z_lewm_seq)
    assert out_lewm_seq.shape == (2, 5, 3)

    score_lewm = head_lewm.compute_control_score(z_lewm_seq)
    assert score_lewm.shape == (2, 5)

    # Spatial tensor (Spatial VAE): [B=4, C=4, H=16, W=16]
    head_vae = LatentRewardHead(is_spatial=True, spatial_channels=4)
    z_vae = torch.randn(4, 4, 16, 16)
    out_vae = head_vae(z_vae)
    assert out_vae.shape == (4, 3)

    # Spatial trajectory: [B=2, H=5, C=4, H=16, W=16]
    z_vae_seq = torch.randn(2, 5, 4, 16, 16)
    out_vae_seq = head_vae(z_vae_seq)
    assert out_vae_seq.shape == (2, 5, 3)


def test_cem_planner_bounds_and_optimization():
    """Verify CEM planner generates bounded actions and optimizes toward higher reward targets."""
    device = torch.device("cpu")
    planner = CEMPlanner(
        horizon=3,
        pop_size=20,
        num_elites=4,
        num_iters=2,
        device=device,
    )

    z0 = torch.zeros(1, 4)

    # Mock linear dynamics: z_{t+1} = z_t + a_t
    def mock_rollout(z_init: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        # z_init: [N, 4], actions: [N, H, 3]
        _n, h, _ = actions.shape
        traj = []
        curr = z_init
        for t in range(h):
            curr = curr + F_pad(actions[:, t, :])
            traj.append(curr)
        return torch.stack(traj, dim=1)  # [N, H, 4]

    def F_pad(a: torch.Tensor) -> torch.Tensor:
        return torch.cat([a, torch.zeros(a.shape[0], 1)], dim=-1)

    # Target: maximize steering (action[0] -> 1.0)
    def mock_score(unrolled_z: torch.Tensor) -> torch.Tensor:
        # Score is unrolled_z[..., 0] (tracks steering)
        return unrolled_z[..., 0]

    action = planner.plan(z0, mock_rollout, mock_score)

    assert action.shape == (3,)
    # Verify action boundaries
    assert -1.0 <= action[0].item() <= 1.0
    assert 0.0 <= action[1].item() <= 1.0
    assert 0.0 <= action[2].item() <= 1.0

    # Because mock_score rewards positive steering, optimized action[0] should be positive
    assert action[0].item() > 0.0
