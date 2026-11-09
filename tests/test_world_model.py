"""
Comprehensive Unit & Integration Tests for AckBot Phase 5 World Models.

Tests:
1. Epps-Pulley Gaussianity Test & SIGReg anti-collapse loss
2. Teacher-Free LeWM joint autograd (no EMA, single shared encoder)
3. Multi-Modal Sensor VAE reconstruction and bounding
4. Action-Conditioned UNet1D & DDIM sampler
5. CEM Planner Ackermann physical boundary compliance
6. Roy's Effective Rank & Ridge Regression linear probe
7. Sharded Dataset loader integration
"""

import math
from pathlib import Path
import tempfile
import numpy as np
import pytest
import torch

from rl_ws.policies.world_model.sigreg import (
    EppsPulleyGaussianityTest,
    SigRegLoss,
    ActionConditionedTransitionLoss,
)
from rl_ws.policies.world_model.representation import (
    compute_effective_rank,
    compute_gaussianity_diagnostics,
    RidgeRegressionProbe,
)
from rl_ws.policies.world_model.sigreg_wm import (
    AckBotSensorEncoder,
    ActionConditionedPredictor,
    TeacherFreeLeWorldModel,
)
from rl_ws.policies.world_model.vae import MultiModalSensorVAE
from rl_ws.policies.vla.diffusion_wm import (
    ActionConditionedUNet1D,
    GaussianDiffusion,
    ActionConditionedDiffusionWorldModel,
)
from rl_ws.policies.world_model.reward_head import LatentRewardHead
from rl_ws.policies.world_model.cem_planner import CEMPlanner
from rl_ws.data.dataset import AckBotTransitionDataset


def get_test_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# 1. SIGReg & Epps-Pulley Analytical Gaussianity Tests
# ---------------------------------------------------------------------------

def test_epps_pulley_exact_standard_normal():
    """EP test on large standard normal sample should be close to 0."""
    ep_test = EppsPulleyGaussianityTest(t_max=3.0, n_knots=17)
    torch.manual_seed(42)
    # 2048 samples from true N(0, 1)
    samples = torch.randn(2048)
    stat = ep_test(samples)

    assert isinstance(stat, torch.Tensor)
    assert stat.item() >= 0.0
    # For standard normal, the statistic scaled by N is small
    assert stat.item() < 1.0


def test_sigreg_anti_collapse_penalty():
    """A collapsed representation must yield a significantly higher SIGReg loss than standard normal."""
    sigreg = SigRegLoss(sigreg_weight=1.0, num_projections=512)
    torch.manual_seed(42)

    # Healthy standard normal representations: [B=128, D=128]
    healthy_z = torch.randn(128, 128)
    loss_healthy = sigreg(healthy_z)

    # Collapsed representation (all zeros or constant)
    collapsed_z = torch.zeros(128, 128)
    loss_collapsed = sigreg(collapsed_z)

    # Rank-collapsed representation (all tokens identical along 127 dims)
    rank_collapsed_z = torch.randn(128, 1) * torch.ones(1, 128)
    loss_rank_collapsed = sigreg(rank_collapsed_z)

    assert loss_collapsed.item() > loss_healthy.item() * 5.0
    assert loss_rank_collapsed.item() > loss_healthy.item() * 2.0


def test_sigreg_gradient_flow():
    """Verify that SIGReg loss propagates non-zero gradients back to representations."""
    sigreg = SigRegLoss(sigreg_weight=1.0, num_projections=256)
    z = torch.randn(32, 64, requires_grad=True)
    loss = sigreg(z)
    loss.backward()

    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
    assert torch.norm(z.grad).item() > 0.0


# ---------------------------------------------------------------------------
# 2. Teacher-Free Le World Model Joint Autograd
# ---------------------------------------------------------------------------

def test_teacher_free_lewm_joint_gradients():
    """
    Verify Teacher-Free LeWM training principles:
    - Single shared encoder for both obs_t and obs_{t+1}
    - Joint backpropagation without stop-gradient or EMA
    """
    device = get_test_device()
    model = TeacherFreeLeWorldModel(
        latent_dim=128,
        action_dim=2,
        cam_dim=64,
        lidar_dim=360,
        proprio_dim=2,
    ).to(device)

    loss_fn = ActionConditionedTransitionLoss(
        loss_type="smooth_l1",
        sigreg_weight=1.0,
    ).to(device)

    b = 8
    obs_t = torch.randn(b, 426, device=device)
    act_t = torch.randn(b, 2, device=device)
    obs_next = torch.randn(b, 426, device=device)

    fwd = model(obs_t, act_t, obs_next)
    losses = loss_fn(
        pred_next_z=fwd["pred_next_z"],
        target_next_z=fwd["target_next_z"],
        regularize_tokens=fwd["all_tokens"],
    )

    loss = losses["loss"]
    loss.backward()

    # Check encoder received non-zero gradients
    encoder_has_grads = any(
        p.grad is not None and torch.norm(p.grad).item() > 0.0
        for p in model.encoder.parameters()
    )
    assert encoder_has_grads, "Encoder parameters must receive gradients jointly!"

    # Check dynamics predictor received non-zero gradients
    predictor_has_grads = any(
        p.grad is not None and torch.norm(p.grad).item() > 0.0
        for p in model.predictor.parameters()
    )
    assert predictor_has_grads, "Predictor parameters must receive gradients jointly!"


def test_lewm_multi_step_rollout_shape():
    """Verify open-loop latent rollout over horizon H."""
    device = get_test_device()
    model = TeacherFreeLeWorldModel(latent_dim=128, action_dim=2).to(device)

    b, h = 4, 7
    obs_0 = torch.randn(b, 426, device=device)
    actions = torch.randn(b, h, 2, device=device)

    trajectory = model.rollout(obs_0, actions)
    assert trajectory.shape == (b, h, 128)
    assert torch.isfinite(trajectory).all()


# ---------------------------------------------------------------------------
# 3. Multi-Modal Sensor VAE Reconstruction
# ---------------------------------------------------------------------------

def test_sensor_vae_forward_and_reconstruction_bounds():
    """
    Verify MultiModalSensorVAE:
    - Encodes [B, 426] -> [B, 4, 32]
    - Decodes back to [B, 426] with correct ranges:
      first 424 features in [0, 1] (depth, lidar)
      last 2 features in [-1, 1] (state v, steer)
    """
    device = get_test_device()
    vae = MultiModalSensorVAE(
        in_dim=426,
        latent_channels=4,
        latent_length=32,
        hidden_dim=128,
        kl_weight=1e-4,
    ).to(device)

    b = 8
    # Valid synthetic observation
    sensors = torch.rand(b, 424, device=device)  # [0, 1]
    states = torch.empty(b, 2, device=device).uniform_(-1.0, 1.0)  # [-1, 1]
    obs = torch.cat([sensors, states], dim=-1)

    res = vae(obs)
    recon = res["recon_x"]

    assert res["z"].shape == (b, 4, 32)
    assert recon.shape == (b, 426)
    assert res["loss"].item() > 0.0
    assert torch.isfinite(res["loss"]).all()

    # Verify sensor range bounds
    recon_sensors = recon[:, :424]
    recon_state = recon[:, 424:]
    assert (recon_sensors >= 0.0).all() and (recon_sensors <= 1.0).all()
    assert (recon_state >= -1.0).all() and (recon_state <= 1.0).all()


# ---------------------------------------------------------------------------
# 4. Action-Conditioned Latent Diffusion & DDIM Sampling
# ---------------------------------------------------------------------------

def test_action_conditioned_diffusion_ddim_sampling():
    """Verify ActionConditionedDiffusionWorldModel AdaGN UNet and fast DDIM sampler."""
    device = get_test_device()
    b = 4
    channels, length = 4, 32
    action_dim = 2

    diff_wm = ActionConditionedDiffusionWorldModel(
        latent_channels=channels,
        latent_length=length,
        action_dim=action_dim,
        base_channels=32,
        num_timesteps=100,
    ).to(device)

    z_t = torch.randn(b, channels, length, device=device)
    z_next = torch.randn(b, channels, length, device=device)
    action = torch.randn(b, action_dim, device=device)

    # 1. Forward training loss
    loss_dict = diff_wm.compute_loss(target_latent=z_next, context_latent=z_t, action=action)
    loss = loss_dict["loss"]
    assert torch.isfinite(loss).all()

    loss.backward()
    assert any(p.grad is not None for p in diff_wm.unet.parameters())

    # 2. Fast DDIM single-step prediction
    pred_z_next = diff_wm.predict_next(z_t, action, ddim_steps=5)
    assert pred_z_next.shape == (b, channels, length)
    assert torch.isfinite(pred_z_next).all()

    # 3. Autoregressive multi-step rollout
    h = 3
    act_seq = torch.randn(b, h, action_dim, device=device)
    traj = diff_wm.rollout_latents(z_t, act_seq, ddim_steps=4)
    assert traj.shape == (b, h, channels, length)


# ---------------------------------------------------------------------------
# 5. CEM Latent Action Planner Bounds
# ---------------------------------------------------------------------------

def test_cem_planner_ackermann_bounds():
    """
    Verify CEMPlanner respects canonical AckBot limits:
    v in [-0.40, 0.80] m/s, delta in [-0.553953, +0.553953] rad.
    """
    device = get_test_device()
    planner = CEMPlanner(
        horizon=5,
        pop_size=20,
        num_elites=3,
        num_iters=2,
        v_min=-0.40,
        v_max=0.80,
        steer_max=0.553953,
        device=device,
    )

    z_0 = torch.randn(1, 128, device=device)

    def dummy_rollout(z_start, actions):
        # returns [N, H, 128]
        return z_start.unsqueeze(1).repeat(1, actions.shape[1], 1)

    def dummy_score(unrolled_z):
        # returns [N, H]
        return -unrolled_z.norm(dim=-1)

    for _ in range(5):
        action = planner.plan(z_0, dummy_rollout, dummy_score)
        assert action.shape == (2,)
        v, steer = action[0].item(), action[1].item()
        assert -0.4001 <= v <= 0.8001, f"Velocity {v} out of bounds"
        assert -0.5540 <= steer <= 0.5540, f"Steer {steer} out of bounds"


# ---------------------------------------------------------------------------
# 6. Representation Diagnostics & Ridge Probing
# ---------------------------------------------------------------------------

def test_effective_rank_calculation():
    """Verify Roy's Participation Ratio for full-rank vs collapsed matrices."""
    # Full rank isotropic Gaussian
    torch.manual_seed(42)
    z_full = torch.randn(500, 64)
    res_full = compute_effective_rank(z_full)
    # PR should be > 75% of D=64
    assert res_full["rank_ratio"] > 0.75

    # 1D collapsed matrix
    z_1d = torch.randn(500, 1) @ torch.randn(1, 64)
    res_1d = compute_effective_rank(z_1d)
    assert res_1d["effective_rank"] < 2.5
    assert res_1d["rank_ratio"] < 0.05


def test_ridge_regression_probe():
    """Verify RidgeRegressionProbe fits linear ground-truth mappings accurately."""
    torch.manual_seed(42)
    n, d, target_d = 200, 32, 4
    X = torch.randn(n, d)
    true_w = torch.randn(d, target_d)
    true_b = torch.tensor([1.0, -2.0, 0.5, 3.0])
    Y = X @ true_w + true_b

    probe = RidgeRegressionProbe(alpha=0.01).fit(X, Y)
    metrics = probe.score_r2(X, Y)

    assert metrics["mean_r2"] > 0.98
    for r2 in metrics["r2_per_dim"]:
        assert r2 > 0.95


# ---------------------------------------------------------------------------
# 7. Sharded Dataset Integration
# ---------------------------------------------------------------------------

def test_sharded_dataset_loading():
    """Verify AckBotTransitionDataset correctly reads compressed NPZ files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_shard = Path(tmpdir) / "shard_0000.npz"
        n = 16
        np.savez_compressed(
            tmp_shard,
            obs=np.random.randn(n, 426).astype(np.float32),
            next_obs=np.random.randn(n, 426).astype(np.float32),
            action=np.random.randn(n, 2).astype(np.float32),
            reward=np.random.randn(n).astype(np.float32),
            terminal=np.zeros(n, dtype=bool),
            telemetry=np.random.randn(n, 8).astype(np.float32),
            next_telemetry=np.random.randn(n, 8).astype(np.float32),
        )

        dataset = AckBotTransitionDataset([tmp_shard])
        assert len(dataset) == n

        sample = dataset[0]
        assert sample["obs"].shape == (426,)
        assert sample["action"].shape == (2,)
        assert sample["telemetry"].shape == (8,)
        assert sample["terminal"].dtype == torch.bool
