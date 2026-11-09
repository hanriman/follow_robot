"""
Unit tests for SIGReg regularizer, TeacherFreeLeWorldModel, and representation health metrics.
"""

import torch

from wm_control.losses.sigreg import (
    ActionConditionedTransitionLoss,
    EppsPulleyGaussianityTest,
    SigRegLoss,
)
from wm_control.metrics.representation import (
    check_representation_health,
    compute_effective_rank,
    compute_gaussianity_diagnostics,
)
from wm_control.models.sigreg_wm import (
    TeacherFreeLeWorldModel,
)


def test_epps_pulley_exact_standard_normal():
    """Verify that ECF test statistic is small on standard Gaussian samples."""
    torch.manual_seed(42)
    ep_test = EppsPulleyGaussianityTest(t_max=3.0, n_knots=17)

    # 1000 samples along 16 random 1D projections
    normal_proj = torch.randn(1000, 16)
    stat = ep_test(normal_proj)

    assert stat.dim() == 0
    # For N=1000 standard normal, statistic should be moderate (< 2.0)
    assert stat.item() < 2.0


def test_sigreg_anti_collapse_penalty():
    """Verify that collapsed representations are heavily penalized compared to isotropic normal."""
    torch.manual_seed(42)
    sigreg = SigRegLoss(sigreg_weight=1.0, num_projections=64)

    # Isotropic Gaussian embeddings [N=500, D=64]
    z_normal = torch.randn(500, 64)
    loss_normal = sigreg(z_normal)

    # Collapsed embeddings near zero
    z_collapsed = torch.randn(500, 64) * 0.01
    loss_collapsed = sigreg(z_collapsed)

    # Collapsed loss should be drastically higher (> 20x)
    assert loss_collapsed.item() > 20.0 * loss_normal.item()


def test_sigreg_gradient_flow():
    """Verify backward gradient propagation through SIGReg to latents."""
    sigreg = SigRegLoss(sigreg_weight=1.0, num_projections=32)
    z = torch.randn(100, 32, requires_grad=True)

    loss = sigreg(z)
    loss.backward()

    assert z.grad is not None
    assert not torch.isnan(z.grad).any()
    assert not torch.isinf(z.grad).any()
    assert (z.grad.abs() > 0.0).any()


def test_sigreg_reproducibility():
    """Verify bitwise determinism when seeded torch.Generator is supplied."""
    sigreg = SigRegLoss(num_projections=64)
    z = torch.randn(64, 32)

    gen1 = torch.Generator().manual_seed(1337)
    gen2 = torch.Generator().manual_seed(1337)

    loss1 = sigreg(z, generator=gen1)
    loss2 = sigreg(z, generator=gen2)

    assert torch.equal(loss1, loss2)


def test_action_conditioned_transition_loss():
    """Test combined transition prediction loss and SIGReg penalty."""
    criterion = ActionConditionedTransitionLoss(
        loss_type="smooth_l1",
        beta=1.0,
        sigreg_weight=0.5,
        num_projections=32,
    )

    B, D = 16, 64
    pred_z = torch.randn(B, D, requires_grad=True)
    target_z = torch.randn(B, D)

    out = criterion(pred_z, target_z)

    assert "loss" in out
    assert "transition_loss" in out
    assert "sigreg_loss" in out
    assert out["loss"].requires_grad

    out["loss"].backward()
    assert pred_z.grad is not None


def test_teacher_free_world_model_joint_gradients():
    """Verify that both encoder and predictor receive joint active gradients without EMA."""
    model = TeacherFreeLeWorldModel(
        latent_dim=32,
        action_dim=3,
        predictor_hidden_dim=64,
        predictor_layers=2,
        in_channels=3,
    )
    criterion = ActionConditionedTransitionLoss(sigreg_weight=1.0, num_projections=32)

    B = 8
    obs_t = torch.randn(B, 3, 64, 64)
    action_t = torch.randn(B, 3)
    obs_next = torch.randn(B, 3, 64, 64)

    out = model(obs_t, action_t, obs_next)
    loss_dict = criterion(out["pred_next_z"], out["target_next_z"], out["all_tokens"])

    loss_dict["loss"].backward()

    # Check encoder received gradients from both current and next observations
    for name, p in model.encoder.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Encoder parameter {name} has no gradient"
            assert not torch.isnan(p.grad).any()

    # Check predictor received gradients
    for name, p in model.predictor.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Predictor parameter {name} has no gradient"
            assert not torch.isnan(p.grad).any()


def test_autoregressive_rollout():
    """Test multi-step autoregressive rollout consistency."""
    model = TeacherFreeLeWorldModel(latent_dim=32, action_dim=3)
    B, H = 4, 10
    initial_obs = torch.randn(B, 3, 64, 64)
    action_sequence = torch.randn(B, H, 3)

    trajectory = model.rollout(initial_obs, action_sequence)
    assert trajectory.shape == (B, H, 32)


def test_effective_rank_metrics():
    """Test effective rank on isotropic normal vs collapsed representation."""
    D = 64
    N = 500

    # Isotropic Gaussian should have high rank ratio (~0.8-1.0)
    z_isotropic = torch.randn(N, D)
    metrics_iso = compute_effective_rank(z_isotropic)
    assert metrics_iso["rank_ratio"] > 0.7
    assert metrics_iso["effective_rank"] > 0.7 * D

    # Collapsed along 2 dimensions
    z_collapsed = torch.randn(N, 2) @ torch.randn(2, D)
    metrics_col = compute_effective_rank(z_collapsed)
    assert metrics_col["effective_rank"] < 5.0
    assert metrics_col["rank_ratio"] < 0.1


def test_gaussianity_diagnostics():
    """Test statistical moments of standard normal."""
    z = torch.randn(1000, 64)
    diag = compute_gaussianity_diagnostics(z)
    assert diag["mean_norm"] < 0.15
    assert diag["var_deviation"] < 0.2
    assert diag["off_diag_corr"] < 0.1


def test_representation_health_check():
    """Test health check threshold detection."""
    z_healthy = torch.randn(500, 32)
    health = check_representation_health(z_healthy)
    assert health["is_healthy"]

    z_unhealthy = torch.randn(500, 32) * 0.05
    health_bad = check_representation_health(z_unhealthy)
    assert not health_bad["is_healthy"]
    assert len(health_bad["issues"]) > 0
