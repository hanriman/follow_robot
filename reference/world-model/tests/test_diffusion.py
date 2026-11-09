"""
Unit tests for Action-Conditioned Latent Diffusion World Model (AC-LDM).

Enforces:
- Section 6.3 of agents.md: Test output isolation in tests/outputs/
- Action controllability divergence test
"""

from pathlib import Path

import torch
import torch.nn.functional as F

from wm_control.models.diffusion_wm import (
    ActionConditionedDiffusionWorldModel,
    ActionConditionedUNet,
    AdaGNResBlock,
    GaussianDiffusion,
    SinusoidalTimeEmbedding,
)


def test_sinusoidal_time_embedding():
    """Verify time embeddings have correct dimensionality and distinguish different steps."""
    emb_module = SinusoidalTimeEmbedding(dim=64)
    t = torch.tensor([0, 10, 50, 99], dtype=torch.long)
    embs = emb_module(t)

    assert embs.shape == (4, 64)
    # Timestep 0 and 99 should have distinct embeddings
    assert not torch.allclose(embs[0], embs[3])


def test_adagn_resblock():
    """Verify AdaGN residual block preserves spatial resolution and modulates by condition."""
    block = AdaGNResBlock(in_ch=16, out_ch=32, cond_dim=64)
    x = torch.randn(2, 16, 16, 16)
    cond1 = torch.randn(2, 64)
    cond2 = torch.randn(2, 64)

    out1 = block(x, cond1)
    out2 = block(x, cond2)

    assert out1.shape == (2, 32, 16, 16)
    assert not torch.allclose(out1, out2)


def test_action_conditioned_unet_shapes_and_gradients():
    """Verify U-Net produces matching latent noise shapes and propagates gradients."""
    unet = ActionConditionedUNet(
        latent_channels=4,
        action_dim=3,
        base_channels=16,
        cond_dim=32,
    )
    unet.train()

    noisy = torch.randn(2, 4, 16, 16, requires_grad=True)
    t = torch.tensor([5, 20], dtype=torch.long)
    context = torch.randn(2, 4, 16, 16, requires_grad=True)
    action = torch.tensor([[0.5, 0.2, 0.0], [-0.5, 0.0, 0.8]], requires_grad=True)

    pred_noise = unet(noisy, t, context, action)

    assert pred_noise.shape == (2, 4, 16, 16)

    loss = pred_noise.sum()
    loss.backward()

    assert noisy.grad is not None
    assert context.grad is not None
    assert action.grad is not None
    assert not torch.isnan(noisy.grad).any()


def test_gaussian_diffusion_schedule():
    """Verify forward noising schedule properties and q_sample shapes."""
    diff = GaussianDiffusion(num_timesteps=50, beta_start=1e-4, beta_end=0.02)
    assert diff.alphas_cumprod[0] > diff.alphas_cumprod[-1]
    assert diff.alphas_cumprod[-1] > 0.0

    x0 = torch.randn(3, 4, 16, 16)
    t = torch.tensor([0, 25, 49], dtype=torch.long)
    noisy, noise = diff.q_sample(x0, t)

    assert noisy.shape == (3, 4, 16, 16)
    assert noise.shape == (3, 4, 16, 16)


def test_action_controllability_divergence():
    """
    Scientific test: Verify that differing action inputs (e.g. hard left vs hard right)
    cause diverging latent predictions from the same starting context latent.
    """
    torch.manual_seed(42)
    model = ActionConditionedDiffusionWorldModel(
        latent_channels=4,
        action_dim=3,
        num_timesteps=20,
        base_channels=16,
    )
    model.eval()

    # Same initial latent
    z0 = torch.randn(1, 4, 16, 16)

    # Opposing actions: hard left [-1, 0, 0] vs hard right [+1, 0, 0]
    action_left = torch.tensor([[-1.0, 0.5, 0.0]])
    action_right = torch.tensor([[1.0, 0.5, 0.0]])

    # Use fixed noise seed for DDIM determinism
    torch.manual_seed(123)
    pred_left = model.predict_next(z0, action_left, ddim_steps=5)

    torch.manual_seed(123)
    pred_right = model.predict_next(z0, action_right, ddim_steps=5)

    divergence = F.l1_loss(pred_left, pred_right).item()
    # Different action conditions must modulate the U-Net features differently
    assert divergence > 0.0, f"Expected non-zero action divergence, got {divergence}"


def test_multistep_rollout_latents():
    """Verify autoregressive multi-step rollout generates expected [B, H, C, H_z, W_z] shape."""
    model = ActionConditionedDiffusionWorldModel(
        latent_channels=4,
        action_dim=3,
        num_timesteps=10,
        base_channels=16,
    )
    model.eval()

    b, h = 2, 4
    z0 = torch.randn(b, 4, 16, 16)
    actions = torch.randn(b, h, 3)

    traj = model.rollout_latents(z0, actions, ddim_steps=3)
    assert traj.shape == (b, h, 4, 16, 16)


def test_diffusion_checkpoint_save_and_load(isolated_test_dir: Path):
    """Verify checkpoint serialization into isolated test directory."""
    model = ActionConditionedDiffusionWorldModel(
        latent_channels=4,
        action_dim=3,
        num_timesteps=10,
        base_channels=16,
    )
    ckpt_path = isolated_test_dir / "test_diffusion.pt"

    torch.save(
        {
            "model_state": model.state_dict(),
            "latent_channels": 4,
            "action_dim": 3,
            "val_loss": 0.05,
        },
        ckpt_path,
    )
    assert ckpt_path.exists()

    loaded = torch.load(ckpt_path)
    assert loaded["val_loss"] == 0.05
    assert "model_state" in loaded
