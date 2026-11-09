"""
Unit tests for SpatialVAE architecture, loss functions, and quality metrics.

Enforces:
- Section 6.3 of agents.md: Test output isolation in tests/outputs/
"""

from pathlib import Path

import torch

from wm_control.losses.vae_loss import (
    VAELoss,
    compute_psnr,
    compute_ssim,
)
from wm_control.models.vae import (
    SpatialDecoder,
    SpatialEncoder,
    SpatialVAE,
)


def test_spatial_encoder_shapes():
    """Verify encoder outputs [B, 4, 16, 16] mean and logvar from [B, 3, 64, 64]."""
    encoder = SpatialEncoder(in_channels=3, latent_channels=4, base_channels=16)
    x = torch.randn(2, 3, 64, 64)
    mean, logvar = encoder(x)

    assert mean.shape == (2, 4, 16, 16)
    assert logvar.shape == (2, 4, 16, 16)


def test_spatial_decoder_shapes_and_bounds():
    """Verify decoder reconstructs [B, 3, 64, 64] with values bounded in [0, 1]."""
    decoder = SpatialDecoder(latent_channels=4, out_channels=3, base_channels=16)
    z = torch.randn(2, 4, 16, 16)
    recon = decoder(z)

    assert recon.shape == (2, 3, 64, 64)
    assert 0.0 <= recon.min().item() and recon.max().item() <= 1.0


def test_vae_reparameterization_gradient_flow():
    """Verify gradients propagate through reparameterization to both mean and logvar."""
    vae = SpatialVAE(in_channels=3, latent_channels=4, base_channels=16)
    vae.train()

    x = torch.randn(2, 3, 64, 64, requires_grad=True)
    out = vae(x)

    loss = out["recon"].sum()
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    for name, p in vae.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Parameter {name} has no gradient"


def test_vae_loss_and_metrics():
    """Verify VAELoss computation, PSNR, and SSIM metrics."""
    criterion = VAELoss(beta_kl=1e-4)

    target = torch.rand(4, 3, 64, 64)
    recon = (target + torch.randn_like(target) * 0.02).clamp(0.0, 1.0)
    mean = torch.zeros(4, 4, 16, 16)
    logvar = torch.zeros(4, 4, 16, 16)

    loss_dict = criterion(recon, target, mean, logvar)

    assert "loss" in loss_dict
    assert "rec_loss" in loss_dict
    assert "kl_loss" in loss_dict
    # Standard normal mean=0, logvar=0 has theoretical KL = 0
    assert abs(loss_dict["kl_loss"].item()) < 1e-4

    psnr = compute_psnr(recon, target)
    ssim = compute_ssim(recon, target)
    assert psnr > 25.0
    assert ssim > 0.90


def test_vae_checkpoint_save_and_load(isolated_test_dir: Path):
    """Verify checkpoint serialization into isolated test directory."""
    vae = SpatialVAE(in_channels=3, latent_channels=4, base_channels=16)
    ckpt_path = isolated_test_dir / "test_vae.pt"

    torch.save(
        {
            "model_state": vae.state_dict(),
            "latent_channels": 4,
            "epoch": 1,
            "val_psnr": 28.5,
        },
        ckpt_path,
    )
    assert ckpt_path.exists()

    ckpt = torch.load(ckpt_path, map_location="cpu")
    vae_new = SpatialVAE(in_channels=3, latent_channels=4, base_channels=16)
    vae_new.load_state_dict(ckpt["model_state"])
    assert ckpt["val_psnr"] == 28.5
