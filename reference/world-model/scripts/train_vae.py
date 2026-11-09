#!/usr/bin/env python3
"""
Training Runner for Spatial Variational Autoencoder (Spatial VAE).

Trains on offline CarRacing-v3 transitions (40,000 train, 5,000 val) and freezes
the continuous spatial encoder for downstream Latent Diffusion modeling.

Outputs (Solid 1-Directory Rule):
- Checkpoints: outputs/experiments/vae/checkpoints/best_vae.pt, latest_vae.pt
- Figures:     outputs/experiments/vae/figures/recon_comparison.png
- Metrics:     outputs/experiments/vae/metrics/eval_metrics.json
- Logs:        outputs/experiments/vae/logs/train.log
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from wm_control.config import get_experiment_dir
from wm_control.data.dataset import get_dataset_splits
from wm_control.losses.vae_loss import (
    VAELoss,
    compute_psnr,
    compute_ssim,
)
from wm_control.models.vae import SpatialVAE
from wm_control.utils.device import get_device
from wm_control.utils.seed import seed_everything


def train_vae(
    epochs: int = 15,
    batch_size: int = 256,
    lr: float = 1e-3,
    beta_kl: float = 1e-4,
    latent_channels: int = 4,
    base_channels: int = 32,
    device_str: str | None = None,
    seed: int = 42,
    save_fig_every: int = 5,
) -> Path:
    """
    Train and checkpoint the Spatial VAE.
    """
    seed_everything(seed)
    device = get_device(device_str)

    # Resolve deterministic canonical output paths
    exp_dir = get_experiment_dir("vae", create_if_missing=True)
    ckpt_dir = exp_dir / "checkpoints"
    fig_dir = exp_dir / "figures"
    metrics_dir = exp_dir / "metrics"
    logs_dir = exp_dir / "logs"

    # Setup logger
    log_file = logs_dir / "train.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger("TrainVAE")
    logger.info(f"Starting Spatial VAE Training on device: {device}")
    logger.info(f"Experiment Directory: {exp_dir}")

    # Load offline dataset splits
    train_ds, val_ds, _ = get_dataset_splits(augment_train=True)
    logger.info(f"Loaded Dataset: {len(train_ds):,} train samples, {len(val_ds):,} val samples")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    # Initialize model, loss, and optimizer
    model = SpatialVAE(
        in_channels=3,
        latent_channels=latent_channels,
        base_channels=base_channels,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Spatial VAE Model Initialized: {total_params:,} trainable parameters")

    criterion = VAELoss(beta_kl=beta_kl, mse_weight=0.5, edge_weight=0.2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_psnr = -1.0
    history: dict[str, list] = {
        "train_loss": [],
        "val_loss": [],
        "val_psnr": [],
        "val_ssim": [],
        "val_l1": [],
    }

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_rec_sum = 0.0
        train_kl_sum = 0.0
        num_train_batches = 0

        # Beta warm-up schedule over first 3 epochs
        beta_scale = min(1.0, epoch / 3.0)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]", leave=False)
        for batch in pbar:
            # Train on both current observation and next observation for rich sample diversity
            x = batch["obs"].to(device)
            optimizer.zero_grad(set_to_none=True)

            out = model(x)
            loss_dict = criterion(
                recon=out["recon"],
                target=x,
                mean=out["mean"],
                logvar=out["logvar"],
                beta_scale=beta_scale,
            )

            loss_dict["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss_dict["loss"].item()
            train_rec_sum += loss_dict["rec_loss"].item()
            train_kl_sum += loss_dict["kl_loss"].item()
            num_train_batches += 1

            pbar.set_postfix(
                loss=f"{loss_dict['loss'].item():.4f}",
                rec=f"{loss_dict['rec_loss'].item():.4f}",
                kl=f"{loss_dict['kl_loss'].item():.4f}",
            )

        scheduler.step()
        avg_train_loss = train_loss_sum / max(1, num_train_batches)

        # Validation Loop
        model.eval()
        val_loss_sum = 0.0
        val_l1_sum = 0.0
        val_psnr_sum = 0.0
        val_ssim_sum = 0.0
        num_val_batches = 0
        sample_recon_pair: tuple[torch.Tensor, torch.Tensor] | None = None

        with torch.no_grad():
            for val_batch in val_loader:
                x_val = val_batch["obs"].to(device)
                out_val = model(x_val)

                val_loss_dict = criterion(
                    recon=out_val["recon"],
                    target=x_val,
                    mean=out_val["mean"],
                    logvar=out_val["logvar"],
                    beta_scale=1.0,
                )

                psnr = compute_psnr(out_val["recon"], x_val)
                ssim = compute_ssim(out_val["recon"], x_val)

                val_loss_sum += val_loss_dict["loss"].item()
                val_l1_sum += val_loss_dict["loss_l1"].item()
                val_psnr_sum += psnr
                val_ssim_sum += ssim
                num_val_batches += 1

                if sample_recon_pair is None:
                    # Save first 8 images for visual verification
                    sample_recon_pair = (x_val[:8].cpu(), out_val["recon"][:8].cpu())

        avg_val_loss = val_loss_sum / max(1, num_val_batches)
        avg_val_l1 = val_l1_sum / max(1, num_val_batches)
        avg_val_psnr = val_psnr_sum / max(1, num_val_batches)
        avg_val_ssim = val_ssim_sum / max(1, num_val_batches)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_psnr"].append(avg_val_psnr)
        history["val_ssim"].append(avg_val_ssim)
        history["val_l1"].append(avg_val_l1)

        logger.info(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val PSNR: {avg_val_psnr:.2f} dB | "
            f"Val SSIM: {avg_val_ssim:.4f} | "
            f"Val L1: {avg_val_l1:.4f}"
        )

        # Save visual comparison grid periodically
        if epoch % save_fig_every == 0 or epoch == epochs:
            if sample_recon_pair is not None:
                orig, rec = sample_recon_pair
                # Interleave original and reconstruction: [O1, R1, O2, R2, ...]
                interleaved = torch.empty((orig.size(0) * 2, *orig.shape[1:]), dtype=orig.dtype)
                interleaved[0::2] = orig
                interleaved[1::2] = rec
                grid = make_grid(interleaved, nrow=4, padding=2, normalize=False)
                fig_path = fig_dir / f"recon_epoch_{epoch:02d}.png"
                save_image(grid, fig_path)
                logger.info(f"Saved visual reconstruction grid to: {fig_path}")

        # Checkpoint: Latest
        latest_ckpt_path = ckpt_dir / "latest_vae.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "latent_channels": latent_channels,
                "base_channels": base_channels,
                "val_psnr": avg_val_psnr,
                "val_ssim": avg_val_ssim,
                "val_loss": avg_val_loss,
            },
            latest_ckpt_path,
        )

        # Checkpoint: Best PSNR
        if avg_val_psnr > best_val_psnr:
            best_val_psnr = avg_val_psnr
            best_ckpt_path = ckpt_dir / "best_vae.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "latent_channels": latent_channels,
                    "base_channels": base_channels,
                    "val_psnr": avg_val_psnr,
                    "val_ssim": avg_val_ssim,
                    "val_loss": avg_val_loss,
                    "val_l1": avg_val_l1,
                },
                best_ckpt_path,
            )
            logger.info(f"[*] New best VAE checkpoint saved with PSNR: {best_val_psnr:.2f} dB -> {best_ckpt_path}")

    total_time = time.time() - start_time
    logger.info(f"\nTraining Complete in {total_time:.1f}s. Best Val PSNR: {best_val_psnr:.2f} dB.")

    # Save metrics JSON
    metrics_file = metrics_dir / "eval_metrics.json"
    summary_metrics = {
        "best_val_psnr": best_val_psnr,
        "final_val_psnr": history["val_psnr"][-1],
        "final_val_ssim": history["val_ssim"][-1],
        "final_val_l1": history["val_l1"][-1],
        "total_epochs": epochs,
        "batch_size": batch_size,
        "training_time_seconds": total_time,
        "history": history,
    }
    with open(metrics_file, "w") as f:
        json.dump(summary_metrics, f, indent=2)
    logger.info(f"Saved evaluation metrics to: {metrics_file}")

    # Copy final comparison to canonical recon_comparison.png
    latest_fig = fig_dir / f"recon_epoch_{epochs:02d}.png"
    if latest_fig.exists():
        canonical_fig = fig_dir / "recon_comparison.png"
        canonical_fig.write_bytes(latest_fig.read_bytes())
        logger.info(f"Canonical figure updated: {canonical_fig}")

    return best_ckpt_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Spatial VAE on offline CarRacing-v3 data.")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size per training step.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--beta-kl", type=float, default=1e-4, help="KL divergence regularization weight.")
    parser.add_argument("--latent-channels", type=int, default=4, help="Number of spatial latent channels.")
    parser.add_argument("--base-channels", type=int, default=32, help="Base conv channels.")
    parser.add_argument("--device", type=str, default=None, help="Explicit device override (cuda, mps, cpu).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    train_vae(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        beta_kl=args.beta_kl,
        latent_channels=args.latent_channels,
        base_channels=args.base_channels,
        device_str=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
