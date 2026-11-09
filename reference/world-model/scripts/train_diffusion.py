#!/usr/bin/env python3
"""
Training Runner for Action-Conditioned Latent Diffusion World Model (AC-LDM).

Trains spatial U-Net denoiser conditioned on context latents and continuous action vectors,
evaluating multi-step latent unrolling and decoding rollouts via frozen Spatial VAE.

Outputs (Solid 1-Directory Rule):
- Checkpoints: outputs/experiments/diffusion/checkpoints/best_diffusion.pt, latest_diffusion.pt
- Figures:     outputs/experiments/diffusion/figures/unrolled_rollout.png
- Metrics:     outputs/experiments/diffusion/metrics/eval_metrics.json
- Logs:        outputs/experiments/diffusion/logs/train.log
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from wm_control.config import get_experiment_dir
from wm_control.data.dataset import get_latent_dataset_splits
from wm_control.models.diffusion_wm import ActionConditionedDiffusionWorldModel
from wm_control.models.vae import SpatialVAE
from wm_control.utils.device import get_device
from wm_control.utils.seed import seed_everything


def train_diffusion(
    epochs: int = 15,
    batch_size: int = 256,
    lr: float = 3e-4,
    num_timesteps: int = 100,
    base_channels: int = 64,
    latent_channels: int = 4,
    action_dim: int = 3,
    ddim_eval_steps: int = 10,
    rollout_horizon: int = 5,
    device_str: str | None = None,
    seed: int = 42,
    save_fig_every: int = 5,
) -> Path:
    """
    Train Action-Conditioned Latent Diffusion Model.
    """
    seed_everything(seed)
    device = get_device(device_str)

    # Resolve deterministic canonical output paths
    exp_dir = get_experiment_dir("diffusion", create_if_missing=True)
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
    logger = logging.getLogger("TrainDiffusion")
    logger.info(f"Starting AC-LDM Training on device: {device}")
    logger.info(f"Experiment Directory: {exp_dir}")

    # Load pre-encoded latent dataset splits
    train_ds, val_ds, _ = get_latent_dataset_splits(augment_train=True)
    logger.info(f"Loaded Latent Dataset: {len(train_ds):,} train samples, {len(val_ds):,} val samples")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    # Load frozen VAE for decoding visual rollouts
    vae_ckpt_path = get_experiment_dir("vae") / "checkpoints" / "best_vae.pt"
    vae_decoder: SpatialVAE | None = None
    if vae_ckpt_path.exists():
        vae_ckpt = torch.load(vae_ckpt_path, map_location=device)
        vae_model = SpatialVAE(
            in_channels=3,
            latent_channels=vae_ckpt.get("latent_channels", 4),
            base_channels=vae_ckpt.get("base_channels", 32),
        ).to(device)
        vae_model.load_state_dict(vae_ckpt["model_state"])
        vae_model.eval()
        for p in vae_model.parameters():
            p.requires_grad = False
        vae_decoder = vae_model
        logger.info(f"Loaded frozen VAE Decoder from: {vae_ckpt_path}")
    else:
        logger.warning(f"VAE checkpoint not found at {vae_ckpt_path}; visual decoding will be skipped.")

    # Initialize AC-LDM
    model = ActionConditionedDiffusionWorldModel(
        latent_channels=latent_channels,
        action_dim=action_dim,
        num_timesteps=num_timesteps,
        base_channels=base_channels,
        vae=vae_decoder,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"AC-LDM Initialized: {total_params:,} trainable parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_loss = float("inf")
    history: dict[str, list] = {
        "train_loss": [],
        "val_loss": [],
        "val_rollout_l1": [],
    }

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        num_train_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]", leave=False)
        for batch in pbar:
            z_t = batch["z_t"].to(device)
            next_z = batch["next_z"].to(device)
            action = batch["action"].to(device)

            optimizer.zero_grad(set_to_none=True)
            loss_dict = model.compute_loss(target_latent=next_z, context_latent=z_t, action=action)

            loss_dict["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss_dict["loss"].item()
            num_train_batches += 1
            pbar.set_postfix(loss=f"{loss_dict['loss'].item():.4f}")

        scheduler.step()
        avg_train_loss = train_loss_sum / max(1, num_train_batches)

        # Validation Loop
        model.eval()
        val_loss_sum = 0.0
        num_val_batches = 0

        with torch.no_grad():
            for val_batch in val_loader:
                z_t_val = val_batch["z_t"].to(device)
                next_z_val = val_batch["next_z"].to(device)
                action_val = val_batch["action"].to(device)

                val_loss_dict = model.compute_loss(
                    target_latent=next_z_val,
                    context_latent=z_t_val,
                    action=action_val,
                )
                val_loss_sum += val_loss_dict["loss"].item()
                num_val_batches += 1

        avg_val_loss = val_loss_sum / max(1, num_val_batches)

        # Multi-Step Rollout Evaluation on first 4 validation samples over horizon H
        val_sample = next(iter(val_loader))
        init_z = val_sample["z_t"][:4].to(device)
        actions_seq = val_sample["action"][:4].unsqueeze(1).repeat(1, rollout_horizon, 1).to(device)
        target_z = val_sample["next_z"][:4].to(device)

        # 1-step prediction error check
        pred_z1 = model.predict_next(init_z, actions_seq[:, 0, :], ddim_steps=ddim_eval_steps)
        step1_l1 = F.l1_loss(pred_z1, target_z).item()

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_rollout_l1"].append(step1_l1)

        logger.info(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Noise MSE: {avg_train_loss:.4f} | "
            f"Val Noise MSE: {avg_val_loss:.4f} | "
            f"Val 1-Step Latent L1: {step1_l1:.4f}"
        )

        # Decode multi-step rollout visually if VAE decoder is available
        if (epoch % save_fig_every == 0 or epoch == epochs) and vae_decoder is not None:
            # Unroll 4 samples over horizon H=5
            with torch.no_grad():
                unrolled = model.rollout_latents(init_z, actions_seq, ddim_steps=ddim_eval_steps)
                # unrolled shape: [B=4, H=5, C=4, H_z=16, W_z=16]
                b_roll, h_roll = unrolled.shape[0], unrolled.shape[1]
                flat_latents = unrolled.view(b_roll * h_roll, *unrolled.shape[2:])
                decoded_frames = vae_decoder.decode(flat_latents)  # [B*H, 3, 64, 64]
                grid = make_grid(decoded_frames.clamp(0.0, 1.0), nrow=h_roll, padding=2)
                fig_path = fig_dir / f"unrolled_rollout_epoch_{epoch:02d}.png"
                save_image(grid, fig_path)
                logger.info(f"Saved visual unrolled rollout to: {fig_path}")

        # Checkpoint: Latest
        latest_ckpt = ckpt_dir / "latest_diffusion.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": avg_val_loss,
                "val_rollout_l1": step1_l1,
            },
            latest_ckpt,
        )

        # Checkpoint: Best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_ckpt = ckpt_dir / "best_diffusion.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "val_loss": avg_val_loss,
                    "val_rollout_l1": step1_l1,
                },
                best_ckpt,
            )
            logger.info(f"[*] New best AC-LDM checkpoint saved with Val MSE: {best_val_loss:.4f} -> {best_ckpt}")

    total_time = time.time() - start_time
    logger.info(f"\nTraining Complete in {total_time:.1f}s. Best Val Noise MSE: {best_val_loss:.4f}.")

    # Save metrics JSON
    metrics_file = metrics_dir / "eval_metrics.json"
    summary_metrics = {
        "best_val_loss": best_val_loss,
        "final_val_loss": history["val_loss"][-1],
        "final_rollout_l1": history["val_rollout_l1"][-1],
        "total_epochs": epochs,
        "batch_size": batch_size,
        "training_time_seconds": total_time,
        "history": history,
    }
    with open(metrics_file, "w") as f:
        json.dump(summary_metrics, f, indent=2)

    latest_fig = fig_dir / f"unrolled_rollout_epoch_{epochs:02d}.png"
    if latest_fig.exists():
        canonical_fig = fig_dir / "unrolled_rollout.png"
        canonical_fig.write_bytes(latest_fig.read_bytes())
        logger.info(f"Canonical figure updated: {canonical_fig}")

    return best_ckpt


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Action-Conditioned Latent Diffusion World Model (AC-LDM).")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size per training step.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--num-timesteps", type=int, default=100, help="Discrete diffusion steps.")
    parser.add_argument("--ddim-eval-steps", type=int, default=10, help="DDIM sampling steps for eval.")
    parser.add_argument("--device", type=str, default=None, help="Device override (cuda, mps, cpu).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    train_diffusion(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_timesteps=args.num_timesteps,
        ddim_eval_steps=args.ddim_eval_steps,
        device_str=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
