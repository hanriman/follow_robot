#!/usr/bin/env python3
"""
Action-Conditioned Latent Diffusion World Model (AC-LDM) Training Script for AckBot.

Trains:
- ActionConditionedUNet1D with AdaGN conditioning on [B, 4, 32] latents
- Conditioned on continuous Ackermann action a_t = [v, delta]
- DDIM sampling acceleration for predictive rollouts

Requires trained MultiModalSensorVAE checkpoint to project sensor observations
to continuous latent manifolds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rl_ws.data.dataset import get_dataset_splits
from rl_ws.policies.world_model.vae import MultiModalSensorVAE
from rl_ws.policies.vla.diffusion_wm import ActionConditionedDiffusionWorldModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs/experiments/diffusion"
DEFAULT_VAE_CKPT = PROJECT_ROOT / "outputs/experiments/vae/checkpoints/best_model.pt"


def parse_args():
    parser = argparse.ArgumentParser(description="Train AckBot Action-Conditioned Diffusion World Model")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--timesteps", type=int, default=1000, help="Diffusion forward noise steps")
    parser.add_argument("--ddim-steps", type=int, default=10, help="DDIM sampling steps for eval")
    parser.add_argument("--vae-checkpoint", type=str, default=str(DEFAULT_VAE_CKPT), help="Path to trained VAE")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUT_DIR), help="Output directory")
    parser.add_argument("--dataset-index", type=str, default=None, help="Path to dataset_index.json")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "mps", "cpu"])
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def select_device(preferred: str) -> torch.device:
    if preferred == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    elif preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    args = parse_args()
    device = select_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("   ACTION-CONDITIONED LATENT DIFFUSION WORLD MODEL (AC-LDM) TRAINING   ")
    print("=" * 72)
    print(f"Device              : {device}")
    print(f"Diffusion Timesteps : {args.timesteps}")
    print(f"Eval DDIM Steps     : {args.ddim_steps}")
    print(f"Epochs              : {args.epochs}")
    print(f"Batch Size          : {args.batch_size}")
    print(f"Output Directory    : {out_dir}")

    # Load pretrained VAE
    vae_path = Path(args.vae_checkpoint)
    if not vae_path.exists():
        fallback = PROJECT_ROOT / "outputs/experiments/vae/checkpoints/latest_model.pt"
        if fallback.exists():
            vae_path = fallback
        else:
            raise FileNotFoundError(
                f"\n[FAIL LOUD] Pretrained VAE checkpoint not found at {vae_path} or {fallback}.\n"
                f"Remediation: Train VAE first with: uv run ackbot-train-vae\n"
            )

    print(f"Loading VAE from    : {vae_path.name}")
    vae_ckpt = torch.load(vae_path, map_location=device, weights_only=False)
    vae = MultiModalSensorVAE(in_dim=426, latent_channels=4, latent_length=32).to(device)
    vae.load_state_dict(vae_ckpt["model_state_dict"])
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False

    # Load dataset
    index_p = Path(args.dataset_index) if args.dataset_index else None
    train_ds, val_ds, test_ds = get_dataset_splits(index_file=index_p)
    print(f"Dataset Index       : {index_p if index_p else 'Default (data/processed/dataset_index.json)'}")
    print(f"Dataset Loaded      : {len(train_ds)} train | {len(val_ds)} val | {len(test_ds)} test")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = ActionConditionedDiffusionWorldModel(
        latent_channels=4,
        latent_length=32,
        action_dim=2,
        base_channels=64,
        num_timesteps=args.timesteps,
        vae=vae,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    history = {
        "train_loss": [],
        "val_loss": [],
        "latent_mse": [],
    }

    best_val_loss = float("inf")

    print("\nBeginning Diffusion Training Loop...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        steps = 0

        for batch in train_loader:
            obs_t = batch["obs"].to(device)
            action_t = batch["action"].to(device)
            obs_next = batch["next_obs"].to(device)

            with torch.no_grad():
                z_t, _ = vae.encode(obs_t)
                z_next, _ = vae.encode(obs_next)

            optimizer.zero_grad()
            res = model.compute_loss(target_latent=z_next, context_latent=z_t, action=action_t)
            loss = res["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss.item()
            steps += 1

        scheduler.step()
        avg_train = train_loss_sum / max(steps, 1)

        # Validation with DDIM sampler
        model.eval()
        val_loss_sum = 0.0
        val_steps = 0
        latent_mses = []

        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                obs_t = batch["obs"].to(device)
                action_t = batch["action"].to(device)
                obs_next = batch["next_obs"].to(device)

                z_t, _ = vae.encode(obs_t)
                z_next, _ = vae.encode(obs_next)

                res = model.compute_loss(target_latent=z_next, context_latent=z_t, action=action_t)
                val_loss_sum += res["loss"].item()
                val_steps += 1

                # On subset of validation, run DDIM to verify transition accuracy
                if i < 3:
                    pred_z_next = model.predict_next(z_t, action_t, ddim_steps=args.ddim_steps)
                    mse = F.mse_loss(pred_z_next, z_next).item()
                    latent_mses.append(mse)

        avg_val = val_loss_sum / max(val_steps, 1)
        avg_latent_mse = float(np.mean(latent_mses)) if latent_mses else 0.0

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["latent_mse"].append(avg_latent_mse)

        print(
            f"Epoch {epoch:2d}/{args.epochs:2d} | "
            f"Train Noise MSE: {avg_train:6.4f} | "
            f"Val Noise MSE: {avg_val:6.4f} | "
            f"DDIM Latent MSE: {avg_latent_mse:6.4f}"
        )

        latest_ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "val_loss": avg_val,
            "ddim_latent_mse": avg_latent_mse,
        }
        torch.save(latest_ckpt, ckpt_dir / "latest_model.pt")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(latest_ckpt, ckpt_dir / "best_model.pt")

    print("\nDiffusion Training completed.")
    print(f"Best Validation Noise MSE: {best_val_loss:6.4f}")
    print(f"Final DDIM Latent MSE     : {history['latent_mse'][-1]:6.4f}")

    summary_file = out_dir / "metrics_summary.json"
    with open(summary_file, "w") as f:
        json.dump(
            {
                "best_val_loss": best_val_loss,
                "final_train_loss": history["train_loss"][-1],
                "final_ddim_latent_mse": history["latent_mse"][-1],
                "history": history,
            },
            f,
            indent=2,
        )

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Noise MSE", color="royalblue")
    plt.plot(history["val_loss"], label="Val Noise MSE", color="crimson")
    plt.title("Diffusion Noise Prediction Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history["latent_mse"], label="DDIM Latent MSE (10 steps)", color="forestgreen", linewidth=2)
    plt.title("DDIM Sampling Rollout Error")
    plt.xlabel("Epoch")
    plt.ylabel("Latent MSE")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    curve_path = out_dir / "diffusion_curve.png"
    plt.savefig(curve_path, dpi=200)
    plt.close()
    print(f"Saved diagnostic plots to: {curve_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
