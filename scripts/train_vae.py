#!/usr/bin/env python3
"""
Multi-Modal Sensor VAE Training Script for AckBot.

Trains:
- MultiModalSensorVAE: Compresses 426-D multi-modal observation ([64 depth, 360 LiDAR, 2 state])
  into structured spatial latent map [B, 4, 32] (128-D total) with reparameterization and decoder.

Logs:
- Reconstruction Loss (Huber / Smooth L1)
- Analytical KL Divergence
- Reconstruction PSNR (dB)
- Telemetry linear probe R^2 from structured latents
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rl_ws.data.dataset import get_dataset_splits, TELEMETRY_KEYS
from rl_ws.policies.world_model.vae import MultiModalSensorVAE
from rl_ws.policies.world_model.representation import RidgeRegressionProbe

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs/experiments/vae"


def parse_args():
    parser = argparse.ArgumentParser(description="Train AckBot Multi-Modal Sensor VAE")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="AdamW weight decay")
    parser.add_argument("--kl-weight", type=float, default=1e-4, help="KL divergence regularization weight")
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
    print("         MULTI-MODAL SENSOR VAE TRAINING PIPELINE                     ")
    print("=" * 72)
    print(f"Device              : {device}")
    print(f"Observation Dim     : 426 (64 depth + 360 LiDAR + 2 state)")
    print(f"Latent Shape        : [4, 32] (128-D total)")
    print(f"KL Weight           : {args.kl_weight}")
    print(f"Epochs              : {args.epochs}")
    print(f"Batch Size          : {args.batch_size}")
    print(f"Output Directory    : {out_dir}")

    index_p = Path(args.dataset_index) if args.dataset_index else None
    train_ds, val_ds, test_ds = get_dataset_splits(index_file=index_p)
    print(f"Dataset Index       : {index_p if index_p else 'Default (data/processed/dataset_index.json)'}")
    print(f"Dataset Loaded      : {len(train_ds)} train | {len(val_ds)} val | {len(test_ds)} test")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = MultiModalSensorVAE(
        in_dim=426,
        latent_channels=4,
        latent_length=32,
        hidden_dim=256,
        kl_weight=args.kl_weight,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    history = {
        "train_loss": [],
        "recon_loss": [],
        "kl_loss": [],
        "val_loss": [],
        "psnr": [],
        "mean_r2": [],
    }

    best_val_loss = float("inf")

    print("\nBeginning VAE Training Loop...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        recon_loss_sum = 0.0
        kl_loss_sum = 0.0
        steps = 0

        for batch in train_loader:
            obs = batch["obs"].to(device)

            optimizer.zero_grad()
            res = model(obs)

            loss = res["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss.item()
            recon_loss_sum += res["recon_loss"].item()
            kl_loss_sum += res["kl_loss"].item()
            steps += 1

        scheduler.step()
        avg_train = train_loss_sum / max(steps, 1)
        avg_recon = recon_loss_sum / max(steps, 1)
        avg_kl = kl_loss_sum / max(steps, 1)

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_steps = 0
        all_psnrs = []
        all_val_z = []
        all_val_telem = []

        with torch.no_grad():
            for batch in val_loader:
                obs = batch["obs"].to(device)
                res = model(obs)
                val_loss_sum += res["loss"].item()
                val_steps += 1

                metrics = model.compute_metrics(obs, res["recon_x"])
                all_psnrs.append(metrics["psnr"])

                z_flat = res["mu"].flatten(start_dim=1)
                all_val_z.append(z_flat)
                all_val_telem.append(batch["telemetry"].to(device))

        avg_val = val_loss_sum / max(val_steps, 1)
        avg_psnr = float(np.mean(all_psnrs)) if all_psnrs else 0.0
        cat_val_z = torch.cat(all_val_z, dim=0)
        cat_val_telem = torch.cat(all_val_telem, dim=0)

        probe = RidgeRegressionProbe(alpha=1.0).fit(cat_val_z, cat_val_telem)
        probe_res = probe.score_r2(cat_val_z, cat_val_telem, labels=TELEMETRY_KEYS)
        mean_r2 = probe_res["mean_r2"]

        history["train_loss"].append(avg_train)
        history["recon_loss"].append(avg_recon)
        history["kl_loss"].append(avg_kl)
        history["val_loss"].append(avg_val)
        history["psnr"].append(avg_psnr)
        history["mean_r2"].append(mean_r2)

        print(
            f"Epoch {epoch:2d}/{args.epochs:2d} | "
            f"Train Loss: {avg_train:6.4f} (Recon: {avg_recon:6.4f}, KL: {avg_kl:6.4f}) | "
            f"Val Loss: {avg_val:6.4f} | "
            f"PSNR: {avg_psnr:5.2f} dB | "
            f"Telem R²: {mean_r2:5.3f}"
        )

        latest_ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "val_loss": avg_val,
            "psnr": avg_psnr,
            "mean_r2": mean_r2,
        }
        torch.save(latest_ckpt, ckpt_dir / "latest_model.pt")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(latest_ckpt, ckpt_dir / "best_model.pt")

    print("\nVAE Training completed.")
    print(f"Best Validation Loss : {best_val_loss:6.4f}")
    print(f"Final PSNR           : {history['psnr'][-1]:5.2f} dB")
    print(f"Final Telemetry R²   : {history['mean_r2'][-1]:5.3f}")

    summary_file = out_dir / "metrics_summary.json"
    with open(summary_file, "w") as f:
        json.dump(
            {
                "best_val_loss": best_val_loss,
                "final_psnr": history["psnr"][-1],
                "final_mean_r2": history["mean_r2"][-1],
                "telemetry_r2_per_dim": probe_res.get("per_dimension_r2", {}),
                "history": history,
            },
            f,
            indent=2,
        )

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss", color="royalblue")
    plt.plot(history["val_loss"], label="Val Loss", color="crimson")
    plt.plot(history["recon_loss"], label="Huber Recon Loss", color="orange", linestyle="--")
    plt.title("Sensor VAE Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history["psnr"], label="Reconstruction PSNR (dB)", color="forestgreen", linewidth=2)
    plt.plot([r * 100 for r in history["mean_r2"]], label="Telemetry R² (%)", color="purple", linewidth=2)
    plt.title("Reconstruction Quality & Decodability")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    curve_path = out_dir / "reconstruction_curve.png"
    plt.savefig(curve_path, dpi=200)
    plt.close()
    print(f"Saved diagnostic plots to: {curve_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
