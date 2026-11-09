#!/usr/bin/env python3
"""
Teacher-Free Le World Model (LeWM) with SIGReg Training Script for AckBot.

Trains:
- Shared AckBotSensorEncoder E_\theta (426-D -> 128-D)
- ActionConditionedPredictor T_\psi (128-D + 2-D Ackermann -> 128-D residual)
using joint end-to-end backpropagation regularized by the analytical Epps-Pulley
Sketched Isotropic Gaussian Regularization (SIGReg) test.

Logs:
- Effective Rank (Roy's Participation Ratio)
- Gaussianity Moments (skewness, kurtosis)
- 8-D Physical Telemetry Linear Probe R^2
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
from rl_ws.policies.world_model.sigreg import ActionConditionedTransitionLoss
from rl_ws.policies.world_model.sigreg_wm import TeacherFreeLeWorldModel
from rl_ws.policies.world_model.representation import (
    compute_effective_rank,
    compute_gaussianity_diagnostics,
    RidgeRegressionProbe,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs/experiments/sigreg"


def parse_args():
    parser = argparse.ArgumentParser(description="Train AckBot Teacher-Free LeWM with SIGReg")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--sigreg-weight", type=float, default=1.0, help="SIGReg regularization multiplier")
    parser.add_argument("--latent-dim", type=int, default=128, help="Latent state dimension")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUT_DIR), help="Output artifact directory")
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
    print("      TEACHER-FREE LE WORLD MODEL (LEWM) + SIGREG TRAINING PIPELINE       ")
    print("=" * 72)
    print(f"Device              : {device}")
    print(f"Latent Dimension    : {args.latent_dim}")
    print(f"SIGReg Loss Weight  : {args.sigreg_weight}")
    print(f"Epochs              : {args.epochs}")
    print(f"Batch Size          : {args.batch_size}")
    print(f"Output Directory    : {out_dir}")

    # Load offline dataset
    index_p = Path(args.dataset_index) if args.dataset_index else None
    train_ds, val_ds, test_ds = get_dataset_splits(index_file=index_p)
    print(f"Dataset Index       : {index_p if index_p else 'Default (data/processed/dataset_index.json)'}")
    print(f"Dataset Loaded      : {len(train_ds)} train | {len(val_ds)} val | {len(test_ds)} test")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = TeacherFreeLeWorldModel(latent_dim=args.latent_dim, action_dim=2).to(device)
    loss_fn = ActionConditionedTransitionLoss(
        loss_type="smooth_l1",
        beta=1.0,
        sigreg_weight=args.sigreg_weight,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    history = {
        "train_loss": [],
        "trans_loss": [],
        "sigreg_loss": [],
        "val_loss": [],
        "effective_rank": [],
        "rank_ratio": [],
        "mean_r2": [],
    }

    best_val_loss = float("inf")

    print("\nBeginning Training Loop...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_total_loss = 0.0
        epoch_trans_loss = 0.0
        epoch_sigreg_loss = 0.0
        steps = 0

        for batch in train_loader:
            obs_t = batch["obs"].to(device)
            action_t = batch["action"].to(device)
            obs_next = batch["next_obs"].to(device)

            optimizer.zero_grad()
            fwd = model(obs_t, action_t, obs_next)

            losses = loss_fn(
                pred_next_z=fwd["pred_next_z"],
                target_next_z=fwd["target_next_z"],
                regularize_tokens=fwd["all_tokens"],
            )

            loss = losses["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_total_loss += loss.item()
            epoch_trans_loss += losses["transition_loss"].item()
            epoch_sigreg_loss += losses["sigreg_loss"].item()
            steps += 1

        scheduler.step()
        avg_train = epoch_total_loss / max(steps, 1)
        avg_trans = epoch_trans_loss / max(steps, 1)
        avg_sigreg = epoch_sigreg_loss / max(steps, 1)

        # Validation & Representation Health Evaluation
        model.eval()
        val_total_loss = 0.0
        val_steps = 0
        all_val_z = []
        all_val_telem = []

        with torch.no_grad():
            for batch in val_loader:
                obs_t = batch["obs"].to(device)
                action_t = batch["action"].to(device)
                obs_next = batch["next_obs"].to(device)

                fwd = model(obs_t, action_t, obs_next)
                losses = loss_fn(
                    pred_next_z=fwd["pred_next_z"],
                    target_next_z=fwd["target_next_z"],
                    regularize_tokens=fwd["all_tokens"],
                )
                val_total_loss += losses["loss"].item()
                val_steps += 1

                all_val_z.append(fwd["z_t"])
                all_val_telem.append(batch["telemetry"].to(device))

        avg_val = val_total_loss / max(val_steps, 1)
        cat_val_z = torch.cat(all_val_z, dim=0)
        cat_val_telem = torch.cat(all_val_telem, dim=0)

        # Representation diagnostics
        rank_diag = compute_effective_rank(cat_val_z)
        eff_rank = rank_diag["effective_rank"]
        rank_ratio = rank_diag["rank_ratio"]

        # Linear probe R^2 on physical telemetry
        probe = RidgeRegressionProbe(alpha=1.0).fit(cat_val_z, cat_val_telem)
        probe_res = probe.score_r2(cat_val_z, cat_val_telem, labels=TELEMETRY_KEYS)
        mean_r2 = probe_res["mean_r2"]

        history["train_loss"].append(avg_train)
        history["trans_loss"].append(avg_trans)
        history["sigreg_loss"].append(avg_sigreg)
        history["val_loss"].append(avg_val)
        history["effective_rank"].append(eff_rank)
        history["rank_ratio"].append(rank_ratio)
        history["mean_r2"].append(mean_r2)

        print(
            f"Epoch {epoch:2d}/{args.epochs:2d} | "
            f"Train Loss: {avg_train:6.4f} (Trans: {avg_trans:6.4f}, SIGReg: {avg_sigreg:6.4f}) | "
            f"Val Loss: {avg_val:6.4f} | "
            f"Eff Rank: {eff_rank:5.1f}/{args.latent_dim} ({rank_ratio*100:4.1f}%) | "
            f"Telem R²: {mean_r2:5.3f}"
        )

        # Save latest checkpoint
        latest_ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "latent_dim": args.latent_dim,
            "val_loss": avg_val,
            "effective_rank": eff_rank,
            "rank_ratio": rank_ratio,
            "mean_r2": mean_r2,
        }
        torch.save(latest_ckpt, ckpt_dir / "latest_model.pt")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(latest_ckpt, ckpt_dir / "best_model.pt")

    print("\nTraining completed.")
    print(f"Best Validation Loss : {best_val_loss:6.4f}")
    print(f"Final Effective Rank : {history['effective_rank'][-1]:5.1f} / {args.latent_dim} ({history['rank_ratio'][-1]*100:.1f}%)")
    print(f"Final Telemetry R²   : {history['mean_r2'][-1]:5.3f}")

    # Save summary metrics
    summary_file = out_dir / "metrics_summary.json"
    with open(summary_file, "w") as f:
        json.dump(
            {
                "best_val_loss": best_val_loss,
                "final_train_loss": history["train_loss"][-1],
                "final_effective_rank": history["effective_rank"][-1],
                "final_rank_ratio": history["rank_ratio"][-1],
                "final_mean_r2": history["mean_r2"][-1],
                "telemetry_r2_per_dim": probe_res.get("per_dimension_r2", {}),
                "history": history,
            },
            f,
            indent=2,
        )

    # Plotting training curves
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss", color="royalblue")
    plt.plot(history["val_loss"], label="Val Loss", color="crimson")
    plt.plot(history["trans_loss"], label="Transition Loss", color="orange", linestyle="--")
    plt.title("LeWM (SIGReg) Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(
        [r * 100 for r in history["rank_ratio"]],
        label="Effective Rank % (Participation Ratio)",
        color="forestgreen",
        linewidth=2,
    )
    plt.plot(
        [r * 100 for r in history["mean_r2"]],
        label="Telemetry Linear Probe R² (%)",
        color="purple",
        linewidth=2,
    )
    plt.axhline(70.0, color="grey", linestyle=":", label="Healthy Rank Target (70%)")
    plt.title("Representation Health & Decodability")
    plt.xlabel("Epoch")
    plt.ylabel("Percentage (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    curve_path = out_dir / "effective_rank_curve.png"
    plt.savefig(curve_path, dpi=200)
    plt.close()
    print(f"Saved diagnostic plots to: {curve_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
