#!/usr/bin/env python3
"""
Latent Reward Head Training Script for AckBot Latent MPC.

Trains:
- LatentRewardHead (LeWM): Maps flat 128-D latent z -> [r_env, dist_error, visual_lock]
- LatentRewardHead (VAE): Maps structured [4, 32] latent z -> [r_env, dist_error, visual_lock]

These heads provide scalar trajectory scoring during receding-horizon CEM-MPC optimization.
Checkpoints saved to outputs/experiments/mpc/checkpoints/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from rl_ws.data.dataset import get_dataset_splits
from rl_ws.policies.world_model.sigreg_wm import TeacherFreeLeWorldModel
from rl_ws.policies.world_model.vae import MultiModalSensorVAE
from rl_ws.policies.world_model.reward_head import LatentRewardHead

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs/experiments/mpc"
DEFAULT_LEWM_CKPT = PROJECT_ROOT / "outputs/experiments/sigreg/checkpoints/best_model.pt"
DEFAULT_VAE_CKPT = PROJECT_ROOT / "outputs/experiments/vae/checkpoints/best_model.pt"


def parse_args():
    parser = argparse.ArgumentParser(description="Train AckBot Latent Reward Heads for MPC")
    parser.add_argument("--epochs", type=int, default=15, help="Epochs per reward head")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--lewm-checkpoint", type=str, default=str(DEFAULT_LEWM_CKPT))
    parser.add_argument("--vae-checkpoint", type=str, default=str(DEFAULT_VAE_CKPT))
    parser.add_argument("--skip-vae", action="store_true", default=False, help="Skip training VAE reward head")
    parser.add_argument("--dataset-index", type=str, default=None, help="Path to dataset_index.json")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "mps", "cpu"])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def select_device(preferred: str) -> torch.device:
    if preferred == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    elif preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_single_head(
    head: LatentRewardHead,
    encoder_fn,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    name: str,
) -> tuple[LatentRewardHead, float]:
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    best_val = float("inf")

    print(f"\nTraining {name} Reward Head...")
    for epoch in range(1, epochs + 1):
        head.train()
        train_loss = 0.0
        steps = 0

        for batch in train_loader:
            obs = batch["obs"].to(device)
            reward = batch["reward"].to(device)
            telem = batch["telemetry"].to(device)

            dist_error = torch.abs(telem[:, 2] - 0.80)
            visual_lock = torch.cos(telem[:, 3])
            targets = torch.stack([reward, dist_error, visual_lock], dim=-1)

            with torch.no_grad():
                z = encoder_fn(obs)

            optimizer.zero_grad()
            preds = head(z)
            loss = F.mse_loss(preds, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            steps += 1

        avg_train = train_loss / max(steps, 1)

        head.eval()
        val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for batch in val_loader:
                obs = batch["obs"].to(device)
                reward = batch["reward"].to(device)
                telem = batch["telemetry"].to(device)

                dist_error = torch.abs(telem[:, 2] - 0.80)
                visual_lock = torch.cos(telem[:, 3])
                targets = torch.stack([reward, dist_error, visual_lock], dim=-1)

                z = encoder_fn(obs)
                preds = head(z)
                loss = F.mse_loss(preds, targets)
                val_loss += loss.item()
                val_steps += 1

        avg_val = val_loss / max(val_steps, 1)
        if avg_val < best_val:
            best_val = avg_val

        print(f"  [{name}] Epoch {epoch:2d}/{epochs:2d} | Train MSE: {avg_train:6.4f} | Val MSE: {avg_val:6.4f}")

    return head, best_val


def main():
    args = parse_args()
    device = select_device(args.device)
    torch.manual_seed(args.seed)

    out_dir = Path(args.output_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("           ACKBOT LATENT REWARD HEAD TRAINING PIPELINE               ")
    print("=" * 72)
    print(f"Device              : {device}")
    print(f"Epochs per Head     : {args.epochs}")
    print(f"Output Directory    : {ckpt_dir}")

    index_p = Path(args.dataset_index) if args.dataset_index else None
    train_ds, val_ds, test_ds = get_dataset_splits(index_file=index_p)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    summary = {}

    # 1. Train LeWM Reward Head
    lewm_path = Path(args.lewm_checkpoint)
    if lewm_path.exists():
        lewm_ckpt = torch.load(lewm_path, map_location=device, weights_only=False)
        lewm = TeacherFreeLeWorldModel(latent_dim=128, action_dim=2).to(device)
        lewm.load_state_dict(lewm_ckpt["model_state_dict"])
        lewm.eval()

        head_lewm = LatentRewardHead(in_dim=128, is_spatial=False).to(device)
        head_lewm, val_mse_lewm = train_single_head(
            head=head_lewm,
            encoder_fn=lambda o: lewm.encode(o),
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            name="LeWM",
        )
        torch.save(
            {"model_state_dict": head_lewm.state_dict(), "val_mse": val_mse_lewm, "is_spatial": False},
            ckpt_dir / "reward_head_lewm.pt",
        )
        summary["lewm_reward_head_val_mse"] = val_mse_lewm
    else:
        print(f"Notice: LeWM checkpoint not found at {lewm_path}. Skipping LeWM head.")

    # 2. Train VAE Reward Head
    vae_path = Path(args.vae_checkpoint)
    if not args.skip_vae and vae_path.exists():
        vae_ckpt = torch.load(vae_path, map_location=device, weights_only=False)
        vae = MultiModalSensorVAE(in_dim=426, latent_channels=4, latent_length=32).to(device)
        vae.load_state_dict(vae_ckpt["model_state_dict"])
        vae.eval()

        head_vae = LatentRewardHead(in_dim=128, is_spatial=True, spatial_channels=4, spatial_length=32).to(device)
        head_vae, val_mse_vae = train_single_head(
            head=head_vae,
            encoder_fn=lambda o: vae.encode(o)[0],  # use mu
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            name="VAE",
        )
        torch.save(
            {"model_state_dict": head_vae.state_dict(), "val_mse": val_mse_vae, "is_spatial": True},
            ckpt_dir / "reward_head_vae.pt",
        )
        summary["vae_reward_head_val_mse"] = val_mse_vae
    else:
        print(f"Notice: VAE checkpoint not found at {vae_path}. Skipping VAE head.")

    summary_file = out_dir / "reward_heads_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nReward Heads training complete.")
    print(f"Saved artifacts to: {ckpt_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()
