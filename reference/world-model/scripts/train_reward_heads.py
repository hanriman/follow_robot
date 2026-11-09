#!/usr/bin/env python3
"""
Train Latent Reward Heads for Latent Model Predictive Control (MPC).

Trains symmetric reward predictors mapping LeWM and Spatial VAE latent representations
to the composite control objective [reward, forward_vel, dist_to_center] on the offline dataset.

Outputs (Solid 1-Directory Rule):
- Checkpoints: outputs/experiments/mpc/checkpoints/reward_head_lewm.pt
               outputs/experiments/mpc/checkpoints/reward_head_vae.pt
- Metrics:     outputs/experiments/mpc/metrics/reward_heads_eval.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from wm_control.config import fail_loud, get_experiment_dir
from wm_control.data.dataset import get_dataset_splits, get_latent_dataset_splits
from wm_control.models.reward_head import LatentRewardHead
from wm_control.models.sigreg_wm import TeacherFreeLeWorldModel
from wm_control.utils.device import get_device
from wm_control.utils.seed import seed_everything


def train_reward_heads(
    epochs: int = 5,
    batch_size: int = 256,
    lr: float = 1e-3,
    device_str: str | None = None,
    seed: int = 42,
) -> tuple[Path, Path]:
    seed_everything(seed)
    device = get_device(device_str)

    mpc_dir = get_experiment_dir("mpc", create_if_missing=True)
    ckpt_dir = mpc_dir / "checkpoints"
    metrics_dir = mpc_dir / "metrics"
    logs_dir = mpc_dir / "logs"

    log_file = logs_dir / "train_rewards.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger("TrainRewardHeads")
    logger.info("Starting Latent Reward Head Training on device: %s", device)

    # 1. Load LeWM
    sigreg_ckpt = get_experiment_dir("sigreg") / "checkpoints" / "best_sigreg.pt"
    if not sigreg_ckpt.exists():
        fail_loud(sigreg_ckpt, "train_reward_heads", "python scripts/train_sigreg.py")

    lewm_state = torch.load(sigreg_ckpt, map_location=device)
    lewm = TeacherFreeLeWorldModel(
        latent_dim=lewm_state.get("latent_dim", 128),
        action_dim=lewm_state.get("action_dim", 3),
        in_channels=3,
    ).to(device)
    lewm.load_state_dict(lewm_state["model_state"])
    lewm.eval()
    for p in lewm.parameters():
        p.requires_grad = False

    # Datasets
    train_ds, val_ds, _ = get_dataset_splits(augment_train=False)
    raw_train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    raw_val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    latent_train_ds, latent_val_ds, _ = get_latent_dataset_splits(augment_train=False)
    latent_train_loader = DataLoader(latent_train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    latent_val_loader = DataLoader(latent_val_ds, batch_size=batch_size, shuffle=False)

    head_lewm = LatentRewardHead(in_dim=128, is_spatial=False).to(device)
    head_vae = LatentRewardHead(is_spatial=True, spatial_channels=4).to(device)

    opt_lewm = torch.optim.AdamW(head_lewm.parameters(), lr=lr, weight_decay=1e-4)
    opt_vae = torch.optim.AdamW(head_vae.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    logger.info("Training LeWM Reward Head (5 epochs)...")
    for ep in range(1, epochs + 1):
        head_lewm.train()
        loss_sum = 0.0
        for batch in tqdm(raw_train_loader, desc=f"LeWM Reward Ep {ep}/{epochs}", leave=False):
            obs = batch["obs"].to(device)
            # targets: [reward, forward_vel (index 6), dist_to_center (index 9)]
            rew = batch["reward"].to(device).unsqueeze(-1)
            telem = batch["telemetry"].to(device)
            fwd_vel = telem[:, 6:7]
            dist_center = telem[:, 9:10]
            targets = torch.cat([rew, fwd_vel, dist_center], dim=-1)

            opt_lewm.zero_grad(set_to_none=True)
            with torch.no_grad():
                z = lewm.encode(obs)
            preds = head_lewm(z)
            loss = criterion(preds, targets)
            loss.backward()
            opt_lewm.step()
            loss_sum += loss.item()

    logger.info("Training Spatial VAE Reward Head (5 epochs)...")
    for ep in range(1, epochs + 1):
        head_vae.train()
        loss_sum = 0.0
        for batch in tqdm(latent_train_loader, desc=f"VAE Reward Ep {ep}/{epochs}", leave=False):
            z_t = batch["z_t"].to(device)
            rew = batch["reward"].to(device).unsqueeze(-1)
            telem = batch["telemetry"].to(device)
            fwd_vel = telem[:, 6:7]
            dist_center = telem[:, 9:10]
            targets = torch.cat([rew, fwd_vel, dist_center], dim=-1)

            opt_vae.zero_grad(set_to_none=True)
            preds = head_vae(z_t)
            loss = criterion(preds, targets)
            loss.backward()
            opt_vae.step()
            loss_sum += loss.item()

    # Evaluate validation losses
    head_lewm.eval()
    head_vae.eval()
    val_loss_lewm, val_loss_vae = 0.0, 0.0
    with torch.no_grad():
        for b_raw, b_lat in zip(raw_val_loader, latent_val_loader, strict=False):
            rew = b_raw["reward"].to(device).unsqueeze(-1)
            telem = b_raw["telemetry"].to(device)
            targets = torch.cat([rew, telem[:, 6:7], telem[:, 9:10]], dim=-1)

            z_lewm = lewm.encode(b_raw["obs"].to(device))
            val_loss_lewm += criterion(head_lewm(z_lewm), targets).item()

            z_vae = b_lat["z_t"].to(device)
            val_loss_vae += criterion(head_vae(z_vae), targets).item()

    val_loss_lewm /= len(raw_val_loader)
    val_loss_vae /= len(latent_val_loader)

    logger.info("Reward Head Validation MSE -> LeWM: %.4f | Spatial VAE: %.4f", val_loss_lewm, val_loss_vae)

    # Save checkpoints
    ckpt_lewm = ckpt_dir / "reward_head_lewm.pt"
    ckpt_vae = ckpt_dir / "reward_head_vae.pt"
    torch.save({"model_state": head_lewm.state_dict(), "val_mse": val_loss_lewm}, ckpt_lewm)
    torch.save({"model_state": head_vae.state_dict(), "val_mse": val_loss_vae}, ckpt_vae)

    metrics_file = metrics_dir / "reward_heads_eval.json"
    with open(metrics_file, "w") as f:
        json.dump(
            {
                "val_mse_lewm": val_loss_lewm,
                "val_mse_vae": val_loss_vae,
                "epochs": epochs,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
            f,
            indent=2,
        )

    logger.info("Reward heads saved to %s and %s", ckpt_lewm, ckpt_vae)
    return ckpt_lewm, ckpt_vae


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Latent Reward Heads.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    train_reward_heads(epochs=args.epochs, batch_size=args.batch_size, device_str=args.device)


if __name__ == "__main__":
    main()
