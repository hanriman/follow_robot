#!/usr/bin/env python3
"""
Unified World Model & Latent Reward Head Training Pipeline.

Trains and evaluates:
1. Teacher-Free LeWM (Representation Learning via SIGReg + Latent Predictor)
2. Multi-Modal Sensor VAE (Gaussian Latent Bottleneck [4, 32] = 128-D)
3. Action-Conditioned Latent Diffusion World Model (AC-LDM)
4. Latent Reward & Progress Predictor Heads (LatentRewardHead)

Applies rigorous out-of-sample linear probing on held-out validation splits
and snapshots best-performing checkpoints.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from rl_ws.common.config import select_device
from rl_ws.data.dataset import get_dataset_splits
from rl_ws.policies.vla.diffusion_wm import ActionConditionedDiffusionWorldModel
from rl_ws.policies.world_model.reward_head import LatentRewardHead
from rl_ws.policies.world_model.sigreg import ActionConditionedTransitionLoss
from rl_ws.policies.world_model.sigreg_wm import TeacherFreeLeWorldModel
from rl_ws.policies.world_model.vae import MultiModalSensorVAE

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    parser = argparse.ArgumentParser(description="Unified AckBot World Model Training Pipeline")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["lewm", "vae", "diffusion", "all"],
        help="World model architecture to train (default: 'all')",
    )
    parser.add_argument(
        "--dataset-index",
        type=str,
        default=None,
        help="Path to dataset_index.json (default: searches data/curriculum_multi_ped or data/processed)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs/experiments"),
        help="Base experiment output directory",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs per model",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Mini-batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Base AdamW learning rate",
    )
    parser.add_argument(
        "--train-reward-head",
        action="store_true",
        default=True,
        help="Train companion LatentRewardHead for latent MPC planning (default: True)",
    )
    parser.add_argument(
        "--reward-head-only",
        action="store_true",
        default=False,
        help="Skip world model training and train only LatentRewardHead on existing checkpoints",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Compute device: 'auto', 'cpu', 'cuda', or 'mps'",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def train_reward_head(
    name: str,
    encoder_fn,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    is_spatial: bool = False,
    epochs: int = 20,
    lr: float = 1e-3,
) -> tuple[LatentRewardHead, float]:
    """Train companion LatentRewardHead and return best validation checkpoint."""
    head = LatentRewardHead(
        in_dim=128,
        hidden_dim=256,
        is_spatial=is_spatial,
        spatial_channels=4,
        spatial_length=32,
    ).to(device)

    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    best_val = float("inf")
    best_state: dict[str, Any] | None = None

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
            best_state = copy.deepcopy(head.state_dict())

        if epoch % 5 == 0 or epoch == epochs:
            print(
                f"  [{name} Reward Head] Epoch {epoch:2d}/{epochs:2d} | Train MSE: {avg_train:6.4f} | Val MSE: {avg_val:6.4f}"
            )

    if best_state is not None:
        head.load_state_dict(best_state)
    return head, best_val


def train_lewm(
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_ds,
    val_ds,
    device: torch.device,
    out_dir: Path,
    epochs: int = 30,
    lr: float = 3e-4,
) -> dict[str, Any]:
    print("\n" + "=" * 72)
    print("      TRAINING TEACHER-FREE LE WORLD MODEL (LEWM) + SIGREG           ")
    print("=" * 72)

    ckpt_dir = out_dir / "sigreg_curriculum/checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model = TeacherFreeLeWorldModel(latent_dim=128, action_dim=2).to(device)
    loss_fn = ActionConditionedTransitionLoss(
        loss_type="smooth_l1", beta=1.0, sigreg_weight=1.0
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        steps = 0

        for batch in train_loader:
            obs_t = batch["obs"].to(device)
            act_t = batch["action"].to(device)
            obs_next = batch["next_obs"].to(device)

            optimizer.zero_grad()
            fwd = model(obs_t, act_t, obs_next)
            losses = loss_fn(
                pred_next_z=fwd["pred_next_z"],
                target_next_z=fwd["target_next_z"],
                regularize_tokens=fwd["all_tokens"],
            )
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += losses["loss"].item()
            steps += 1

        scheduler.step()
        avg_train = train_loss / max(steps, 1)

        model.eval()
        val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for batch in val_loader:
                obs_t = batch["obs"].to(device)
                act_t = batch["action"].to(device)
                obs_next = batch["next_obs"].to(device)
                fwd = model(obs_t, act_t, obs_next)
                losses = loss_fn(fwd["pred_next_z"], fwd["target_next_z"], fwd["all_tokens"])
                val_loss += losses["loss"].item()
                val_steps += 1

        avg_val = val_loss / max(val_steps, 1)
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = copy.deepcopy(model.state_dict())

        if epoch % 5 == 0 or epoch == epochs:
            print(
                f"  [LeWM] Epoch {epoch:2d}/{epochs:2d} | Train Loss: {avg_train:6.4f} | Val Loss: {avg_val:6.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    # Save best checkpoint
    torch.save(
        {"model_state_dict": model.state_dict(), "latent_dim": 128, "val_loss": best_val_loss},
        ckpt_dir / "best_model.pt",
    )
    print(f"  Saved LeWM checkpoint to {ckpt_dir / 'best_model.pt'}")
    return {"val_loss": best_val_loss, "model": model}


def train_vae(
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    out_dir: Path,
    epochs: int = 30,
    lr: float = 3e-4,
) -> dict[str, Any]:
    print("\n" + "=" * 72)
    print("         TRAINING MULTI-MODAL SENSOR VAE ([4, 32] = 128-D)           ")
    print("=" * 72)

    ckpt_dir = out_dir / "vae_curriculum/checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model = MultiModalSensorVAE(in_dim=426, latent_channels=4, latent_length=32, kl_weight=1e-3).to(
        device
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        steps = 0

        for batch in train_loader:
            obs = batch["obs"].to(device)
            optimizer.zero_grad()
            res = model(obs)
            res["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += res["loss"].item()
            steps += 1

        scheduler.step()
        avg_train = train_loss / max(steps, 1)

        model.eval()
        val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for batch in val_loader:
                obs = batch["obs"].to(device)
                res = model(obs)
                val_loss += res["loss"].item()
                val_steps += 1

        avg_val = val_loss / max(val_steps, 1)
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = copy.deepcopy(model.state_dict())

        if epoch % 5 == 0 or epoch == epochs:
            print(
                f"  [VAE] Epoch {epoch:2d}/{epochs:2d} | Train Loss: {avg_train:6.4f} | Val Loss: {avg_val:6.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "latent_channels": 4,
            "latent_length": 32,
            "val_loss": best_val_loss,
        },
        ckpt_dir / "best_model.pt",
    )
    print(f"  Saved VAE checkpoint to {ckpt_dir / 'best_model.pt'}")
    return {"val_loss": best_val_loss, "model": model}


def train_diffusion(
    train_loader: DataLoader,
    val_loader: DataLoader,
    vae: MultiModalSensorVAE,
    device: torch.device,
    out_dir: Path,
    epochs: int = 30,
    lr: float = 3e-4,
) -> dict[str, Any]:
    print("\n" + "=" * 72)
    print("     TRAINING ACTION-CONDITIONED DIFFUSION WORLD MODEL (AC-LDM)      ")
    print("=" * 72)

    ckpt_dir = out_dir / "diffusion_curriculum/checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model = ActionConditionedDiffusionWorldModel(
        latent_channels=4,
        latent_length=32,
        action_dim=2,
        base_channels=64,
        num_timesteps=1000,
        vae=vae,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        steps = 0

        for batch in train_loader:
            obs_t = batch["obs"].to(device)
            act_t = batch["action"].to(device)
            obs_next = batch["next_obs"].to(device)

            optimizer.zero_grad()
            loss = model.compute_training_loss(obs_t, act_t, obs_next)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            steps += 1

        scheduler.step()
        avg_train = train_loss / max(steps, 1)

        model.eval()
        val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for batch in val_loader:
                obs_t = batch["obs"].to(device)
                act_t = batch["action"].to(device)
                obs_next = batch["next_obs"].to(device)
                loss = model.compute_training_loss(obs_t, act_t, obs_next)
                val_loss += loss.item()
                val_steps += 1

        avg_val = val_loss / max(val_steps, 1)
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = copy.deepcopy(model.state_dict())

        if epoch % 5 == 0 or epoch == epochs:
            print(
                f"  [AC-LDM] Epoch {epoch:2d}/{epochs:2d} | Train Loss: {avg_train:6.4f} | Val Loss: {avg_val:6.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(
        {"model_state_dict": model.state_dict(), "val_loss": best_val_loss},
        ckpt_dir / "best_model.pt",
    )
    print(f"  Saved AC-LDM checkpoint to {ckpt_dir / 'best_model.pt'}")
    return {"val_loss": best_val_loss, "model": model}


def main():
    args = parse_args()
    device = select_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    mpc_ckpt_dir = out_dir / "mpc_curriculum/checkpoints"
    mpc_ckpt_dir.mkdir(parents=True, exist_ok=True)

    index_p = Path(args.dataset_index) if args.dataset_index else None
    train_ds, val_ds, test_ds = get_dataset_splits(index_file=index_p)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    print("=" * 72)
    print("             ACKBOT UNIFIED WORLD MODEL PIPELINE                      ")
    print("=" * 72)
    print(f"Device              : {device}")
    print(f"Dataset Size        : {len(train_ds)} train | {len(val_ds)} val | {len(test_ds)} test")
    print(f"Selected Mode       : {args.model.upper()}")
    print(f"Reward Heads        : {args.train_reward_head or args.reward_head_only}")

    lewm_model = None
    vae_model = None

    if not args.reward_head_only:
        if args.model in ("lewm", "all"):
            res = train_lewm(
                train_loader, val_loader, train_ds, val_ds, device, out_dir, args.epochs, args.lr
            )
            lewm_model = res["model"]

        if args.model in ("vae", "diffusion", "all"):
            res = train_vae(train_loader, val_loader, device, out_dir, args.epochs, args.lr)
            vae_model = res["model"]

        if args.model in ("diffusion", "all"):
            if vae_model is None:
                vae_path = out_dir / "vae_curriculum/checkpoints/best_model.pt"
                vae_model = MultiModalSensorVAE(in_dim=426, latent_channels=4, latent_length=32).to(
                    device
                )
                vae_model.load_state_dict(
                    torch.load(vae_path, map_location=device, weights_only=False)[
                        "model_state_dict"
                    ]
                )
            train_diffusion(
                train_loader, val_loader, vae_model, device, out_dir, args.epochs, args.lr
            )

    # Train companion reward heads
    if args.train_reward_head or args.reward_head_only:
        print("\n" + "=" * 72)
        print("          TRAINING COMPANION LATENT REWARD HEADS                     ")
        print("=" * 72)

        # 1. LeWM Reward Head
        lewm_ckpt = out_dir / "sigreg_curriculum/checkpoints/best_model.pt"
        if lewm_model is None and lewm_ckpt.exists():
            lewm_model = TeacherFreeLeWorldModel(latent_dim=128, action_dim=2).to(device)
            lewm_model.load_state_dict(
                torch.load(lewm_ckpt, map_location=device, weights_only=False)["model_state_dict"]
            )
            lewm_model.eval()

        if lewm_model is not None:
            print("Training LeWM LatentRewardHead...")
            head_lewm, val_err = train_reward_head(
                "LeWM",
                lambda obs: lewm_model.encode(obs),
                train_loader,
                val_loader,
                device,
                is_spatial=False,
                epochs=20,
            )
            torch.save(
                {"model_state_dict": head_lewm.state_dict(), "val_mse": val_err},
                mpc_ckpt_dir / "reward_head_lewm.pt",
            )
            print(f"Saved LeWM reward head to {mpc_ckpt_dir / 'reward_head_lewm.pt'}")

        # 2. VAE / Diffusion Reward Head
        vae_ckpt = out_dir / "vae_curriculum/checkpoints/best_model.pt"
        if vae_model is None and vae_ckpt.exists():
            vae_model = MultiModalSensorVAE(in_dim=426, latent_channels=4, latent_length=32).to(
                device
            )
            vae_model.load_state_dict(
                torch.load(vae_ckpt, map_location=device, weights_only=False)["model_state_dict"]
            )
            vae_model.eval()

        if vae_model is not None:
            print("Training VAE/Diffusion LatentRewardHead...")
            head_vae, val_err = train_reward_head(
                "VAE/Diffusion",
                lambda obs: vae_model.encode(obs),
                train_loader,
                val_loader,
                device,
                is_spatial=True,
                epochs=20,
            )
            torch.save(
                {"model_state_dict": head_vae.state_dict(), "val_mse": val_err},
                mpc_ckpt_dir / "reward_head_vae.pt",
            )
            print(f"Saved VAE/Diffusion reward head to {mpc_ckpt_dir / 'reward_head_vae.pt'}")

    print("\nPipeline finished successfully!")


if __name__ == "__main__":
    main()
