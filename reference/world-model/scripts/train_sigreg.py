#!/usr/bin/env python3
"""
Training Runner for Teacher-Free Le World Model (LeWM) with SIGReg.

Jointly trains visual encoder and action-conditioned transition dynamics end-to-end
without EMA target encoders, stop-gradient operators, or negative pairs, preventing
representation collapse via Sketched Isotropic Gaussian Regularization (SIGReg).

Outputs (Solid 1-Directory Rule):
- Checkpoints: outputs/experiments/sigreg/checkpoints/best_sigreg.pt, latest_sigreg.pt
- Figures:     outputs/experiments/sigreg/figures/effective_rank_evolution.png
- Metrics:     outputs/experiments/sigreg/metrics/eval_metrics.json
- Logs:        outputs/experiments/sigreg/logs/train.log
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from wm_control.config import DATA_PROCESSED_DIR, fail_loud, get_experiment_dir
from wm_control.data.dataset import get_dataset_splits
from wm_control.losses.sigreg import ActionConditionedTransitionLoss
from wm_control.metrics.representation import (
    compute_effective_rank,
    compute_gaussianity_diagnostics,
)
from wm_control.models.sigreg_wm import TeacherFreeLeWorldModel
from wm_control.utils.device import get_device
from wm_control.utils.seed import seed_everything

matplotlib.use("Agg")


def train_sigreg(
    epochs: int = 15,
    batch_size: int = 256,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    sigreg_weight: float = 1.0,
    num_projections: int = 1024,
    latent_dim: int = 128,
    action_dim: int = 3,
    predictor_hidden_dim: int = 512,
    predictor_layers: int = 3,
    rollout_horizon: int = 5,
    device_str: str | None = None,
    seed: int = 42,
) -> Path:
    """
    Train Teacher-Free Le World Model with SIGReg regularizer.
    """
    seed_everything(seed)
    device = get_device(device_str)

    # Verify dataset split index exists
    index_file = DATA_PROCESSED_DIR / "dataset_index.json"
    if not index_file.exists():
        fail_loud(
            missing_resource=index_file,
            caller_context="train_sigreg",
            remediation_cmd="python scripts/harvest_data.py --num-transitions 50000",
        )

    # Resolve deterministic canonical output directories
    exp_dir = get_experiment_dir("sigreg", create_if_missing=True)
    ckpt_dir = exp_dir / "checkpoints"
    fig_dir = exp_dir / "figures"
    metrics_dir = exp_dir / "metrics"
    logs_dir = exp_dir / "logs"

    # Logger setup
    log_file = logs_dir / "train.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger("TrainSIGReg")
    logger.info(f"Starting Teacher-Free LeWM (SIGReg) Training on device: {device}")
    logger.info(f"Experiment Directory: {exp_dir}")

    # Load dataset splits (identical 40k/5k/5k split as generative model)
    train_ds, val_ds, _ = get_dataset_splits(augment_train=True)
    logger.info(f"Loaded Dataset: {len(train_ds):,} train samples, {len(val_ds):,} val samples")

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

    # Initialize model
    model = TeacherFreeLeWorldModel(
        latent_dim=latent_dim,
        action_dim=action_dim,
        predictor_hidden_dim=predictor_hidden_dim,
        predictor_layers=predictor_layers,
        in_channels=3,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Teacher-Free LeWM Initialized: {total_params:,} trainable parameters")

    criterion = ActionConditionedTransitionLoss(
        loss_type="smooth_l1",
        sigreg_weight=sigreg_weight,
        num_projections=num_projections,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_trans_loss = float("inf")
    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_trans_loss": [],
        "train_sigreg_loss": [],
        "val_loss": [],
        "val_trans_loss": [],
        "val_sigreg_loss": [],
        "val_effective_rank": [],
        "val_rank_ratio": [],
        "val_rollout_l1": [],
    }

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_total_loss = 0.0
        train_trans_loss_sum = 0.0
        train_sigreg_loss_sum = 0.0
        num_train_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]", leave=False)
        for batch in pbar:
            obs_t = batch["obs"].to(device)
            next_obs = batch["next_obs"].to(device)
            action_t = batch["action"].to(device)

            optimizer.zero_grad(set_to_none=True)

            # End-to-end forward pass through single encoder
            fwd_dict = model(obs_t=obs_t, action_t=action_t, obs_next=next_obs)
            pred_next_z = fwd_dict["pred_next_z"]
            target_next_z = fwd_dict["target_next_z"]
            all_tokens = fwd_dict["all_tokens"]

            loss_dict = criterion(
                pred_next_z=pred_next_z,
                target_next_z=target_next_z,
                regularize_tokens=all_tokens,
            )

            loss = loss_dict["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_total_loss += loss.item()
            train_trans_loss_sum += loss_dict["transition_loss"].item()
            train_sigreg_loss_sum += loss_dict["sigreg_loss"].item()
            num_train_batches += 1

            pbar.set_postfix(
                trans=f"{loss_dict['transition_loss'].item():.4f}",
                sigreg=f"{loss_dict['sigreg_loss'].item():.4f}",
            )

        scheduler.step()
        avg_train_loss = train_total_loss / max(1, num_train_batches)
        avg_train_trans = train_trans_loss_sum / max(1, num_train_batches)
        avg_train_sigreg = train_sigreg_loss_sum / max(1, num_train_batches)

        # Validation Loop
        model.eval()
        val_total_loss = 0.0
        val_trans_sum = 0.0
        val_sigreg_sum = 0.0
        num_val_batches = 0
        val_latents_list: list[torch.Tensor] = []

        with torch.no_grad():
            for val_batch in val_loader:
                obs_t_val = val_batch["obs"].to(device)
                next_obs_val = val_batch["next_obs"].to(device)
                action_t_val = val_batch["action"].to(device)

                fwd_val = model(obs_t=obs_t_val, action_t=action_t_val, obs_next=next_obs_val)
                val_loss_dict = criterion(
                    pred_next_z=fwd_val["pred_next_z"],
                    target_next_z=fwd_val["target_next_z"],
                    regularize_tokens=fwd_val["all_tokens"],
                )

                val_total_loss += val_loss_dict["loss"].item()
                val_trans_sum += val_loss_dict["transition_loss"].item()
                val_sigreg_sum += val_loss_dict["sigreg_loss"].item()
                num_val_batches += 1

                if len(val_latents_list) < 4:  # Accumulate ~1000 samples for rank diagnostics
                    val_latents_list.append(fwd_val["z_t"].cpu())

        avg_val_loss = val_total_loss / max(1, num_val_batches)
        avg_val_trans = val_trans_sum / max(1, num_val_batches)
        avg_val_sigreg = val_sigreg_sum / max(1, num_val_batches)

        # Quantitative Dimensionality & Health Metrics
        accum_val_latents = torch.cat(val_latents_list, dim=0)
        rank_metrics = compute_effective_rank(accum_val_latents)
        gauss_metrics = compute_gaussianity_diagnostics(accum_val_latents)

        # Multi-Step Rollout Evaluation on validation batch over horizon H
        val_sample = next(iter(val_loader))
        init_obs = val_sample["obs"][:16].to(device)
        actions_seq = val_sample["action"][:16].unsqueeze(1).repeat(1, rollout_horizon, 1).to(device)
        target_z1 = model.encode(val_sample["next_obs"][:16].to(device))

        with torch.no_grad():
            unrolled_z = model.rollout(init_obs, actions_seq)  # [16, H, D]
            step1_l1 = torch.nn.functional.l1_loss(unrolled_z[:, 0, :], target_z1).item()

        history["train_loss"].append(avg_train_loss)
        history["train_trans_loss"].append(avg_train_trans)
        history["train_sigreg_loss"].append(avg_train_sigreg)
        history["val_loss"].append(avg_val_loss)
        history["val_trans_loss"].append(avg_val_trans)
        history["val_sigreg_loss"].append(avg_val_sigreg)
        history["val_effective_rank"].append(rank_metrics["effective_rank"])
        history["val_rank_ratio"].append(rank_metrics["rank_ratio"])
        history["val_rollout_l1"].append(step1_l1)

        logger.info(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Val Trans: {avg_val_trans:.4f} | "
            f"SIGReg: {avg_val_sigreg:.4f} | "
            f"EffRank: {rank_metrics['effective_rank']:.1f}/{latent_dim} ({rank_metrics['rank_ratio'] * 100:.1f}%) | "
            f"MeanNorm: {gauss_metrics['mean_norm']:.3f} | "
            f"VarDev: {gauss_metrics['var_deviation']:.3f}"
        )

        # Checkpoint: Latest
        latest_ckpt = ckpt_dir / "latest_sigreg.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "latent_dim": latent_dim,
                "val_trans_loss": avg_val_trans,
                "val_effective_rank": rank_metrics["effective_rank"],
            },
            latest_ckpt,
        )

        # Checkpoint: Best
        if avg_val_trans < best_val_trans_loss:
            best_val_trans_loss = avg_val_trans
            best_ckpt = ckpt_dir / "best_sigreg.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "latent_dim": latent_dim,
                    "action_dim": action_dim,
                    "val_trans_loss": avg_val_trans,
                    "val_effective_rank": rank_metrics["effective_rank"],
                },
                best_ckpt,
            )
            logger.info(
                f"[*] New best Teacher-Free LeWM saved with Val Trans Loss: {best_val_trans_loss:.4f} -> {best_ckpt}"
            )

    total_time = time.time() - start_time
    logger.info(f"\nTraining Complete in {total_time:.1f}s. Best Val Trans Loss: {best_val_trans_loss:.4f}.")

    # Plot Effective Rank & Loss Evolution Figure
    _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ep_axis = range(1, epochs + 1)

    ax1.plot(ep_axis, history["train_trans_loss"], label="Train Transition Loss", marker="o")
    ax1.plot(ep_axis, history["val_trans_loss"], label="Val Transition Loss", marker="s")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Transition Loss (Smooth L1)")
    ax1.set_title("Dynamics Predictor Convergence")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(ep_axis, history["val_effective_rank"], label="Validation Effective Rank", color="green", marker="^")
    ax2.axhline(y=latent_dim, color="red", linestyle="--", label=f"Max Capacity (D={latent_dim})")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Participation Ratio (Effective Rank)")
    ax2.set_title("SIGReg Anti-Collapse Verification")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    fig_path = fig_dir / "effective_rank_evolution.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()
    logger.info(f"Saved diagnostic figure to: {fig_path}")

    # Metrics JSON
    metrics_file = metrics_dir / "eval_metrics.json"
    summary_metrics = {
        "best_val_trans_loss": best_val_trans_loss,
        "final_val_trans_loss": history["val_trans_loss"][-1],
        "final_val_sigreg_loss": history["val_sigreg_loss"][-1],
        "final_effective_rank": history["val_effective_rank"][-1],
        "final_rank_ratio": history["val_rank_ratio"][-1],
        "total_epochs": epochs,
        "batch_size": batch_size,
        "training_time_seconds": total_time,
        "history": history,
    }
    with open(metrics_file, "w") as f:
        json.dump(summary_metrics, f, indent=2)

    return best_ckpt


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Teacher-Free Le World Model with SIGReg.")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size per step.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--sigreg-weight", type=float, default=1.0, help="SIGReg penalty weight.")
    parser.add_argument("--latent-dim", type=int, default=128, help="Latent dimension D.")
    parser.add_argument("--device", type=str, default=None, help="Device override (cuda, mps, cpu).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    train_sigreg(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        sigreg_weight=args.sigreg_weight,
        latent_dim=args.latent_dim,
        device_str=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
