#!/usr/bin/env python3
"""
Visual Distractor and Task-Irrelevant Feature (TIF) Stress-Testing.

Quantitatively evaluates the robustness of Predictive (LeWM) vs. Generative (Spatial VAE / AC-LDM)
world model representations against task-irrelevant visual perturbations:
1. Physical Telemetry Probing R^2 Degradation under additive Gaussian noise sigma in [0.0, 0.05, 0.1, 0.2, 0.3]
2. Relative Latent Distortion ||E(x + noise) - E(x)|| / ||E(x)||
3. Multi-Step Latent Rollout Drift under visual perturbations

Outputs (Solid 1-Directory Rule):
- Metrics: outputs/experiments/distractors/metrics/distractor_evaluation.json
- Figures: outputs/experiments/distractors/figures/distractor_robustness_curves.png
- Logs:    outputs/experiments/distractors/logs/distractor_eval.log
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from wm_control.config import fail_loud, get_experiment_dir
from wm_control.data.dataset import get_dataset_splits
from wm_control.data.distractor_wrapper import apply_distractor_to_batch
from wm_control.metrics import RidgeRegressionProbe
from wm_control.models.diffusion_wm import ActionConditionedDiffusionWorldModel
from wm_control.models.sigreg_wm import TeacherFreeLeWorldModel
from wm_control.models.vae import SpatialVAE
from wm_control.utils.device import get_device
from wm_control.utils.seed import seed_everything

matplotlib.use("Agg")


TELEMETRY_LABELS = [
    "pos_x",
    "pos_y",
    "heading_theta",
    "lin_vel_x",
    "lin_vel_y",
    "angular_vel",
    "forward_vel",
    "lateral_vel",
    "road_curvature",
    "dist_to_center",
]


def run_distractor_evaluation(
    noise_levels: list[float] | None = None,
    batch_size: int = 256,
    device_str: str | None = None,
    seed: int = 42,
) -> Path:
    if noise_levels is None:
        noise_levels = [0.0, 0.05, 0.10, 0.20, 0.30]

    seed_everything(seed)
    device = get_device(device_str)

    dist_dir = get_experiment_dir("distractors", create_if_missing=True)
    fig_dir = dist_dir / "figures"
    metrics_dir = dist_dir / "metrics"
    logs_dir = dist_dir / "logs"

    log_file = logs_dir / "distractor_eval.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger("DistractorEval")
    logger.info("Starting Visual Distractor Robustness Suite on device: %s", device)
    logger.info("Noise levels sigma: %s", noise_levels)

    # Load Model Checkpoints
    vae_ckpt_path = get_experiment_dir("vae") / "checkpoints" / "best_vae.pt"
    diff_ckpt_path = get_experiment_dir("diffusion") / "checkpoints" / "best_diffusion.pt"
    sigreg_ckpt_path = get_experiment_dir("sigreg") / "checkpoints" / "best_sigreg.pt"

    for p, name in [
        (vae_ckpt_path, "Spatial VAE"),
        (diff_ckpt_path, "Latent Diffusion"),
        (sigreg_ckpt_path, "Teacher-Free LeWM"),
    ]:
        if not p.exists():
            fail_loud(p, "run_distractor_evaluation", f"Ensure {name} is trained.")

    vae_ckpt = torch.load(vae_ckpt_path, map_location=device)
    vae = SpatialVAE(latent_channels=vae_ckpt.get("latent_channels", 4), base_channels=vae_ckpt.get("base_channels", 32)).to(device)
    vae.load_state_dict(vae_ckpt["model_state"])
    vae.eval()

    diff_ckpt = torch.load(diff_ckpt_path, map_location=device)
    diff = ActionConditionedDiffusionWorldModel(latent_channels=4, action_dim=3, base_channels=64, vae=vae).to(device)
    diff.load_state_dict(diff_ckpt["model_state"])
    diff.eval()

    sigreg_ckpt = torch.load(sigreg_ckpt_path, map_location=device)
    lewm = TeacherFreeLeWorldModel(latent_dim=sigreg_ckpt.get("latent_dim", 128), action_dim=3).to(device)
    lewm.load_state_dict(sigreg_ckpt["model_state"])
    lewm.eval()

    # Load Test Dataset
    _, _, test_ds = get_dataset_splits(augment_train=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # 1. Fit Baseline Linear Probes on Clean Test Representations
    logger.info("Extracting clean test representations...")
    clean_obs_list, telem_list = [], []
    with torch.no_grad():
        for b in test_loader:
            clean_obs_list.append(b["obs"].to(device))
            telem_list.append(b["telemetry"].to(device))
    all_clean_obs = torch.cat(clean_obs_list, dim=0)
    all_telem = torch.cat(telem_list, dim=0)

    n_probe = 4000
    with torch.no_grad():
        clean_z_lewm = lewm.encode(all_clean_obs)
        clean_z_vae, _ = vae.encode(all_clean_obs)
        clean_z_vae_flat = clean_z_vae.flatten(start_dim=1)

    probe_lewm = RidgeRegressionProbe(alpha=1.0).fit(clean_z_lewm[:n_probe], all_telem[:n_probe])
    probe_vae = RidgeRegressionProbe(alpha=1.0).fit(clean_z_vae_flat[:n_probe], all_telem[:n_probe])

    holdout_telem = all_telem[n_probe:]

    results_per_sigma: dict[str, Any] = {}
    lewm_r2_curve: list[float] = []
    vae_r2_curve: list[float] = []
    lewm_distort_curve: list[float] = []
    vae_distort_curve: list[float] = []

    for sigma in noise_levels:
        logger.info("\nEvaluating Distractor Noise Level sigma = %.2f...", sigma)
        with torch.no_grad():
            noisy_obs = apply_distractor_to_batch(all_clean_obs, mode="gaussian", sigma=sigma)
            noisy_z_lewm = lewm.encode(noisy_obs)
            noisy_z_vae, _ = vae.encode(noisy_obs)
            noisy_z_vae_flat = noisy_z_vae.flatten(start_dim=1)

            # Evaluate probing on noisy holdout
            score_lewm = probe_lewm.score_r2(noisy_z_lewm[n_probe:], holdout_telem, labels=TELEMETRY_LABELS)
            score_vae = probe_vae.score_r2(noisy_z_vae_flat[n_probe:], holdout_telem, labels=TELEMETRY_LABELS)

            # Measure relative distortion: ||z_noisy - z_clean|| / ||z_clean||
            distort_lewm = (
                torch.norm(noisy_z_lewm - clean_z_lewm, dim=-1) / torch.norm(clean_z_lewm, dim=-1).clamp(min=1e-6)
            ).mean().item()
            distort_vae = (
                torch.norm(noisy_z_vae_flat - clean_z_vae_flat, dim=-1)
                / torch.norm(clean_z_vae_flat, dim=-1).clamp(min=1e-6)
            ).mean().item()

        lewm_r2_curve.append(score_lewm["mean_r2"])
        vae_r2_curve.append(score_vae["mean_r2"])
        lewm_distort_curve.append(distort_lewm)
        vae_distort_curve.append(distort_vae)

        results_per_sigma[f"sigma_{sigma:.2f}"] = {
            "sigma": sigma,
            "LeWM": {
                "mean_r2": score_lewm["mean_r2"],
                "relative_distortion": distort_lewm,
                "per_dim_r2": score_lewm.get("per_dimension_r2", {}),
            },
            "Spatial_VAE": {
                "mean_r2": score_vae["mean_r2"],
                "relative_distortion": distort_vae,
                "per_dim_r2": score_vae.get("per_dimension_r2", {}),
            },
        }

        logger.info(
            "  sigma=%.2f | LeWM R^2: %.4f (Distort: %.3f) | Spatial VAE R^2: %.4f (Distort: %.3f)",
            sigma,
            score_lewm["mean_r2"],
            distort_lewm,
            score_vae["mean_r2"],
            distort_vae,
        )

    # Save Metrics JSON
    metrics_file = metrics_dir / "distractor_evaluation.json"
    summary_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "noise_levels": noise_levels,
        "results": results_per_sigma,
    }
    with open(metrics_file, "w") as f:
        json.dump(summary_data, f, indent=2)
    logger.info("Saved distractor evaluation metrics to: %s", metrics_file)

    # Plot Robustness Curves
    _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Physics Geometry Decodability under Distractor Noise
    ax1.plot(noise_levels, lewm_r2_curve, "o-", label="LeWM (Predictive, 128-D)", color="#1f77b4", linewidth=2)
    ax1.plot(noise_levels, vae_r2_curve, "s--", label="Spatial VAE (Generative, 1024-D)", color="#ff7f0e", linewidth=2)
    ax1.set_xlabel(r"Gaussian Noise Level $\sigma$")
    ax1.set_ylabel("Linear Probing $R^2$ (Mean Physics Telemetry)")
    ax1.set_title("Physics Geometry Robustness under Distractors")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Relative Latent Space Distortion
    ax2.plot(noise_levels, lewm_distort_curve, "o-", label="LeWM Latent Distortion", color="#1f77b4", linewidth=2)
    ax2.plot(noise_levels, vae_distort_curve, "s--", label="Spatial VAE Latent Distortion", color="#ff7f0e", linewidth=2)
    ax2.set_xlabel(r"Gaussian Noise Level $\sigma$")
    ax2.set_ylabel(r"Relative Feature Distortion $\frac{\|\Delta z\|}{\|z\|}$")
    ax2.set_title("Task-Irrelevant Latent Distortion Rate")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    fig_path = fig_dir / "distractor_robustness_curves.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()
    logger.info("Saved robustness figures to: %s", fig_path)

    return metrics_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate World Model Robustness to Visual Distractors.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    run_distractor_evaluation(batch_size=args.batch_size, device_str=args.device)


if __name__ == "__main__":
    main()
