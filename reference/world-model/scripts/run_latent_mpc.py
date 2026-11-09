#!/usr/bin/env python3
"""
Closed-Loop Latent Model Predictive Control (MPC) Benchmark on CarRacing-v3.

Executes live receding-horizon trajectory optimization using learned world models
(Predictive LeWM vs. Generative AC-LDM) paired with the Cross-Entropy Method (CEM)
and compares against Random and Waypoint Heuristic baselines across multi-seed runs.

Outputs (Solid 1-Directory Rule):
- Metrics: outputs/experiments/mpc/metrics/mpc_benchmark_results.json
- Figures: outputs/experiments/mpc/figures/mpc_cumulative_rewards.png
           outputs/experiments/mpc/figures/mpc_planning_latency.png
- Logs:    outputs/experiments/mpc/logs/mpc_eval.log
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
import numpy as np
import torch

from wm_control.config import fail_loud, get_experiment_dir
from wm_control.controllers.cem_planner import CEMPlanner
from wm_control.data.env_wrapper import make_carracing_env
from wm_control.data.heuristic_policy import CarRacingHeuristicPolicy
from wm_control.models.diffusion_wm import ActionConditionedDiffusionWorldModel
from wm_control.models.reward_head import LatentRewardHead
from wm_control.models.sigreg_wm import TeacherFreeLeWorldModel
from wm_control.models.vae import SpatialVAE
from wm_control.utils.device import get_device

matplotlib.use("Agg")


def run_episode(
    controller_name: str,
    env_seed: int,
    max_steps: int = 200,
    lewm_model: TeacherFreeLeWorldModel | None = None,
    head_lewm: LatentRewardHead | None = None,
    diff_model: ActionConditionedDiffusionWorldModel | None = None,
    vae_model: SpatialVAE | None = None,
    head_vae: LatentRewardHead | None = None,
    planner_horizon: int = 5,
    planner_pop: int = 50,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Execute a single live closed-loop control episode in CarRacing-v3."""
    env = make_carracing_env(seed=env_seed)
    obs_t, info = env.reset(seed=env_seed)

    planner = None
    if controller_name in ("LeWM_MPC", "AC_LDM_MPC"):
        planner = CEMPlanner(
            horizon=planner_horizon,
            pop_size=planner_pop,
            num_elites=max(2, planner_pop // 10),
            num_iters=3,
            device=device,
        )

    heuristic = CarRacingHeuristicPolicy(noise_scale=0.0) if controller_name == "Oracle" else None

    total_reward = 0.0
    latencies: list[float] = []
    forward_vels: list[float] = []
    off_track_count = 0
    steps_taken = 0

    for _step in range(max_steps):
        steps_taken += 1
        t0 = time.perf_counter()

        if controller_name == "Random":
            # Uniform random in steering [-1, 1], gas [0, 1], brake [0, 1]
            act = np.array([
                np.random.uniform(-1.0, 1.0),
                np.random.uniform(0.0, 1.0),
                np.random.uniform(0.0, 0.3),
            ], dtype=np.float32)
            latencies.append((time.perf_counter() - t0) * 1000.0)

        elif controller_name == "Oracle":
            act = heuristic.select_action(env)
            latencies.append((time.perf_counter() - t0) * 1000.0)

        elif controller_name == "LeWM_MPC":
            obs_tensor = torch.from_numpy(obs_t).unsqueeze(0).to(device)
            with torch.no_grad():
                z_0 = lewm_model.encode(obs_tensor)

            def lewm_rollout(z_init: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
                # z_init: [N, 128], actions: [N, H, 3]
                traj = []
                curr = z_init
                for h in range(actions.shape[1]):
                    curr = lewm_model.predict_next(curr, actions[:, h, :])
                    traj.append(curr)
                return torch.stack(traj, dim=1)

            def lewm_score(unrolled_z: torch.Tensor) -> torch.Tensor:
                # unrolled_z: [N, H, 128]
                return head_lewm.compute_control_score(unrolled_z)

            act_t = planner.plan(z_0, lewm_rollout, lewm_score)
            act = act_t.cpu().numpy()
            latencies.append((time.perf_counter() - t0) * 1000.0)

        elif controller_name == "AC_LDM_MPC":
            obs_tensor = torch.from_numpy(obs_t).unsqueeze(0).to(device)
            with torch.no_grad():
                z_0, _ = vae_model.encode(obs_tensor)

            def ldm_rollout(z_init: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
                traj = []
                curr = z_init
                for h in range(actions.shape[1]):
                    # Accelerated 1-step DDIM rollout for online planning
                    curr = diff_model.predict_next(curr, actions[:, h, :], ddim_steps=1)
                    traj.append(curr)
                return torch.stack(traj, dim=1)

            def ldm_score(unrolled_z: torch.Tensor) -> torch.Tensor:
                return head_vae.compute_control_score(unrolled_z)

            act_t = planner.plan(z_0, ldm_rollout, ldm_score)
            act = act_t.cpu().numpy()
            latencies.append((time.perf_counter() - t0) * 1000.0)
        else:
            raise ValueError(f"Unknown controller: {controller_name}")

        obs_next, rew, term, trunc, info = env.step(act)
        total_reward += rew
        obs_t = obs_next

        # Telemetry stats
        telem = info.get("telemetry", {})
        fwd_v = float(telem.get("forward_velocity", 0.0)) if isinstance(telem, dict) else 0.0
        forward_vels.append(fwd_v)
        if rew < -0.05:
            off_track_count += 1

        if term or trunc:
            break

    env.close()
    mean_lat = float(np.mean(latencies)) if latencies else 0.0
    mean_fwd = float(np.mean(forward_vels)) if forward_vels else 0.0
    off_track_ratio = float(off_track_count / max(1, steps_taken))

    return {
        "controller": controller_name,
        "seed": env_seed,
        "steps": steps_taken,
        "cumulative_reward": float(total_reward),
        "mean_latency_ms": mean_lat,
        "mean_forward_vel": mean_fwd,
        "off_track_ratio": off_track_ratio,
    }


def run_mpc_benchmark(
    seeds: list[int] | None = None,
    max_steps: int = 200,
    horizon: int = 5,
    pop_size: int = 50,
    device_str: str | None = None,
) -> Path:
    if seeds is None:
        seeds = [101, 102, 103, 104, 105]
    device = get_device(device_str)

    mpc_dir = get_experiment_dir("mpc", create_if_missing=True)
    ckpt_dir = mpc_dir / "checkpoints"
    fig_dir = mpc_dir / "figures"
    metrics_dir = mpc_dir / "metrics"
    logs_dir = mpc_dir / "logs"

    log_file = logs_dir / "mpc_eval.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger("LatentMPC")
    logger.info("Initializing Closed-Loop Latent MPC Benchmark across seeds: %s", seeds)

    # Load World Models & Reward Heads
    vae_ckpt_path = get_experiment_dir("vae") / "checkpoints" / "best_vae.pt"
    diff_ckpt_path = get_experiment_dir("diffusion") / "checkpoints" / "best_diffusion.pt"
    sigreg_ckpt_path = get_experiment_dir("sigreg") / "checkpoints" / "best_sigreg.pt"
    head_lewm_path = ckpt_dir / "reward_head_lewm.pt"
    head_vae_path = ckpt_dir / "reward_head_vae.pt"

    for p, name in [
        (vae_ckpt_path, "Spatial VAE"),
        (diff_ckpt_path, "Latent Diffusion"),
        (sigreg_ckpt_path, "Teacher-Free LeWM"),
        (head_lewm_path, "LeWM Reward Head"),
        (head_vae_path, "VAE Reward Head"),
    ]:
        if not p.exists():
            fail_loud(p, "run_mpc_benchmark", f"Ensure {name} is trained.")

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

    head_lewm = LatentRewardHead(in_dim=128, is_spatial=False).to(device)
    head_lewm.load_state_dict(torch.load(head_lewm_path, map_location=device)["model_state"])
    head_lewm.eval()

    head_vae = LatentRewardHead(is_spatial=True, spatial_channels=4).to(device)
    head_vae.load_state_dict(torch.load(head_vae_path, map_location=device)["model_state"])
    head_vae.eval()

    controllers = ["Random", "Oracle", "LeWM_MPC", "AC_LDM_MPC"]
    benchmark_results: dict[str, list[dict[str, Any]]] = {c: [] for c in controllers}

    for c_name in controllers:
        logger.info("\n--- Evaluating Controller: %s ---", c_name)
        for s in seeds:
            res = run_episode(
                controller_name=c_name,
                env_seed=s,
                max_steps=max_steps,
                lewm_model=lewm,
                head_lewm=head_lewm,
                diff_model=diff,
                vae_model=vae,
                head_vae=head_vae,
                planner_horizon=horizon,
                planner_pop=pop_size,
                device=device,
            )
            benchmark_results[c_name].append(res)
            logger.info(
                "  [%s | Seed %d] Reward: %.2f | Fwd Vel: %.2f | Off-Track: %.1f%% | Step Latency: %.1f ms",
                c_name,
                s,
                res["cumulative_reward"],
                res["mean_forward_vel"],
                res["off_track_ratio"] * 100.0,
                res["mean_latency_ms"],
            )

    # Compute Aggregate Stats
    summary_stats: dict[str, dict[str, float]] = {}
    for c_name, episodes in benchmark_results.items():
        rewards = [e["cumulative_reward"] for e in episodes]
        latencies = [e["mean_latency_ms"] for e in episodes]
        fwd_vels = [e["mean_forward_vel"] for e in episodes]
        off_tracks = [e["off_track_ratio"] for e in episodes]

        summary_stats[c_name] = {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "mean_latency_ms": float(np.mean(latencies)),
            "mean_forward_vel": float(np.mean(fwd_vels)),
            "mean_off_track_ratio": float(np.mean(off_tracks)),
        }

    # Save Metrics JSON
    results_json = metrics_dir / "mpc_benchmark_results.json"
    with open(results_json, "w") as f:
        json.dump(
            {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "seeds": seeds,
                "max_steps": max_steps,
                "horizon": horizon,
                "population_size": pop_size,
                "summary": summary_stats,
                "raw_episodes": benchmark_results,
            },
            f,
            indent=2,
        )
    logger.info("Saved benchmark metrics to %s", results_json)

    # Generate Publication Figures
    names = list(summary_stats.keys())
    means = [summary_stats[k]["mean_reward"] for k in names]
    stds = [summary_stats[k]["std_reward"] for k in names]
    lats = [summary_stats[k]["mean_latency_ms"] for k in names]

    # Figure 1: Cumulative Return Comparison with Std Bars
    _fig1, ax1 = plt.subplots(figsize=(8, 5))
    bars = ax1.bar(names, means, yerr=stds, capsize=6, color=["#7f7f7f", "#2ca02c", "#1f77b4", "#ff7f0e"])
    ax1.set_ylabel("Episodic Cumulative Return")
    ax1.set_title("Closed-Loop Latent MPC Trajectory Return (5 Seeds)")
    ax1.grid(True, alpha=0.3, axis="y")
    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, h / 2.0, f"{h:.1f}", ha="center", va="center", color="white", fontweight="bold")
    plt.tight_layout()
    fig1_path = fig_dir / "mpc_cumulative_rewards.png"
    plt.savefig(fig1_path, dpi=200)
    plt.close()

    # Figure 2: Step Latency vs Control Performance
    _fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.bar(names, lats, color=["#7f7f7f", "#2ca02c", "#1f77b4", "#ff7f0e"])
    ax2.set_ylabel("Decision Latency per Step (ms)")
    ax2.set_yscale("log")
    ax2.set_title("Decision Latency (Log Scale)")
    ax2.grid(True, alpha=0.3, axis="y", which="both")
    for i, v in enumerate(lats):
        ax2.text(i, v * 1.2, f"{v:.1f} ms", ha="center", fontsize=9)
    plt.tight_layout()
    fig2_path = fig_dir / "mpc_planning_latency.png"
    plt.savefig(fig2_path, dpi=200)
    plt.close()

    logger.info("Saved comparison figures to %s and %s", fig1_path, fig2_path)
    return results_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Live Closed-Loop Latent MPC Benchmark.")
    parser.add_argument("--steps", type=int, default=150, help="Max steps per episode.")
    parser.add_argument("--horizon", type=int, default=5, help="Planning horizon H.")
    parser.add_argument("--pop-size", type=int, default=50, help="CEM population size.")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    run_mpc_benchmark(
        seeds=[101, 102, 103, 104, 105],
        max_steps=args.steps,
        horizon=args.horizon,
        pop_size=args.pop_size,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
