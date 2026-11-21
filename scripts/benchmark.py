#!/usr/bin/env python3
"""
Unified Evaluation and Benchmark Runner for AckBot Person Following.

Consolidates all evaluation suites:
1. 'comparative' : Representation learning diagnostics (effective rank, out-of-sample R^2 probe, Gaussianity).
2. 'occlusion'   : Blind corner and U-turn target re-acquisition benchmarks.
3. 'robustness'  : Sim-to-real perturbations (sensor blackout, actuator delay, friction shift).
4. 'mpc'         : Closed-loop MPC control trajectories and curriculum stage benchmarks.
5. 'all'         : Runs all suites and saves consolidated reports to outputs/reports/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import matplotlib

import rl_ws.gym_ackbot.envs  # noqa: F401 (registers AckBotFollow-v0)

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from rl_ws.common.agents import (
    BaseAgent,
    HeuristicPursuitAgent,
    LatentMPCAgent,
    PPOAgent,
    RandomAgent,
    load_reward_head,
    load_world_model,
)
from rl_ws.common.config import select_device
from rl_ws.data.dataset import get_dataset_splits
from rl_ws.policies.world_model.representation import (
    RidgeRegressionProbe,
    compute_effective_rank,
    compute_gaussianity_diagnostics,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    parser = argparse.ArgumentParser(description="AckBot Unified Evaluation & Benchmark Runner")
    parser.add_argument(
        "--suite",
        type=str,
        default="all",
        choices=["comparative", "occlusion", "robustness", "mpc", "all"],
        help="Benchmark suite to execute (default: 'all')",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes per scenario/policy (default: 5)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs/reports"),
        help="Output directory for generated JSON reports and plots",
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
        help="Random seed for evaluation rollouts",
    )
    return parser.parse_args()


def run_episode(
    env: gym.Env,
    agent: BaseAgent,
    seed: int = 42,
    sensor_dropout_step: int = -1,
    sensor_dropout_len: int = 0,
    actuator_delay: int = 0,
) -> dict[str, Any]:
    """Execute a single evaluation rollout with optional sim-to-real perturbations."""
    obs, info = env.reset(seed=seed)
    agent.reset()

    done = False
    ep_reward = 0.0
    ep_steps = 0
    dists = []
    speeds = []
    in_fov_list = []
    robot_traj = []
    target_traj = []
    collisions = 0
    action_queue: list[np.ndarray] = []

    dt = env.unwrapped.dt  # 0.05s

    while not done:
        robot_traj.append((env.unwrapped.robot_x, env.unwrapped.robot_y))
        target_traj.append((env.unwrapped.target_x, env.unwrapped.target_y))

        # Sensor blackout perturbation
        perturbed_obs = obs.copy()
        if sensor_dropout_step <= ep_steps < (sensor_dropout_step + sensor_dropout_len):
            perturbed_obs[:64] = 0.0  # Camera depth blackout

        raw_action = agent.act(perturbed_obs, info=info)

        # Actuator latency queue
        if actuator_delay > 0:
            action_queue.append(raw_action)
            if len(action_queue) > actuator_delay:
                action = action_queue.pop(0)
            else:
                action = np.zeros_like(raw_action)
        else:
            action = raw_action

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        ep_reward += reward
        ep_steps += 1
        dists.append(info.get("target_dist", 0.0))
        speeds.append(abs(info.get("robot_v", 0.0)))
        in_fov_list.append(1.0 if info.get("target_in_fov", False) else 0.0)

        if info.get("collision", False):
            collisions += 1

    return {
        "reward": float(ep_reward),
        "steps": int(ep_steps),
        "duration_sec": float(ep_steps * dt),
        "mean_dist": float(np.mean(dists)) if dists else 0.0,
        "mean_speed": float(np.mean(speeds)) if speeds else 0.0,
        "visual_contact_pct": float(np.mean(in_fov_list) * 100.0) if in_fov_list else 0.0,
        "collisions": int(collisions),
        "robot_traj": robot_traj,
        "target_traj": target_traj,
        "target_locked": bool(info.get("target_locked", False)),
    }


def run_comparative_suite(device: torch.device, out_dir: Path) -> dict[str, Any]:
    print("\n" + "=" * 72)
    print("      SUITE 1: COMPARATIVE REPRESENTATION LEARNING BENCHMARK          ")
    print("=" * 72)

    try:
        train_ds, val_ds, _test_ds = get_dataset_splits()
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as e:
        print(f"Could not load dataset splits: {e}. Skipping comparative suite.")
        return {}

    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=False)

    # Collect training and validation batches for linear probe
    obs_train_list, telem_train_list = [], []
    for b in train_loader:
        obs_train_list.append(b["obs"])
        telem_train_list.append(b["telemetry"])
        if len(obs_train_list) >= 10:  # Use first 2560 samples for fast probe fitting
            break
    x_train = torch.cat(obs_train_list, dim=0).to(device)
    y_train = torch.cat(telem_train_list, dim=0).to(device)[:, :2]  # [rel_x, rel_y]

    obs_val_list, telem_val_list = [], []
    for b in val_loader:
        obs_val_list.append(b["obs"])
        telem_val_list.append(b["telemetry"])
    x_val = torch.cat(obs_val_list, dim=0).to(device)
    y_val = torch.cat(telem_val_list, dim=0).to(device)[:, :2]

    results = {}

    # 1. Raw Sensory Baseline
    probe_raw = RidgeRegressionProbe(in_dim=426, out_dim=2, alpha=1.0)
    probe_raw.fit(x_train, y_train)
    r2_raw = float(probe_raw.score_r2(x_val, y_val).item())
    results["raw_sensor"] = {
        "latent_dim": 426,
        "effective_rank": 426.0,
        "linear_probe_r2": r2_raw,
    }

    # 2. VAE Representation
    vae_path = PROJECT_ROOT / "outputs/experiments/vae_curriculum/checkpoints/best_model.pt"
    if not vae_path.exists():
        vae_path = PROJECT_ROOT / "outputs/experiments/vae/checkpoints/best_model.pt"

    if vae_path.exists():
        vae = load_world_model("vae", vae_path, device=device)
        with torch.no_grad():
            z_train = vae.encode(x_train)
            z_val = vae.encode(x_val)
        rank_diag = compute_effective_rank(z_val)
        gauss_diag = compute_gaussianity_diagnostics(z_val)

        z_train_flat = z_train.reshape(z_train.shape[0], -1)
        z_val_flat = z_val.reshape(z_val.shape[0], -1)
        probe_vae = RidgeRegressionProbe(in_dim=128, out_dim=2, alpha=1.0)
        probe_vae.fit(z_train_flat, y_train)
        r2_vae = float(probe_vae.score_r2(z_val_flat, y_val).item())

        results["vae"] = {
            "latent_dim": 128,
            "effective_rank": float(rank_diag["effective_rank"]),
            "rank_ratio": float(rank_diag["rank_ratio"]),
            "linear_probe_r2": r2_vae,
            "excess_kurtosis": float(gauss_diag["excess_kurtosis"]),
            "skewness_mean": float(gauss_diag["skewness_mean"]),
        }

    # 3. LeWM Representation
    lewm_path = PROJECT_ROOT / "outputs/experiments/sigreg_curriculum/checkpoints/best_model.pt"
    if not lewm_path.exists():
        lewm_path = PROJECT_ROOT / "outputs/experiments/sigreg/checkpoints/best_model.pt"

    if lewm_path.exists():
        lewm = load_world_model("lewm", lewm_path, device=device)
        with torch.no_grad():
            z_train = lewm.encode(x_train)
            z_val = lewm.encode(x_val)
        rank_diag = compute_effective_rank(z_val)
        gauss_diag = compute_gaussianity_diagnostics(z_val)

        probe_lewm = RidgeRegressionProbe(in_dim=128, out_dim=2, alpha=1.0)
        probe_lewm.fit(z_train, y_train)
        r2_lewm = float(probe_lewm.score_r2(z_val, y_val).item())

        results["lewm"] = {
            "latent_dim": 128,
            "effective_rank": float(rank_diag["effective_rank"]),
            "rank_ratio": float(rank_diag["rank_ratio"]),
            "linear_probe_r2": r2_lewm,
            "excess_kurtosis": float(gauss_diag["excess_kurtosis"]),
            "skewness_mean": float(gauss_diag["skewness_mean"]),
        }

    report_path = out_dir / "comparative_benchmark.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved comparative report to {report_path}")
    return results


def run_occlusion_suite(
    device: torch.device, out_dir: Path, episodes: int = 5, seed: int = 42
) -> dict[str, Any]:
    print("\n" + "=" * 72)
    print("      SUITE 2: OCCLUSION & TARGET RE-ACQUISITION BENCHMARK            ")
    print("=" * 72)

    env = gym.make("AckBotFollow-v0")
    results = {}

    # Prepare candidate policies
    policies: dict[str, BaseAgent] = {
        "Heuristic (Pursuit)": HeuristicPursuitAgent(),
        "Random Baseline": RandomAgent(seed=seed),
    }

    ppo_path = (
        PROJECT_ROOT
        / "outputs/experiments/exp_20261108_154753_curriculum_ppo_active_follow/checkpoints/best_model.pt"
    )
    if ppo_path.exists():
        policies["PPO (Reactive)"] = PPOAgent(ppo_path, device=device)

    lewm_path = PROJECT_ROOT / "outputs/experiments/sigreg_curriculum/checkpoints/best_model.pt"
    rew_lewm_path = (
        PROJECT_ROOT / "outputs/experiments/mpc_curriculum/checkpoints/reward_head_lewm.pt"
    )
    if lewm_path.exists() and rew_lewm_path.exists():
        wm = load_world_model("lewm", lewm_path, device=device)
        rew = load_reward_head(rew_lewm_path, device=device, is_spatial=False)
        policies["LeWM (SIGReg) MPC"] = LatentMPCAgent(
            wm, rew, model_type="lewm", horizon=5, pop_size=25, num_iters=2, device=device
        )

    # Evaluate across Stage 2 (Blind Corner) and Stage 3 (U-Turn Corridor)
    for stage_num, stage_name in [(2, "Stage 2 (Blind Corner)"), (3, "Stage 3 (U-Turn Corridor)")]:
        results[stage_name] = {}
        for pol_name, agent in policies.items():
            ep_metrics = []
            for i in range(episodes):
                env.unwrapped.stage = stage_num
                res = run_episode(env, agent, seed=seed + i)
                ep_metrics.append(res)

            success_rate = float(
                np.mean(
                    [1.0 if m["collisions"] == 0 and m["steps"] > 50 else 0.0 for m in ep_metrics]
                )
                * 100.0
            )
            reacq_rate = float(np.mean([m["visual_contact_pct"] for m in ep_metrics]))
            mean_dist = float(np.mean([m["mean_dist"] for m in ep_metrics]))
            total_collisions = int(sum([m["collisions"] for m in ep_metrics]))

            results[stage_name][pol_name] = {
                "success_rate_pct": success_rate,
                "reacquisition_contact_pct": reacq_rate,
                "mean_distance_m": mean_dist,
                "total_collisions": total_collisions,
            }

    env.close()
    report_path = out_dir / "occlusion_benchmark.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved occlusion report to {report_path}")
    return results


def run_robustness_suite(
    device: torch.device, out_dir: Path, episodes: int = 5, seed: int = 42
) -> dict[str, Any]:
    print("\n" + "=" * 72)
    print("      SUITE 3: SIM-TO-REAL ROBUSTNESS & BLACKOUT BENCHMARK            ")
    print("=" * 72)

    env = gym.make("AckBotFollow-v0")
    results = {}

    policies: dict[str, BaseAgent] = {
        "Heuristic": HeuristicPursuitAgent(),
    }
    ppo_path = (
        PROJECT_ROOT
        / "outputs/experiments/exp_20261108_154753_curriculum_ppo_active_follow/checkpoints/best_model.pt"
    )
    if ppo_path.exists():
        policies["PPO (Reactive)"] = PPOAgent(ppo_path, device=device)

    lewm_path = PROJECT_ROOT / "outputs/experiments/sigreg_curriculum/checkpoints/best_model.pt"
    rew_lewm_path = (
        PROJECT_ROOT / "outputs/experiments/mpc_curriculum/checkpoints/reward_head_lewm.pt"
    )
    if lewm_path.exists() and rew_lewm_path.exists():
        wm = load_world_model("lewm", lewm_path, device=device)
        rew = load_reward_head(rew_lewm_path, device=device, is_spatial=False)
        policies["LeWM MPC"] = LatentMPCAgent(
            wm, rew, model_type="lewm", horizon=5, pop_size=20, num_iters=2, device=device
        )

    # Scenarios: Nominal, Sensor Blackout (1.0s = 20 steps at 20Hz), Actuator Delay (100ms = 2 steps at 20Hz)
    scenarios = [
        (
            "Nominal Tracking",
            {"sensor_dropout_step": -1, "sensor_dropout_len": 0, "actuator_delay": 0},
        ),
        (
            "Sensor Blackout (1.0s)",
            {"sensor_dropout_step": 20, "sensor_dropout_len": int(1.0 / 0.05), "actuator_delay": 0},
        ),
        (
            "Actuator Delay (100ms)",
            {
                "sensor_dropout_step": -1,
                "sensor_dropout_len": 0,
                "actuator_delay": int(0.10 / 0.05),
            },
        ),
    ]

    for scen_name, kwargs in scenarios:
        results[scen_name] = {}
        for pol_name, agent in policies.items():
            ep_metrics = []
            for i in range(episodes):
                res = run_episode(env, agent, seed=seed + i, **kwargs)
                ep_metrics.append(res)

            results[scen_name][pol_name] = {
                "mean_reward": float(np.mean([m["reward"] for m in ep_metrics])),
                "mean_distance": float(np.mean([m["mean_dist"] for m in ep_metrics])),
                "visual_contact_pct": float(np.mean([m["visual_contact_pct"] for m in ep_metrics])),
                "collisions": int(sum([m["collisions"] for m in ep_metrics])),
            }

    env.close()
    report_path = out_dir / "robustness_benchmark.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved robustness report to {report_path}")
    return results


def run_mpc_suite(
    device: torch.device, out_dir: Path, episodes: int = 5, seed: int = 42
) -> dict[str, Any]:
    print("\n" + "=" * 72)
    print("      SUITE 4: CLOSED-LOOP MPC CURRICULUM TRAJECTORY BENCHMARK        ")
    print("=" * 72)

    env = gym.make("AckBotFollow-v0")
    results = {}
    trajectories_to_plot = {}

    policies: dict[str, BaseAgent] = {
        "Heuristic": HeuristicPursuitAgent(),
    }
    ppo_path = (
        PROJECT_ROOT
        / "outputs/experiments/exp_20261108_154753_curriculum_ppo_active_follow/checkpoints/best_model.pt"
    )
    if ppo_path.exists():
        policies["PPO (Reactive)"] = PPOAgent(ppo_path, device=device)

    lewm_path = PROJECT_ROOT / "outputs/experiments/sigreg_curriculum/checkpoints/best_model.pt"
    rew_lewm_path = (
        PROJECT_ROOT / "outputs/experiments/mpc_curriculum/checkpoints/reward_head_lewm.pt"
    )
    if lewm_path.exists() and rew_lewm_path.exists():
        wm = load_world_model("lewm", lewm_path, device=device)
        rew = load_reward_head(rew_lewm_path, device=device, is_spatial=False)
        policies["LeWM (SIGReg) MPC"] = LatentMPCAgent(
            wm, rew, model_type="lewm", horizon=5, pop_size=25, num_iters=2, device=device
        )

    for pol_name, agent in policies.items():
        ep_list = []
        for i in range(episodes):
            res = run_episode(env, agent, seed=seed + i)
            ep_list.append(res)
        results[pol_name] = {
            "mean_reward": float(np.mean([e["reward"] for e in ep_list])),
            "mean_dist": float(np.mean([e["mean_dist"] for e in ep_list])),
            "mean_speed": float(np.mean([e["mean_speed"] for e in ep_list])),
            "visual_contact_pct": float(np.mean([e["visual_contact_pct"] for e in ep_list])),
            "collisions": int(sum([e["collisions"] for e in ep_list])),
        }
        trajectories_to_plot[pol_name] = ep_list[0]

    # Generate dynamic trajectory plot with real environment obstacles
    plt.figure(figsize=(9, 9))
    plt.title(
        f"AckBot Closed-Loop Following Trajectories (Stage {env.unwrapped.stage})",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlim(-5.2, 5.2)
    plt.ylim(-5.2, 5.2)
    plt.grid(True, linestyle="--", alpha=0.5)

    # Draw real obstacles from environment
    for ox, oy, obs_r in env.unwrapped.static_obstacles:
        circ = plt.Circle((ox, oy), obs_r, color="gray", alpha=0.6)
        plt.gca().add_patch(circ)
    for x1, y1, x2, y2 in env.unwrapped.internal_walls:
        plt.plot([x1, x2], [y1, y2], color="black", linewidth=3)

    colors = {"Heuristic": "orange", "PPO (Reactive)": "blue", "LeWM (SIGReg) MPC": "green"}
    for pol_name, ep_data in trajectories_to_plot.items():
        rt = np.array(ep_data["robot_traj"])
        tt = np.array(ep_data["target_traj"])
        c = colors.get(pol_name, "purple")
        plt.plot(rt[:, 0], rt[:, 1], label=f"{pol_name} (Robot)", color=c, linewidth=2.0)
        plt.plot(
            tt[:, 0], tt[:, 1], label=f"{pol_name} (Target)", color=c, linestyle=":", alpha=0.7
        )

    plt.legend(loc="upper right")
    plt.tight_layout()
    plot_path = out_dir / "mpc_tracking_trajectories.png"
    plt.savefig(plot_path, dpi=180)
    plt.close()

    env.close()
    report_path = out_dir / "mpc_evaluation_results.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved MPC results to {report_path} and trajectory plot to {plot_path}")
    return results


def main():
    args = parse_args()
    device = select_device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("               ACKBOT UNIFIED BENCHMARK SUITE                         ")
    print("=" * 72)
    print(f"Device              : {device}")
    print(f"Suite Selection     : {args.suite.upper()}")
    print(f"Episodes / Run      : {args.episodes}")
    print(f"Output Directory    : {out_dir}")

    if args.suite in ("comparative", "all"):
        run_comparative_suite(device, out_dir)

    if args.suite in ("occlusion", "all"):
        run_occlusion_suite(device, out_dir, episodes=args.episodes, seed=args.seed)

    if args.suite in ("robustness", "all"):
        run_robustness_suite(device, out_dir, episodes=args.episodes, seed=args.seed)

    if args.suite in ("mpc", "all"):
        run_mpc_suite(device, out_dir, episodes=args.episodes, seed=args.seed)

    print("\nBenchmark suite execution completed successfully!")


if __name__ == "__main__":
    main()
