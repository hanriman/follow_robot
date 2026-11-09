#!/usr/bin/env python3
"""
AckBot Occlusion & Multi-Pedestrian Benchmark: Reactive PPO vs. LeWM CEM-MPC.

Rigorously compares pure reactive PPO against Teacher-Free LeWM CEM-MPC on two critical challenges:
1. Stage 2 (Blind Corner Occlusion): Wall divider causes complete visual line-of-sight drop-out.
2. Stage 4 (Multi-Pedestrian Crossings): Distractors cross paths, casting dynamic ray shadows.

Metrics evaluated:
- Re-acquisition Latency (seconds to regain visual lock after corner occlusion)
- Visual Lock % (percentage of time target is tracked in FOV)
- Tracking Distance Error (|d - d_desired|)
- Episodic Return
- Collision Count & Minimum Obstacle Clearance

Outputs:
- outputs/experiments/mpc_curriculum/benchmark_results.json
- outputs/experiments/mpc_curriculum/corner_occlusion_trajectories.png
- outputs/experiments/mpc_curriculum/multi_ped_occlusion_trajectories.png
- outputs/experiments/mpc_curriculum/comparative_metrics_bar.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import rl_ws.gym_ackbot  # registers AckBotFollow-v0
from rl_ws.policies.ppo.actor_critic import ActorCritic
from rl_ws.policies.world_model.sigreg_wm import TeacherFreeLeWorldModel
from rl_ws.policies.world_model.reward_head import LatentRewardHead
from rl_ws.policies.world_model.cem_planner import CEMPlanner

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PPO_CKPT = PROJECT_ROOT / "outputs/experiments/exp_20261108_154753_curriculum_ppo_active_follow/checkpoints/best_model.pt"
DEFAULT_LEWM_CKPT = PROJECT_ROOT / "outputs/experiments/sigreg_curriculum/checkpoints/best_model.pt"
DEFAULT_REW_LEWM = PROJECT_ROOT / "outputs/experiments/mpc_curriculum/checkpoints/reward_head_lewm.pt"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs/experiments/mpc_curriculum"


def parse_args():
    parser = argparse.ArgumentParser(description="Occlusion & Multi-Pedestrian Benchmark")
    parser.add_argument("--episodes", type=int, default=5, help="Episodes per evaluation scenario")
    parser.add_argument("--horizon", type=int, default=5, help="CEM planning horizon H")
    parser.add_argument("--pop-size", type=int, default=40, help="CEM population size")
    parser.add_argument("--num-iters", type=int, default=3, help="CEM refinement iterations")
    parser.add_argument("--ppo-checkpoint", type=str, default=str(DEFAULT_PPO_CKPT))
    parser.add_argument("--lewm-checkpoint", type=str, default=str(DEFAULT_LEWM_CKPT))
    parser.add_argument("--reward-head-lewm", type=str, default=str(DEFAULT_REW_LEWM))
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


def run_benchmark_episode(
    env: gym.Env,
    policy_fn,
    stage: int,
    num_distractors: int,
    seed: int,
    max_steps: int = 250,
    reset_fn=None,
) -> dict:
    obs, info = env.reset(seed=seed, options={"stage": stage, "num_distractors": num_distractors})
    if reset_fn is not None:
        reset_fn()

    done = False
    ep_reward = 0.0
    step = 0

    dists = []
    in_fov_flags = []
    robot_path = []
    target_path = []
    distractor_paths = [[] for _ in range(num_distractors)]
    collisions = 0
    min_clearance = float("inf")

    # Occlusion latency tracking
    had_initial_lock = False
    occlusion_start = None
    reacquisition_step = None
    occlusion_latencies = []

    while not done and step < max_steps:
        rx, ry = env.unwrapped.robot_x, env.unwrapped.robot_y
        tx, ty = env.unwrapped.target_x, env.unwrapped.target_y
        robot_path.append((rx, ry))
        target_path.append((tx, ty))

        for d_i, d in enumerate(getattr(env.unwrapped, "distractors", [])):
            if d_i < len(distractor_paths):
                distractor_paths[d_i].append((d.x, d.y))

        action = policy_fn(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        step += 1
        ep_reward += reward

        target_dist = info.get("target_dist", math.hypot(tx - rx, ty - ry))
        in_fov = bool(info.get("target_in_fov", False))
        collision = bool(info.get("collision", False))

        dists.append(target_dist)
        in_fov_flags.append(1.0 if in_fov else 0.0)
        if collision:
            collisions += 1

        # Check clearance to obstacles & target
        min_clearance = min(min_clearance, target_dist)

        # Track occlusion and re-acquisition
        if in_fov:
            had_initial_lock = True
            if occlusion_start is not None:
                latency_steps = step - occlusion_start
                occlusion_latencies.append(latency_steps * 0.1)  # 10 Hz -> seconds
                occlusion_start = None
        else:
            if had_initial_lock and occlusion_start is None:
                occlusion_start = step

        done = terminated or truncated

    if occlusion_start is not None:
        # Occluded until episode end
        occlusion_latencies.append((step - occlusion_start) * 0.1)

    mean_dist = float(np.mean(dists)) if dists else 0.0
    dist_error = float(np.mean([abs(d - 0.80) for d in dists])) if dists else 0.0
    visual_lock_pct = float(np.mean(in_fov_flags) * 100.0) if in_fov_flags else 0.0
    avg_reacquisition_latency = float(np.mean(occlusion_latencies)) if occlusion_latencies else 0.0

    return {
        "reward": float(ep_reward),
        "steps": step,
        "mean_dist": mean_dist,
        "dist_error": dist_error,
        "visual_lock_pct": visual_lock_pct,
        "collisions": collisions,
        "min_clearance": float(min_clearance),
        "reacquisition_latencies": occlusion_latencies,
        "avg_reacquisition_latency": avg_reacquisition_latency,
        "robot_path": robot_path,
        "target_path": target_path,
        "distractor_paths": distractor_paths,
    }


def plot_scenario_trajectories(
    ppo_ep: dict,
    lewm_ep: dict,
    stage: int,
    env: gym.Env,
    save_path: Path,
):
    plt.figure(figsize=(9, 8))
    ax = plt.subplot(1, 1, 1)

    # Plot arena boundary
    arena = env.unwrapped.arena_size / 2.0
    ax.set_xlim(-arena - 0.2, arena + 0.2)
    ax.set_ylim(-arena - 0.2, arena + 0.2)
    ax.set_aspect("equal")

    # Plot internal walls
    for x1, y1, x2, y2 in env.unwrapped.internal_walls:
        ax.plot([x1, x2], [y1, y2], color="black", linewidth=4, label="Walls" if "Walls" not in [l.get_label() for l in ax.lines] else "")

    # Plot static obstacles
    for ox, oy, r in env.unwrapped.static_obstacles:
        circle = plt.Circle((ox, oy), r, color="dimgray", alpha=0.7)
        ax.add_patch(circle)

    # Plot Target Path
    tx, ty = zip(*ppo_ep["target_path"])
    ax.plot(tx, ty, "g--", linewidth=2.5, label="Human Target Path")
    ax.scatter([tx[0]], [ty[0]], color="green", marker="o", s=80, label="Target Start")
    ax.scatter([tx[-1]], [ty[-1]], color="darkgreen", marker="X", s=90, label="Target End")

    # Plot Distractors if any
    for i, dpath in enumerate(ppo_ep.get("distractor_paths", [])):
        if dpath:
            dx, dy = zip(*dpath)
            ax.plot(dx, dy, color="gray", linestyle=":", alpha=0.6, label=f"Distractor {i+1}" if i == 0 else "")

    # Plot PPO Trajectory
    px, py = zip(*ppo_ep["robot_path"])
    ax.plot(px, py, color="crimson", linewidth=2.0, label=f"Reactive PPO (Lock: {ppo_ep['visual_lock_pct']:.1f}%)")
    ax.scatter([px[0]], [py[0]], color="crimson", marker="s", s=60)

    # Plot LeWM CEM-MPC Trajectory
    lx, ly = zip(*lewm_ep["robot_path"])
    ax.plot(lx, ly, color="royalblue", linewidth=2.2, label=f"LeWM CEM-MPC (Lock: {lewm_ep['visual_lock_pct']:.1f}%)")
    ax.scatter([lx[0]], [ly[0]], color="royalblue", marker="^", s=70)

    stage_title = "Stage 2: Blind Corner Occlusion" if stage == 2 else "Stage 4: Multi-Pedestrian Crossings"
    ax.set_title(f"{stage_title} Trajectory Tracking", fontsize=13, fontweight="bold")
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"  Saved trajectory visualization: {save_path.name}")


def main():
    args = parse_args()
    device = select_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print("      ACKBOT OCCLUSION BENCHMARK: REACTIVE PPO VS. LEWM CEM-MPC       ")
    print("=" * 76)
    print(f"Device              : {device}")
    print(f"Evaluation Episodes : {args.episodes}")
    print(f"CEM Horizon         : {args.horizon} | Pop: {args.pop_size} | Iters: {args.num_iters}")
    print(f"PPO Checkpoint      : {args.ppo_checkpoint}")
    print(f"LeWM Checkpoint     : {args.lewm_checkpoint}")
    print(f"Reward Head         : {args.reward_head_lewm}")

    env = gym.make("AckBotFollow-v0")

    # Load PPO model
    ppo_path = Path(args.ppo_checkpoint)
    assert ppo_path.exists(), f"PPO checkpoint not found at {ppo_path}"
    ppo_ckpt = torch.load(ppo_path, map_location=device, weights_only=False)
    ppo_model = ActorCritic(obs_dim=426, act_dim=2).to(device)
    ppo_model.load_state_dict(ppo_ckpt["model_state_dict"])
    ppo_model.eval()

    def ppo_policy(obs):
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            return ppo_model.get_deterministic_action(obs_t).squeeze(0).cpu().numpy()

    # Load LeWM and Latent Reward Head
    lewm_path = Path(args.lewm_checkpoint)
    rew_path = Path(args.reward_head_lewm)
    assert lewm_path.exists(), f"LeWM checkpoint not found at {lewm_path}"
    assert rew_path.exists(), f"Reward head not found at {rew_path}"

    lewm_ckpt = torch.load(lewm_path, map_location=device, weights_only=False)
    lewm = TeacherFreeLeWorldModel(latent_dim=128, action_dim=2).to(device)
    lewm.load_state_dict(lewm_ckpt["model_state_dict"])
    lewm.eval()

    rew_ckpt = torch.load(rew_path, map_location=device, weights_only=False)
    rew_head = LatentRewardHead(in_dim=128, is_spatial=False).to(device)
    rew_head.load_state_dict(rew_ckpt["model_state_dict"])
    rew_head.eval()

    planner = CEMPlanner(
        horizon=args.horizon,
        pop_size=args.pop_size,
        num_elites=5,
        num_iters=args.num_iters,
        device=device,
    )

    def lewm_policy(obs):
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            z_0 = lewm.encode(obs_t)
            act_t = planner.plan(
                initial_z=z_0,
                rollout_fn=lambda z, acts: lewm.predictor.rollout(z, acts),
                score_fn=lambda unrolled: rew_head.compute_control_score(unrolled),
            )
            return act_t.cpu().numpy()

    results = {}

    # -------------------------------------------------------------
    # Scenario 1: Stage 2 Blind Corner Occlusion
    # -------------------------------------------------------------
    print("\n[Scenario 1/2] Evaluating Stage 2: Blind Corner Occlusion...")
    s2_ppo_runs = []
    s2_lewm_runs = []

    for ep in range(args.episodes):
        ep_seed = args.seed + ep * 100
        run_p = run_benchmark_episode(env, ppo_policy, stage=2, num_distractors=0, seed=ep_seed)
        run_l = run_benchmark_episode(env, lewm_policy, stage=2, num_distractors=0, seed=ep_seed, reset_fn=planner.reset)
        s2_ppo_runs.append(run_p)
        s2_lewm_runs.append(run_l)

    results["Stage_2_Blind_Corner"] = {
        "PPO": {
            "mean_reward": float(np.mean([r["reward"] for r in s2_ppo_runs])),
            "visual_lock_pct": float(np.mean([r["visual_lock_pct"] for r in s2_ppo_runs])),
            "mean_dist_error": float(np.mean([r["dist_error"] for r in s2_ppo_runs])),
            "avg_reacquisition_latency": float(np.mean([r["avg_reacquisition_latency"] for r in s2_ppo_runs])),
            "collisions": int(sum([r["collisions"] for r in s2_ppo_runs])),
        },
        "LeWM_MPC": {
            "mean_reward": float(np.mean([r["reward"] for r in s2_lewm_runs])),
            "visual_lock_pct": float(np.mean([r["visual_lock_pct"] for r in s2_lewm_runs])),
            "mean_dist_error": float(np.mean([r["dist_error"] for r in s2_lewm_runs])),
            "avg_reacquisition_latency": float(np.mean([r["avg_reacquisition_latency"] for r in s2_lewm_runs])),
            "collisions": int(sum([r["collisions"] for r in s2_lewm_runs])),
        },
    }

    plot_scenario_trajectories(
        s2_ppo_runs[0],
        s2_lewm_runs[0],
        stage=2,
        env=env,
        save_path=out_dir / "corner_occlusion_trajectories.png",
    )

    # -------------------------------------------------------------
    # Scenario 2: Stage 4 Multi-Pedestrian Crossings
    # -------------------------------------------------------------
    print("\n[Scenario 2/2] Evaluating Stage 4: Multi-Pedestrian Crossings...")
    s4_ppo_runs = []
    s4_lewm_runs = []

    for ep in range(args.episodes):
        ep_seed = args.seed + 1000 + ep * 100
        run_p = run_benchmark_episode(env, ppo_policy, stage=4, num_distractors=2, seed=ep_seed)
        run_l = run_benchmark_episode(env, lewm_policy, stage=4, num_distractors=2, seed=ep_seed, reset_fn=planner.reset)
        s4_ppo_runs.append(run_p)
        s4_lewm_runs.append(run_l)

    results["Stage_4_Multi_Pedestrian"] = {
        "PPO": {
            "mean_reward": float(np.mean([r["reward"] for r in s4_ppo_runs])),
            "visual_lock_pct": float(np.mean([r["visual_lock_pct"] for r in s4_ppo_runs])),
            "mean_dist_error": float(np.mean([r["dist_error"] for r in s4_ppo_runs])),
            "avg_reacquisition_latency": float(np.mean([r["avg_reacquisition_latency"] for r in s4_ppo_runs])),
            "collisions": int(sum([r["collisions"] for r in s4_ppo_runs])),
        },
        "LeWM_MPC": {
            "mean_reward": float(np.mean([r["reward"] for r in s4_lewm_runs])),
            "visual_lock_pct": float(np.mean([r["visual_lock_pct"] for r in s4_lewm_runs])),
            "mean_dist_error": float(np.mean([r["dist_error"] for r in s4_lewm_runs])),
            "avg_reacquisition_latency": float(np.mean([r["avg_reacquisition_latency"] for r in s4_lewm_runs])),
            "collisions": int(sum([r["collisions"] for r in s4_lewm_runs])),
        },
    }

    plot_scenario_trajectories(
        s4_ppo_runs[0],
        s4_lewm_runs[0],
        stage=4,
        env=env,
        save_path=out_dir / "multi_ped_occlusion_trajectories.png",
    )

    env.close()

    # Save benchmark JSON
    json_path = out_dir / "benchmark_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Plot Comparative Bar Chart
    plt.figure(figsize=(10, 5))
    metrics = ["Visual Lock %", "Re-acq Latency (s)", "Dist Error (m)"]
    stages = ["Stage 2 Corner", "Stage 4 Multi-Ped"]

    ppo_vals = [
        [results["Stage_2_Blind_Corner"]["PPO"]["visual_lock_pct"], results["Stage_4_Multi_Pedestrian"]["PPO"]["visual_lock_pct"]],
        [results["Stage_2_Blind_Corner"]["PPO"]["avg_reacquisition_latency"], results["Stage_4_Multi_Pedestrian"]["PPO"]["avg_reacquisition_latency"]],
        [results["Stage_2_Blind_Corner"]["PPO"]["mean_dist_error"], results["Stage_4_Multi_Pedestrian"]["PPO"]["mean_dist_error"]],
    ]

    lewm_vals = [
        [results["Stage_2_Blind_Corner"]["LeWM_MPC"]["visual_lock_pct"], results["Stage_4_Multi_Pedestrian"]["LeWM_MPC"]["visual_lock_pct"]],
        [results["Stage_2_Blind_Corner"]["LeWM_MPC"]["avg_reacquisition_latency"], results["Stage_4_Multi_Pedestrian"]["LeWM_MPC"]["avg_reacquisition_latency"]],
        [results["Stage_2_Blind_Corner"]["LeWM_MPC"]["mean_dist_error"], results["Stage_4_Multi_Pedestrian"]["LeWM_MPC"]["mean_dist_error"]],
    ]

    for m_idx, metric_name in enumerate(metrics):
        plt.subplot(1, 3, m_idx + 1)
        x = np.arange(len(stages))
        w = 0.35
        plt.bar(x - w / 2, ppo_vals[m_idx], width=w, label="Reactive PPO", color="crimson", alpha=0.85)
        plt.bar(x + w / 2, lewm_vals[m_idx], width=w, label="LeWM CEM-MPC", color="royalblue", alpha=0.85)
        plt.xticks(x, stages, rotation=15, fontsize=8)
        plt.title(metric_name, fontsize=11, fontweight="bold")
        if m_idx == 0:
            plt.ylabel("Value")
            plt.legend(fontsize=8)
        plt.grid(True, linestyle=":", alpha=0.5)

    plt.suptitle("Occlusion Robustness: Reactive PPO vs. LeWM CEM-MPC", fontsize=13, fontweight="bold")
    plt.tight_layout()
    bar_path = out_dir / "comparative_metrics_bar.png"
    plt.savefig(bar_path, dpi=200)
    plt.close()
    print(f"  Saved comparative metrics bar chart: {bar_path.name}")

    print("\n" + "=" * 76)
    print("                    BENCHMARK RESULTS SUMMARY                         ")
    print("=" * 76)
    print("STAGE 2: BLIND CORNER OCCLUSION:")
    print(f"  Reactive PPO : Visual Lock = {results['Stage_2_Blind_Corner']['PPO']['visual_lock_pct']:5.1f}% | Latency = {results['Stage_2_Blind_Corner']['PPO']['avg_reacquisition_latency']:4.2f}s | Dist Error = {results['Stage_2_Blind_Corner']['PPO']['mean_dist_error']:4.2f}m")
    print(f"  LeWM CEM-MPC : Visual Lock = {results['Stage_2_Blind_Corner']['LeWM_MPC']['visual_lock_pct']:5.1f}% | Latency = {results['Stage_2_Blind_Corner']['LeWM_MPC']['avg_reacquisition_latency']:4.2f}s | Dist Error = {results['Stage_2_Blind_Corner']['LeWM_MPC']['mean_dist_error']:4.2f}m")
    print("-" * 76)
    print("STAGE 4: MULTI-PEDESTRIAN CROSSINGS:")
    print(f"  Reactive PPO : Visual Lock = {results['Stage_4_Multi_Pedestrian']['PPO']['visual_lock_pct']:5.1f}% | Latency = {results['Stage_4_Multi_Pedestrian']['PPO']['avg_reacquisition_latency']:4.2f}s | Dist Error = {results['Stage_4_Multi_Pedestrian']['PPO']['mean_dist_error']:4.2f}m")
    print(f"  LeWM CEM-MPC : Visual Lock = {results['Stage_4_Multi_Pedestrian']['LeWM_MPC']['visual_lock_pct']:5.1f}% | Latency = {results['Stage_4_Multi_Pedestrian']['LeWM_MPC']['avg_reacquisition_latency']:4.2f}s | Dist Error = {results['Stage_4_Multi_Pedestrian']['LeWM_MPC']['mean_dist_error']:4.2f}m")
    print("=" * 76)


if __name__ == "__main__":
    main()
