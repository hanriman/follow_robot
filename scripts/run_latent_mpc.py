#!/usr/bin/env python3
"""
AckBot Latent Model Predictive Control (MPC) Benchmark Runner.

Executes closed-loop evaluation in AckBotFollowEnv comparing:
1. Random Baseline: Uniform random actions within Ackermann constraints
2. PPO Reactive Policy: Trained actor-critic benchmark
3. LeWM CEM-MPC: Latent trajectory optimization with Teacher-Free Le World Model
4. AC-LDM CEM-MPC: Latent trajectory optimization with Action-Conditioned Diffusion World Model

Outputs:
- outputs/experiments/mpc/mpc_evaluation_results.json
- outputs/experiments/mpc/mpc_tracking_trajectories.png
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
from rl_ws.policies.world_model.vae import MultiModalSensorVAE
from rl_ws.policies.vla.diffusion_wm import ActionConditionedDiffusionWorldModel
from rl_ws.policies.world_model.reward_head import LatentRewardHead
from rl_ws.policies.world_model.cem_planner import CEMPlanner

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs/experiments/mpc"
DEFAULT_PPO_CKPT = PROJECT_ROOT / "outputs/experiments/exp_20261103_170712_ppo_active_follow/checkpoints/best_model.pt"
DEFAULT_LEWM_CKPT = PROJECT_ROOT / "outputs/experiments/sigreg/checkpoints/best_model.pt"
DEFAULT_VAE_CKPT = PROJECT_ROOT / "outputs/experiments/vae/checkpoints/best_model.pt"
DEFAULT_DIFF_CKPT = PROJECT_ROOT / "outputs/experiments/diffusion/checkpoints/best_model.pt"
DEFAULT_REW_LEWM = PROJECT_ROOT / "outputs/experiments/mpc/checkpoints/reward_head_lewm.pt"
DEFAULT_REW_VAE = PROJECT_ROOT / "outputs/experiments/mpc/checkpoints/reward_head_vae.pt"


def parse_args():
    parser = argparse.ArgumentParser(description="AckBot Latent MPC Closed-Loop Benchmark")
    parser.add_argument("--episodes", type=int, default=3, help="Evaluation episodes per policy")
    parser.add_argument("--horizon", type=int, default=5, help="MPC planning horizon H")
    parser.add_argument("--pop-size", type=int, default=40, help="CEM population size")
    parser.add_argument("--num-iters", type=int, default=3, help="CEM refinement iterations")
    parser.add_argument("--ppo-checkpoint", type=str, default=str(DEFAULT_PPO_CKPT))
    parser.add_argument("--lewm-checkpoint", type=str, default=str(DEFAULT_LEWM_CKPT))
    parser.add_argument("--vae-checkpoint", type=str, default=str(DEFAULT_VAE_CKPT))
    parser.add_argument("--diffusion-checkpoint", type=str, default=str(DEFAULT_DIFF_CKPT))
    parser.add_argument("--reward-head-lewm", type=str, default=str(DEFAULT_REW_LEWM))
    parser.add_argument("--reward-head-vae", type=str, default=str(DEFAULT_REW_VAE))
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


def run_episode(
    env: gym.Env,
    policy_fn,
    reset_fn=None,
    seed: int = 42,
) -> dict:
    obs, info = env.reset(seed=seed)
    if reset_fn is not None:
        reset_fn()

    done = False
    ep_reward = 0.0
    ep_steps = 0
    dists = []
    speeds = []
    in_fov_list = []
    robot_traj = []
    target_traj = []
    collisions = 0

    while not done:
        robot_traj.append((env.unwrapped.robot_x, env.unwrapped.robot_y))
        target_traj.append((env.unwrapped.target_x, env.unwrapped.target_y))

        action = policy_fn(obs)
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
        "reward": ep_reward,
        "steps": ep_steps,
        "mean_dist": float(np.mean(dists)) if dists else 0.0,
        "mean_speed": float(np.mean(speeds)) if speeds else 0.0,
        "visual_contact_pct": float(np.mean(in_fov_list) * 100.0) if in_fov_list else 0.0,
        "collisions": collisions,
        "robot_traj": robot_traj,
        "target_traj": target_traj,
    }


def main():
    args = parse_args()
    device = select_device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("           ACKBOT CLOSED-LOOP LATENT MPC CONTROL BENCHMARK             ")
    print("=" * 72)
    print(f"Device              : {device}")
    print(f"Episodes per Policy : {args.episodes}")
    print(f"CEM Horizon         : {args.horizon} steps | Pop Size: {args.pop_size}")
    print(f"Output Directory    : {out_dir}")

    env = gym.make("AckBotFollow-v0")
    unwrapped = env.unwrapped
    v_min, v_max = unwrapped.v_min, unwrapped.v_max
    steer_max = unwrapped.max_steering_angle

    results = {}
    trajectories_to_plot = {}

    # Policy 1: Random Baseline
    print("\n[1/4] Evaluating Random Action Baseline...")
    def random_policy(obs):
        return np.array([
            np.random.uniform(v_min, v_max),
            np.random.uniform(-steer_max, steer_max),
        ], dtype=np.float32)

    rand_eps = [run_episode(env, random_policy, seed=args.seed + i) for i in range(args.episodes)]
    results["Random"] = {
        "mean_reward": float(np.mean([e["reward"] for e in rand_eps])),
        "mean_dist": float(np.mean([e["mean_dist"] for e in rand_eps])),
        "mean_speed": float(np.mean([e["mean_speed"] for e in rand_eps])),
        "visual_contact_pct": float(np.mean([e["visual_contact_pct"] for e in rand_eps])),
        "collisions": int(sum([e["collisions"] for e in rand_eps])),
    }
    trajectories_to_plot["Random"] = rand_eps[0]

    # Policy 2: Reactive PPO Policy
    ppo_path = Path(args.ppo_checkpoint)
    if ppo_path.exists():
        print("\n[2/4] Evaluating Reactive PPO Policy...")
        ppo_ckpt = torch.load(ppo_path, map_location=device, weights_only=False)
        ppo_model = ActorCritic(obs_dim=426, act_dim=2).to(device)
        ppo_model.load_state_dict(ppo_ckpt["model_state_dict"])
        ppo_model.eval()

        def ppo_policy(obs):
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                return ppo_model.get_deterministic_action(obs_t).squeeze(0).cpu().numpy()

        ppo_eps = [run_episode(env, ppo_policy, seed=args.seed + i) for i in range(args.episodes)]
        results["PPO (Reactive)"] = {
            "mean_reward": float(np.mean([e["reward"] for e in ppo_eps])),
            "mean_dist": float(np.mean([e["mean_dist"] for e in ppo_eps])),
            "mean_speed": float(np.mean([e["mean_speed"] for e in ppo_eps])),
            "visual_contact_pct": float(np.mean([e["visual_contact_pct"] for e in ppo_eps])),
            "collisions": int(sum([e["collisions"] for e in ppo_eps])),
        }
        trajectories_to_plot["PPO (Reactive)"] = ppo_eps[0]
    else:
        print(f"\n[2/4] PPO checkpoint not found at {ppo_path}. Skipping PPO.")

    # Policy 3: LeWM Latent CEM-MPC
    lewm_path = Path(args.lewm_checkpoint)
    rew_lewm_path = Path(args.reward_head_lewm)
    if lewm_path.exists() and rew_lewm_path.exists():
        print("\n[3/4] Evaluating LeWM (SIGReg) CEM-MPC...")
        lewm_ckpt = torch.load(lewm_path, map_location=device, weights_only=False)
        lewm = TeacherFreeLeWorldModel(latent_dim=128, action_dim=2).to(device)
        lewm.load_state_dict(lewm_ckpt["model_state_dict"])
        lewm.eval()

        rew_ckpt = torch.load(rew_lewm_path, map_location=device, weights_only=False)
        rew_head = LatentRewardHead(in_dim=128, is_spatial=False).to(device)
        rew_head.load_state_dict(rew_ckpt["model_state_dict"])
        rew_head.eval()

        planner_lewm = CEMPlanner(
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

                action_t = planner_lewm.plan(
                    initial_z=z_0,
                    rollout_fn=lambda z, acts: lewm.predictor.rollout(z, acts),
                    score_fn=lambda unrolled: rew_head.compute_control_score(unrolled),
                )
                return action_t.cpu().numpy()

        lewm_eps = [run_episode(env, lewm_policy, reset_fn=planner_lewm.reset, seed=args.seed + i) for i in range(args.episodes)]
        results["LeWM (SIGReg) MPC"] = {
            "mean_reward": float(np.mean([e["reward"] for e in lewm_eps])),
            "mean_dist": float(np.mean([e["mean_dist"] for e in lewm_eps])),
            "mean_speed": float(np.mean([e["mean_speed"] for e in lewm_eps])),
            "visual_contact_pct": float(np.mean([e["visual_contact_pct"] for e in lewm_eps])),
            "collisions": int(sum([e["collisions"] for e in lewm_eps])),
        }
        trajectories_to_plot["LeWM (SIGReg) MPC"] = lewm_eps[0]
    else:
        print(f"\n[3/4] Checkpoint missing for LeWM MPC ({lewm_path} or {rew_lewm_path}). Skipping.")

    # Policy 4: AC-LDM Latent CEM-MPC
    diff_path = Path(args.diffusion_checkpoint)
    vae_path = Path(args.vae_checkpoint)
    rew_vae_path = Path(args.reward_head_vae)
    if diff_path.exists() and vae_path.exists() and rew_vae_path.exists():
        print("\n[4/4] Evaluating AC-LDM Diffusion CEM-MPC...")
        vae_ckpt = torch.load(vae_path, map_location=device, weights_only=False)
        vae = MultiModalSensorVAE(in_dim=426, latent_channels=4, latent_length=32).to(device)
        vae.load_state_dict(vae_ckpt["model_state_dict"])
        vae.eval()

        diff_ckpt = torch.load(diff_path, map_location=device, weights_only=False)
        diff_wm = ActionConditionedDiffusionWorldModel(
            latent_channels=4,
            latent_length=32,
            action_dim=2,
            base_channels=64,
            num_timesteps=1000,
            vae=vae,
        ).to(device)
        diff_wm.load_state_dict(diff_ckpt["model_state_dict"])
        diff_wm.eval()

        rew_vae_ckpt = torch.load(rew_vae_path, map_location=device, weights_only=False)
        rew_head_vae = LatentRewardHead(in_dim=128, is_spatial=True, spatial_channels=4, spatial_length=32).to(device)
        rew_head_vae.load_state_dict(rew_vae_ckpt["model_state_dict"])
        rew_head_vae.eval()

        # For diffusion MPC, use lighter pop_size & 5 DDIM steps to balance speed
        planner_diff = CEMPlanner(
            horizon=3,
            pop_size=15,
            num_elites=3,
            num_iters=2,
            device=device,
        )

        def diff_policy(obs):
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                z_0, _ = vae.encode(obs_t)

                action_t = planner_diff.plan(
                    initial_z=z_0,
                    rollout_fn=lambda z, acts: diff_wm.rollout_latents(z, acts, ddim_steps=5),
                    score_fn=lambda unrolled: rew_head_vae.compute_control_score(unrolled),
                )
                return action_t.cpu().numpy()

        diff_eps = [run_episode(env, diff_policy, reset_fn=planner_diff.reset, seed=args.seed + i) for i in range(args.episodes)]
        results["AC-LDM (Diffusion) MPC"] = {
            "mean_reward": float(np.mean([e["reward"] for e in diff_eps])),
            "mean_dist": float(np.mean([e["mean_dist"] for e in diff_eps])),
            "mean_speed": float(np.mean([e["mean_speed"] for e in diff_eps])),
            "visual_contact_pct": float(np.mean([e["visual_contact_pct"] for e in diff_eps])),
            "collisions": int(sum([e["collisions"] for e in diff_eps])),
        }
        trajectories_to_plot["AC-LDM (Diffusion) MPC"] = diff_eps[0]
    else:
        print(f"\n[4/4] Checkpoints missing for AC-LDM MPC ({diff_path}, {vae_path}, or {rew_vae_path}). Skipping.")

    env.close()

    # Save benchmark JSON
    benchmark_file = out_dir / "mpc_evaluation_results.json"
    with open(benchmark_file, "w") as f:
        json.dump(results, f, indent=2)

    # Print Comparative Results Table
    print("\n" + "=" * 80)
    print("                  CLOSED-LOOP CONTROL BENCHMARK RESULTS                       ")
    print("=" * 80)
    print(f"{'Policy':<25} | {'Reward':>9} | {'Mean Dist':>9} | {'Mean Speed':>10} | {'Visual Lock':>11} | {'Collisions':>10}")
    print("-" * 80)
    for pol_name, m in results.items():
        print(
            f"{pol_name:<25} | "
            f"{m['mean_reward']:>9.1f} | "
            f"{m['mean_dist']:>8.2f}m | "
            f"{m['mean_speed']:>8.2f}m/s | "
            f"{m['visual_contact_pct']:>10.1f}% | "
            f"{m['collisions']:>10d}"
        )
    print("=" * 80)

    # Plot trajectories
    if trajectories_to_plot:
        plt.figure(figsize=(9, 9))
        colors = {
            "Random": "grey",
            "PPO (Reactive)": "dodgerblue",
            "LeWM (SIGReg) MPC": "forestgreen",
            "AC-LDM (Diffusion) MPC": "darkorange",
        }

        # Plot obstacles
        for ox, oy, r in [(1.8, 1.8, 0.35), (-2.0, 1.5, 0.40), (2.0, -2.0, 0.45), (-2.2, -1.8, 0.35), (0.0, 3.0, 0.40), (0.0, -3.0, 0.40)]:
            circle = plt.Circle((ox, oy), r, color="darkred", alpha=0.3)
            plt.gca().add_patch(circle)

        # Plot target from first trajectory
        first_ep = next(iter(trajectories_to_plot.values()))
        tgt_x = [p[0] for p in first_ep["target_traj"]]
        tgt_y = [p[1] for p in first_ep["target_traj"]]
        plt.plot(tgt_x, tgt_y, color="crimson", linestyle="--", linewidth=2.5, label="Target Walking Path")
        plt.scatter(tgt_x[0], tgt_y[0], color="crimson", marker="*", s=150, zorder=5, label="Target Start")

        for pol_name, ep_data in trajectories_to_plot.items():
            rob_x = [p[0] for p in ep_data["robot_traj"]]
            rob_y = [p[1] for p in ep_data["robot_traj"]]
            col = colors.get(pol_name, "blue")
            plt.plot(rob_x, rob_y, color=col, linewidth=2, label=f"AckBot: {pol_name}")
            plt.scatter(rob_x[0], rob_y[0], color=col, marker="o", s=80, zorder=5)

        plt.xlim(-5.2, 5.2)
        plt.ylim(-5.2, 5.2)
        plt.title("Closed-Loop Target Following Trajectories (MPC vs Baselines)")
        plt.xlabel("X Position (m)")
        plt.ylabel("Y Position (m)")
        plt.legend(loc="lower right")
        plt.grid(True, linestyle="--", alpha=0.4)

        traj_plot_path = out_dir / "mpc_tracking_trajectories.png"
        plt.savefig(traj_plot_path, dpi=200)
        plt.close()
        print(f"Saved tracking trajectory plot to: {traj_plot_path}")


if __name__ == "__main__":
    main()
