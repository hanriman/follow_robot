#!/usr/bin/env python3
"""
AckBot Sim-to-Real Robustness & Hardware Disturbance Benchmark.

Evaluates controller robustness under realistic hardware perturbations:
1. Gaussian Sensor Noise: Simulated RealSense depth jitter & RPLiDAR range noise.
2. Camera Dropout Bursts: Prolonged USB/driver frame dropouts (0.3s to 1.5s blackouts).
3. LiDAR Blind-Spot Occlusions: Angular masking from dirt/cables/occluding objects.
4. Actuator Transport Delay: 1-step CAN/serial latency (100 ms).

Compares:
- Reactive PPO (Curriculum checkpoint)
- Teacher-Free LeWM CEM-MPC (SIGReg curriculum checkpoint)
- AC-LDM Diffusion CEM-MPC (Diffusion curriculum checkpoint)

Outputs:
- outputs/reports/sim2real_robustness/robustness_metrics.json
- outputs/reports/sim2real_robustness/robustness_comparison_bar.png
- outputs/reports/sim2real_robustness/dropout_recovery_curve.png
"""

from __future__ import annotations

import argparse
from collections import deque
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
DEFAULT_PPO = PROJECT_ROOT / "outputs/experiments/exp_20261108_154753_curriculum_ppo_active_follow/checkpoints/best_model.pt"
DEFAULT_LEWM = PROJECT_ROOT / "outputs/experiments/sigreg_curriculum/checkpoints/best_model.pt"
DEFAULT_VAE = PROJECT_ROOT / "outputs/experiments/vae_curriculum/checkpoints/best_model.pt"
DEFAULT_DIFF = PROJECT_ROOT / "outputs/experiments/diffusion_curriculum/checkpoints/best_model.pt"
DEFAULT_REW_LEWM = PROJECT_ROOT / "outputs/experiments/mpc_curriculum/checkpoints/reward_head_lewm.pt"
DEFAULT_REW_VAE = PROJECT_ROOT / "outputs/experiments/mpc_curriculum/checkpoints/reward_head_vae.pt"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs/reports/sim2real_robustness"


class Sim2RealPerturbationWrapper(gym.Wrapper):
    """
    Gymnasium Wrapper injecting real-world sensor and actuator degradations.
    """

    def __init__(
        self,
        env: gym.Env,
        noise_std: float = 0.0,
        dropout_prob: float = 0.0,
        dropout_len: int = 0,
        lidar_occlusion_deg: float = 0.0,
        actuator_delay: int = 0,
    ):
        super().__init__(env)
        self.noise_std = float(noise_std)
        self.dropout_prob = float(dropout_prob)
        self.dropout_len = int(dropout_len)
        self.lidar_occlusion_deg = float(lidar_occlusion_deg)
        self.actuator_delay = int(actuator_delay)

        self.dropout_timer = 0
        self.action_queue: deque = deque()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.dropout_timer = 0
        self.action_queue.clear()
        for _ in range(self.actuator_delay):
            self.action_queue.append(np.zeros(2, dtype=np.float32))
        return self._perturb_observation(obs), info

    def step(self, action: np.ndarray):
        if self.actuator_delay > 0:
            self.action_queue.append(action.copy())
            exec_action = self.action_queue.popleft()
        else:
            exec_action = action

        obs, reward, terminated, truncated, info = self.env.step(exec_action)
        perturbed_obs = self._perturb_observation(obs)
        return perturbed_obs, reward, terminated, truncated, info

    def _perturb_observation(self, obs: np.ndarray) -> np.ndarray:
        p_obs = obs.copy()

        # 1. Camera Drop-Out Burst (zero out / max out camera depth 0:64)
        if self.dropout_timer > 0:
            p_obs[:64] = 1.0  # Max distance (no target signature)
            self.dropout_timer -= 1
        elif self.dropout_prob > 0.0 and np.random.rand() < self.dropout_prob:
            self.dropout_timer = self.dropout_len
            p_obs[:64] = 1.0

        # 2. LiDAR Blind-Spot Occlusion
        if self.lidar_occlusion_deg > 0.0:
            # Occlude front sector [-deg/2, +deg/2]
            n_rays = int((self.lidar_occlusion_deg / 360.0) * 360)
            half = n_rays // 2
            # 64:424 is LiDAR (0 is front)
            p_obs[64 : 64 + half] = 1.0
            p_obs[424 - half : 424] = 1.0

        # 3. Gaussian Sensor Noise
        if self.noise_std > 0.0:
            noise = np.random.normal(0.0, self.noise_std, size=p_obs.shape).astype(np.float32)
            # Sensors 0:424 are in [0, 1]
            p_obs[:424] = np.clip(p_obs[:424] + noise[:424], 0.0, 1.0)
            # Proprioception 424:426 is in [-1, 1]
            p_obs[424:] = np.clip(p_obs[424:] + noise[424:] * 0.5, -1.0, 1.0)

        return p_obs


def evaluate_policy_on_regime(
    env: gym.Env,
    policy_fn,
    episodes: int = 3,
    reset_fn=None,
    seed: int = 42,
    stage: int = 2,
    num_distractors: int = 0,
) -> dict:
    rewards = []
    dist_errors = []
    visual_locks = []
    collisions = []

    for ep in range(episodes):
        ep_seed = seed + ep * 100
        obs, info = env.reset(seed=ep_seed, options={"stage": stage, "num_distractors": num_distractors})
        if reset_fn:
            reset_fn()

        ep_rew = 0.0
        step_dists = []
        step_locks = []
        ep_coll = 0
        done = False
        steps = 0

        while not done and steps < 250:
            act = policy_fn(obs)
            obs, rew, term, trunc, info = env.step(act)
            steps += 1
            ep_rew += rew
            d = info.get("target_dist", 1.0)
            step_dists.append(abs(d - 0.80))
            step_locks.append(1.0 if info.get("target_in_fov", False) else 0.0)
            if info.get("collision", False):
                ep_coll += 1
            done = term or trunc

        rewards.append(ep_rew)
        dist_errors.append(float(np.mean(step_dists)) if step_dists else 0.0)
        visual_locks.append(float(np.mean(step_locks) * 100.0) if step_locks else 0.0)
        collisions.append(ep_coll)

    return {
        "mean_reward": float(np.mean(rewards)),
        "mean_dist_error": float(np.mean(dist_errors)),
        "visual_lock_pct": float(np.mean(visual_locks)),
        "collisions": int(sum(collisions)),
    }


def main():
    parser = argparse.ArgumentParser(description="Sim-to-Real Robustness Benchmark")
    parser.add_argument("--episodes", type=int, default=3, help="Episodes per perturbation regime")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "mps", "cpu"])
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("mps") if args.device == "mps" or (args.device == "auto" and torch.backends.mps.is_available()) else torch.device("cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print("      ACKBOT SIM-TO-REAL HARDWARE ROBUSTNESS BENCHMARK SUITE          ")
    print("=" * 76)
    print(f"Device              : {device}")
    print(f"Evaluation Episodes : {args.episodes}")
    print(f"Output Directory    : {out_dir}")

    # Load Policies
    # 1. PPO Policy
    ppo_ckpt = torch.load(DEFAULT_PPO, map_location=device, weights_only=False)
    ppo_model = ActorCritic(obs_dim=426, act_dim=2).to(device)
    ppo_model.load_state_dict(ppo_ckpt["model_state_dict"])
    ppo_model.eval()

    def ppo_policy(obs):
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            return ppo_model.get_deterministic_action(obs_t).squeeze(0).cpu().numpy()

    # 2. LeWM CEM-MPC
    lewm_ckpt = torch.load(DEFAULT_LEWM, map_location=device, weights_only=False)
    lewm = TeacherFreeLeWorldModel(latent_dim=128, action_dim=2).to(device)
    lewm.load_state_dict(lewm_ckpt["model_state_dict"])
    lewm.eval()

    rew_lewm_ckpt = torch.load(DEFAULT_REW_LEWM, map_location=device, weights_only=False)
    rew_head_lewm = LatentRewardHead(in_dim=128, is_spatial=False).to(device)
    rew_head_lewm.load_state_dict(rew_lewm_ckpt["model_state_dict"])
    rew_head_lewm.eval()

    planner_lewm = CEMPlanner(horizon=5, pop_size=35, num_elites=5, num_iters=3, device=device)

    def lewm_policy(obs):
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            z_0 = lewm.encode(obs_t)
            act_t = planner_lewm.plan(
                initial_z=z_0,
                rollout_fn=lambda z, acts: lewm.predictor.rollout(z, acts),
                score_fn=lambda unrolled: rew_head_lewm.compute_control_score(unrolled),
            )
            return act_t.cpu().numpy()

    # 3. AC-LDM Diffusion CEM-MPC
    vae_ckpt = torch.load(DEFAULT_VAE, map_location=device, weights_only=False)
    vae = MultiModalSensorVAE(in_dim=426, latent_channels=4, latent_length=32).to(device)
    vae.load_state_dict(vae_ckpt["model_state_dict"])
    vae.eval()

    diff_ckpt = torch.load(DEFAULT_DIFF, map_location=device, weights_only=False)
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

    rew_vae_ckpt = torch.load(DEFAULT_REW_VAE, map_location=device, weights_only=False)
    rew_head_vae = LatentRewardHead(in_dim=128, is_spatial=True, spatial_channels=4, spatial_length=32).to(device)
    rew_head_vae.load_state_dict(rew_vae_ckpt["model_state_dict"])
    rew_head_vae.eval()

    planner_diff = CEMPlanner(horizon=3, pop_size=15, num_elites=3, num_iters=2, device=device)

    def diff_policy(obs):
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            z_0, _ = vae.encode(obs_t)
            act_t = planner_diff.plan(
                initial_z=z_0,
                rollout_fn=lambda z, acts: diff_wm.rollout_latents(z, acts, ddim_steps=4),
                score_fn=lambda unrolled: rew_head_vae.compute_control_score(unrolled),
            )
            return act_t.cpu().numpy()

    # Define Perturbation Regimes
    regimes = {
        "Nominal (Clean)": {
            "noise_std": 0.0,
            "dropout_prob": 0.0,
            "dropout_len": 0,
            "lidar_occlusion_deg": 0.0,
            "actuator_delay": 0,
        },
        "Sensor Noise & Occlusion": {
            "noise_std": 0.05,
            "dropout_prob": 0.0,
            "dropout_len": 0,
            "lidar_occlusion_deg": 45.0,
            "actuator_delay": 0,
        },
        "Severe Camera Dropouts (1.0s)": {
            "noise_std": 0.03,
            "dropout_prob": 0.10,
            "dropout_len": 10,  # 10 steps = 1.0 second blackout
            "lidar_occlusion_deg": 0.0,
            "actuator_delay": 0,
        },
        "Harsh Compound (Delay+Noise+Dropouts)": {
            "noise_std": 0.05,
            "dropout_prob": 0.08,
            "dropout_len": 8,
            "lidar_occlusion_deg": 30.0,
            "actuator_delay": 1,  # 100 ms transport delay
        },
    }

    base_env = gym.make("AckBotFollow-v0")
    results = {}

    for regime_name, params in regimes.items():
        print(f"\nEvaluating Regime: {regime_name}...")
        w_env = Sim2RealPerturbationWrapper(base_env, **params)

        res_ppo = evaluate_policy_on_regime(w_env, ppo_policy, episodes=args.episodes, seed=args.seed)
        res_lewm = evaluate_policy_on_regime(w_env, lewm_policy, episodes=args.episodes, reset_fn=planner_lewm.reset, seed=args.seed)
        res_diff = evaluate_policy_on_regime(w_env, diff_policy, episodes=args.episodes, reset_fn=planner_diff.reset, seed=args.seed)

        results[regime_name] = {
            "PPO": res_ppo,
            "LeWM_MPC": res_lewm,
            "AC_LDM_MPC": res_diff,
        }

        print(f"  PPO        : Return = {res_ppo['mean_reward']:6.1f} | Lock = {res_ppo['visual_lock_pct']:5.1f}% | DistErr = {res_ppo['mean_dist_error']:4.2f}m")
        print(f"  LeWM MPC   : Return = {res_lewm['mean_reward']:6.1f} | Lock = {res_lewm['visual_lock_pct']:5.1f}% | DistErr = {res_lewm['mean_dist_error']:4.2f}m")
        print(f"  AC-LDM MPC : Return = {res_diff['mean_reward']:6.1f} | Lock = {res_diff['visual_lock_pct']:5.1f}% | DistErr = {res_diff['mean_dist_error']:4.2f}m")

    base_env.close()

    # Save JSON summary
    json_path = out_dir / "robustness_metrics.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Plot Comparison Bar Chart
    plt.figure(figsize=(13, 5))
    regime_labels = ["Nominal", "Noise+Occl", "Dropouts (1.0s)", "Compound"]
    x = np.arange(len(regime_labels))
    w = 0.25

    # Subplot 1: Return Retention
    plt.subplot(1, 3, 1)
    ppo_ret = [results[r]["PPO"]["mean_reward"] for r in regimes]
    lewm_ret = [results[r]["LeWM_MPC"]["mean_reward"] for r in regimes]
    diff_ret = [results[r]["AC_LDM_MPC"]["mean_reward"] for r in regimes]

    plt.bar(x - w, ppo_ret, width=w, label="Reactive PPO", color="crimson", alpha=0.85)
    plt.bar(x, lewm_ret, width=w, label="LeWM CEM-MPC", color="royalblue", alpha=0.85)
    plt.bar(x + w, diff_ret, width=w, label="AC-LDM MPC", color="mediumseagreen", alpha=0.85)
    plt.xticks(x, regime_labels, rotation=20, fontsize=8)
    plt.ylabel("Episodic Return")
    plt.title("Episodic Return across Regimes", fontsize=11, fontweight="bold")
    plt.legend(fontsize=8)
    plt.grid(True, linestyle=":", alpha=0.5)

    # Subplot 2: Visual Lock Retention
    plt.subplot(1, 3, 2)
    ppo_lock = [results[r]["PPO"]["visual_lock_pct"] for r in regimes]
    lewm_lock = [results[r]["LeWM_MPC"]["visual_lock_pct"] for r in regimes]
    diff_lock = [results[r]["AC_LDM_MPC"]["visual_lock_pct"] for r in regimes]

    plt.bar(x - w, ppo_lock, width=w, label="Reactive PPO", color="crimson", alpha=0.85)
    plt.bar(x, lewm_lock, width=w, label="LeWM CEM-MPC", color="royalblue", alpha=0.85)
    plt.bar(x + w, diff_lock, width=w, label="AC-LDM MPC", color="mediumseagreen", alpha=0.85)
    plt.xticks(x, regime_labels, rotation=20, fontsize=8)
    plt.ylabel("Visual Lock %")
    plt.title("Visual Lock % Under Disturbances", fontsize=11, fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.5)

    # Subplot 3: Target Distance Error
    plt.subplot(1, 3, 3)
    ppo_err = [results[r]["PPO"]["mean_dist_error"] for r in regimes]
    lewm_err = [results[r]["LeWM_MPC"]["mean_dist_error"] for r in regimes]
    diff_err = [results[r]["AC_LDM_MPC"]["mean_dist_error"] for r in regimes]

    plt.bar(x - w, ppo_err, width=w, label="Reactive PPO", color="crimson", alpha=0.85)
    plt.bar(x, lewm_err, width=w, label="LeWM CEM-MPC", color="royalblue", alpha=0.85)
    plt.bar(x + w, diff_err, width=w, label="AC-LDM MPC", color="mediumseagreen", alpha=0.85)
    plt.xticks(x, regime_labels, rotation=20, fontsize=8)
    plt.ylabel("Distance Error (meters)")
    plt.title("Tracking Distance Error (|d - 0.8m|)", fontsize=11, fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.5)

    plt.suptitle("AckBot Sim-to-Real Robustness Benchmark (Noise, Dropouts, Blind Sectors, Latency)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    chart_path = out_dir / "robustness_comparison_bar.png"
    plt.savefig(chart_path, dpi=200)
    plt.close()
    print(f"\nSaved robustness comparison bar chart: {chart_path.name}")
    print("=" * 76)


if __name__ == "__main__":
    main()
