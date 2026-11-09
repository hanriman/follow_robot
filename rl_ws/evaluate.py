#!/usr/bin/env python3
"""
Evaluation and Visual Playback script for trained AckBot PPO policies.
Loads checkpoint weights, runs deterministic rollouts, and renders live action.
"""

import sys
import time
import math
import argparse
from pathlib import Path
import numpy as np
import torch

import gymnasium as gym
import rl_ws.gym_ackbot  # registers AckBotFollow-v0
from rl_ws.policies.ppo.actor_critic import ActorCritic


def parse_args():
    parser = argparse.ArgumentParser(description="AckBot Policy Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint file")
    parser.add_argument("--episodes", type=int, default=5, help="Number of test episodes")
    parser.add_argument("--render", action="store_true", help="Open live visual rendering window")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "mps", "cpu"])
    parser.add_argument("--seed", type=int, default=100, help="Random seed")
    return parser.parse_args()


def select_device(preferred):
    if preferred == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    elif preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Error: Checkpoint file '{ckpt_path}' does not exist.")
        sys.exit(1)

    device = select_device(args.device)

    # Load Checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    step = checkpoint.get("step", "N/A")

    print("=" * 68)
    print("               ACKBOT TRAINED POLICY EVALUATION                      ")
    print("=" * 68)
    print(f"Checkpoint       : {ckpt_path.name} (Step: {step})")
    print(f"Device           : {device}")
    print(f"Test Episodes    : {args.episodes}")
    print(f"Visual Rendering : {'ENABLED (Live Window)' if args.render else 'DISABLED (Headless)'}")
    print("=" * 68)

    render_mode = "human" if args.render else None
    env = gym.make("AckBotFollow-v0", render_mode=render_mode)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    model = ActorCritic(obs_dim=obs_dim, act_dim=act_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_rewards = []
    all_lengths = []
    all_target_dists = []
    all_speeds = []
    collisions = 0

    print(f"{'Ep':>3} | {'Reward':>8} | {'Steps':>6} | {'Avg Dist':>9} | {'Avg Speed':>10} | {'Outcome':>10}")
    print("-" * 68)

    for ep in range(1, args.episodes + 1):
        obs, info = env.reset(seed=args.seed + ep)
        done = False
        ep_ret = 0.0
        ep_len = 0
        ep_dists = []
        ep_speeds = []

        while not done:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                # Deterministic greedy action
                action = model.get_deterministic_action(obs_t).squeeze(0).cpu().numpy()

            obs, reward, terminated, truncated, info = env.step(action)
            ep_ret += reward
            ep_len += 1
            ep_dists.append(info.get("target_dist", 0.0))
            ep_speeds.append(abs(info.get("robot_v", 0.0)))

            if args.render:
                env.render()

            done = terminated or truncated
            if terminated and info.get("collision"):
                collisions += 1

        outcome = "COLLISION" if (terminated and info.get("collision")) else "SUCCESS"
        avg_dist = np.mean(ep_dists)
        avg_speed = np.mean(ep_speeds)

        all_rewards.append(ep_ret)
        all_lengths.append(ep_len)
        all_target_dists.append(avg_dist)
        all_speeds.append(avg_speed)

        print(
            f"{ep:3d} | "
            f"{ep_ret:8.2f} | "
            f"{ep_len:6d} | "
            f"{avg_dist:7.2f}m | "
            f"{avg_speed:7.2f}m/s | "
            f"{outcome:>10}"
        )

    env.close()

    print("=" * 68)
    print("BENCHMARK EVALUATION SUMMARY:")
    print(f"  • Mean Return          : {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}")
    print(f"  • Mean Target Distance : {np.mean(all_target_dists):.2f} m (Desired: 0.80 m)")
    print(f"  • Mean Speed           : {np.mean(all_speeds):.2f} m/s")
    print(f"  • Collision Rate       : {collisions / args.episodes * 100:.1f}%")
    print("=" * 68)


if __name__ == "__main__":
    main()
