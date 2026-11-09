#!/usr/bin/env python3
"""
AckBot Offline Transition Dataset Harvester.

Collects diverse, multi-modal transitions from AckBotFollowEnv using a blend of:
1. Pretrained PPO policy with exploration noise (expert tracking)
2. Heuristic pursuit controller with obstacle avoidance (geometric recovery)
3. Exploratory random actions (out-of-distribution state coverage)

Each transition records:
- Pure sensor observation (426-D): camera depth (64) + LiDAR (360) + state (2)
- Action executed (2-D): [target_v, target_steer]
- Reward and terminal flags
- 8-D Ground-truth physical telemetry: [rel_x, rel_y, target_dist, rel_heading, v, steer, target_vx, target_vy]

Outputs compressed NPZ shards and data/processed/dataset_index.json.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym

import rl_ws.gym_ackbot  # registers AckBotFollow-v0
from rl_ws.policies.ppo.actor_critic import ActorCritic

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CURRICULUM_PPO_CKPT = PROJECT_ROOT / "outputs/experiments/exp_20261108_154753_curriculum_ppo_active_follow/checkpoints/best_model.pt"
LEGACY_PPO_CKPT = PROJECT_ROOT / "outputs/experiments/exp_20261103_170712_ppo_active_follow/checkpoints/best_model.pt"
DEFAULT_PPO_CKPT = CURRICULUM_PPO_CKPT if CURRICULUM_PPO_CKPT.exists() else LEGACY_PPO_CKPT
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/curriculum_multi_ped"


def parse_args():
    parser = argparse.ArgumentParser(description="Harvest AckBot offline transitions for World Models")
    parser.add_argument("--num-transitions", type=int, default=25000, help="Total transitions to collect")
    parser.add_argument("--shard-size", type=int, default=5000, help="Transitions per NPZ shard")
    parser.add_argument("--ppo-checkpoint", type=str, default=str(DEFAULT_PPO_CKPT), help="Path to PPO model")
    parser.add_argument("--noise-std", type=float, default=0.15, help="Action exploration Gaussian noise std")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for environment and policy")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "mps", "cpu"])
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    parser.add_argument("--curriculum-mix", action="store_true", default=True, help="Harvest across 4 curriculum stages with distractors")
    parser.add_argument("--distractors", type=int, default=2, help="Number of distractors in stage 4")
    return parser.parse_args()


def select_device(preferred: str) -> torch.device:
    if preferred == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    elif preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def extract_telemetry(env: gym.Env) -> np.ndarray:
    """Extract ground-truth 8-D physical telemetry vector."""
    unwrapped = env.unwrapped
    rel_x, rel_y, rel_heading = unwrapped._get_relative_target_state()
    target_dist = math.hypot(unwrapped.target_x - unwrapped.robot_x, unwrapped.target_y - unwrapped.robot_y)
    return np.array(
        [
            rel_x,
            rel_y,
            target_dist,
            rel_heading,
            unwrapped.robot_v,
            unwrapped.robot_steer,
            unwrapped.target_vx,
            unwrapped.target_vy,
        ],
        dtype=np.float32,
    )


def heuristic_pursuit_action(env: gym.Env) -> np.ndarray:
    """Robust proportional pursuit policy with simple obstacle reflex."""
    unwrapped = env.unwrapped
    rel_x, rel_y, rel_heading = unwrapped._get_relative_target_state()
    target_dist = math.hypot(unwrapped.target_x - unwrapped.robot_x, unwrapped.target_y - unwrapped.robot_y)

    # Proportional steering towards relative heading
    steer_gain = 1.2
    target_steer = np.clip(steer_gain * rel_heading, -unwrapped.max_steering_angle, unwrapped.max_steering_angle)

    # Speed profile based on distance
    dist_error = target_dist - unwrapped.target_follow_distance
    if dist_error > 0.1:
        target_v = np.clip(0.3 + 0.5 * dist_error, 0.0, unwrapped.v_max)
    elif dist_error < -0.2:
        target_v = unwrapped.v_min * 0.5  # Reverse slightly if too close
    else:
        target_v = 0.25  # Cruise at target speed

    # LiDAR obstacle avoidance reflex: check front 60 degrees (rays 330 to 30)
    # in 360-sample lidar: 0 is front, angles advance CCW
    obs = unwrapped._get_observation()
    lidar_scan = obs[64:424] * 5.0  # Denormalize to meters (max 5.0m)
    front_rays = np.concatenate([lidar_scan[:30], lidar_scan[330:]])
    min_front_dist = np.min(front_rays)

    if min_front_dist < 0.45:
        # Obstacle reflex: steer away from closest side and slow down
        left_dist = np.mean(lidar_scan[10:50])
        right_dist = np.mean(lidar_scan[310:350])
        avoid_steer = unwrapped.max_steering_angle if left_dist > right_dist else -unwrapped.max_steering_angle
        target_steer = 0.7 * avoid_steer + 0.3 * target_steer
        target_v = min(target_v, 0.15)

    return np.array([target_v, target_steer], dtype=np.float32)


def main():
    args = parse_args()
    device = select_device(args.device)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    data_raw_dir = output_dir / "raw"
    data_processed_dir = output_dir / "processed"
    data_raw_dir.mkdir(parents=True, exist_ok=True)
    data_processed_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("              ACKBOT TRANSITION DATASET HARVESTER                     ")
    print("=" * 70)
    print(f"Target Transitions : {args.num_transitions}")
    print(f"Shard Size         : {args.shard_size}")
    print(f"Exploration Noise  : {args.noise_std}")
    print(f"Device             : {device}")
    print(f"Storage Raw Dir    : {data_raw_dir}")
    print(f"Curriculum Mix     : {args.curriculum_mix}")

    # Load PPO model if available
    ppo_model = None
    ppo_path = Path(args.ppo_checkpoint)
    if ppo_path.exists():
        print(f"Loading PPO Policy : {ppo_path.name}")
        checkpoint = torch.load(ppo_path, map_location=device, weights_only=False)
        ppo_model = ActorCritic(obs_dim=426, act_dim=2).to(device)
        ppo_model.load_state_dict(checkpoint["model_state_dict"])
        ppo_model.eval()
    else:
        print(f"Warning: PPO checkpoint not found at {ppo_path}. Using pure heuristic pursuit!")

    env = gym.make("AckBotFollow-v0")
    unwrapped = env.unwrapped
    v_min, v_max = unwrapped.v_min, unwrapped.v_max
    steer_max = unwrapped.max_steering_angle

    # Curriculum quota targets
    stage_targets = {
        1: int(args.num_transitions * 0.20),
        2: int(args.num_transitions * 0.20),
        3: int(args.num_transitions * 0.20),
        4: args.num_transitions - 3 * int(args.num_transitions * 0.20),
    }
    stage_counts = {1: 0, 2: 0, 3: 0, 4: 0}

    def choose_stage() -> int:
        if not args.curriculum_mix:
            return 1
        rem = {s: max(0, stage_targets[s] - stage_counts[s]) for s in [1, 2, 3, 4]}
        tot_rem = sum(rem.values())
        if tot_rem == 0:
            return int(np.random.choice([1, 2, 3, 4]))
        probs = [rem[s] / tot_rem for s in [1, 2, 3, 4]]
        return int(np.random.choice([1, 2, 3, 4], p=probs))

    # Buffers for current shard
    shard_obs = []
    shard_next_obs = []
    shard_actions = []
    shard_rewards = []
    shard_terminals = []
    shard_telemetry = []
    shard_next_telemetry = []

    shard_paths = []
    total_collected = 0
    shard_idx = 0
    episode_step = 0

    current_stage = choose_stage()
    current_distractors = args.distractors if current_stage == 4 else 0
    obs, _ = env.reset(seed=args.seed, options={"stage": current_stage, "num_distractors": current_distractors})
    current_telem = extract_telemetry(env)

    print("\nHarvesting transitions...")

    while total_collected < args.num_transitions:
        mode_roll = np.random.rand()

        if ppo_model is not None and mode_roll < 0.65:
            # Mode 1: PPO with Gaussian exploration noise
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                base_act = ppo_model.get_deterministic_action(obs_t).squeeze(0).cpu().numpy()
            noise = np.random.normal(0.0, args.noise_std, size=base_act.shape)
            action = base_act + noise
        elif mode_roll < 0.90 or ppo_model is None:
            # Mode 2: Heuristic pursuit controller
            base_act = heuristic_pursuit_action(env)
            noise = np.random.normal(0.0, args.noise_std * 0.8, size=base_act.shape)
            action = base_act + noise
        else:
            # Mode 3: Uniform exploration / recovery
            action = np.array(
                [
                    np.random.uniform(v_min, v_max),
                    np.random.uniform(-steer_max, steer_max),
                ],
                dtype=np.float32,
            )

        # Enforce physical kinematic limits
        action[0] = np.clip(action[0], v_min, v_max)
        action[1] = np.clip(action[1], -steer_max, steer_max)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        next_telem = extract_telemetry(env)
        episode_step += 1
        done = terminated or truncated or (episode_step >= 150)

        shard_obs.append(obs)
        shard_next_obs.append(next_obs)
        shard_actions.append(action)
        shard_rewards.append(reward)
        shard_terminals.append(terminated or truncated)
        shard_telemetry.append(current_telem)
        shard_next_telemetry.append(next_telem)

        total_collected += 1
        stage_counts[current_stage] += 1

        if total_collected % 2500 == 0 or total_collected == args.num_transitions:
            pct = 100.0 * total_collected / args.num_transitions
            print(
                f"  [{pct:5.1f}%] Harvested {total_collected}/{args.num_transitions} | "
                f"S1: {stage_counts[1]} | S2: {stage_counts[2]} | S3: {stage_counts[3]} | S4: {stage_counts[4]}"
            )

        # Save shard when full
        if len(shard_obs) >= args.shard_size or total_collected >= args.num_transitions:
            shard_file = data_raw_dir / f"shard_{shard_idx:04d}.npz"
            np.savez_compressed(
                shard_file,
                obs=np.array(shard_obs, dtype=np.float32),
                next_obs=np.array(shard_next_obs, dtype=np.float32),
                action=np.array(shard_actions, dtype=np.float32),
                reward=np.array(shard_rewards, dtype=np.float32),
                terminal=np.array(shard_terminals, dtype=bool),
                telemetry=np.array(shard_telemetry, dtype=np.float32),
                next_telemetry=np.array(shard_next_telemetry, dtype=np.float32),
            )
            shard_paths.append(str(shard_file.resolve()))
            print(f"  --> Saved shard {shard_idx:04d} with {len(shard_obs)} transitions: {shard_file.name}")

            shard_idx += 1
            shard_obs.clear()
            shard_next_obs.clear()
            shard_actions.clear()
            shard_rewards.clear()
            shard_terminals.clear()
            shard_telemetry.clear()
            shard_next_telemetry.clear()

        if done:
            episode_step = 0
            current_stage = choose_stage()
            current_distractors = args.distractors if current_stage == 4 else 0
            obs, _ = env.reset(options={"stage": current_stage, "num_distractors": current_distractors})
            current_telem = extract_telemetry(env)
        else:
            obs = next_obs
            current_telem = next_telem

    env.close()

    # Create dataset splits: 80% train, 10% val, 10% test across shards
    n_shards = len(shard_paths)
    if n_shards >= 5:
        n_test = max(1, int(round(n_shards * 0.10)))
        n_val = max(1, int(round(n_shards * 0.10)))
        train_shards = shard_paths[: n_shards - n_val - n_test]
        val_shards = shard_paths[n_shards - n_val - n_test : n_shards - n_test]
        test_shards = shard_paths[n_shards - n_test :]
    else:
        # Single or few shards: train on all, val/test point to first/last
        train_shards = shard_paths
        val_shards = shard_paths[:1]
        test_shards = shard_paths[-1:]

    index_data = {
        "total_transitions": total_collected,
        "num_shards": len(shard_paths),
        "shard_size": args.shard_size,
        "stage_distribution": stage_counts,
        "train_shards": train_shards,
        "val_shards": val_shards,
        "test_shards": test_shards,
    }

    index_path = data_processed_dir / "dataset_index.json"
    with open(index_path, "w") as f:
        json.dump(index_data, f, indent=2)

    print("\nDataset index written successfully:")
    print(f"  Index file  : {index_path}")
    print(f"  Stage breakdown: {stage_counts}")
    print(f"  Train shards: {len(train_shards)} | Val shards: {len(val_shards)} | Test shards: {len(test_shards)}")
    print("=" * 70)
    print("=" * 70)


if __name__ == "__main__":
    main()
