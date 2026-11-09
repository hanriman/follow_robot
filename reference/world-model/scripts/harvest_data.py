#!/usr/bin/env python3
"""
Trajectory Rollout Harvester for Gymnasium CarRacing-v3.

Generates 50,000 verified offline transitions with synchronized physical telemetry:
- Centerline / waypoint heuristic policy with Ornstein-Uhlenbeck action noise
- Automatic shard serialization into data/raw/ (uint8 compressed)
- Train / Validation / Test split generation into data/processed/dataset_index.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from wm_control.config import (
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    EnvConfig,
)
from wm_control.data.dataset import (
    TELEMETRY_KEYS,
    telemetry_dict_to_array,
)
from wm_control.data.env_wrapper import make_carracing_env
from wm_control.data.heuristic_policy import CarRacingHeuristicPolicy
from wm_control.utils.seed import seed_everything


def harvest_trajectories(
    num_transitions: int = 50_000,
    shard_size: int = 2_500,
    raw_dir: Path = DATA_RAW_DIR,
    processed_dir: Path = DATA_PROCESSED_DIR,
    seed: int = 42,
    noise_scale: float = 0.15,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> Path:
    """
    Harvest offline transitions and write chunked shards to disk.
    """
    seed_everything(seed)
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    env_config = EnvConfig(target_height=64, target_width=64, crop_dashboard=True)
    env = make_carracing_env(config=env_config)
    policy = CarRacingHeuristicPolicy(noise_scale=noise_scale)

    total_shards = math.ceil(num_transitions / shard_size)
    print(
        f"\nStarting Trajectory Harvesting:\n"
        f"  Total Transitions: {num_transitions:,}\n"
        f"  Shard Size:        {shard_size:,}\n"
        f"  Expected Shards:   {total_shards}\n"
        f"  Raw Destination:   {raw_dir}\n"
        f"  Index Destination: {processed_dir}\n"
    )

    # Buffers for current shard
    buf_obs: list[np.ndarray] = []
    buf_next_obs: list[np.ndarray] = []
    buf_actions: list[np.ndarray] = []
    buf_rewards: list[float] = []
    buf_terminals: list[bool] = []
    buf_telemetry: list[np.ndarray] = []
    buf_next_telemetry: list[np.ndarray] = []

    shard_paths: list[str] = []
    current_shard_idx = 0
    total_harvested = 0
    episode_idx = 0

    pbar = tqdm(total=num_transitions, desc="Harvesting transitions", unit="steps")
    start_time = time.time()

    while total_harvested < num_transitions:
        episode_seed = seed + episode_idx
        obs, info = env.reset(seed=episode_seed)
        policy.reset()
        episode_idx += 1
        telem = info.get("telemetry", env.get_telemetry())
        telem_vec = telemetry_dict_to_array(telem)

        # Pre-roll 30 steps with neutral gas to clear starting zoom-in animation
        for _ in range(30):
            obs, _, _, _, info = env.step(np.array([0.0, 0.5, 0.0], dtype=np.float32))
            telem = info.get("telemetry", env.get_telemetry())
            telem_vec = telemetry_dict_to_array(telem)

        episode_steps = 0
        done = False

        while not done and total_harvested < num_transitions:
            action = policy.select_action(env, telem)
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated

            next_telem = next_info.get("telemetry", env.get_telemetry())
            next_telem_vec = telemetry_dict_to_array(next_telem)

            # Store in buffer: convert float [0, 1] frame to uint8 [0, 255] for 4x disk compression
            obs_uint8 = np.clip(obs * 255.0, 0, 255).astype(np.uint8)
            next_obs_uint8 = np.clip(next_obs * 255.0, 0, 255).astype(np.uint8)

            buf_obs.append(obs_uint8)
            buf_next_obs.append(next_obs_uint8)
            buf_actions.append(action)
            buf_rewards.append(reward)
            buf_terminals.append(done)
            buf_telemetry.append(telem_vec)
            buf_next_telemetry.append(next_telem_vec)

            obs = next_obs
            telem = next_telem
            telem_vec = next_telem_vec

            total_harvested += 1
            episode_steps += 1
            pbar.update(1)

            # Flush buffer to disk when shard_size is reached
            if len(buf_obs) >= shard_size or total_harvested >= num_transitions:
                shard_filename = f"shard_{current_shard_idx:04d}.npz"
                shard_path = raw_dir / shard_filename

                np.savez_compressed(
                    shard_path,
                    obs=np.stack(buf_obs, axis=0),
                    next_obs=np.stack(buf_next_obs, axis=0),
                    action=np.stack(buf_actions, axis=0),
                    reward=np.array(buf_rewards, dtype=np.float32),
                    terminal=np.array(buf_terminals, dtype=np.bool_),
                    telemetry=np.stack(buf_telemetry, axis=0),
                    next_telemetry=np.stack(buf_next_telemetry, axis=0),
                )
                shard_paths.append(str(shard_path))
                current_shard_idx += 1

                # Clear buffers
                buf_obs.clear()
                buf_next_obs.clear()
                buf_actions.clear()
                buf_rewards.clear()
                buf_terminals.clear()
                buf_telemetry.clear()
                buf_next_telemetry.clear()

            # Safeguard maximum steps per episode to maintain track variety
            if episode_steps >= env_config.max_episode_steps:
                break

    pbar.close()
    env.close()

    elapsed = time.time() - start_time
    fps = total_harvested / max(elapsed, 1e-3)
    print(f"\nHarvesting Complete: {total_harvested:,} transitions across {len(shard_paths)} shards in {elapsed:.1f}s ({fps:.1f} FPS).")

    # Generate Train / Val / Test Splits (deterministic shard assignment)
    num_shards = len(shard_paths)
    if num_shards >= 3:
        n_train = max(1, round(num_shards * train_ratio))
        n_val = max(1, round(num_shards * val_ratio))
        if n_train + n_val >= num_shards:
            n_train = max(1, num_shards - 2)
            n_val = 1
        train_shards = shard_paths[:n_train]
        val_shards = shard_paths[n_train : n_train + n_val]
        test_shards = shard_paths[n_train + n_val :]
    elif num_shards == 2:
        train_shards = [shard_paths[0]]
        val_shards = [shard_paths[1]]
        test_shards = [shard_paths[1]]
    else:
        train_shards = shard_paths
        val_shards = shard_paths
        test_shards = shard_paths

    index_data = {
        "total_transitions": total_harvested,
        "num_shards": num_shards,
        "shard_size": shard_size,
        "telemetry_keys": TELEMETRY_KEYS,
        "train_shards": train_shards,
        "val_shards": val_shards,
        "test_shards": test_shards,
        "train_transitions": len(train_shards) * shard_size,
        "val_transitions": len(val_shards) * shard_size,
        "test_transitions": total_harvested - (len(train_shards) + len(val_shards)) * shard_size,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    index_file = processed_dir / "dataset_index.json"
    with open(index_file, "w") as f:
        json.dump(index_data, f, indent=2)

    print(f"Dataset index written to: {index_file}")
    print(f"  Train: {len(train_shards)} shards ({len(train_shards)*shard_size:,} transitions)")
    print(f"  Val:   {len(val_shards)} shards")
    print(f"  Test:  {len(test_shards)} shards")

    return index_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Harvest offline transitions from CarRacing-v3.")
    parser.add_argument("--num-transitions", type=int, default=50_000, help="Total transitions to harvest.")
    parser.add_argument("--shard-size", type=int, default=2_500, help="Transitions per compressed shard file.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument("--noise-scale", type=float, default=0.15, help="Action noise scale.")
    parser.add_argument("--raw-dir", type=Path, default=DATA_RAW_DIR, help="Raw shard directory.")
    parser.add_argument("--processed-dir", type=Path, default=DATA_PROCESSED_DIR, help="Processed index directory.")
    args = parser.parse_args()

    harvest_trajectories(
        num_transitions=args.num_transitions,
        shard_size=args.shard_size,
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        seed=args.seed,
        noise_scale=args.noise_scale,
    )


if __name__ == "__main__":
    main()
