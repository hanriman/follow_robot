"""
Offline Sharded Transition Dataset for AckBot World Models.

Handles:
- Multi-modal transition loading from compressed NPZ shards
- Ground truth 8-D physical telemetry extraction for linear probing
- Deterministic train / val / test splitting
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"

TELEMETRY_KEYS = [
    "rel_x",
    "rel_y",
    "target_dist",
    "rel_heading",
    "robot_v",
    "robot_steer",
    "target_vx",
    "target_vy",
]


class AckBotTransitionDataset(Dataset):
    """
    In-memory or shard-loaded PyTorch Dataset of verified AckBot transitions.
    """

    def __init__(
        self,
        shard_paths: list[str | Path],
        augment_noise: float = 0.0,
    ) -> None:
        self.shard_paths = [Path(p) for p in shard_paths]
        self.augment_noise = float(augment_noise)

        # Buffers loaded into memory
        all_obs = []
        all_next_obs = []
        all_actions = []
        all_rewards = []
        all_terminals = []
        all_telem = []
        all_next_telem = []

        for p in self.shard_paths:
            if not p.exists():
                raise FileNotFoundError(f"Shard not found: {p}")
            data = np.load(p)
            all_obs.append(data["obs"])
            all_next_obs.append(data["next_obs"])
            all_actions.append(data["action"])
            all_rewards.append(data["reward"])
            all_terminals.append(data["terminal"])
            all_telem.append(data["telemetry"])
            all_next_telem.append(data["next_telemetry"])

        self.obs = torch.from_numpy(np.concatenate(all_obs, axis=0)).float()
        self.next_obs = torch.from_numpy(np.concatenate(all_next_obs, axis=0)).float()
        self.actions = torch.from_numpy(np.concatenate(all_actions, axis=0)).float()
        self.rewards = torch.from_numpy(np.concatenate(all_rewards, axis=0)).float()
        self.terminals = torch.from_numpy(np.concatenate(all_terminals, axis=0)).bool()
        self.telemetry = torch.from_numpy(np.concatenate(all_telem, axis=0)).float()
        self.next_telemetry = torch.from_numpy(np.concatenate(all_next_telem, axis=0)).float()

        self.length = self.obs.shape[0]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        obs = self.obs[idx]
        next_obs = self.next_obs[idx]

        if self.augment_noise > 0.0:
            # Add small Gaussian noise: clamp exteroceptive sensors to [0, 1] and proprioception to [-1, 1]
            noise_obs = obs + torch.randn_like(obs) * self.augment_noise
            obs = torch.cat(
                [
                    torch.clamp(noise_obs[:424], 0.0, 1.0),
                    torch.clamp(noise_obs[424:], -1.0, 1.0),
                ]
            )

            noise_next = next_obs + torch.randn_like(next_obs) * self.augment_noise
            next_obs = torch.cat(
                [
                    torch.clamp(noise_next[:424], 0.0, 1.0),
                    torch.clamp(noise_next[424:], -1.0, 1.0),
                ]
            )

        return {
            "obs": obs,
            "next_obs": next_obs,
            "action": self.actions[idx],
            "reward": self.rewards[idx],
            "terminal": self.terminals[idx],
            "telemetry": self.telemetry[idx],
            "next_telemetry": self.next_telemetry[idx],
        }


def get_dataset_splits(
    index_file: Path | None = None,
    augment_train: bool = True,
    noise_std: float = 0.01,
) -> tuple[AckBotTransitionDataset, AckBotTransitionDataset, AckBotTransitionDataset]:
    """
    Load canonical train, validation, and test datasets according to dataset_index.json.
    """
    if index_file is None:
        index_file = DATA_PROCESSED_DIR / "dataset_index.json"

    if not index_file.exists():
        raise FileNotFoundError(
            f"\n[FAIL LOUD] Dataset index file not found: {index_file}\n"
            f"Remediation: Run data harvesting first:\n"
            f"  >> uv run ackbot-harvest --num-transitions 50000\n"
        )

    with open(index_file, "r") as f:
        meta = json.load(f)

    train_shards = meta["train_shards"]
    val_shards = meta["val_shards"]
    test_shards = meta["test_shards"]

    train_ds = AckBotTransitionDataset(
        train_shards, augment_noise=noise_std if augment_train else 0.0
    )
    val_ds = AckBotTransitionDataset(val_shards, augment_noise=0.0)
    test_ds = AckBotTransitionDataset(test_shards, augment_noise=0.0)

    return train_ds, val_ds, test_ds
