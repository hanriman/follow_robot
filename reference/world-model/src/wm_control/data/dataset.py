"""
PyTorch Dataset and DataLoaders for Continuous CarRacing-v3 Trajectories.

Features:
- Memory-efficient uint8 on-disk storage with dynamic float32 normalization.
- Ground-truth Box2D physics telemetry vectors for linear probing (R²).
- Synchronized spatial-action augmentations (horizontal flip with sign inversion on steering & lateral drift).
- Deterministic train/validation/test shard splits.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from wm_control.config import DATA_PROCESSED_DIR, DATA_RAW_DIR, fail_loud

# Telemetry vector ordering
TELEMETRY_KEYS = [
    "velocity_x",
    "velocity_y",
    "speed",
    "forward_velocity",
    "lateral_velocity",
    "angular_velocity",
    "angle",
    "position_x",
    "position_y",
    "steering_angle",
]
TELEMETRY_DIM = len(TELEMETRY_KEYS)


def telemetry_dict_to_array(t_dict: Dict[str, float]) -> np.ndarray:
    """Pack a telemetry dictionary into a consistent 10-D float32 vector."""
    return np.array([float(t_dict.get(k, 0.0)) for k in TELEMETRY_KEYS], dtype=np.float32)


def telemetry_array_to_dict(arr: np.ndarray) -> Dict[str, float]:
    """Unpack a 10-D vector back into a telemetry dictionary."""
    return {k: float(arr[i]) for i, k in enumerate(TELEMETRY_KEYS)}


class CarRacingTransitionDataset(Dataset):
    """
    Offline Trajectory Transition Dataset yielding (o_t, a_t, o_{t+1}, r_t, d_t, telemetry_t, telemetry_{t+1}).

    Args:
        shard_paths: List of file paths to `.npz` shards.
        augment_hflip: Whether to apply synchronized horizontal flipping augmentation during training.
        normalize_to_minus_one_one: If True, normalizes pixel frames to [-1.0, 1.0], else [0.0, 1.0].
    """

    def __init__(
        self,
        shard_paths: List[Union[str, Path]],
        augment_hflip: bool = False,
        normalize_to_minus_one_one: bool = False,
    ) -> None:
        super().__init__()
        self.shard_paths = [Path(p) for p in shard_paths]
        self.augment_hflip = augment_hflip
        self.normalize_to_minus_one_one = normalize_to_minus_one_one

        if not self.shard_paths:
            raise ValueError("shard_paths cannot be empty.")

        # Load and concatenate shards into memory (or memory-map if large)
        self.obs: List[np.ndarray] = []
        self.next_obs: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.rewards: List[np.ndarray] = []
        self.terminals: List[np.ndarray] = []
        self.telemetry: List[np.ndarray] = []
        self.next_telemetry: List[np.ndarray] = []

        for p in self.shard_paths:
            if not p.exists():
                fail_loud(
                    missing_resource=p,
                    caller_context="CarRacingTransitionDataset.__init__",
                    remediation_cmd="python scripts/harvest_data.py --num-transitions 50000",
                )
            data = np.load(p)
            self.obs.append(data["obs"])
            self.next_obs.append(data["next_obs"])
            self.actions.append(data["action"])
            self.rewards.append(data["reward"])
            self.terminals.append(data["terminal"])
            self.telemetry.append(data["telemetry"])
            self.next_telemetry.append(data["next_telemetry"])

        self.obs = np.concatenate(self.obs, axis=0)
        self.next_obs = np.concatenate(self.next_obs, axis=0)
        self.actions = np.concatenate(self.actions, axis=0).astype(np.float32)
        self.rewards = np.concatenate(self.rewards, axis=0).astype(np.float32)
        self.terminals = np.concatenate(self.terminals, axis=0).astype(np.bool_)
        self.telemetry = np.concatenate(self.telemetry, axis=0).astype(np.float32)
        self.next_telemetry = np.concatenate(self.next_telemetry, axis=0).astype(np.float32)

        self.length = len(self.obs)

    def __len__(self) -> int:
        return self.length

    def _process_frame(self, frame_uint8: np.ndarray) -> torch.Tensor:
        """Convert [C, H, W] uint8 to float32 tensor."""
        t = torch.from_numpy(frame_uint8).float() / 255.0
        if self.normalize_to_minus_one_one:
            t = t * 2.0 - 1.0
        return t

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        obs = self.obs[idx].copy()
        next_obs = self.next_obs[idx].copy()
        action = self.actions[idx].copy()
        reward = self.rewards[idx]
        terminal = self.terminals[idx]
        telem = self.telemetry[idx].copy()
        next_telem = self.next_telemetry[idx].copy()

        # Synchronized horizontal flip augmentation
        if self.augment_hflip and np.random.rand() > 0.5:
            # Flip image along width (last dimension: [C, H, W])
            obs = np.flip(obs, axis=-1).copy()
            next_obs = np.flip(next_obs, axis=-1).copy()

            # Invert steering in action: [steering, gas, brake]
            action[0] = -action[0]

            # Invert lateral velocity and angular velocity in telemetry:
            # indices: lateral_velocity=4, angular_velocity=5, steering_angle=9
            telem[4] = -telem[4]
            telem[5] = -telem[5]
            telem[9] = -telem[9]

            next_telem[4] = -next_telem[4]
            next_telem[5] = -next_telem[5]
            next_telem[9] = -next_telem[9]

        return {
            "obs": self._process_frame(obs),
            "next_obs": self._process_frame(next_obs),
            "action": torch.from_numpy(action),
            "reward": torch.tensor(reward, dtype=torch.float32),
            "terminal": torch.tensor(terminal, dtype=torch.bool),
            "telemetry": torch.from_numpy(telem),
            "next_telemetry": torch.from_numpy(next_telem),
        }


def get_dataset_splits(
    index_file: Optional[Union[str, Path]] = None,
    augment_train: bool = True,
    normalize_to_minus_one_one: bool = False,
) -> Tuple[CarRacingTransitionDataset, CarRacingTransitionDataset, CarRacingTransitionDataset]:
    """
    Load official train, validation, and test datasets according to dataset_index.json.
    """
    if index_file is None:
        index_file = DATA_PROCESSED_DIR / "dataset_index.json"
    index_path = Path(index_file)

    if not index_path.exists():
        fail_loud(
            missing_resource=index_path,
            caller_context="get_dataset_splits",
            remediation_cmd="python scripts/harvest_data.py --num-transitions 50000",
        )

    with open(index_path, "r") as f:
        meta = json.load(f)

    train_shards = meta["train_shards"]
    val_shards = meta["val_shards"]
    test_shards = meta["test_shards"]

    train_ds = CarRacingTransitionDataset(
        shard_paths=train_shards,
        augment_hflip=augment_train,
        normalize_to_minus_one_one=normalize_to_minus_one_one,
    )
    val_ds = CarRacingTransitionDataset(
        shard_paths=val_shards,
        augment_hflip=False,
        normalize_to_minus_one_one=normalize_to_minus_one_one,
    )
    test_ds = CarRacingTransitionDataset(
        shard_paths=test_shards,
        augment_hflip=False,
        normalize_to_minus_one_one=normalize_to_minus_one_one,
    )

    return train_ds, val_ds, test_ds


class CarRacingLatentDataset(Dataset):
    """
    Dataset of pre-encoded spatial VAE latents (z_t, a_t, z_{t+1}, r_t, d_t, telem_t, telem_{t+1}).
    """

    def __init__(
        self,
        shard_paths: List[Union[str, Path]],
        augment_hflip: bool = False,
    ) -> None:
        super().__init__()
        self.shard_paths = [Path(p) for p in shard_paths]
        self.augment_hflip = augment_hflip

        if not self.shard_paths:
            raise ValueError("shard_paths cannot be empty.")

        self.z_t: List[np.ndarray] = []
        self.next_z: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.rewards: List[np.ndarray] = []
        self.terminals: List[np.ndarray] = []
        self.telemetry: List[np.ndarray] = []
        self.next_telemetry: List[np.ndarray] = []

        for p in self.shard_paths:
            if not p.exists():
                fail_loud(
                    missing_resource=p,
                    caller_context="CarRacingLatentDataset.__init__",
                    remediation_cmd="python scripts/preencode_latents.py",
                )
            data = np.load(p)
            self.z_t.append(data["z_t"])
            self.next_z.append(data["next_z"])
            self.actions.append(data["action"])
            self.rewards.append(data["reward"])
            self.terminals.append(data["terminal"])
            self.telemetry.append(data["telemetry"])
            self.next_telemetry.append(data["next_telemetry"])

        self.z_t = np.concatenate(self.z_t, axis=0).astype(np.float32)
        self.next_z = np.concatenate(self.next_z, axis=0).astype(np.float32)
        self.actions = np.concatenate(self.actions, axis=0).astype(np.float32)
        self.rewards = np.concatenate(self.rewards, axis=0).astype(np.float32)
        self.terminals = np.concatenate(self.terminals, axis=0).astype(np.bool_)
        self.telemetry = np.concatenate(self.telemetry, axis=0).astype(np.float32)
        self.next_telemetry = np.concatenate(self.next_telemetry, axis=0).astype(np.float32)

        self.length = len(self.z_t)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        z = self.z_t[idx].copy()
        next_z = self.next_z[idx].copy()
        act = self.actions[idx].copy()
        rew = self.rewards[idx]
        term = self.terminals[idx]
        telem = self.telemetry[idx].copy()
        next_telem = self.next_telemetry[idx].copy()

        # Synchronized horizontal spatial flip
        if self.augment_hflip and np.random.rand() > 0.5:
            z = np.flip(z, axis=-1).copy()
            next_z = np.flip(next_z, axis=-1).copy()
            act[0] = -act[0]
            telem[4] = -telem[4]
            telem[5] = -telem[5]
            telem[9] = -telem[9]
            next_telem[4] = -next_telem[4]
            next_telem[5] = -next_telem[5]
            next_telem[9] = -next_telem[9]

        return {
            "z_t": torch.from_numpy(z),
            "next_z": torch.from_numpy(next_z),
            "action": torch.from_numpy(act),
            "reward": torch.tensor(rew, dtype=torch.float32),
            "terminal": torch.tensor(term, dtype=torch.bool),
            "telemetry": torch.from_numpy(telem),
            "next_telemetry": torch.from_numpy(next_telem),
        }


def get_latent_dataset_splits(
    index_file: Optional[Union[str, Path]] = None,
    augment_train: bool = True,
) -> Tuple[CarRacingLatentDataset, CarRacingLatentDataset, CarRacingLatentDataset]:
    """
    Load official latent train, validation, and test datasets.
    """
    if index_file is None:
        index_file = DATA_PROCESSED_DIR / "latent_dataset_index.json"
    index_path = Path(index_file)

    if not index_path.exists():
        fail_loud(
            missing_resource=index_path,
            caller_context="get_latent_dataset_splits",
            remediation_cmd="python scripts/preencode_latents.py",
        )

    with open(index_path, "r") as f:
        meta = json.load(f)

    train_ds = CarRacingLatentDataset(shard_paths=meta["train_shards"], augment_hflip=augment_train)
    val_ds = CarRacingLatentDataset(shard_paths=meta["val_shards"], augment_hflip=False)
    test_ds = CarRacingLatentDataset(shard_paths=meta["test_shards"], augment_hflip=False)

    return train_ds, val_ds, test_ds
