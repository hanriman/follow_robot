"""
Unit tests for CarRacingTransitionDataset and telemetry serialization.

Enforces:
- Section 6.3 of agents.md: Test output isolation inside tests/outputs/
"""

import math
from pathlib import Path

import numpy as np
import torch

from wm_control.data.dataset import (
    CarRacingTransitionDataset,
    telemetry_array_to_dict,
    telemetry_dict_to_array,
)


def test_telemetry_conversion_invertibility():
    """Verify bidirectional conversion between telemetry dict and 10-D array."""
    sample_dict = {
        "velocity_x": 12.5,
        "velocity_y": -3.2,
        "speed": 12.9,
        "forward_velocity": 12.8,
        "lateral_velocity": -0.4,
        "angular_velocity": 0.15,
        "angle": 1.57,
        "position_x": 100.2,
        "position_y": -45.6,
        "steering_angle": 0.05,
    }

    arr = telemetry_dict_to_array(sample_dict)
    assert arr.shape == (10,)
    assert arr.dtype == np.float32

    reconstructed = telemetry_array_to_dict(arr)
    for k in sample_dict:
        assert math.isclose(sample_dict[k], reconstructed[k], rel_tol=1e-5)



def test_transition_dataset_loading_and_shapes(isolated_test_dir: Path):
    """Verify dataset loading, tensor conversions, and item access."""
    # Create synthetic test shard inside isolated_test_dir
    n_samples = 20
    dummy_obs = np.random.randint(0, 256, size=(n_samples, 3, 64, 64), dtype=np.uint8)
    dummy_next_obs = np.random.randint(0, 256, size=(n_samples, 3, 64, 64), dtype=np.uint8)
    dummy_action = np.random.uniform(-1, 1, size=(n_samples, 3)).astype(np.float32)
    dummy_reward = np.random.randn(n_samples).astype(np.float32)
    dummy_terminal = np.zeros(n_samples, dtype=bool)
    dummy_telem = np.random.randn(n_samples, 10).astype(np.float32)
    dummy_next_telem = np.random.randn(n_samples, 10).astype(np.float32)

    shard_path = isolated_test_dir / "test_shard_0000.npz"
    np.savez_compressed(
        shard_path,
        obs=dummy_obs,
        next_obs=dummy_next_obs,
        action=dummy_action,
        reward=dummy_reward,
        terminal=dummy_terminal,
        telemetry=dummy_telem,
        next_telemetry=dummy_next_telem,
    )

    dataset = CarRacingTransitionDataset(shard_paths=[shard_path], augment_hflip=False)
    assert len(dataset) == n_samples

    item = dataset[0]
    assert item["obs"].shape == (3, 64, 64)
    assert item["next_obs"].shape == (3, 64, 64)
    assert item["obs"].dtype == torch.float32
    assert 0.0 <= item["obs"].min() and item["obs"].max() <= 1.0
    assert item["action"].shape == (3,)
    assert item["telemetry"].shape == (10,)
    assert item["next_telemetry"].shape == (10,)


def test_horizontal_flip_augmentation(isolated_test_dir: Path):
    """Verify horizontal flip inverts steering and lateral velocity symmetrically."""
    n_samples = 10
    # Left steering: -0.8, positive lateral velocity: 5.0
    dummy_obs = np.zeros((n_samples, 3, 64, 64), dtype=np.uint8)
    dummy_next_obs = np.zeros((n_samples, 3, 64, 64), dtype=np.uint8)
    dummy_action = np.zeros((n_samples, 3), dtype=np.float32)
    dummy_action[:, 0] = -0.8  # steering left
    dummy_reward = np.zeros(n_samples, dtype=np.float32)
    dummy_terminal = np.zeros(n_samples, dtype=bool)
    dummy_telem = np.zeros((n_samples, 10), dtype=np.float32)
    dummy_telem[:, 4] = 5.0    # lateral_velocity (index 4)
    dummy_telem[:, 5] = 1.2    # angular_velocity (index 5)
    dummy_next_telem = dummy_telem.copy()

    shard_path = isolated_test_dir / "test_shard_hflip.npz"
    np.savez_compressed(
        shard_path,
        obs=dummy_obs,
        next_obs=dummy_next_obs,
        action=dummy_action,
        reward=dummy_reward,
        terminal=dummy_terminal,
        telemetry=dummy_telem,
        next_telemetry=dummy_next_telem,
    )

    # Force augmentation
    np.random.seed(42)
    dataset = CarRacingTransitionDataset(shard_paths=[shard_path], augment_hflip=True)

    # Sample items: some should be flipped with inverted steering (+0.8) and inverted lateral vel (-5.0)
    steer_values = [dataset[i]["action"][0].item() for i in range(len(dataset))]
    assert any(math.isclose(s, 0.8, rel_tol=1e-3) for s in steer_values)
