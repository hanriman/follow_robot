"""
Unit tests for Visual Distractor Wrapper and Vectorized Batch Perturbations.

Enforces:
- Section 6.3 of agents.md: Test output isolation in tests/outputs/
"""

import gymnasium as gym
import numpy as np
import torch

from wm_control.data.distractor_wrapper import (
    VisualDistractorWrapper,
    apply_distractor_to_batch,
)


class DummyEnv(gym.Env):
    """Mock environment emitting synthetic (3, 64, 64) frames."""

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(3, 64, 64), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)

    def reset(self, **kwargs):
        return np.ones((3, 64, 64), dtype=np.float32) * 0.5, {}

    def step(self, action):
        return np.ones((3, 64, 64), dtype=np.float32) * 0.5, 1.0, False, False, {}


def test_visual_distractor_wrapper_modes():
    """Verify that distractor wrapper perturbs observations within [0, 1] bounds."""
    base_env = DummyEnv()

    # 1. Gaussian noise
    env_gauss = VisualDistractorWrapper(base_env, mode="gaussian", sigma=0.1)
    obs, _ = env_gauss.reset()
    assert obs.shape == (3, 64, 64)
    assert 0.0 <= obs.min() and obs.max() <= 1.0
    assert not np.allclose(obs, 0.5)

    # 2. Patch distractor
    env_patch = VisualDistractorWrapper(base_env, mode="patch", num_patches=2, patch_size=8)
    obs, _ = env_patch.reset()
    assert obs.shape == (3, 64, 64)
    assert 0.0 <= obs.min() and obs.max() <= 1.0


def test_apply_distractor_to_batch():
    """Verify vectorized PyTorch distractor injection."""
    batch = torch.full((4, 3, 64, 64), 0.5)

    # Zero noise should be identity
    clean = apply_distractor_to_batch(batch, mode="gaussian", sigma=0.0)
    assert torch.allclose(clean, batch)

    # Gaussian noise should stay bounded in [0, 1]
    noisy = apply_distractor_to_batch(batch, mode="gaussian", sigma=0.2)
    assert noisy.shape == (4, 3, 64, 64)
    assert noisy.min().item() >= 0.0 and noisy.max().item() <= 1.0
    assert not torch.allclose(noisy, batch)
