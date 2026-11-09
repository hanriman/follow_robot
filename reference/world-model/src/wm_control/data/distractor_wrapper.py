"""
Visual Distractor and Task-Irrelevant Feature (TIF) Perturbation Wrapper.

Applies controlled visual noise, color shifts, and background distractors to
continuous visual control observations to test whether world model representations
overfit to passive pixel textures or preserve invariant physical dynamics.
"""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torchvision.transforms.functional as TF


class VisualDistractorWrapper(gym.ObservationWrapper):
    """
    Gymnasium Observation Wrapper injecting controllable visual distractors.

    Supported Perturbation Modes:
    - 'gaussian': Additive Gaussian pixel noise N(0, sigma^2).
    - 'color_jitter': Global brightness, contrast, and hue shifts.
    - 'patch': Random floating colored square patches in visual field.
    - 'combined': Simultaneous Gaussian noise + color jitter.

    Args:
        env: Gymnasium environment instance.
        mode: Distractor type ('gaussian', 'color_jitter', 'patch', 'combined').
        sigma: Standard deviation for Gaussian noise in [0.0, 1.0].
        jitter_factor: Scale factor for brightness/contrast jitter in [0.0, 1.0].
        num_patches: Number of floating distractor patches.
        patch_size: Square size of distractor patches in pixels.
        seed: Random seed for deterministic perturbation patterns.
    """

    def __init__(
        self,
        env: gym.Env,
        mode: str = "gaussian",
        sigma: float = 0.1,
        jitter_factor: float = 0.3,
        num_patches: int = 3,
        patch_size: int = 6,
        seed: int = 42,
    ) -> None:
        super().__init__(env)
        self.mode = mode
        self.sigma = float(sigma)
        self.jitter_factor = float(jitter_factor)
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.rng = np.random.default_rng(seed)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """
        Apply distractor transform to observation [3, H, W] float32 in [0, 1].
        """
        obs = observation.copy().astype(np.float32)

        if self.mode == "gaussian":
            noise = self.rng.normal(0.0, self.sigma, size=obs.shape).astype(np.float32)
            obs = np.clip(obs + noise, 0.0, 1.0)

        elif self.mode == "color_jitter":
            # Convert to PyTorch tensor for torchvision functional transforms
            t_obs = torch.from_numpy(obs)
            # Brightness & contrast jitter
            b_factor = 1.0 + float(self.rng.uniform(-self.jitter_factor, self.jitter_factor))
            c_factor = 1.0 + float(self.rng.uniform(-self.jitter_factor, self.jitter_factor))
            t_jitter = TF.adjust_brightness(t_obs, b_factor)
            t_jitter = TF.adjust_contrast(t_jitter, c_factor)
            obs = np.clip(t_jitter.numpy(), 0.0, 1.0)

        elif self.mode == "patch":
            _, h, w = obs.shape
            for _ in range(self.num_patches):
                px = self.rng.integers(0, max(1, w - self.patch_size))
                py = self.rng.integers(0, max(1, h - self.patch_size))
                color = self.rng.uniform(0.0, 1.0, size=(3, 1, 1)).astype(np.float32)
                obs[:, py : py + self.patch_size, px : px + self.patch_size] = color

        elif self.mode == "combined":
            # Add Gaussian noise
            noise = self.rng.normal(0.0, self.sigma, size=obs.shape).astype(np.float32)
            obs = np.clip(obs + noise, 0.0, 1.0)
            # Add patch
            _, h, w = obs.shape
            for _ in range(self.num_patches):
                px = self.rng.integers(0, max(1, w - self.patch_size))
                py = self.rng.integers(0, max(1, h - self.patch_size))
                color = self.rng.uniform(0.0, 1.0, size=(3, 1, 1)).astype(np.float32)
                obs[:, py : py + self.patch_size, px : px + self.patch_size] = color

        return obs.astype(np.float32)


def apply_distractor_to_batch(
    images: torch.Tensor,
    mode: str = "gaussian",
    sigma: float = 0.1,
    jitter_factor: float = 0.3,
) -> torch.Tensor:
    """
    Vectorized batch transform for PyTorch tensors [B, 3, H, W] in [0, 1].
    """
    if sigma <= 0.0 and mode == "gaussian":
        return images

    device = images.device
    out = images.clone()

    if mode == "gaussian":
        noise = torch.randn_like(out) * sigma
        out = torch.clamp(out + noise, 0.0, 1.0)
    elif mode == "color_jitter":
        # Random scale shift per batch item
        b_mult = 1.0 + (torch.rand(out.shape[0], 1, 1, 1, device=device) - 0.5) * 2.0 * jitter_factor
        out = torch.clamp(out * b_mult, 0.0, 1.0)
    return out
