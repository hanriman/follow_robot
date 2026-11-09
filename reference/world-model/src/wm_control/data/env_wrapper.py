"""
Gymnasium CarRacing-v3 Environment Wrapper with Physics Telemetry Extraction.

Preprocessing:
- Crops out bottom 12 dashboard rows (isolating track and vehicle)
- Resizes road observations to (64, 64, 3) -> (3, 64, 64) normalized in [0, 1]
- Synchronizes Box2D ground-truth physical state telemetry on every step
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from wm_control.config import EnvConfig


class CarRacingObservationWrapper(gym.Wrapper):
    """
    Standardized visual control wrapper for CarRacing-v3.

    Features:
    1. Crops the bottom dashboard panel (top 84 rows kept) to focus purely on visual track dynamics.
    2. Resizes observation to (target_height, target_width), default 64x64, channel-first (C, H, W).
    3. Normalizes pixel values to [0.0, 1.0] float32.
    4. Extracts exact Box2D physics state telemetry (linear and angular velocities, heading, coordinates).
    """

    def __init__(
        self,
        env: Optional[gym.Env] = None,
        config: Optional[EnvConfig] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        if config is None:
            config = EnvConfig()
        self.config = config

        if env is None:
            # Attempt requested env_id first, fallback if deprecated or unversioned
            env_id = self.config.env_id
            try:
                env = gym.make(env_id, continuous=True, render_mode=render_mode)
            except (gym.error.NameNotFound, gym.error.DeprecatedEnv, gym.error.VersionNotFound):
                fallback_id = "CarRacing-v3" if "v2" in env_id else "CarRacing-v2"
                env = gym.make(fallback_id, continuous=True, render_mode=render_mode)

        super().__init__(env)

        self.crop_dashboard = self.config.crop_dashboard
        self.target_height = self.config.target_height
        self.target_width = self.config.target_width

        # Update observation space: [C, H, W] in [0.0, 1.0] float32
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(3, self.target_height, self.target_width),
            dtype=np.float32,
        )

    def _process_frame(self, raw_frame: np.ndarray) -> np.ndarray:
        """
        Crop dashboard and resize frame to (3, target_height, target_width) float32 in [0, 1].
        """
        # Raw frame is (96, 96, 3) uint8
        if self.crop_dashboard and raw_frame.shape[0] >= 84:
            frame = raw_frame[:84, :, :]
        else:
            frame = raw_frame

        # Convert to torch tensor for fast bilinear interpolation: (1, 3, H, W)
        tensor = (
            torch.from_numpy(frame)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .div(255.0)
        )
        resized = F.interpolate(
            tensor,
            size=(self.target_height, self.target_width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        return resized.cpu().numpy().astype(np.float32)

    def get_telemetry(self) -> Dict[str, float]:
        """
        Extract ground-truth Box2D physical state telemetry from car body.
        """
        try:
            car = self.env.unwrapped.car
            if car is None or not hasattr(car, "hull") or car.hull is None:
                return self._empty_telemetry()

            # World frame velocities and positions
            pos_x = float(car.hull.position[0])
            pos_y = float(car.hull.position[1])
            vel_x = float(car.hull.linearVelocity[0])
            vel_y = float(car.hull.linearVelocity[1])
            speed = float(math.hypot(vel_x, vel_y))
            angle = float(car.hull.angle)
            angular_vel = float(car.hull.angularVelocity)

            # Local car frame velocities (longitudinal and lateral drift)
            local_vel = car.hull.GetLocalVector(car.hull.linearVelocity)
            lateral_vel = float(local_vel[0])
            forward_vel = float(local_vel[1])

            # Steering angle from front wheels
            steering_angle = 0.0
            if len(car.wheels) > 0 and hasattr(car.wheels[0], "joint") and car.wheels[0].joint is not None:
                steering_angle = float(car.wheels[0].joint.angle)

            return {
                "velocity_x": vel_x,
                "velocity_y": vel_y,
                "speed": speed,
                "forward_velocity": forward_vel,
                "lateral_velocity": lateral_vel,
                "angular_velocity": angular_vel,
                "angle": angle,
                "position_x": pos_x,
                "position_y": pos_y,
                "steering_angle": steering_angle,
            }
        except Exception:
            return self._empty_telemetry()

    def _empty_telemetry(self) -> Dict[str, float]:
        return {
            "velocity_x": 0.0,
            "velocity_y": 0.0,
            "speed": 0.0,
            "forward_velocity": 0.0,
            "lateral_velocity": 0.0,
            "angular_velocity": 0.0,
            "angle": 0.0,
            "position_x": 0.0,
            "position_y": 0.0,
            "steering_angle": 0.0,
        }

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        raw_obs, info = super().reset(seed=seed, options=options)
        processed_obs = self._process_frame(raw_obs)
        info["telemetry"] = self.get_telemetry()
        return processed_obs, info

    def step(
        self,
        action: Union[np.ndarray, list],
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Ensure action is float32 numpy array within valid continuous bounds
        act = np.asarray(action, dtype=np.float32)
        if act.shape != (3,):
            raise ValueError(f"Action must be shape (3,), got {act.shape}")

        act[0] = np.clip(act[0], -1.0, 1.0)  # steering
        act[1] = np.clip(act[1], 0.0, 1.0)   # gas
        act[2] = np.clip(act[2], 0.0, 1.0)   # brake

        raw_obs, reward, terminated, truncated, info = super().step(act)
        processed_obs = self._process_frame(raw_obs)
        info["telemetry"] = self.get_telemetry()
        return processed_obs, float(reward), terminated, truncated, info


def make_carracing_env(
    config: Optional[EnvConfig] = None,
    seed: Optional[int] = None,
    render_mode: Optional[str] = None,
) -> CarRacingObservationWrapper:
    """
    Factory helper to instantiate a wrapped CarRacing environment.
    """
    if config is None:
        config = EnvConfig()
    wrapped_env = CarRacingObservationWrapper(config=config, render_mode=render_mode)
    if seed is not None:
        wrapped_env.reset(seed=seed)
    return wrapped_env
