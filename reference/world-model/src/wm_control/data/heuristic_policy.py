"""
Automated Waypoint-Tracking Heuristic Policy with Exploration Perturbations.

Provides diverse, track-following offline continuous control rollouts:
1. Waypoint Tracking: Projects car heading toward lookahead track centerline coordinates.
2. Speed Regulation: Modulates throttle and braking based on track curvature.
3. Exploration Noise: Injects Ornstein-Uhlenbeck continuous action noise.
4. Off-Road Recovery: Detects off-track excursion and steers back toward centerline.
"""

from __future__ import annotations

import math
from typing import Any, List, Optional, Tuple

import gymnasium as gym
import numpy as np


class OrnsteinUhlenbeckActionNoise:
    """
    Ornstein-Uhlenbeck process for temporally correlated continuous exploration noise.
    dx = theta * (mu - x) * dt + sigma * dW
    """

    def __init__(
        self,
        action_dim: int = 3,
        mu: float = 0.0,
        theta: float = 0.15,
        sigma: float = 0.2,
        dt: float = 1e-2,
    ) -> None:
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.state = np.ones(self.action_dim, dtype=np.float32) * self.mu
        self.reset()

    def reset(self) -> None:
        self.state = np.ones(self.action_dim, dtype=np.float32) * self.mu

    def sample(self) -> np.ndarray:
        dx = (
            self.theta * (self.mu - self.state) * self.dt
            + self.sigma * np.sqrt(self.dt) * np.random.randn(self.action_dim).astype(np.float32)
        )
        self.state = self.state + dx
        return self.state.copy()


class CarRacingHeuristicPolicy:
    """
    Automated heuristic controller for continuous CarRacing-v3.

    Args:
        lookahead_tiles: Number of track tiles to look ahead along the track (default: 8).
        kp_steer: Proportional gain for steering angle error (default: 1.8).
        kd_steer: Derivative gain for steering angle rate of change (default: 0.2).
        target_straight_speed: Target speed on straight road segments (default: 38.0).
        target_corner_speed: Target speed on tight corners (default: 20.0).
        noise_scale: Multiplier for exploration perturbation noise.
        random_kick_prob: Probability of injecting random action pulses for recovery diversity.
    """

    def __init__(
        self,
        lookahead_tiles: int = 8,
        kp_steer: float = 1.8,
        kd_steer: float = 0.2,
        target_straight_speed: float = 38.0,
        target_corner_speed: float = 20.0,
        noise_scale: float = 0.15,
        random_kick_prob: float = 0.03,
    ) -> None:
        self.lookahead_tiles = lookahead_tiles
        self.kp_steer = kp_steer
        self.kd_steer = kd_steer
        self.target_straight_speed = target_straight_speed
        self.target_corner_speed = target_corner_speed
        self.noise_scale = noise_scale
        self.random_kick_prob = random_kick_prob

        self.ou_noise = OrnsteinUhlenbeckActionNoise(action_dim=3, sigma=0.25)
        self.last_angle_diff = 0.0
        self.current_tile_idx = 0

    def reset(self) -> None:
        self.ou_noise.reset()
        self.last_angle_diff = 0.0
        self.current_tile_idx = 0

    def select_action(
        self,
        env: gym.Env,
        telemetry: Optional[dict[str, float]] = None,
    ) -> np.ndarray:
        """
        Compute continuous action [steering, gas, brake] based on car physics and track geometry.
        """
        unwrapped = env.unwrapped
        car = getattr(unwrapped, "car", None)
        track = getattr(unwrapped, "track", None)

        # Fallback if car or track is uninitialized
        if car is None or track is None or len(track) == 0:
            return np.array([0.0, 0.5, 0.0], dtype=np.float32)

        car_x = float(car.hull.position[0])
        car_y = float(car.hull.position[1])
        car_angle = float(car.hull.angle)
        speed = math.hypot(car.hull.linearVelocity[0], car.hull.linearVelocity[1])

        num_tiles = len(track)

        # 1. Locate closest track tile around current index (local search window)
        search_range = 25
        search_indices = [
            (self.current_tile_idx + i) % num_tiles
            for i in range(-search_range, search_range + 1)
        ]
        dists = [
            math.hypot(track[idx][2] - car_x, track[idx][3] - car_y)
            for idx in search_indices
        ]
        min_dist_idx = int(np.argmin(dists))
        closest_tile_idx = search_indices[min_dist_idx]
        closest_dist = dists[min_dist_idx]
        self.current_tile_idx = closest_tile_idx

        # 2. Check for off-road excursion (CarRacing road width is ~8-10 units)
        is_off_road = closest_dist > 6.0

        if is_off_road:
            # Aggressive recovery: steer directly back toward the nearest track tile
            target_x = float(track[closest_tile_idx][2])
            target_y = float(track[closest_tile_idx][3])
            lookahead_dist = closest_dist
        else:
            # Normal driving: target waypoint is lookahead_tiles ahead
            lookahead_idx = (closest_tile_idx + self.lookahead_tiles) % num_tiles
            target_x = float(track[lookahead_idx][2])
            target_y = float(track[lookahead_idx][3])
            lookahead_dist = math.hypot(target_x - car_x, target_y - car_y)

        # 3. Calculate target heading vs car forward vector
        dx = target_x - car_x
        dy = target_y - car_y
        angle_to_target = math.atan2(dy, dx)
        # In Box2D CarRacing, car forward direction is car.hull.angle + pi/2
        car_forward_angle = car_angle + math.pi / 2.0
        angle_diff = (angle_to_target - car_forward_angle + math.pi) % (2.0 * math.pi) - math.pi

        # Derivative of angular error
        angle_rate = angle_diff - self.last_angle_diff
        self.last_angle_diff = angle_diff

        # 4. PID Steering computation
        steer = self.kp_steer * angle_diff + self.kd_steer * angle_rate
        steer = float(np.clip(steer, -1.0, 1.0))

        # 5. Throttle and Braking Regulation
        abs_err = abs(angle_diff)
        if is_off_road:
            # Low speed on grass to regain traction and steer onto road
            target_speed = 12.0
        elif abs_err > 0.4:
            # Sharp curve: slow down
            target_speed = self.target_corner_speed
        else:
            # Smooth road / straightaway
            target_speed = self.target_straight_speed

        if speed < target_speed:
            gas = 0.6 if is_off_road else (0.8 if speed < target_speed * 0.7 else 0.4)
            brake = 0.0
        else:
            gas = 0.0
            brake = 0.3 if speed > target_speed + 6.0 else 0.0

        # 6. Exploration Noise Injection (Ornstein-Uhlenbeck)
        noise = self.ou_noise.sample() * self.noise_scale
        steer = float(np.clip(steer + noise[0], -1.0, 1.0))
        gas = float(np.clip(gas + noise[1] * 0.5, 0.0, 1.0))
        brake = float(np.clip(brake + noise[2] * 0.3, 0.0, 1.0))

        # 7. Occasional Random Action Kicks (creates diverse perturbation recovery data)
        if np.random.rand() < self.random_kick_prob:
            kick_type = np.random.choice(["burst_gas", "hard_steer", "tap_brake"])
            if kick_type == "burst_gas":
                gas = 1.0
                brake = 0.0
            elif kick_type == "hard_steer":
                steer = float(np.random.choice([-0.9, 0.9]))
            elif kick_type == "tap_brake":
                brake = 0.6
                gas = 0.0

        return np.array([steer, gas, brake], dtype=np.float32)
