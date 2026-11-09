"""
Bipedal Human Walker Kinematics and Dynamic Appearance Model for AckBot.

Simulates:
1. Two distinct physical legs/feet: Left (x_L, y_L, r_leg) and Right (x_R, y_R, r_leg)
2. Realistic walking gait cycle with cadence f ~ 1.6 Hz, stance width 0.22m, stride length 0.45m
3. Pelvis/torso center, heading kinematics, and turn dynamics
4. Dynamic clothing and appearance color variation randomized per episode
"""

from __future__ import annotations

import math
from typing import NamedTuple
import numpy as np


class LegState(NamedTuple):
    x: float
    y: float
    radius: float
    phase: float


CLOTHING_PALETTE = [
    (0.15, 0.30, 0.65),  # Navy Blue
    (0.80, 0.20, 0.20),  # Crimson Red
    (0.15, 0.60, 0.35),  # Emerald Green
    (0.85, 0.55, 0.15),  # Amber Orange
    (0.35, 0.20, 0.55),  # Royal Purple
    (0.20, 0.65, 0.70),  # Cyan / Teal
    (0.25, 0.25, 0.28),  # Charcoal
    (0.75, 0.75, 0.78),  # Light Grey
]


class BipedalHumanTarget:
    """
    Realistic Bipedal Human Walker with alternating 2-foot gait dynamics.
    """

    def __init__(
        self,
        x: float = 2.0,
        y: float = 0.0,
        yaw: float = 0.0,
        speed: float = 0.28,
        cadence_hz: float = 1.60,
        stride_length: float = 0.45,
        stance_width: float = 0.22,
        leg_radius: float = 0.09,
        torso_radius: float = 0.18,
        height: float = 1.75,
        color_rgb: tuple[float, float, float] | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.x = float(x)
        self.y = float(y)
        self.yaw = float(yaw)
        self.speed = float(speed)
        self.cadence_hz = float(cadence_hz)
        self.stride_length = float(stride_length)
        self.stance_width = float(stance_width)
        self.leg_radius = float(leg_radius)
        self.torso_radius = float(torso_radius)
        self.height = float(height)

        _rng = rng if rng is not None else np.random.default_rng()

        # Gait phase phi in [0, 2*pi)
        self.gait_phase = 0.0

        # Clothing / appearance color
        if color_rgb is not None:
            self.color_rgb = color_rgb
        else:
            self.color_rgb = CLOTHING_PALETTE[int(_rng.integers(len(CLOTHING_PALETTE)))]

        # Velocity components
        self.vx = self.speed * math.cos(self.yaw)
        self.vy = self.speed * math.sin(self.yaw)
        self.omega = 0.0

    def reset(
        self,
        x: float,
        y: float,
        yaw: float,
        speed: float = 0.28,
        color_idx: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Reset target state and sample new appearance."""
        _rng = rng if rng is not None else np.random.default_rng()
        self.x = float(x)
        self.y = float(y)
        self.yaw = float(yaw)
        self.speed = float(speed)
        self.gait_phase = float(_rng.uniform(0.0, 2.0 * np.pi))

        if color_idx is not None:
            self.color_rgb = CLOTHING_PALETTE[color_idx % len(CLOTHING_PALETTE)]
        else:
            self.color_rgb = CLOTHING_PALETTE[int(_rng.integers(len(CLOTHING_PALETTE)))]

        self.vx = self.speed * math.cos(self.yaw)
        self.vy = self.speed * math.sin(self.yaw)
        self.omega = 0.0

    def step(self, dt: float = 0.05) -> None:
        """Advance walker position, heading, and periodic bipedal gait."""
        # Update heading and position
        self.yaw = (self.yaw + self.omega * dt + math.pi) % (2.0 * math.pi) - math.pi
        self.vx = self.speed * math.cos(self.yaw)
        self.vy = self.speed * math.sin(self.yaw)

        self.x += self.vx * dt
        self.y += self.vy * dt

        # Advance gait phase proportionally to distance walked:
        # phi_dot = 2*pi * cadence * (speed / nominal_speed)
        if self.speed > 0.01:
            step_phase_delta = 2.0 * math.pi * self.cadence_hz * dt * (self.speed / 0.28)
            self.gait_phase = (self.gait_phase + step_phase_delta) % (2.0 * math.pi)

    def get_feet_positions(self) -> tuple[LegState, LegState]:
        """
        Calculate current world-frame coordinates of left and right feet/legs.

        Returns:
            (left_leg, right_leg) LegState tuples (x, y, radius, phase).
        """
        # Heading unit vectors
        cos_y = math.cos(self.yaw)
        sin_y = math.sin(self.yaw)

        # Forward unit vector: (cos_y, sin_y)
        # Left lateral unit vector: (-sin_y, cos_y)
        forward_x = cos_y
        forward_y = sin_y
        left_x = -sin_y
        left_y = cos_y

        half_stance = self.stance_width / 2.0
        half_stride = self.stride_length / 2.0

        # Sinusoidal swing offset along forward direction
        # Left leg swings forward when sin(phi) > 0; right leg is antiphase (pi offset)
        swing_l = half_stride * math.sin(self.gait_phase)
        swing_r = -swing_l

        # Left foot
        x_left = self.x + swing_l * forward_x + half_stance * left_x
        y_left = self.y + swing_l * forward_y + half_stance * left_y

        # Right foot
        x_right = self.x + swing_r * forward_x - half_stance * left_x
        y_right = self.y + swing_r * forward_y - half_stance * left_y

        left_leg = LegState(x=x_left, y=y_left, radius=self.leg_radius, phase=self.gait_phase)
        right_leg = LegState(
            x=x_right,
            y=y_right,
            radius=self.leg_radius,
            phase=(self.gait_phase + math.pi) % (2.0 * math.pi),
        )

        return left_leg, right_leg

    def ray_intersection(
        self,
        ray_origin_x: float,
        ray_origin_y: float,
        cos_angle: float,
        sin_angle: float,
        max_range: float,
    ) -> float:
        """
        Compute minimum intersection distance of a 2D ray against the bipedal walker.
        Tests intersections against both left and right cylindrical legs.
        """
        closest_hit = max_range
        left_leg, right_leg = self.get_feet_positions()

        for leg in (left_leg, right_leg):
            dx = leg.x - ray_origin_x
            dy = leg.y - ray_origin_y
            proj = dx * cos_angle + dy * sin_angle
            if proj > 0:
                perp_sq = (dx * dx + dy * dy) - (proj * proj)
                r_sq = leg.radius * leg.radius
                if perp_sq < r_sq:
                    hit = proj - math.sqrt(max(0.0, r_sq - perp_sq))
                    if 0.0 < hit < closest_hit:
                        closest_hit = hit

        return closest_hit
