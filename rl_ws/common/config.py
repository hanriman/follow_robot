"""Unified Robot Configuration and Device Selection for AckBot."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


@dataclass(frozen=True)
class RobotConfig:
    # Chassis Geometry & Mass
    chassis_length: float = 0.30
    chassis_width: float = 0.20
    chassis_mass: float = 5.0
    total_mass: float = 6.6
    ground_clearance: float = 0.06

    # Wheels & Axles
    wheel_diameter: float = 0.12
    wheel_radius: float = 0.06
    wheel_width: float = 0.06
    wheel_mass: float = 0.40
    wheelbase: float = 0.30
    track_width: float = 0.30
    overall_width: float = 0.36
    chassis_gap: float = 0.02

    # Kinematics & Limits
    min_turning_radius: float = 0.485
    max_steering_angle: float = 0.553953  # rad (~31.74 deg)
    v_max: float = 0.80  # m/s
    v_min: float = -0.40  # m/s
    max_steering_rate: float = 1.0471975512  # rad/s (60 deg/s)
    dt: float = 0.05  # 20 Hz simulation / control loop

    # Camera RGB-D Parameters
    cam_pos_x: float = 0.15
    cam_pos_y: float = 0.00
    cam_pos_z: float = 0.25
    cam_fov_deg: float = 85.0
    cam_fov_rad: float = math.radians(85.0)
    cam_range_min: float = 0.15
    cam_range_max: float = 5.00
    cam_samples: int = 64

    # LiDAR 2D Parameters
    lidar_pos_x: float = -0.15
    lidar_pos_y: float = 0.00
    lidar_pos_z: float = 0.30
    lidar_fov_deg: float = 360.0
    lidar_range_min: float = 0.10
    lidar_range_max: float = 3.00
    lidar_samples: int = 360

    # Observation & Action Dimensions
    obs_dim: int = 64 + 360 + 2  # 426
    act_dim: int = 2

    # Following Goal Target
    target_follow_distance: float = 0.80

    def denormalize_cam(self, cam_norm: np.ndarray) -> np.ndarray:
        """Denormalize [0, 1] camera depth back to meters."""
        return cam_norm * (self.cam_range_max - self.cam_range_min) + self.cam_range_min

    def normalize_cam(self, cam_meters: np.ndarray) -> np.ndarray:
        """Normalize camera depth meters to [0, 1]."""
        clipped = np.clip(cam_meters, self.cam_range_min, self.cam_range_max)
        return (clipped - self.cam_range_min) / (self.cam_range_max - self.cam_range_min)

    def denormalize_lidar(self, lidar_norm: np.ndarray) -> np.ndarray:
        """Denormalize [0, 1] LiDAR range back to meters."""
        return lidar_norm * (self.lidar_range_max - self.lidar_range_min) + self.lidar_range_min

    def normalize_lidar(self, lidar_meters: np.ndarray) -> np.ndarray:
        """Normalize LiDAR range meters to [0, 1]."""
        clipped = np.clip(lidar_meters, self.lidar_range_min, self.lidar_range_max)
        return (clipped - self.lidar_range_min) / (self.lidar_range_max - self.lidar_range_min)

    def normalize_action(self, v: float, steer: float) -> np.ndarray:
        """Map physical [v, steer] to normalized [-1, 1] space."""
        v_scale = (self.v_max - self.v_min) / 2.0
        v_bias = (self.v_max + self.v_min) / 2.0
        norm_v = (v - v_bias) / v_scale
        norm_steer = steer / self.max_steering_angle
        return np.array(
            [np.clip(norm_v, -1.0, 1.0), np.clip(norm_steer, -1.0, 1.0)], dtype=np.float32
        )

    def unscale_action(self, norm_v: float, norm_steer: float) -> tuple[float, float]:
        """Map normalized action [-1, 1] back to physical [v, steer]."""
        v_scale = (self.v_max - self.v_min) / 2.0
        v_bias = (self.v_max + self.v_min) / 2.0
        v = float(np.clip(norm_v * v_scale + v_bias, self.v_min, self.v_max))
        steer = float(
            np.clip(
                norm_steer * self.max_steering_angle,
                -self.max_steering_angle,
                self.max_steering_angle,
            )
        )
        return v, steer


_CACHED_CONFIG: RobotConfig | None = None


def get_robot_config(config_path: str | Path | None = None) -> RobotConfig:
    """Load and cache the canonical RobotConfig from YAML, with robust fallback."""
    global _CACHED_CONFIG
    if _CACHED_CONFIG is not None and config_path is None:
        return _CACHED_CONFIG

    if config_path is None:
        # Search relative to repository root or common module location
        candidates = [
            Path(__file__).resolve().parent.parent / "configs" / "robot_params.yaml",
            Path.cwd() / "rl_ws" / "configs" / "robot_params.yaml",
            Path.cwd() / "configs" / "robot_params.yaml",
        ]
        for c in candidates:
            if c.exists():
                config_path = c
                break

    if config_path and Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            data: dict[str, Any] = yaml.safe_load(f)

        chassis = data.get("chassis", {})
        wheels = data.get("wheels", {})
        kinematics = data.get("kinematics", {})
        camera = data.get("camera_rgbd", {})
        lidar = data.get("lidar_2d", {})

        cfg = RobotConfig(
            chassis_length=float(chassis.get("length_x", 0.30)),
            chassis_width=float(chassis.get("width_y", 0.20)),
            chassis_mass=float(chassis.get("mass", 5.0)),
            total_mass=float(chassis.get("total_mass", 6.6)),
            ground_clearance=float(chassis.get("ground_clearance", 0.06)),
            wheel_diameter=float(wheels.get("diameter", 0.12)),
            wheel_radius=float(wheels.get("radius", 0.06)),
            wheel_width=float(wheels.get("width", 0.06)),
            wheel_mass=float(wheels.get("mass", 0.40)),
            wheelbase=float(wheels.get("wheelbase", 0.30)),
            track_width=float(wheels.get("track_width_front", 0.30)),
            overall_width=float(wheels.get("overall_width", 0.36)),
            chassis_gap=float(wheels.get("chassis_gap", 0.02)),
            min_turning_radius=float(kinematics.get("min_turning_radius", 0.485)),
            max_steering_angle=float(kinematics.get("max_steering_angle", 0.553953)),
            v_max=float(kinematics.get("max_linear_speed", 0.80)),
            v_min=float(kinematics.get("min_linear_speed", -0.40)),
            max_steering_rate=float(kinematics.get("max_steering_rate", 1.0471975512)),
            dt=0.05,
            cam_pos_x=float(camera.get("pos_x", 0.15)),
            cam_pos_y=float(camera.get("pos_y", 0.00)),
            cam_pos_z=float(camera.get("pos_z", 0.25)),
            cam_fov_deg=float(camera.get("fov_h_deg", 85.0)),
            cam_fov_rad=math.radians(float(camera.get("fov_h_deg", 85.0))),
            cam_range_min=float(camera.get("depth_min", 0.15)),
            cam_range_max=float(camera.get("depth_max", 5.00)),
            cam_samples=64,
            lidar_pos_x=float(lidar.get("pos_x", -0.15)),
            lidar_pos_y=float(lidar.get("pos_y", 0.00)),
            lidar_pos_z=float(lidar.get("pos_z", 0.30)),
            lidar_fov_deg=float(lidar.get("fov_deg", 360.0)),
            lidar_range_min=float(lidar.get("range_min", 0.10)),
            lidar_range_max=float(lidar.get("range_max", 3.00)),
            lidar_samples=int(lidar.get("samples", 360)),
        )
    else:
        cfg = RobotConfig()

    _CACHED_CONFIG = cfg
    return cfg


def select_device(requested: str = "auto") -> torch.device:
    """Select the best available compute device (MPS on macOS, CUDA on Linux, or CPU)."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
