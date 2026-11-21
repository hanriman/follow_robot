"""
Target Registration and Re-ID Tracking Architecture for AckBot.

Provides:
1. Target Registration & Identity Locking (color signature + initial coordinates)
2. Constant-Velocity 2D Kalman Filter (x, y, vx, vy)
3. Blind-Corner Occlusion Prediction: projects state forward during visual dropouts
4. Mahalanobis Gated Data Association to reject static obstacles and distractor pillars
"""

from __future__ import annotations

import math
import uuid

import numpy as np


class TargetRegistrationTracker:
    """
    Target Registration Tracker maintaining identity and kinematic state
    of a registered human walker even across blind-corner visual occlusions.
    """

    def __init__(
        self,
        dt: float = 0.05,
        process_noise_pos: float = 0.05,
        process_noise_vel: float = 0.20,
        measurement_noise: float = 0.10,
        gate_threshold: float = 9.21,  # Chi-squared 99% gate for 2 DOF
    ) -> None:
        self.dt = float(dt)
        self.gate_threshold = float(gate_threshold)

        # State vector: [x, y, vx, vy]
        self.state = np.zeros(4, dtype=np.float64)
        self.covariance = np.eye(4, dtype=np.float64) * 0.5

        # Transition matrix F
        self.F = np.array(
            [
                [1.0, 0.0, self.dt, 0.0],
                [0.0, 1.0, 0.0, self.dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        # Process noise Q
        q_pos = process_noise_pos**2
        q_vel = process_noise_vel**2
        self.Q = np.diag([q_pos, q_pos, q_vel, q_vel]).astype(np.float64)

        # Measurement matrix H (observing x, y)
        self.H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )

        # Measurement noise R
        r_var = measurement_noise**2
        self.R = np.diag([r_var, r_var]).astype(np.float64)

        # Target registration metadata
        self.target_id: str | None = None
        self.registered_color: tuple[float, float, float] | None = None
        self.is_registered: bool = False
        self.is_occluded: bool = True
        self.occlusion_steps: int = 0
        self.total_tracked_steps: int = 0

    def register_target(
        self,
        x: float,
        y: float,
        vx: float = 0.0,
        vy: float = 0.0,
        color_rgb: tuple[float, float, float] | None = None,
        target_id: str | None = None,
    ) -> str:
        """Register and lock onto target."""
        self.target_id = target_id or f"target_{uuid.uuid4().hex[:6]}"
        self.registered_color = color_rgb or (0.2, 0.4, 0.8)
        self.state = np.array([x, y, vx, vy], dtype=np.float64)
        self.covariance = np.diag([0.05, 0.05, 0.50, 0.50]).astype(np.float64)
        self.is_registered = True
        self.is_occluded = False
        self.occlusion_steps = 0
        self.total_tracked_steps = 0
        return self.target_id

    def predict(self) -> np.ndarray:
        """Propagate state and covariance forward by dt (used during occlusions)."""
        if not self.is_registered:
            return self.state

        self.state = self.F @ self.state
        self.covariance = self.F @ self.covariance @ self.F.T + self.Q
        self.occlusion_steps += 1
        self.is_occluded = True
        self.total_tracked_steps += 1
        return self.state

    def update(self, measured_x: float, measured_y: float) -> bool:
        """
        Update Kalman filter with observation measurement [x, y].
        Applies continuous prediction step followed by Mahalanobis distance gating.

        Returns:
            True if measurement passed gate and updated filter, False if rejected.
        """
        if not self.is_registered:
            self.register_target(measured_x, measured_y)
            return True

        # 1. State and covariance prediction step
        self.state = self.F @ self.state
        self.covariance = self.F @ self.covariance @ self.F.T + self.Q

        z = np.array([measured_x, measured_y], dtype=np.float64)
        z_pred = self.H @ self.state
        y_residual = z - z_pred

        # Innovation covariance S = H P H^T + R
        S = self.H @ self.covariance @ self.H.T + self.R

        # Mahalanobis distance squared d^2 = y^T S^-1 y
        try:
            S_inv = np.linalg.inv(S)
            d2 = float(y_residual.T @ S_inv @ y_residual)
        except np.linalg.LinAlgError:
            d2 = 999.0

        if d2 > self.gate_threshold:
            # Measurement too far from predicted tracklet (rejected)
            self.is_occluded = True
            self.occlusion_steps += 1
            self.total_tracked_steps += 1
            return False

        # Kalman gain K = P H^T S^-1
        K = self.covariance @ self.H.T @ S_inv
        self.state = self.state + K @ y_residual
        eye = np.eye(4, dtype=np.float64)
        self.covariance = (eye - K @ self.H) @ self.covariance

        self.is_occluded = False
        self.occlusion_steps = 0
        self.total_tracked_steps += 1
        return True

    def update_with_candidates(
        self,
        candidates: list[dict],
        color_weight: float = 6.0,
        color_threshold: float = 0.40,
    ) -> dict:
        """
        Gated Re-ID data association across multiple candidate detections.
        Jointly evaluates kinematic Mahalanobis distance d^2 and visual color signature.

        Args:
            candidates: list of dicts with {"x": float, "y": float, "color_rgb": tuple[float, float, float] | None, "id": str | None}
            color_weight: weighting factor for appearance distance in association cost
            color_threshold: maximum allowed Euclidean color difference in RGB space

        Returns:
            Association diagnostic dict.
        """
        if not self.is_registered or not candidates:
            self.predict()
            return {
                "associated": False,
                "selected_candidate": None,
                "d2": 999.0,
                "color_dist": 999.0,
                "cost": 999.0,
            }

        # 1. Prediction step before evaluating candidate measurements
        self.state = self.F @ self.state
        self.covariance = self.F @ self.covariance @ self.F.T + self.Q

        # Predict innovation covariance S = H P H^T + R
        S = self.H @ self.covariance @ self.H.T + self.R
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            self.is_occluded = True
            self.occlusion_steps += 1
            self.total_tracked_steps += 1
            return {
                "associated": False,
                "selected_candidate": None,
                "d2": 999.0,
                "color_dist": 999.0,
                "cost": 999.0,
            }

        z_pred = self.H @ self.state
        best_cand = None
        best_cost = float("inf")
        best_d2 = float("inf")
        best_cdist = float("inf")

        reg_color = (
            np.array(self.registered_color, dtype=np.float64) if self.registered_color else None
        )

        for cand in candidates:
            z = np.array([cand["x"], cand["y"]], dtype=np.float64)
            y_res = z - z_pred
            d2 = float(y_res.T @ S_inv @ y_res)

            if d2 > self.gate_threshold:
                # Outside kinematic validation gate
                continue

            color_dist = 0.0
            if reg_color is not None and "color_rgb" in cand and cand["color_rgb"] is not None:
                cand_color = np.array(cand["color_rgb"], dtype=np.float64)
                color_dist = float(np.linalg.norm(cand_color - reg_color))
                if color_dist > color_threshold:
                    # Color mismatch (different clothing / appearance)
                    continue

            total_cost = d2 + color_weight * color_dist
            if total_cost < best_cost:
                best_cost = total_cost
                best_cand = cand
                best_d2 = d2
                best_cdist = color_dist

        if best_cand is not None:
            # Update filter with the selected candidate
            z = np.array([best_cand["x"], best_cand["y"]], dtype=np.float64)
            y_res = z - z_pred
            K = self.covariance @ self.H.T @ S_inv
            self.state = self.state + K @ y_res
            eye = np.eye(4, dtype=np.float64)
            self.covariance = (eye - K @ self.H) @ self.covariance

            self.is_occluded = False
            self.occlusion_steps = 0
            self.total_tracked_steps += 1
            return {
                "associated": True,
                "selected_candidate": best_cand,
                "d2": best_d2,
                "color_dist": best_cdist,
                "cost": best_cost,
            }
        else:
            # No candidate passed both spatial gate and color gate
            self.is_occluded = True
            self.occlusion_steps += 1
            self.total_tracked_steps += 1
            return {
                "associated": False,
                "selected_candidate": None,
                "d2": 999.0,
                "color_dist": 999.0,
                "cost": 999.0,
            }

    def get_status(
        self,
        robot_x: float = 0.0,
        robot_y: float = 0.0,
        robot_yaw: float = 0.0,
    ) -> dict:
        """Return tracking telemetry and confidence relative to robot pose."""
        if not self.is_registered:
            return {
                "registered": False,
                "target_id": None,
                "is_occluded": True,
                "occlusion_steps": self.occlusion_steps,
                "estimated_pos": (0.0, 0.0),
                "estimated_vel": (0.0, 0.0),
                "estimated_dist": 0.0,
                "estimated_bearing": 0.0,
                "estimated_speed": 0.0,
                "confidence": 0.0,
                "color_rgb": self.registered_color,
            }

        x, y, vx, vy = self.state
        speed = math.hypot(vx, vy)

        # Transform to ego-centric coordinates relative to robot pose
        dx = x - robot_x
        dy = y - robot_y
        cos_yaw = math.cos(-robot_yaw)
        sin_yaw = math.sin(-robot_yaw)
        rel_x = dx * cos_yaw - dy * sin_yaw
        rel_y = dx * sin_yaw + dy * cos_yaw

        dist = math.hypot(dx, dy)
        bearing = math.atan2(rel_y, rel_x)

        # Confidence decays during extended occlusions
        pos_var = float(np.trace(self.covariance[:2, :2]))
        confidence = max(
            0.0, min(1.0, 1.0 / (1.0 + 2.0 * pos_var) * math.exp(-0.05 * self.occlusion_steps))
        )

        return {
            "registered": True,
            "target_id": self.target_id,
            "is_occluded": bool(self.is_occluded),
            "occlusion_steps": self.occlusion_steps,
            "estimated_pos": (float(x), float(y)),
            "estimated_vel": (float(vx), float(vy)),
            "estimated_dist": float(dist),
            "estimated_bearing": float(bearing),
            "estimated_speed": float(speed),
            "confidence": float(confidence),
            "color_rgb": self.registered_color,
        }
