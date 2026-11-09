#!/usr/bin/env python3
"""
Multi-View Synchronized Simulation Media Recorder for AckBot in Gazebo Harmonic.

Subscribes to live simulation streams and records a synchronized composite canvas:
- Left Panel (720x720): Overhead 3D aerial spectator view of arena.
- Top-Right (480x320): Robot front RealSense RGB camera with tracking reticle.
- Middle-Right (480x240): Real-time 2D polar LiDAR radar scan sweep.
- Bottom-Right (480x160): LeWM CEM-MPC actuation & tracking telemetry HUD.

Outputs:
- outputs/reports/media/gazebo_lewm_tracking.mp4 (H.264/MP4 Video)
- outputs/reports/media/gazebo_lewm_tracking.gif (Optimized Animated GIF)
- outputs/reports/media/gazebo_frame_XXXs.png (Presentation Keyframes)
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image as PILImage

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image as RosImage, LaserScan
    from geometry_msgs.msg import Twist
    from std_msgs.msg import String
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False
    Node = object


class SimulationMediaRecorder(Node):
    """
    ROS 2 Node that synchronizes multi-modal Gazebo topics into a presentation canvas.
    """

    def __init__(
        self,
        duration: float = 15.0,
        fps: float = 15.0,
        output_dir: Path = Path("outputs/reports/media"),
        mp4_filename: str = "gazebo_lewm_tracking.mp4",
        gif_filename: str = "gazebo_lewm_tracking.gif",
        gif_fps: float = 10.0,
        keyframe_interval: float = 3.0,
        keyframe_prefix: str = "gazebo_frame_",
    ):
        if not HAS_ROS2:
            raise RuntimeError("ROS 2 (rclpy) is required to run the media recorder.")

        super().__init__("simulation_media_recorder")

        self.duration = duration
        self.fps = fps
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mp4_path = self.output_dir / mp4_filename
        self.gif_path = self.output_dir / gif_filename
        self.gif_fps = gif_fps
        self.keyframe_interval = keyframe_interval
        self.keyframe_prefix = keyframe_prefix

        # Canvas Dimensions
        self.canvas_w = 1200
        self.canvas_h = 720
        self.left_w = 720
        self.right_w = 480
        self.cam_h = 320
        self.lidar_h = 240
        self.hud_h = 160

        # Cached latest topic messages
        self.overhead_frame: Optional[np.ndarray] = None
        self.front_frame: Optional[np.ndarray] = None
        self.lidar_ranges: Optional[np.ndarray] = None
        self.lidar_angle_min = -math.pi
        self.lidar_angle_max = math.pi
        self.linear_v = 0.0
        self.steer_angle_deg = 0.0
        self.wheelbase = 0.30
        self.mpc_status_str = "Initializing..."
        self.target_status = "SEARCHING"
        self.target_dist = 0.0
        self.target_conf = 0.0
        self.mpc_latency_ms = 0.0

        # Subscriptions
        self.sub_overhead = self.create_subscription(
            RosImage, "/overhead_camera/image_raw", self._cb_overhead, 10
        )
        self.sub_front = self.create_subscription(
            RosImage, "/camera/image_raw", self._cb_front, 10
        )
        self.sub_scan = self.create_subscription(
            LaserScan, "/scan", self._cb_scan, 10
        )
        self.sub_cmd = self.create_subscription(
            Twist, "/cmd_vel", self._cb_cmd, 10
        )
        self.sub_status = self.create_subscription(
            String, "/ackbot/mpc_status", self._cb_status, 10
        )

        # Recording state
        self.recording_started = False
        self.start_time = 0.0
        self.frames_captured = 0
        self.max_frames = int(self.duration * self.fps)
        self.next_keyframe_time = 0.0

        # Frame storage
        self.gif_frames: list[PILImage.Image] = []
        self.video_writer: Optional[cv2.VideoWriter] = None

        # Setup capture timer
        self.timer_period = 1.0 / self.fps
        self.timer = self.create_timer(self.timer_period, self._on_timer)

        self.get_logger().info("=" * 60)
        self.get_logger().info("   ACKBOT SIMULATION MEDIA RECORDER NODE INITIALIZED    ")
        self.get_logger().info(f"Target Duration: {self.duration:.1f}s @ {self.fps:.1f} FPS ({self.max_frames} frames)")
        self.get_logger().info(f"Output MP4: {self.mp4_path}")
        self.get_logger().info(f"Output GIF: {self.gif_path} ({self.gif_fps} FPS)")
        self.get_logger().info("Waiting for overhead and front camera streams...")
        self.get_logger().info("=" * 60)

    # -------------------------------------------------------------------------
    # Topic Callbacks
    # -------------------------------------------------------------------------

    def _cb_overhead(self, msg: RosImage) -> None:
        try:
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, -1))
            if msg.encoding in ("rgb8", "RGB8"):
                self.overhead_frame = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            elif msg.encoding in ("bgr8", "BGR8"):
                self.overhead_frame = arr
            elif msg.encoding in ("mono8", "MONO8"):
                self.overhead_frame = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        except Exception as e:
            self.get_logger().warn(f"Failed to decode overhead image: {e}")

    def _cb_front(self, msg: RosImage) -> None:
        try:
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, -1))
            if msg.encoding in ("rgb8", "RGB8"):
                self.front_frame = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            elif msg.encoding in ("bgr8", "BGR8"):
                self.front_frame = arr
            elif msg.encoding in ("mono8", "MONO8"):
                self.front_frame = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        except Exception as e:
            self.get_logger().warn(f"Failed to decode front image: {e}")

    def _cb_scan(self, msg: LaserScan) -> None:
        self.lidar_ranges = np.array(msg.ranges, dtype=np.float32)
        self.lidar_angle_min = float(msg.angle_min)
        self.lidar_angle_max = float(msg.angle_max)

    def _cb_cmd(self, msg: Twist) -> None:
        self.linear_v = float(msg.linear.x)
        # delta = atan(w * L / v)
        w = float(msg.angular.z)
        if abs(self.linear_v) > 0.05:
            steer_rad = math.atan2(w * self.wheelbase, abs(self.linear_v))
            self.steer_angle_deg = math.degrees(steer_rad)
        else:
            self.steer_angle_deg = 0.0

    def _cb_status(self, msg: String) -> None:
        self.mpc_status_str = msg.data
        # Parse status string:
        # e.g.: [Step 02240] MPC: v=+0.80m/s, steer=-30.9° | Target: LOCKED (d=4.4m, conf=0.99) | 1096.2 ms
        try:
            m_steer = re.search(r"steer=([+-]?\d+\.?\d*)°", msg.data)
            if m_steer:
                self.steer_angle_deg = float(m_steer.group(1))

            m_v = re.search(r"v=([+-]?\d+\.?\d*)m/s", msg.data)
            if m_v:
                self.linear_v = float(m_v.group(1))

            m_target = re.search(r"Target:\s*(\w+)\s*\(d=(\d+\.?\d*)m,\s*conf=(\d+\.?\d*)\)", msg.data)
            if m_target:
                self.target_status = m_target.group(1)
                self.target_dist = float(m_target.group(2))
                self.target_conf = float(m_target.group(3))

            m_ms = re.search(r"\|\s*(\d+\.?\d*)\s*ms", msg.data)
            if m_ms:
                self.mpc_latency_ms = float(m_ms.group(1))
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # Render Panels
    # -------------------------------------------------------------------------

    def _render_overhead_panel(self, t_elapsed: float) -> np.ndarray:
        """Render Left Panel (720x720): Overhead 3D camera with annotations."""
        if self.overhead_frame is not None:
            panel = cv2.resize(self.overhead_frame, (self.left_w, self.canvas_h))
        else:
            panel = np.full((self.canvas_h, self.left_w, 3), 30, dtype=np.uint8)
            cv2.putText(
                panel, "Waiting for Overhead Camera...",
                (120, self.canvas_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2
            )

        # Top Header Bar (Semi-transparent overlay)
        header_h = 56
        overlay = panel.copy()
        cv2.rectangle(overlay, (0, 0), (self.left_w, header_h), (18, 22, 28), -1)
        cv2.addWeighted(overlay, 0.85, panel, 0.15, 0, panel)

        cv2.putText(
            panel, "A2R AckBot: Gazebo Harmonic Multi-Pedestrian Tracking",
            (16, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA
        )
        cv2.putText(
            panel, "Receding Horizon LeWM CEM-MPC  |  Aerial 3D Spectator (z=8.0m)",
            (16, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 220, 255), 1, cv2.LINE_AA
        )

        # Live Timestamp badge in top right of left panel
        t_badge = f"T = {t_elapsed:04.1f}s / {self.duration:04.1f}s"
        cv2.rectangle(panel, (self.left_w - 180, 10), (self.left_w - 12, 42), (32, 40, 52), -1)
        cv2.rectangle(panel, (self.left_w - 180, 10), (self.left_w - 12, 42), (0, 180, 255), 1)
        cv2.putText(
            panel, t_badge, (self.left_w - 170, 32),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA
        )

        # Arena compass indicator (bottom-left)
        cv2.circle(panel, (40, self.canvas_h - 40), 20, (35, 45, 55), -1)
        cv2.circle(panel, (40, self.canvas_h - 40), 20, (100, 120, 140), 1)
        cv2.arrowedLine(panel, (40, self.canvas_h - 26), (40, self.canvas_h - 52), (0, 255, 128), 2, tipLength=0.3)
        cv2.putText(panel, "N", (36, self.canvas_h - 54), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 128), 1)

        return panel

    def _render_front_camera_panel(self) -> np.ndarray:
        """Render Top-Right Panel (480x320): Front RealSense camera."""
        if self.front_frame is not None:
            panel = cv2.resize(self.front_frame, (self.right_w, self.cam_h))
        else:
            panel = np.full((self.cam_h, self.right_w, 3), 24, dtype=np.uint8)
            cv2.putText(
                panel, "Waiting for Front Camera...",
                (80, self.cam_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1
            )

        # Header banner
        overlay = panel.copy()
        cv2.rectangle(overlay, (0, 0), (self.right_w, 32), (18, 22, 28), -1)
        cv2.addWeighted(overlay, 0.85, panel, 0.15, 0, panel)

        cv2.putText(
            panel, "Robot Front RealSense Camera (RGB 640x480)",
            (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA
        )

        # Draw central tracking reticle / crosshair
        cx, cy = self.right_w // 2, self.cam_h // 2
        reticle_color = (0, 255, 0) if self.target_status == "LOCKED" else (0, 165, 255)
        # Bounding crosshair
        cv2.line(panel, (cx - 25, cy), (cx - 8, cy), reticle_color, 1)
        cv2.line(panel, (cx + 8, cy), (cx + 25, cy), reticle_color, 1)
        cv2.line(panel, (cx, cy - 25), (cx, cy - 8), reticle_color, 1)
        cv2.line(panel, (cx, cy + 8), (cx, cy + 25), reticle_color, 1)
        cv2.circle(panel, (cx, cy), 14, reticle_color, 1)

        # Target classification tag
        tag = f"TARGET: {self.target_status}"
        cv2.putText(
            panel, tag, (cx - 45, cy + 34),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, reticle_color, 1, cv2.LINE_AA
        )

        return panel

    def _render_lidar_panel(self) -> np.ndarray:
        """Render Middle-Right Panel (480x240): Polar 2D LiDAR radar sweep."""
        panel = np.full((self.lidar_h, self.right_w, 3), (16, 20, 26), dtype=np.uint8)

        # Header
        cv2.rectangle(panel, (0, 0), (self.right_w, 26), (24, 30, 40), -1)
        cv2.putText(
            panel, "360° 2D LiDAR Polar Radar Scope (3.0m max)",
            (12, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 220, 255), 1, cv2.LINE_AA
        )

        # Radar origin (AckBot location at bottom center of scope)
        ox = self.right_w // 2
        oy = self.lidar_h - 32
        scale = 55.0  # pixels per meter (3m = 165px)

        # Range rings: 1m, 2m, 3m
        for r_m in [1.0, 2.0, 3.0]:
            r_px = int(r_m * scale)
            cv2.circle(panel, (ox, oy), r_px, (40, 50, 65), 1)
            cv2.putText(
                panel, f"{r_m:.0f}m", (ox + r_px - 14, oy + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30, (80, 100, 130), 1
            )

        # Camera FOV cone (+-42.5 deg)
        fov_half = math.radians(42.5)
        fov_len = int(3.0 * scale)
        p_left = (int(ox - fov_len * math.sin(fov_half)), int(oy - fov_len * math.cos(fov_half)))
        p_right = (int(ox + fov_len * math.sin(fov_half)), int(oy - fov_len * math.cos(fov_half)))
        cv2.line(panel, (ox, oy), p_left, (35, 45, 60), 1)
        cv2.line(panel, (ox, oy), p_right, (35, 45, 60), 1)

        # Plot LiDAR returns
        if self.lidar_ranges is not None and len(self.lidar_ranges) > 0:
            num_rays = len(self.lidar_ranges)
            angles = np.linspace(self.lidar_angle_min, self.lidar_angle_max, num_rays)
            for dist, ang in zip(self.lidar_ranges, angles):
                if 0.10 <= dist <= 3.0:
                    # LaserScan coordinate: 0 rad is forward (+x, upwards on display)
                    # +pi/2 is left (-x on display)
                    px = int(ox - dist * math.sin(ang) * scale)
                    py = int(oy - dist * math.cos(ang) * scale)
                    if 0 <= px < self.right_w and 0 <= py < self.lidar_h:
                        # Color points by distance: close = red/orange, far = cyan/green
                        c = (
                            int(np.clip(255 * (1.0 - dist / 3.0), 0, 255)),
                            int(np.clip(255 * (dist / 3.0), 100, 255)),
                            int(np.clip(180 * (dist / 3.0), 0, 255))
                        )
                        cv2.circle(panel, (px, py), 1, c, -1)

        # AckBot footprint symbol at center
        cv2.rectangle(panel, (ox - 6, oy - 10), (ox + 6, oy + 6), (0, 200, 255), -1)
        cv2.arrowedLine(panel, (ox, oy + 4), (ox, oy - 14), (255, 255, 255), 1, tipLength=0.4)

        return panel

    def _render_hud_panel(self) -> np.ndarray:
        """Render Bottom-Right Panel (480x160): LeWM CEM-MPC Telemetry & HUD."""
        panel = np.full((self.hud_h, self.right_w, 3), (20, 24, 32), dtype=np.uint8)

        # Header
        cv2.rectangle(panel, (0, 0), (self.right_w, 24), (28, 35, 48), -1)
        cv2.putText(
            panel, "LeWM CEM-MPC Telemetry & Kinematics",
            (12, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA
        )

        # Row 1: Target Tracking Status Pill Badge
        y1 = 48
        is_locked = (self.target_status == "LOCKED")
        badge_bg = (24, 96, 40) if is_locked else (24, 60, 120)
        badge_border = (40, 200, 80) if is_locked else (40, 140, 255)
        cv2.rectangle(panel, (14, y1 - 16), (115, y1 + 6), badge_bg, -1)
        cv2.rectangle(panel, (14, y1 - 16), (115, y1 + 6), badge_border, 1)
        cv2.putText(
            panel, f"● {self.target_status}", (20, y1 - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA
        )

        tracking_info = f"Range: {self.target_dist:04.1f}m   Conf: {self.target_conf:04.2f}"
        cv2.putText(
            panel, tracking_info, (130, y1 - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 220, 235), 1, cv2.LINE_AA
        )

        # Row 2: Linear Velocity Gauge ([-0.4, +0.8] m/s)
        y2 = 82
        cv2.putText(
            panel, f"Speed: {self.linear_v:+05.2f} m/s", (14, y2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA
        )
        bar_x, bar_w, bar_h = 160, 180, 12
        cv2.rectangle(panel, (bar_x, y2 - 10), (bar_x + bar_w, y2 + 2), (40, 48, 62), -1)
        # Normalize: -0.4 -> 0.0, 0.0 -> 0.33, +0.8 -> 1.0
        v_norm = np.clip((self.linear_v + 0.4) / 1.2, 0.0, 1.0)
        zero_x = bar_x + int(0.333 * bar_w)
        fill_x = bar_x + int(v_norm * bar_w)
        bar_col = (0, 220, 100) if self.linear_v >= 0 else (60, 60, 240)
        if fill_x >= zero_x:
            cv2.rectangle(panel, (zero_x, y2 - 10), (fill_x, y2 + 2), bar_col, -1)
        else:
            cv2.rectangle(panel, (fill_x, y2 - 10), (zero_x, y2 + 2), bar_col, -1)
        cv2.line(panel, (zero_x, y2 - 12), (zero_x, y2 + 4), (200, 200, 200), 1)

        # Row 3: Steering Angle Gauge ([-31.7, +31.7] deg)
        y3 = 114
        cv2.putText(
            panel, f"Steer: {self.steer_angle_deg:+05.1f}°", (14, y3),
            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA
        )
        steer_norm = np.clip((self.steer_angle_deg + 31.74) / 63.48, 0.0, 1.0)
        center_x = bar_x + bar_w // 2
        steer_fill_x = bar_x + int(steer_norm * bar_w)
        cv2.rectangle(panel, (bar_x, y3 - 10), (bar_x + bar_w, y3 + 2), (40, 48, 62), -1)
        if steer_fill_x >= center_x:
            cv2.rectangle(panel, (center_x, y3 - 10), (steer_fill_x, y3 + 2), (255, 180, 0), -1)
        else:
            cv2.rectangle(panel, (steer_fill_x, y3 - 10), (center_x, y3 + 2), (255, 180, 0), -1)
        cv2.line(panel, (center_x, y3 - 12), (center_x, y3 + 4), (200, 200, 200), 1)

        # Row 4: LeWM Model & Planner Summary
        y4 = 144
        model_str = f"LeWM CEM-MPC (H=5, N=35) | Cycle: {self.mpc_latency_ms:04.1f}ms"
        cv2.putText(
            panel, model_str, (14, y4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 160, 185), 1, cv2.LINE_AA
        )

        return panel

    def _compose_frame(self, t_elapsed: float) -> np.ndarray:
        """Compose the complete 1200x720 canvas from panels."""
        canvas = np.zeros((self.canvas_h, self.canvas_w, 3), dtype=np.uint8)

        # Left: Overhead Spectator (720x720)
        canvas[:, :self.left_w] = self._render_overhead_panel(t_elapsed)

        # Top-Right: Front Camera (480x320)
        canvas[:self.cam_h, self.left_w:] = self._render_front_camera_panel()

        # Middle-Right: LiDAR Polar Scope (480x240)
        canvas[self.cam_h:self.cam_h + self.lidar_h, self.left_w:] = self._render_lidar_panel()

        # Bottom-Right: HUD (480x160)
        canvas[self.cam_h + self.lidar_h:, self.left_w:] = self._render_hud_panel()

        # Subtle divider lines
        cv2.line(canvas, (self.left_w, 0), (self.left_w, self.canvas_h), (50, 60, 75), 2)
        cv2.line(canvas, (self.left_w, self.cam_h), (self.canvas_w, self.cam_h), (50, 60, 75), 1)
        cv2.line(canvas, (self.left_w, self.cam_h + self.lidar_h), (self.canvas_w, self.cam_h + self.lidar_h), (50, 60, 75), 1)

        return canvas

    # -------------------------------------------------------------------------
    # Main Recording Loop
    # -------------------------------------------------------------------------

    def _on_timer(self) -> None:
        """Periodic frame capture at target FPS."""
        # Wait until camera stream is available
        if self.overhead_frame is None and self.front_frame is None:
            return

        if not self.recording_started:
            self.recording_started = True
            self.start_time = time.monotonic()
            self.next_keyframe_time = 0.0

            # Initialize MP4 VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.video_writer = cv2.VideoWriter(
                str(self.mp4_path), fourcc, self.fps, (self.canvas_w, self.canvas_h)
            )
            self.get_logger().info(f"==> Started synchronized recording to {self.mp4_path}...")

        t_elapsed = time.monotonic() - self.start_time

        # Render full canvas
        canvas = self._compose_frame(t_elapsed)

        # Write to video
        if self.video_writer is not None:
            self.video_writer.write(canvas)

        # Capture keyframe snapshots
        if t_elapsed >= self.next_keyframe_time:
            keyframe_name = f"{self.keyframe_prefix}{int(self.next_keyframe_time):03d}s.png"
            keyframe_path = self.output_dir / keyframe_name
            cv2.imwrite(str(keyframe_path), canvas)
            self.get_logger().info(f"[Keyframe] Saved {keyframe_name} at T={t_elapsed:.1f}s")
            self.next_keyframe_time += self.keyframe_interval

        # Store for GIF (subsample according to gif_fps)
        gif_sample_ratio = max(1, int(round(self.fps / self.gif_fps)))
        if self.frames_captured % gif_sample_ratio == 0:
            # Downscale slightly for smooth, lightweight web GIF
            gif_w = 960
            gif_h = int(self.canvas_h * (gif_w / self.canvas_w))
            rgb_small = cv2.cvtColor(cv2.resize(canvas, (gif_w, gif_h)), cv2.COLOR_BGR2RGB)
            self.gif_frames.append(PILImage.fromarray(rgb_small))

        self.frames_captured += 1

        if self.frames_captured % 15 == 0:
            self.get_logger().info(
                f"[Recording] Frame {self.frames_captured:03d}/{self.max_frames} "
                f"({t_elapsed:04.1f}s / {self.duration:04.1f}s)"
            )

        # Check if duration completed
        if self.frames_captured >= self.max_frames or t_elapsed >= self.duration:
            self._finalize_recording()

    def _finalize_recording(self) -> None:
        """Finalize video, save animated GIF, and exit."""
        self.timer.cancel()
        self.get_logger().info("=" * 60)
        self.get_logger().info("   RECORDING COMPLETE - FINALIZING MEDIA ARTIFACTS      ")
        self.get_logger().info("=" * 60)

        # 1. Release Video Writer
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            mp4_sz = self.mp4_path.stat().st_size / (1024 * 1024)
            self.get_logger().info(f"[SUCCESS] MP4 Video saved: {self.mp4_path} ({mp4_sz:.2f} MB)")

        # 2. Save Final Keyframe
        final_keyframe = self.output_dir / f"{self.keyframe_prefix}{int(self.duration):03d}s.png"
        if not final_keyframe.exists() and self.frames_captured > 0:
            last_canvas = self._compose_frame(self.duration)
            cv2.imwrite(str(final_keyframe), last_canvas)
            self.get_logger().info(f"[Keyframe] Saved final keyframe {final_keyframe.name}")

        # 3. Export Optimized Animated GIF
        if self.gif_frames:
            self.get_logger().info(f"Generating animated GIF ({len(self.gif_frames)} frames @ {self.gif_fps} FPS)...")
            duration_ms = int(1000.0 / self.gif_fps)
            # Optimize GIF palette using PIL quantize
            self.gif_frames[0].save(
                str(self.gif_path),
                save_all=True,
                append_images=self.gif_frames[1:],
                optimize=True,
                duration=duration_ms,
                loop=0,
            )
            gif_sz = self.gif_path.stat().st_size / (1024 * 1024)
            self.get_logger().info(f"[SUCCESS] Animated GIF saved: {self.gif_path} ({gif_sz:.2f} MB)")

        self.get_logger().info("All media artifacts exported successfully.")
        # Terminate node cleanly
        rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(description="Record synchronized Gazebo simulation media.")
    parser.add_argument("--duration", type=float, default=15.0, help="Recording duration in seconds")
    parser.add_argument("--fps", type=float, default=15.0, help="Video capture FPS")
    parser.add_argument("--gif-fps", type=float, default=10.0, help="GIF frame rate")
    parser.add_argument("--keyframe-interval", type=float, default=3.0, help="Keyframe interval in seconds")
    parser.add_argument("--output-dir", type=str, default="outputs/reports/media", help="Output directory")
    parser.add_argument("--mp4-name", type=str, default="gazebo_lewm_tracking.mp4", help="MP4 filename")
    parser.add_argument("--gif-name", type=str, default="gazebo_lewm_tracking.gif", help="GIF filename")
    parser.add_argument("--keyframe-prefix", type=str, default="gazebo_frame_", help="Keyframe filename prefix")
    args = parser.parse_args()

    if not HAS_ROS2:
        print("[ERROR] rclpy is required to run this script. Please run inside ROS 2 environment.", file=sys.stderr)
        sys.exit(1)

    rclpy.init()
    node = SimulationMediaRecorder(
        duration=args.duration,
        fps=args.fps,
        output_dir=Path(args.output_dir),
        mp4_filename=args.mp4_name,
        gif_filename=args.gif_name,
        gif_fps=args.gif_fps,
        keyframe_interval=args.keyframe_interval,
        keyframe_prefix=args.keyframe_prefix,
    )

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
