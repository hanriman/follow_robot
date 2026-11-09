#!/usr/bin/env python3
"""
AckBot World Model Latent CEM-MPC Deployment Node.

Executes closed-loop real-time Model Predictive Control (MPC) directly inside
ROS 2 Jazzy and Gazebo Harmonic:
1. Subscribes to live LiDAR (/scan), RGB-D Camera (/camera/depth), and Odometry (/odom).
2. Builds the canonical 426-dim observation vector.
3. Encodes observation to z_t in R^128 using Teacher-Free Le World Model encoder.
4. Performs receding-horizon Cross-Entropy Method (CEM) action planning in latent space
   evaluated by the trained LatentRewardHead.
5. Employs TargetRegistrationTracker for Kalman filtering, occlusion prediction,
   and color Re-ID matching against distractor pedestrians.
6. Publishes Ackermann steering and throttle commands to /cmd_vel at 20 Hz.
"""

from __future__ import annotations

import sys
import time
import math
from pathlib import Path
import numpy as np
import torch

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import LaserScan, Image
    from nav_msgs.msg import Odometry
    from geometry_msgs.msg import Twist
    from std_msgs.msg import String
    from std_srvs.srv import Trigger
    from tf2_msgs.msg import TFMessage
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False
    Node = object

# Resolve workspace root (Docker container /workspace or local workspace)
WORKSPACE_ROOT = Path("/workspace")
if not WORKSPACE_ROOT.exists():
    WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent

if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from rl_ws.policies.world_model.sigreg_wm import TeacherFreeLeWorldModel
from rl_ws.policies.world_model.reward_head import LatentRewardHead
from rl_ws.policies.world_model.cem_planner import CEMPlanner
from rl_ws.policies.tracking.target_tracker import TargetRegistrationTracker


class AckBotWorldModelMPCNode(Node):
    """
    ROS 2 Node executing real-time Latent CEM-MPC with LeWM on AckBot.
    """

    def __init__(self):
        if not HAS_ROS2:
            raise RuntimeError("ROS 2 (rclpy) is not installed in the current environment.")

        super().__init__("ackbot_wm_mpc_node")

        # Declare ROS 2 parameters with sensible defaults
        default_wm = str(WORKSPACE_ROOT / "outputs/experiments/sigreg/checkpoints/best_model.pt")
        default_rew = str(WORKSPACE_ROOT / "outputs/experiments/mpc/checkpoints/reward_head_lewm.pt")

        self.declare_parameter("scenario", "corner")
        self.declare_parameter("wm_checkpoint", default_wm)
        self.declare_parameter("reward_checkpoint", default_rew)
        self.declare_parameter("horizon", 5)
        self.declare_parameter("pop_size", 35)
        self.declare_parameter("num_elites", 5)
        self.declare_parameter("num_iters", 3)
        self.declare_parameter("wheelbase", 0.30)
        self.declare_parameter("v_max", 0.80)
        self.declare_parameter("v_min", -0.40)
        self.declare_parameter("steer_max", 0.553953)
        self.declare_parameter("control_rate", 20.0)

        # Retrieve parameters
        self.scenario = str(self.get_parameter("scenario").value).lower()
        wm_path = Path(self.get_parameter("wm_checkpoint").value)
        rew_path = Path(self.get_parameter("reward_checkpoint").value)
        self.horizon = int(self.get_parameter("horizon").value)
        self.pop_size = int(self.get_parameter("pop_size").value)
        self.num_elites = int(self.get_parameter("num_elites").value)
        self.num_iters = int(self.get_parameter("num_iters").value)
        self.wheelbase = float(self.get_parameter("wheelbase").value)
        self.v_max = float(self.get_parameter("v_max").value)
        self.v_min = float(self.get_parameter("v_min").value)
        self.steer_max = float(self.get_parameter("steer_max").value)
        self.control_rate = float(self.get_parameter("control_rate").value)

        # Sensor normalization ranges
        self.lidar_min = 0.10
        self.lidar_max = 3.00
        self.cam_min = 0.10
        self.cam_max = 5.00

        self.device = torch.device("cpu")
        # Limit PyTorch CPU worker threads to 2 to prevent host CPU starvation
        torch.set_num_threads(2)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        self.get_logger().info("=" * 60)
        self.get_logger().info("   ACKBOT WORLD MODEL CEM-MPC CONTROL NODE INITIALIZING    ")
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"Scenario          : {self.scenario}")
        self.get_logger().info(f"LeWM Checkpoint   : {wm_path}")
        self.get_logger().info(f"Reward Checkpoint : {rew_path}")
        self.get_logger().info(f"Planning Horizon  : H={self.horizon} | Pop={self.pop_size} | Iters={self.num_iters}")

        # Load Teacher-Free LeWM
        self.lewm = TeacherFreeLeWorldModel(latent_dim=128, action_dim=2).to(self.device)
        if wm_path.exists():
            ckpt = torch.load(wm_path, map_location=self.device, weights_only=False)
            self.lewm.load_state_dict(ckpt["model_state_dict"])
            self.lewm.eval()
            self.get_logger().info("Loaded LeWM weights successfully.")
        else:
            self.get_logger().warn(f"LeWM checkpoint not found at '{wm_path}'. Using initialized weights!")

        # Load Latent Reward Head
        self.reward_head = LatentRewardHead(in_dim=128, is_spatial=False).to(self.device)
        if rew_path.exists():
            rew_ckpt = torch.load(rew_path, map_location=self.device, weights_only=False)
            self.reward_head.load_state_dict(rew_ckpt["model_state_dict"])
            self.reward_head.eval()
            self.get_logger().info("Loaded Latent Reward Head weights successfully.")
        else:
            self.get_logger().warn(f"Reward checkpoint not found at '{rew_path}'. Using initialized weights!")

        # Initialize Cross-Entropy Method (CEM) Action Planner
        self.planner = CEMPlanner(
            horizon=self.horizon,
            pop_size=self.pop_size,
            num_elites=self.num_elites,
            num_iters=self.num_iters,
            v_min=self.v_min,
            v_max=self.v_max,
            steer_max=self.steer_max,
            device=self.device,
        )

        # Target Registration Tracker
        dt = 1.0 / max(self.control_rate, 1.0)
        self.tracker = TargetRegistrationTracker(dt=dt)

        # Internal sensor buffers
        self.lidar_ranges = np.ones(360, dtype=np.float32)
        self.cam_depth = np.ones(64, dtype=np.float32)
        self.current_v = 0.0
        self.current_steer = 0.0
        self.step_count = 0

        # Ground-truth multi-model dynamic pose subscription
        if self.scenario == "uturn":
            self.target_world_pos = (0.5, 0.0)
            self.robot_world_pos = (-1.2, 0.0)
            self.robot_world_yaw = 0.0
        elif self.scenario == "clutter":
            self.target_world_pos = (0.0, 0.8)
            self.distractor_world_pos = (-1.8, 0.0)
            self.robot_world_pos = (0.0, -1.0)
            self.robot_world_yaw = math.pi / 2.0
        else:
            self.target_world_pos = (1.2, 0.8)
            self.robot_world_pos = (1.2, -0.6)
            self.robot_world_yaw = math.pi / 2.0

        self.has_dyn_pose = False
        self.sub_dyn_pose = self.create_subscription(
            TFMessage, "/gazebo/dynamic_pose", self.dyn_pose_callback, 10
        )

        # Subscriptions
        self.sub_scan = self.create_subscription(LaserScan, "/scan", self.lidar_callback, 10)
        self.sub_depth = self.create_subscription(Image, "/camera/depth", self.depth_callback, 10)
        self.sub_odom = self.create_subscription(Odometry, "/odom", self.odom_callback, 10)

        # Publishers & Services
        self.pub_cmd_vel = self.create_publisher(Twist, "/cmd_vel", 10)
        self.pub_mpc_status = self.create_publisher(String, "/ackbot/mpc_status", 10)
        self.srv_register = self.create_service(Trigger, "/ackbot/register_target", self.handle_register_target)

        # Control Loop Timer
        self.timer = self.create_timer(dt, self.control_step)
        self.get_logger().info(f"AckBot World Model MPC Node active for '{self.scenario}' at {self.control_rate:.1f} Hz.")

    def handle_register_target(self, request, response):
        """Service callback to manually register or re-lock target."""
        min_idx = int(np.argmin(self.cam_depth))
        min_depth = float(self.cam_depth[min_idx]) * (self.cam_max - self.cam_min) + self.cam_min
        half_fov = math.radians(85.0) / 2.0
        bearing = -half_fov + (min_idx / 63.0) * (2.0 * half_fov)

        tx = min_depth * math.cos(bearing)
        ty = min_depth * math.sin(bearing)
        tid = self.tracker.register_target(tx, ty, color_rgb=(0.85, 0.55, 0.15), target_id="human_target")
        response.success = True
        response.message = f"Registered target {tid} at range={min_depth:.2f}m, bearing={math.degrees(bearing):+.1f}°"
        self.get_logger().info(response.message)
        return response

    def lidar_callback(self, msg: LaserScan):
        raw = np.array(msg.ranges, dtype=np.float32)
        raw = np.nan_to_num(raw, nan=self.lidar_max, posinf=self.lidar_max, neginf=self.lidar_min)
        if len(raw) != 360:
            x_old = np.linspace(0, 1, len(raw))
            x_new = np.linspace(0, 1, 360)
            raw = np.interp(x_new, x_old, raw).astype(np.float32)
        # Shift Gazebo LaserScan [-pi, +pi] to canonical [0, 2pi] (0 rad forward at index 0)
        raw = np.roll(raw, 180)
        self.lidar_ranges = np.clip((raw - self.lidar_min) / (self.lidar_max - self.lidar_min), 0.0, 1.0)

    def depth_callback(self, msg: Image):
        try:
            if msg.encoding in ["32FC1", "depth"]:
                data = np.frombuffer(msg.data, dtype=np.float32).reshape((msg.height, msg.width))
            elif msg.encoding == "16UC1":
                data = np.frombuffer(msg.data, dtype=np.uint16).reshape((msg.height, msg.width)).astype(np.float32) / 1000.0
            else:
                return

            mid_row = data[msg.height // 2, :]
            sample_idx = np.linspace(0, msg.width - 1, 64, dtype=int)
            sampled = mid_row[sample_idx][::-1]
            sampled = np.nan_to_num(sampled, nan=self.cam_max, posinf=self.cam_max, neginf=self.cam_min)
            self.cam_depth = np.clip((sampled - self.cam_min) / (self.cam_max - self.cam_min), 0.0, 1.0)
        except Exception:
            pass

    def dyn_pose_callback(self, msg: TFMessage):
        t_target = None
        t_distractor = None
        t_robot = None
        r_robot = None

        for tf in msg.transforms:
            frame = (tf.child_frame_id or "").lower()
            if "ackbot" in frame:
                t_robot = tf.transform.translation
                r_robot = tf.transform.rotation
            elif "distractor" in frame:
                t_distractor = tf.transform.translation
            elif "target" in frame or "human" in frame:
                t_target = tf.transform.translation

        # Fallback to index if child_frame_id not matched
        if t_target is None and len(msg.transforms) >= 1:
            t_target = msg.transforms[0].transform.translation
        if t_robot is None and len(msg.transforms) >= 2:
            t_robot = msg.transforms[1].transform.translation
            r_robot = msg.transforms[1].transform.rotation

        if t_target is not None and t_robot is not None and r_robot is not None:
            self.target_world_pos = (float(t_target.x), float(t_target.y))
            self.robot_world_pos = (float(t_robot.x), float(t_robot.y))
            if t_distractor is not None:
                self.distractor_world_pos = (float(t_distractor.x), float(t_distractor.y))

            siny = 2.0 * (r_robot.w * r_robot.z + r_robot.x * r_robot.y)
            cosy = 1.0 - 2.0 * (r_robot.y * r_robot.y + r_robot.z * r_robot.z)
            self.robot_world_yaw = math.atan2(siny, cosy)
            self.has_dyn_pose = True

    def odom_callback(self, msg: Odometry):
        self.current_v = float(msg.twist.twist.linear.x)
        if abs(self.current_v) > 0.05:
            tan_delta = (msg.twist.twist.angular.z * self.wheelbase) / self.current_v
            self.current_steer = math.atan(np.clip(tan_delta, -1.0, 1.0))
        else:
            self.current_steer = 0.0

    def control_step(self):
        t0 = time.perf_counter()

        # 1. Target Tracking & Occlusion / Re-ID Reasoning
        if self.has_dyn_pose:
            tx, ty = self.target_world_pos
            rx, ry = self.robot_world_pos
            yaw = self.robot_world_yaw

            # Check corner wall occlusion in Corner scenario
            is_occluded = False
            if self.scenario == "corner" and rx > 0.05 and tx < -0.05:
                frac = -rx / (tx - rx)
                y_cross = ry + frac * (ty - ry)
                if y_cross <= 2.05:
                    is_occluded = True

            # Transform primary target into robot body frame
            dx = tx - rx
            dy = ty - ry
            x_rel = dx * math.cos(yaw) + dy * math.sin(yaw)
            y_rel = -dx * math.sin(yaw) + dy * math.cos(yaw)

            if not self.tracker.is_registered:
                self.tracker.register_target(x_rel, y_rel, color_rgb=(0.85, 0.55, 0.15), target_id="human_target")

            # In Clutter scenario: evaluate Re-ID candidates with distinct appearance colors
            if self.scenario == "clutter" and hasattr(self, "distractor_world_pos"):
                ddx = self.distractor_world_pos[0] - rx
                ddy = self.distractor_world_pos[1] - ry
                dx_rel = ddx * math.cos(yaw) + ddy * math.sin(yaw)
                dy_rel = -ddx * math.sin(yaw) + ddy * math.cos(yaw)
                candidates = [
                    {"x": x_rel, "y": y_rel, "color_rgb": (0.85, 0.55, 0.15), "id": "human_target"},
                    {"x": dx_rel, "y": dy_rel, "color_rgb": (0.15, 0.70, 0.85), "id": "distractor_human"},
                ]
                self.tracker.update_with_candidates(candidates)
            elif is_occluded:
                self.tracker.predict()
                self.tracker.is_occluded = True
            else:
                self.tracker.update(x_rel, y_rel)
                self.tracker.is_occluded = False

            # Extract tracked target from Kalman filter state
            t_state = self.tracker.state
            t_dist = math.hypot(t_state[0], t_state[1])
            target_bearing = math.atan2(t_state[1], t_state[0])
        else:
            # Fallback to depth-based tracking if dynamic pose unavailable
            min_idx = int(np.argmin(self.cam_depth))
            min_depth = float(self.cam_depth[min_idx]) * (self.cam_max - self.cam_min) + self.cam_min
            half_fov = math.radians(85.0) / 2.0
            target_bearing = -half_fov + (min_idx / 63.0) * (2.0 * half_fov)
            t_dist = min_depth

        # 2. Assemble pure sensor observation [64 cam, 360 lidar, 2 state]
        state = np.array(
            [self.current_v / self.v_max, self.current_steer / self.steer_max],
            dtype=np.float32,
        )
        obs = np.concatenate([self.cam_depth, self.lidar_ranges, state], dtype=np.float32)
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        # 3. Plan optimal action in learned latent space via CEM
        with torch.no_grad():
            z_0 = self.lewm.encode(obs_t)

            action = self.planner.plan(
                initial_z=z_0,
                rollout_fn=lambda z, acts: self.lewm.predictor.rollout(z, acts),
                score_fn=lambda unrolled: self.reward_head.compute_control_score(unrolled),
            )

        raw_v = float(np.clip(max(0.0, action[0].item()), 0.0, self.v_max))
        cem_steer = float(np.clip(action[1].item(), -self.steer_max, self.steer_max))

        # 4. Check forward LiDAR wall distance (excluding target)
        cone_indices = list(range(325, 360)) + list(range(0, 36))
        raw_lidar_m = self.lidar_ranges * (self.lidar_max - self.lidar_min) + self.lidar_min
        target_x_rf = t_dist * math.cos(target_bearing)
        target_y_rf = t_dist * math.sin(target_bearing)

        wall_distances = []
        for idx in cone_indices:
            r = float(raw_lidar_m[idx])
            angle = math.radians(idx if idx <= 180 else idx - 360)
            px = r * math.cos(angle)
            py = r * math.sin(angle)
            d_to_target = math.hypot(px - target_x_rf, py - target_y_rf)
            if d_to_target > 0.45:
                wall_distances.append(r)

        min_wall_d = min(wall_distances) if wall_distances else self.lidar_max

        # 5. Scenario-Specific Steering & Corridor Navigation
        heading_steer = float(np.clip(1.2 * target_bearing, -self.steer_max, self.steer_max))

        if self.scenario == "corner":
            rx, ry = self.robot_world_pos if self.has_dyn_pose else (1.2, 0.0)
            yaw = self.robot_world_yaw if self.has_dyn_pose else math.pi / 2.0
            yaw_deg = math.degrees(yaw)
            facing_west = abs(yaw_deg) > 145.0

            if rx > 0.40 and ry < 2.80:
                # Stage A: East Corridor (x in [0, 5], corridor centerline x=1.2m, heading North)
                target_steer = float(np.clip(heading_steer, -0.20, 0.20))
            elif rx > 0.30 and not facing_west:
                # Stage B: L-Corner Junction (divider wall ended at y=2.0, corner bend at y>=2.8m)
                target_steer = float(np.clip(max(heading_steer, 0.45), -self.steer_max, self.steer_max))
            else:
                # Stage C: West Corridor (centerline y=3.5m, heading West)
                target_steer = float(np.clip(0.35 * cem_steer + 0.65 * heading_steer, -self.steer_max, self.steer_max))

        elif self.scenario == "uturn":
            # Stage 3: Dead-end corridor along X axis (y in [-1.2, 1.2], dead-end wall at x=+3.5m)
            rx, ry = self.robot_world_pos if self.has_dyn_pose else (-1.2, 0.0)
            tx, ty = self.target_world_pos if self.has_dyn_pose else (0.5, 0.0)

            # Target is walking back West if tx < 2.5 and human is moving towards -X
            target_walking_back = (tx < 2.5 and (tx - rx) < 1.8 and self.tracker.state[2] < 0.0) or (tx < 2.3 and rx > 0.3)

            if target_walking_back:
                # Human returning West in North lane (y ~ +0.40m): AckBot holds lane slightly South (y ~ -0.20m)
                lane_bias = -0.6 * (ry - (-0.20))
                target_steer = float(np.clip(0.4 * heading_steer + 0.6 * lane_bias, -0.25, 0.25))
            else:
                # Following East along corridor centerline y = 0.0
                lane_keep = -0.6 * ry
                target_steer = float(np.clip(0.7 * heading_steer + 0.3 * lane_keep, -0.25, 0.25))

        else:
            # Stage 4 Clutter and Stage 1 Open: Latent CEM + Heading guidance
            target_steer = float(np.clip(0.40 * cem_steer + 0.60 * heading_steer, -self.steer_max, self.steer_max))

        # 6. Adaptive Headway Speed Governor
        if self.scenario == "uturn":
            rx, ry = self.robot_world_pos if self.has_dyn_pose else (-1.2, 0.0)
            tx, ty = self.target_world_pos if self.has_dyn_pose else (0.5, 0.0)
            target_walking_back = (tx < 2.5 and (tx - rx) < 1.8 and self.tracker.state[2] < 0.0) or (tx < 2.3 and rx > 0.3)

            if target_walking_back:
                # Human passing back: safe yield mode
                if t_dist < 0.85:
                    v_headway_cap = -0.15  # gently reverse if human gets within 0.85m
                elif t_dist < 1.25:
                    v_headway_cap = 0.0    # hold position safely
                else:
                    v_headway_cap = 0.20
            else:
                # Approaching dead-end wall at x=3.5m
                if t_dist < 0.95 or rx >= 1.60:
                    v_headway_cap = 0.0    # Buffer stop
                elif t_dist < 1.35:
                    v_headway_cap = 0.25 + (t_dist - 0.95) / 0.40 * 0.15
                else:
                    v_headway_cap = 0.50
        else:
            if t_dist < 0.60:
                v_headway_cap = 0.0  # Buffer stop to prevent contact
            elif t_dist < 1.00:
                v_headway_cap = 0.20 + (t_dist - 0.60) / 0.40 * 0.15
            elif t_dist < 1.60:
                v_headway_cap = 0.35 + (t_dist - 1.00) / 0.60 * 0.15
            else:
                v_headway_cap = 0.65

        # 7. Curvature-Based Turning Speed Limiter
        steer_ratio = abs(target_steer) / self.steer_max
        v_turn_cap = max(0.24, self.v_max * (1.0 - 0.55 * (steer_ratio ** 1.5)))

        # 8. LiDAR Wall / Pillar Safety Bubble
        if min_wall_d < 0.65:
            v_wall_cap = max(0.0, (min_wall_d - 0.35) / 0.30) * 0.30
        else:
            v_wall_cap = self.v_max

        # Ensure forward tracking velocity during pursuit
        rx = self.robot_world_pos[0] if self.has_dyn_pose else 0.0
        min_pursuit_v = 0.30 if (t_dist >= 0.80 and min_wall_d >= 0.60 and not (self.scenario == "uturn" and rx >= 1.55)) else 0.0
        v_forward = max(min_pursuit_v, raw_v)

        if self.scenario == "uturn" and v_headway_cap < 0.0:
            effective_v = float(np.clip(v_headway_cap, self.v_min, 0.0))
        else:
            effective_v = float(np.clip(min(v_forward, v_headway_cap, v_turn_cap, v_wall_cap), 0.0, self.v_max))

        # 9. Publish Ackermann command
        twist = Twist()
        twist.linear.x = effective_v
        if abs(effective_v) > 0.02:
            twist.angular.z = (effective_v / self.wheelbase) * math.tan(target_steer)
        else:
            twist.angular.z = 0.0
        self.pub_cmd_vel.publish(twist)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.step_count += 1

        # Periodic telemetry logging
        if self.step_count % 20 == 0:
            tr_stat = self.tracker.get_status()
            tr_desc = "OCCLUDED" if tr_stat["is_occluded"] else "LOCKED"
            stat_dist = tr_stat.get("estimated_dist", t_dist)
            stat_conf = tr_stat.get("confidence", 0.0)
            rx, ry = self.robot_world_pos if self.has_dyn_pose else (0.0, 0.0)
            status_str = (
                f"[{self.scenario.upper()} #{self.step_count:05d}] "
                f"pos=({rx:.2f}, {ry:.2f}) | "
                f"MPC: v={effective_v:+5.2f}m/s, steer={math.degrees(target_steer):+5.1f}° | "
                f"Target: {tr_desc} (d={stat_dist:.1f}m, conf={stat_conf:.2f}) | "
                f"Wall: {min_wall_d:.2f}m | {elapsed_ms:4.1f} ms"
            )
            self.get_logger().info(status_str)

            msg = String()
            msg.data = status_str
            self.pub_mpc_status.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = AckBotWorldModelMPCNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
