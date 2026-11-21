#!/usr/bin/env python3
"""
AckBot Trained PPO Policy Deployment Node.
Subscribes to live LiDAR and Camera topics, feeds sensor readings into the PyTorch ActorCritic model,
and publishes real-time Ackermann steering commands to /cmd_vel.
"""

import sys
import math
from pathlib import Path
import numpy as np
import torch
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

# Path resolution for ActorCritic architecture
WORKSPACE_ROOT = Path("/workspace")
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.append(str(WORKSPACE_ROOT))

from rl_ws.policies.ppo.actor_critic import ActorCritic


class AckBotPolicyNode(Node):
    def __init__(self):
        super().__init__("ackbot_policy_node")

        # Declare ROS parameters
        self.declare_parameter(
            "checkpoint_path",
            "/workspace/outputs/experiments/exp_20261108_154753_curriculum_ppo_active_follow/checkpoints/best_model.pt",
        )
        self.declare_parameter("wheelbase", 0.30)
        self.declare_parameter("v_max", 0.80)
        self.declare_parameter("steer_max", 0.553953)
        self.declare_parameter("lidar_min", 0.10)
        self.declare_parameter("lidar_max", 3.00)
        self.declare_parameter("cam_min", 0.15)
        self.declare_parameter("cam_max", 5.00)

        ckpt_path = Path(self.get_parameter("checkpoint_path").value)
        self.wheelbase = float(self.get_parameter("wheelbase").value)
        self.v_max = float(self.get_parameter("v_max").value)
        self.steer_max = float(self.get_parameter("steer_max").value)
        self.lidar_min = float(self.get_parameter("lidar_min").value)
        self.lidar_max = float(self.get_parameter("lidar_max").value)
        self.cam_min = float(self.get_parameter("cam_min").value)
        self.cam_max = float(self.get_parameter("cam_max").value)

        self.get_logger().info(f"Loading policy checkpoint: {ckpt_path}")

        # Initialize Neural Network
        self.device = torch.device("cpu")
        self.model = ActorCritic(obs_dim=426, act_dim=2, cam_dim=64, lidar_dim=360).to(self.device)

        if ckpt_path.exists():
            checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()
            self.get_logger().info("Policy checkpoint loaded successfully into memory.")
        else:
            self.get_logger().warn(f"Checkpoint not found at '{ckpt_path}'. Running randomly initialized policy!")

        # Internal sensor buffers
        self.lidar_ranges = np.ones(360, dtype=np.float32)
        self.cam_depth = np.ones(64, dtype=np.float32)
        self.current_v = 0.0
        self.current_steer = 0.0

        # Subscriptions
        self.sub_scan = self.create_subscription(LaserScan, "/scan", self.lidar_callback, 10)
        self.sub_odom = self.create_subscription(Odometry, "/odom", self.odom_callback, 10)
        self.sub_depth = self.create_subscription(Image, "/camera/depth", self.depth_callback, 10)

        # Publisher
        self.pub_cmd_vel = self.create_publisher(Twist, "/cmd_vel", 10)

        # 20 Hz Inference Timer (matching 0.05s step)
        self.timer = self.create_timer(0.05, self.inference_step)
        self.get_logger().info("AckBot Policy Inference Node initialized and active at 20 Hz.")

    def lidar_callback(self, msg: LaserScan):
        raw = np.array(msg.ranges, dtype=np.float32)
        # Handle infs and NaNs
        raw = np.nan_to_num(raw, nan=self.lidar_max, posinf=self.lidar_max, neginf=self.lidar_min)
        # Interpolate or resample to exactly 360 rays if needed
        if len(raw) != 360:
            x_old = np.linspace(0, 1, len(raw))
            x_new = np.linspace(0, 1, 360)
            raw = np.interp(x_new, x_old, raw).astype(np.float32)
        # Shift 180 degrees so index 0 corresponds to front heading (matching Gym env)
        raw = np.roll(raw, len(raw) // 2)
        # Normalize to [0, 1]
        self.lidar_ranges = np.clip((raw - self.lidar_min) / (self.lidar_max - self.lidar_min), 0.0, 1.0)

    def depth_callback(self, msg: Image):
        # Extract 64 horizontal depth samples across middle scanline
        try:
            if msg.encoding in ["32FC1", "depth"]:
                data = np.frombuffer(msg.data, dtype=np.float32).reshape((msg.height, msg.width))
            elif msg.encoding == "16UC1":
                data = np.frombuffer(msg.data, dtype=np.uint16).reshape((msg.height, msg.width)).astype(np.float32) / 1000.0
            else:
                return

            mid_row = data[msg.height // 2, :]
            # Sample 64 points across 85 deg FOV, reversing so right=-FOV/2 to left=+FOV/2 matches Gym
            sample_idx = np.linspace(0, msg.width - 1, 64, dtype=int)
            sampled = mid_row[sample_idx][::-1]
            sampled = np.nan_to_num(sampled, nan=self.cam_max, posinf=self.cam_max, neginf=self.cam_min)
            self.cam_depth = np.clip((sampled - self.cam_min) / (self.cam_max - self.cam_min), 0.0, 1.0)
        except Exception:
            pass

    def odom_callback(self, msg: Odometry):
        self.current_v = float(msg.twist.twist.linear.x)
        # Approximate steering from angular velocity: w = (v / L) * tan(steer)
        if abs(self.current_v) > 0.05:
            tan_delta = (msg.twist.twist.angular.z * self.wheelbase) / self.current_v
            self.current_steer = math.atan(np.clip(tan_delta, -1.0, 1.0))
        else:
            self.current_steer = 0.0

    def inference_step(self):
        # Build 426-dim observation vector [64 cam, 360 lidar, 2 state]
        state = np.array(
            [self.current_v / self.v_max, self.current_steer / self.steer_max],
            dtype=np.float32,
        )
        obs = np.concatenate([self.cam_depth, self.lidar_ranges, state], dtype=np.float32)

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            action = self.model.get_deterministic_action(obs_t).squeeze(0).cpu().numpy()

        target_v = float(action[0])
        target_steer = float(action[1])

        # Publish Ackermann Twist command
        twist = Twist()
        twist.linear.x = target_v
        # Yaw rate for Ackermann bicycle model: w = (v / L) * tan(delta)
        twist.angular.z = (target_v / self.wheelbase) * math.tan(target_steer)

        self.pub_cmd_vel.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = AckBotPolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
