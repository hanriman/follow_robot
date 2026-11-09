#!/usr/bin/env python3
"""
Diagnostic trajectory recorder to track exact positions, headway, and clearances.
"""

import math
import sys
import time
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
import numpy as np


class TrajectoryDiagnoser(Node):
    def __init__(self, duration=15.0):
        super().__init__("trajectory_diagnoser")
        self.duration = duration
        self.t0 = time.time()
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.robot_v = 0.0
        self.robot_w = 0.0
        self.cmd_v = 0.0
        self.cmd_w = 0.0
        self.mpc_status = ""
        self.min_fwd_lidar = 99.0
        self.min_all_lidar = 99.0

        self.sub_odom = self.create_subscription(Odometry, "/odom", self._cb_odom, 10)
        self.sub_cmd = self.create_subscription(Twist, "/cmd_vel", self._cb_cmd, 10)
        self.sub_scan = self.create_subscription(LaserScan, "/scan", self._cb_scan, 10)
        self.sub_stat = self.create_subscription(String, "/ackbot/mpc_status", self._cb_stat, 10)

        self.timer = self.create_timer(0.25, self._on_timer)
        self.samples = []

    def _cb_odom(self, msg: Odometry):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.robot_yaw = math.atan2(siny_cosp, cosy_cosp)
        self.robot_v = msg.twist.twist.linear.x
        self.robot_w = msg.twist.twist.angular.z

    def _cb_cmd(self, msg: Twist):
        self.cmd_v = msg.linear.x
        self.cmd_w = msg.angular.z

    def _cb_scan(self, msg: LaserScan):
        raw = np.array(msg.ranges, dtype=np.float32)
        raw = np.nan_to_num(raw, nan=3.0, posinf=3.0, neginf=0.1)
        if len(raw) != 360:
            x_old = np.linspace(0, 1, len(raw))
            x_new = np.linspace(0, 1, 360)
            raw = np.interp(x_new, x_old, raw).astype(np.float32)
        raw = np.roll(raw, 180)
        fwd = np.concatenate([raw[330:], raw[:31]])
        self.min_fwd_lidar = float(np.min(fwd))
        self.min_all_lidar = float(np.min(raw))

    def _cb_stat(self, msg: String):
        self.mpc_status = msg.data

    def _on_timer(self):
        t_el = time.time() - self.t0
        if t_el > self.duration:
            self._print_summary()
            sys.exit(0)

        steer_deg = 0.0
        if abs(self.cmd_v) > 0.02:
            steer_rad = math.atan2(self.cmd_w * 0.30, self.cmd_v)
            steer_deg = math.degrees(steer_rad)

        line = (
            f"T={t_el:04.1f}s | Robot: x={self.robot_x:+.2f}, y={self.robot_y:+.2f}, yaw={math.degrees(self.robot_yaw):+5.1f}° | "
            f"Cmd: v={self.cmd_v:+.2f}m/s, steer={steer_deg:+5.1f}° | "
            f"LiDAR: fwd={self.min_fwd_lidar:.2f}m, min={self.min_all_lidar:.2f}m"
        )
        print(line)
        self.samples.append(line)

    def _print_summary(self):
        print("\n" + "=" * 60)
        print("Trajectory Diagnosis Finished.")
        print(f"Final Robot Pose: x={self.robot_x:.2f}, y={self.robot_y:.2f}, yaw={math.degrees(self.robot_yaw):.1f}°")
        print("=" * 60)


def main():
    rclpy.init()
    node = TrajectoryDiagnoser(duration=15.0)
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
