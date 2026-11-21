#!/usr/bin/env python3
"""
Inspect live camera depth and lidar ranges received from Gazebo bridges.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan


class SensorInspector(Node):
    def __init__(self):
        super().__init__("sensor_inspector")
        self.sub_depth = self.create_subscription(Image, "/camera/depth", self.cb_depth, 10)
        self.sub_scan = self.create_subscription(LaserScan, "/scan", self.cb_scan, 10)
        self.cam_min = 0.10
        self.cam_max = 5.00
        self.count = 0

    def cb_depth(self, msg: Image):
        try:
            if msg.encoding in ["32FC1", "depth"]:
                data = np.frombuffer(msg.data, dtype=np.float32).reshape((msg.height, msg.width))
            elif msg.encoding == "16UC1":
                data = (
                    np.frombuffer(msg.data, dtype=np.uint16)
                    .reshape((msg.height, msg.width))
                    .astype(np.float32)
                    / 1000.0
                )
            else:
                return

            mid_row = data[msg.height // 2, :]
            sample_idx = np.linspace(0, msg.width - 1, 64, dtype=int)
            sampled = mid_row[sample_idx][::-1]
            _cam_depth = np.clip((sampled - self.cam_min) / (self.cam_max - self.cam_min), 0.0, 1.0)

            min_i = int(np.argmin(sampled))
            min_val = float(sampled[min_i])
            self.count += 1
            if self.count % 10 == 1:
                print(
                    f"[Depth #{self.count:03d}] Min depth = {min_val:.2f}m at index {min_i}/64 | Center 5 = {sampled[30:35]}"
                )
        except Exception as e:
            print(f"Error in cb_depth: {e}")

    def cb_scan(self, msg: LaserScan):
        pass


def main():
    rclpy.init()
    node = SensorInspector()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
