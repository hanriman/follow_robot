#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


class LidarInspector(Node):
    def __init__(self):
        super().__init__("lidar_inspector")
        self.sub_scan = self.create_subscription(LaserScan, "/scan", self.cb_scan, 10)
        self.lidar_min = 0.10
        self.lidar_max = 3.00
        self.count = 0

    def cb_scan(self, msg: LaserScan):
        raw = np.array(msg.ranges, dtype=np.float32)
        raw = np.nan_to_num(raw, nan=self.lidar_max, posinf=self.lidar_max, neginf=self.lidar_min)
        if len(raw) != 360:
            x_old = np.linspace(0, 1, len(raw))
            x_new = np.linspace(0, 1, 360)
            raw = np.interp(x_new, x_old, raw).astype(np.float32)
        raw = np.roll(raw, 180)
        self.count += 1
        if self.count % 10 == 1:
            # Check 4 cardinal directions:
            # index 0: Front
            # index 90: Left (+90 deg)
            # index 180: Back
            # index 270: Right (-90 deg)
            print(
                f"[Scan #{self.count:03d}] Front(idx 0)={raw[0]:.2f}m | "
                f"Left(idx 90)={raw[90]:.2f}m | "
                f"Back(idx 180)={raw[180]:.2f}m | "
                f"Right(idx 270)={raw[270]:.2f}m | "
                f"Min={np.min(raw):.2f}m at idx {np.argmin(raw)}"
            )


def main():
    rclpy.init()
    node = LidarInspector()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
