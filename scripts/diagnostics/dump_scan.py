#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


class ScanDump(Node):
    def __init__(self):
        super().__init__("scan_dump")
        self.sub = self.create_subscription(LaserScan, "/scan", self.cb, 10)

    def cb(self, msg: LaserScan):
        raw = np.array(msg.ranges, dtype=np.float32)
        raw = np.nan_to_num(raw, nan=3.0, posinf=3.0, neginf=0.1)
        raw = np.roll(raw, 180)
        # Beams in front: indices 345..359 and 0..15 (approx +-15 degrees)
        fwd_idx = list(range(345, 360)) + list(range(16))
        fwd_vals = [(i, raw[i]) for i in fwd_idx if raw[i] < 2.9]
        print(f"Forward hits (<2.9m): {fwd_vals}")
        rclpy.shutdown()


rclpy.init()
node = ScanDump()
rclpy.spin(node)
