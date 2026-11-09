#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, LaserScan


class OdomChecker(Node):
    def __init__(self):
        super().__init__("odom_checker")
        self.sub_odom = self.create_subscription(Odometry, "/odom", self.cb_odom, 10)
        self.sub_scan = self.create_subscription(LaserScan, "/scan", self.cb_scan, 10)
        self.count = 0

    def cb_odom(self, msg: Odometry):
        self.count += 1
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        lin_v = msg.twist.twist.linear.x
        ang_w = msg.twist.twist.angular.z
        if self.count % 5 == 1:
            print(
                f"[Odom #{self.count:03d}] Pos: ({p.x:+.3f}, {p.y:+.3f}, {p.z:+.3f}) | "
                f"Quat: (x={q.x:+.3f}, y={q.y:+.3f}, z={q.z:+.3f}, w={q.w:+.3f}) | "
                f"Yaw: {math.degrees(yaw):+6.1f}° | Vel: lin={lin_v:+.3f}, ang={ang_w:+.3f}"
            )

    def cb_scan(self, msg: LaserScan):
        pass


def main():
    rclpy.init()
    node = OdomChecker()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
