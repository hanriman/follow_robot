#!/usr/bin/env python3
"""
Test VelocityControl response of human_target in Gazebo Harmonic.
"""

import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node


def main():
    rclpy.init()
    node = Node("test_vel_ctrl")
    pub = node.create_publisher(Twist, "/human_target/cmd_vel", 10)

    # Send pure angular.z
    print("Sending pure angular.z = 1.0 rad/s for 3 seconds...")
    twist = Twist()
    twist.angular.z = 1.0
    for _ in range(30):
        pub.publish(twist)
        time.sleep(0.1)

    # Send linear.x
    print("Sending linear.x = 0.5 m/s for 3 seconds...")
    twist = Twist()
    twist.linear.x = 0.5
    for _ in range(30):
        pub.publish(twist)
        time.sleep(0.1)

    print("Done test.")


if __name__ == "__main__":
    main()
