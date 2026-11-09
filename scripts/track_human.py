#!/usr/bin/env python3
import time
import subprocess
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

def main():
    rclpy.init()
    node = Node("human_tracker")
    
    # We can subscribe to /overhead_camera/image_raw or check Gazebo model pose
    # In Gazebo, we can run `gz model -m human_target -p` or check the spectator frames!
    print("Checking human position...")

if __name__ == "__main__":
    main()
