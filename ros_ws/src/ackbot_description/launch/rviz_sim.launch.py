#!/usr/bin/env python3
"""
Launch RViz2 configured for live AckBot Gazebo simulation inspection:
LiDAR scan, RGB camera viewport, Odometry path, and TF tree.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare("ackbot_description")
    use_sim_time = LaunchConfiguration("use_sim_time", default="true")
    rviz_config = PathJoinSubstitution([pkg_share, "rviz", "sim_view.rviz"])

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
        arguments=["-d", rviz_config],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="true",
                description="Use simulation (Gazebo) clock if true",
            ),
            rviz_node,
        ]
    )
