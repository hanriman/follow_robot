#!/usr/bin/env python3
"""
Launch file for AckBot policy deployment node.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    checkpoint_path = LaunchConfiguration(
        "checkpoint_path",
        default="/workspace/outputs/experiments/exp_20261103_170712_ppo_active_follow/checkpoints/best_model.pt",
    )
    wander = LaunchConfiguration("wander", default="true")

    policy_node = Node(
        package="ackbot_policy",
        executable="policy_node",
        name="ackbot_policy_node",
        output="screen",
        parameters=[
            {"checkpoint_path": checkpoint_path},
            {"wheelbase": 0.30},
            {"v_max": 0.80},
            {"steer_max": 0.553953},
        ],
    )

    wander_node = Node(
        package="ackbot_policy",
        executable="target_wander_node",
        name="target_wander_node",
        output="screen",
        condition=IfCondition(wander),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("checkpoint_path", default_value=checkpoint_path),
            DeclareLaunchArgument(
                "wander",
                default_value="true",
                description="Enable autonomous wandering for human target beacon",
            ),
            policy_node,
            wander_node,
        ]
    )
