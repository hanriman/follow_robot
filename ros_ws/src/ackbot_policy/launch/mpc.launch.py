#!/usr/bin/env python3
"""
Launch file for AckBot World Model Latent CEM-MPC deployment node.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    wm_checkpoint = LaunchConfiguration(
        "wm_checkpoint",
        default="/workspace/outputs/experiments/sigreg_curriculum/checkpoints/best_model.pt",
    )
    reward_checkpoint = LaunchConfiguration(
        "reward_checkpoint",
        default="/workspace/outputs/experiments/mpc_curriculum/checkpoints/reward_head_lewm.pt",
    )
    horizon = LaunchConfiguration("horizon", default="5")
    pop_size = LaunchConfiguration("pop_size", default="35")
    wander = LaunchConfiguration("wander", default="true")

    mpc_node = Node(
        package="ackbot_policy",
        executable="wm_mpc_node",
        name="ackbot_wm_mpc_node",
        output="screen",
        parameters=[
            {"wm_checkpoint": wm_checkpoint},
            {"reward_checkpoint": reward_checkpoint},
            {"horizon": horizon},
            {"pop_size": pop_size},
            {"wheelbase": 0.30},
            {"v_max": 0.80},
            {"steer_max": 0.553953},
            {"control_rate": 20.0},
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
            DeclareLaunchArgument("wm_checkpoint", default_value=wm_checkpoint),
            DeclareLaunchArgument("reward_checkpoint", default_value=reward_checkpoint),
            DeclareLaunchArgument("horizon", default_value="5"),
            DeclareLaunchArgument("pop_size", default_value="35"),
            DeclareLaunchArgument(
                "wander",
                default_value="true",
                description="Enable autonomous wandering for human target beacon",
            ),
            mpc_node,
            wander_node,
        ]
    )
