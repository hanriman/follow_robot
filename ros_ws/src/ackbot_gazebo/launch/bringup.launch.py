#!/usr/bin/env python3
"""
Master Bringup Launch File:
Runs 3D Gazebo Harmonic simulation arena, bridges, autonomous wandering target,
and trained follower policy node for all curriculum scenarios:
- Stage 1 ('open'): Wide obstacle-free tracking
- Stage 2 ('corner'): L-Corner Blind Bend with visual occlusion
- Stage 3 ('uturn'): Dead-end corridor 180-deg reversal and safe yielding
- Stage 4 ('clutter'): Cluttered pillar arena with appearance Re-ID distractor
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EqualsSubstitution, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def setup_bringup(context, *args, **kwargs):
    pkg_gz = FindPackageShare("ackbot_gazebo")
    pkg_policy = FindPackageShare("ackbot_policy")
    pkg_desc = FindPackageShare("ackbot_description")

    use_sim_time = LaunchConfiguration("use_sim_time")
    controller = LaunchConfiguration("controller").perform(context).lower()
    scenario = LaunchConfiguration("scenario").perform(context).lower()
    checkpoint = LaunchConfiguration("checkpoint").perform(context)
    wm_checkpoint = LaunchConfiguration("wm_checkpoint").perform(context)
    reward_checkpoint = LaunchConfiguration("reward_checkpoint").perform(context)
    wander = LaunchConfiguration("wander").perform(context).lower() in ("true", "1", "yes")
    rviz = LaunchConfiguration("rviz").perform(context).lower() in ("true", "1", "yes")
    headless = LaunchConfiguration("headless")

    # Configure scenario spawn pose and world file
    if scenario == "uturn":
        world_name = "follow_uturn.sdf"
        spawn_x = "-1.2"
        spawn_y = "0.0"
        spawn_yaw = "0.0"
    elif scenario == "clutter":
        world_name = "follow_clutter.sdf"
        spawn_x = "0.0"
        spawn_y = "-1.0"
        spawn_yaw = "1.5707963"
    elif scenario == "open":
        world_name = "follow_arena.sdf"
        spawn_x = "0.0"
        spawn_y = "-1.0"
        spawn_yaw = "1.5707963"
    else:  # default: corner
        scenario = "corner"
        world_name = "follow_curriculum.sdf"
        spawn_x = "1.2"
        spawn_y = "-0.6"
        spawn_yaw = "1.5707963"

    nodes_to_launch = []

    # 1. Gazebo Harmonic Simulation & Bridges
    sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_gz, "launch", "sim.launch.py"])
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "world_name": world_name,
            "spawn_x": spawn_x,
            "spawn_y": spawn_y,
            "spawn_z": "0.06",
            "spawn_yaw": spawn_yaw,
            "headless": headless,
        }.items(),
    )
    nodes_to_launch.append(sim_launch)

    # 2. Autonomous Target Wandering Node
    if wander:
        wander_node = Node(
            package="ackbot_policy",
            executable="target_wander_node",
            name="target_wander_node",
            output="screen",
            parameters=[
                {"scenario": scenario},
                {"use_sim_time": use_sim_time},
            ],
        )
        nodes_to_launch.append(wander_node)

    # 3. Autonomous Follower Policy Node
    if controller == "mpc":
        mpc_node = Node(
            package="ackbot_policy",
            executable="wm_mpc_node",
            name="ackbot_wm_mpc_node",
            output="screen",
            emulate_tty=True,
            parameters=[
                {
                    "scenario": scenario,
                    "wm_checkpoint": wm_checkpoint,
                    "reward_checkpoint": reward_checkpoint,
                    "horizon": 5,
                    "pop_size": 35,
                    "num_elites": 5,
                    "num_iters": 3,
                    "wheelbase": 0.30,
                    "v_max": 0.80,
                    "steer_max": 0.553953,
                    "control_rate": 20.0,
                    "use_sim_time": use_sim_time,
                }
            ],
        )
        nodes_to_launch.append(mpc_node)
    elif controller == "ppo":
        ppo_node = Node(
            package="ackbot_policy",
            executable="policy_node",
            name="ackbot_policy_node",
            output="screen",
            parameters=[
                {"checkpoint_path": checkpoint},
                {"wheelbase": 0.30},
                {"v_max": 0.80},
                {"steer_max": 0.553953},
                {"use_sim_time": use_sim_time},
            ],
        )
        nodes_to_launch.append(ppo_node)

    # 4. Optional RViz2 Dashboard
    if rviz:
        rviz_node = Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            output="screen",
            parameters=[{"use_sim_time": use_sim_time}],
            arguments=["-d", PathJoinSubstitution([pkg_desc, "rviz", "sim_view.rviz"])],
        )
        nodes_to_launch.append(rviz_node)

    return nodes_to_launch


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument(
                "controller",
                default_value="mpc",
                description="Controller type: 'mpc' (LeWM Latent MPC) or 'ppo' (Reactive PPO)",
            ),
            DeclareLaunchArgument(
                "scenario",
                default_value="corner",
                description="Curriculum scenario: 'open', 'corner', 'uturn', 'clutter'",
            ),
            DeclareLaunchArgument(
                "checkpoint",
                default_value="/workspace/outputs/experiments/exp_20261108_154753_curriculum_ppo_active_follow/checkpoints/best_model.pt",
                description="Path to PPO checkpoint .pt",
            ),
            DeclareLaunchArgument(
                "wm_checkpoint",
                default_value="/workspace/outputs/experiments/sigreg_curriculum/checkpoints/best_model.pt",
                description="Path to LeWM checkpoint .pt",
            ),
            DeclareLaunchArgument(
                "reward_checkpoint",
                default_value="/workspace/outputs/experiments/mpc_curriculum/checkpoints/reward_head_lewm.pt",
                description="Path to Latent Reward Head checkpoint .pt",
            ),
            DeclareLaunchArgument(
                "wander",
                default_value="true",
                description="Launch wandering human target beacon",
            ),
            DeclareLaunchArgument(
                "rviz",
                default_value="false",
                description="Launch RViz2 display inside noVNC",
            ),
            DeclareLaunchArgument(
                "headless",
                default_value="false",
                description="Run Gazebo in headless server-only mode (saves ~250% CPU)",
            ),
            OpaqueFunction(function=setup_bringup),
        ]
    )
