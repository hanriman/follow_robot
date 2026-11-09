#!/usr/bin/env python3
"""
Master Launch File: AckBot Gazebo Harmonic 3D Simulation Arena & ROS 2 Bridges.
Supports dynamic world loading, parameterized spawn poses, and multi-agent bridges.
"""

from pathlib import Path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def setup_simulation(context, *args, **kwargs):
    pkg_desc = FindPackageShare("ackbot_description")
    pkg_gz = FindPackageShare("ackbot_gazebo")
    pkg_ros_gz_sim = FindPackageShare("ros_gz_sim")

    use_sim_time = LaunchConfiguration("use_sim_time")
    headless = LaunchConfiguration("headless")
    world_name_str = LaunchConfiguration("world_name").perform(context)
    spawn_x = LaunchConfiguration("spawn_x").perform(context)
    spawn_y = LaunchConfiguration("spawn_y").perform(context)
    spawn_z = LaunchConfiguration("spawn_z").perform(context)
    spawn_yaw = LaunchConfiguration("spawn_yaw").perform(context)

    # 1. World file
    world_file = PathJoinSubstitution([pkg_gz, "worlds", world_name_str]).perform(context)
    world_base = Path(world_name_str).stem  # e.g. follow_curriculum, follow_uturn, follow_clutter

    # 2. Xacro Robot Description
    xacro_file = PathJoinSubstitution([pkg_desc, "urdf", "ackbot.urdf.xacro"])
    robot_description_content = Command(["xacro ", xacro_file])
    robot_description_param = ParameterValue(robot_description_content, value_type=str)

    # 3. Gazebo Sim Node
    is_headless = headless.perform(context).lower() in ("true", "1", "yes")
    gz_flag = "-s -r " if is_headless else "-r "
    gz_sim_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_ros_gz_sim, "launch", "gz_sim.launch.py"])
        ),
        launch_arguments={"gz_args": f"{gz_flag}{world_file}"}.items(),
    )

    # 4. Robot State Publisher Node
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[
            {"robot_description": robot_description_param, "use_sim_time": use_sim_time}
        ],
    )

    # 5. Spawn AckBot in Gazebo
    spawn_ackbot = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=[
            "-name", "ackbot",
            "-string", robot_description_content,
            "-x", spawn_x,
            "-y", spawn_y,
            "-z", spawn_z,
            "-Y", spawn_yaw,
        ],
    )

    # 6. ROS - Gazebo Bridge
    bridge_args = [
        # Simulation Clock: GZ -> ROS 2
        "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
        # LiDAR: GZ -> ROS 2
        "/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan",
        # Camera RGB: GZ -> ROS 2
        "/camera/image@sensor_msgs/msg/Image[gz.msgs.Image",
        # Camera Depth: GZ -> ROS 2
        "/camera/depth_image@sensor_msgs/msg/Image[gz.msgs.Image",
        # Odometry & TF: GZ -> ROS 2
        "/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry",
        "/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V",
        # Joint States: GZ -> ROS 2
        "/joint_states@sensor_msgs/msg/JointState[gz.msgs.Model",
        # Control: ROS 2 -> GZ
        "/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist",
        # Primary Target Motion: ROS 2 -> GZ
        "/human_target/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist",
        # Secondary Distractor Motion: ROS 2 -> GZ
        "/distractor_human/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist",
        # Aerial Spectator Camera: GZ -> ROS 2
        "/overhead_camera/image@sensor_msgs/msg/Image[gz.msgs.Image",
        # Dynamic Poses for Active World: GZ -> ROS 2
        f"/world/{world_base}/dynamic_pose/info@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V",
    ]

    remappings = [
        ("/camera/image", "/camera/image_raw"),
        ("/camera/depth_image", "/camera/depth"),
        ("/overhead_camera/image", "/overhead_camera/image_raw"),
        (f"/world/{world_base}/dynamic_pose/info", "/gazebo/dynamic_pose"),
    ]

    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
        arguments=bridge_args,
        remappings=remappings,
    )

    # 7. Foxglove Studio Bridge
    foxglove_bridge = Node(
        package="foxglove_bridge",
        executable="foxglove_bridge",
        name="foxglove_bridge",
        output="screen",
        parameters=[
            {"port": 8765},
            {"use_sim_time": use_sim_time},
            {"send_buffer_limit": 10000000},
        ],
    )

    return [
        gz_sim_node,
        robot_state_publisher,
        spawn_ackbot,
        bridge,
        foxglove_bridge,
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("headless", default_value="false"),
            DeclareLaunchArgument("world_name", default_value="follow_curriculum.sdf"),
            DeclareLaunchArgument("spawn_x", default_value="1.2"),
            DeclareLaunchArgument("spawn_y", default_value="-0.6"),
            DeclareLaunchArgument("spawn_z", default_value="0.06"),
            DeclareLaunchArgument("spawn_yaw", default_value="1.5707963"),
            OpaqueFunction(function=setup_simulation),
        ]
    )
