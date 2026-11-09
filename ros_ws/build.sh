#!/usr/bin/env bash
# Helper script to build the ROS 2 workspace inside the container

set -e
WS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${WS_DIR}"

# Source ROS 2 base environment
if [ -f "/opt/ros/jazzy/setup.bash" ]; then
    source "/opt/ros/jazzy/setup.bash"
fi

echo "===================================================================="
echo "               BUILDING ACKBOT ROS 2 WORKSPACE                     "
echo "===================================================================="

colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release

echo "===================================================================="
echo "Build complete! Source workspace with: source /workspace/ros_ws/install/setup.bash"
echo "===================================================================="
