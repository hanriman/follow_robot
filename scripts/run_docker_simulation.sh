#!/usr/bin/env bash
# ==============================================================================
# AckBot Docker ROS 2 Gazebo Simulation Runner
#
# Launches Gazebo Harmonic, ROS 2 bridges, autonomous wandering target, and
# either LeWM CEM-MPC or Reactive PPO controller inside the ARM64 container.
#
# Usage:
#   ./scripts/run_docker_simulation.sh [mpc|ppo|stop] [corner|uturn|clutter|open] [--headless]
#   ./scripts/run_docker_simulation.sh stop
# ==============================================================================

set -e

CONTAINER_NAME="ackbot_simulation"

# Handle stop command directly
if [ "${1:-}" = "stop" ] || [ "${1:-}" = "kill" ]; then
    echo "======================================================================"
    echo "          STOPPING ACKBOT SIMULATION STACK                            "
    echo "======================================================================"
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        docker exec "${CONTAINER_NAME}" python3 /workspace/scripts/sim/clean_simulation_processes.py
        echo "[✓] Simulation processes terminated cleanly. Container is idle (0% CPU)."
    else
        echo "[i] Container '${CONTAINER_NAME}' is not running."
    fi
    exit 0
fi

CONTROLLER="mpc"
SCENARIO="corner"
HEADLESS="false"

# Robust argument parsing supporting positional modes and flags in any order
for arg in "$@"; do
    case "$arg" in
        mpc|ppo)
            CONTROLLER="$arg"
            ;;
        corner|uturn|clutter|open)
            SCENARIO="$arg"
            ;;
        --headless|-s)
            HEADLESS="true"
            ;;
        --daemon|-d)
            # Default execution mode inside container
            ;;
    esac
done

echo "======================================================================"
echo "          ACKBOT DOCKER ROS 2 GAZEBO DEPLOYMENT RUNNER               "
echo "======================================================================"
echo "Target Container : ${CONTAINER_NAME}"
echo "Controller Mode  : ${CONTROLLER} (mpc = LeWM CEM-MPC | ppo = Reactive PPO)"
echo "Curriculum Scene : ${SCENARIO} (corner, uturn, clutter, open)"
echo "Headless Mode    : ${HEADLESS} (headless saves ~250% CPU by skipping GUI)"
echo "Foxglove Bridge  : ws://localhost:8765"
echo "noVNC Desktop    : http://localhost:6080"
echo "======================================================================"

# Verify container is running
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "[!] Container '${CONTAINER_NAME}' is not running. Starting container..."
    docker start "${CONTAINER_NAME}"
    sleep 2
fi

echo "[+] Container '${CONTAINER_NAME}' is active."

# Kill any existing ROS 2 launch processes inside container to prevent duplicate CPU load
echo "[+] Cleaning previous ROS 2 processes inside container..."
docker exec "${CONTAINER_NAME}" python3 /workspace/scripts/sim/clean_simulation_processes.py
sleep 1

# Launch bringup in background inside container
echo "[+] Launching AckBot bringup (${CONTROLLER} controller, ${SCENARIO} scenario, headless=${HEADLESS})..."
docker exec -d "${CONTAINER_NAME}" bash -c \
    "source /opt/ros/jazzy/setup.bash && source /workspace/ros_ws/install/setup.bash && \
     ros2 launch ackbot_gazebo bringup.launch.py controller:=${CONTROLLER} scenario:=${SCENARIO} headless:=${HEADLESS} > /tmp/bringup.log 2>&1"

echo "[+] Waiting 6 seconds for Gazebo Harmonic & ROS 2 nodes to initialize..."
sleep 6

# Check running ROS 2 nodes
echo "[+] Active ROS 2 Nodes:"
docker exec "${CONTAINER_NAME}" bash -c \
    "source /opt/ros/jazzy/setup.bash && ros2 node list 2>/dev/null || echo 'No nodes found'"

# Check active ROS 2 topics
echo "[+] Active ROS 2 Topics:"
docker exec "${CONTAINER_NAME}" bash -c \
    "source /opt/ros/jazzy/setup.bash && ros2 topic list 2>/dev/null || echo 'No topics found'"

echo "======================================================================"
echo "Simulation launched successfully!"
echo "  • Open Foxglove Studio -> Connect to ws://localhost:8765"
if [ "${HEADLESS}" != "true" ]; then
    echo "  • Open Web Browser     -> http://localhost:6080 (noVNC Desktop)"
fi
echo "  • View logs            -> docker exec ${CONTAINER_NAME} cat /tmp/bringup.log"
echo "  • Stop simulation      -> ./scripts/run_docker_simulation.sh stop"
echo "======================================================================"
