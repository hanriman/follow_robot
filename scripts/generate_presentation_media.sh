#!/usr/bin/env bash
# ==============================================================================
# Script: generate_presentation_media.sh
# Purpose: Master automated pipeline to launch Gazebo Harmonic simulation,
#          run LeWM CEM-MPC multi-pedestrian tracking, record 18s synchronized
#          composite media (MP4/GIF/Keyframes) for curriculum scenarios,
#          and automatically shut down background processes to keep CPU at 0% idle.
#
# Usage:
#   ./scripts/generate_presentation_media.sh [--scenario <corner|uturn|clutter|all>] [--keep-alive] [--gui]
# ==============================================================================

set -euo pipefail

CONTAINER_NAME="ackbot_simulation"
MEDIA_DIR="outputs/reports/media"
DURATION=18.0
FPS=15.0
GIF_FPS=10.0
KEYFRAME_INTERVAL=3.0

KEEP_ALIVE=false
HEADLESS=true
SCENARIO="corner"

# Parse command line flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        --scenario)
            SCENARIO="$2"
            shift 2
            ;;
        --scenario=*)
            SCENARIO="${1#*=}"
            shift 1
            ;;
        --keep-alive)
            KEEP_ALIVE=true
            shift 1
            ;;
        --gui)
            HEADLESS=false
            shift 1
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

echo "======================================================================"
echo "    ACKBOT 3D GAZEBO HARMONIC PRESENTATION MEDIA GENERATOR"
echo "======================================================================"
echo "Scenario       : ${SCENARIO}"
echo "Mode           : Headless=${HEADLESS} (saves ~250% GUI rendering CPU)"
echo "Keep-Alive     : ${KEEP_ALIVE} (auto-cleanup releases CPU when done)"
echo "Target Media   : ${MEDIA_DIR} (${DURATION}s @ ${FPS} FPS)"
echo "======================================================================"

# 1. Ensure Docker Container is Active
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "[!] Container '${CONTAINER_NAME}' is not running. Starting..."
    docker start "${CONTAINER_NAME}"
    sleep 3
fi

echo "[✓] Container '${CONTAINER_NAME}' is active."

# Cleanup handler function
cleanup() {
    if [ "${KEEP_ALIVE}" != "true" ]; then
        echo ""
        echo "[+] Automatically shutting down simulation stack to release CPU..."
        docker exec "${CONTAINER_NAME}" python3 /workspace/scripts/sim/clean_simulation_processes.py || true
        echo "[✓] Simulation stopped. Host CPU returned to 0% idle."
    else
        echo ""
        echo "[i] Keep-alive active. Simulation is still running in background."
        echo "    To stop it manually, run: ./scripts/run_docker_simulation.sh stop"
    fi
}

trap cleanup EXIT

# Function to record a single scenario
record_scenario() {
    local sc="$1"
    local mp4_name
    local gif_name
    local kf_prefix

    if [ "$sc" == "uturn" ]; then
        mp4_name="gazebo_uturn_tracking.mp4"
        gif_name="gazebo_uturn_tracking.gif"
        kf_prefix="gazebo_uturn_frame_"
    elif [ "$sc" == "clutter" ]; then
        mp4_name="gazebo_clutter_tracking.mp4"
        gif_name="gazebo_clutter_tracking.gif"
        kf_prefix="gazebo_clutter_frame_"
    else
        mp4_name="gazebo_lewm_tracking.mp4"
        gif_name="gazebo_lewm_tracking.gif"
        kf_prefix="gazebo_frame_"
    fi

    echo ""
    echo "======================================================================"
    echo "    RECORDING SCENARIO: ${sc^^} (${mp4_name})"
    echo "======================================================================"

    # Kill any stale duplicate simulation instances
    echo "[+] Cleaning previous simulation processes..."
    docker exec "${CONTAINER_NAME}" python3 /workspace/scripts/sim/clean_simulation_processes.py
    sleep 1

    # Launch Bringup in Container
    echo "[+] Launching simulation stack in container for scenario: ${sc}..."
    docker exec -d "${CONTAINER_NAME}" bash -c \
        "source /opt/ros/jazzy/setup.bash && source /workspace/ros_ws/install/setup.bash && \
         ros2 launch ackbot_gazebo bringup.launch.py controller:=mpc scenario:=${sc} headless:=${HEADLESS} > /tmp/bringup.log 2>&1"

    echo "[i] Waiting 8s for Gazebo Harmonic world and camera bridges to initialize..."
    sleep 8

    # Create host media output directory
    mkdir -p "${MEDIA_DIR}"

    # Execute Multi-View Synchronized Media Recorder
    echo "[*] Launching Synchronized Media Recorder (${DURATION}s @ ${FPS} FPS)..."
    docker exec -t "${CONTAINER_NAME}" bash -c \
        "source /opt/ros/jazzy/setup.bash && source /workspace/ros_ws/install/setup.bash && \
         python3 /workspace/scripts/sim/record_simulation_media.py \
            --duration ${DURATION} \
            --fps ${FPS} \
            --gif-fps ${GIF_FPS} \
            --keyframe-interval ${KEYFRAME_INTERVAL} \
            --mp4-name ${mp4_name} \
            --gif-name ${gif_name} \
            --keyframe-prefix ${kf_prefix} \
            --output-dir ${MEDIA_DIR}"

    echo "[✓] Scenario ${sc^^} recording completed."
    docker exec "${CONTAINER_NAME}" python3 /workspace/scripts/sim/clean_simulation_processes.py
    sleep 2
}

if [ "$SCENARIO" == "all" ]; then
    record_scenario "corner"
    record_scenario "uturn"
    record_scenario "clutter"
else
    record_scenario "$SCENARIO"
fi

echo ""
echo "======================================================================"
echo "    GENERATED PRESENTATION MEDIA ARTIFACTS"
echo "======================================================================"
ls -lh "${MEDIA_DIR}"

echo ""
echo "[✓] Media generation completed successfully!"
