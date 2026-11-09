#!/bin/bash
set -e

# Setup VNC environment if not already running
if ! pgrep -x "Xtigervnc" > /dev/null; then
    mkdir -p ~/.vnc
    VNC_PASSWD="${VNC_PASSWORD:-ackbot2026}"
    if command -v vncpasswd > /dev/null; then
        echo "$VNC_PASSWD" | vncpasswd -f > ~/.vnc/passwd
        chmod 600 ~/.vnc/passwd
        vncserver :1 -geometry 1920x1080 -depth 24 -SecurityTypes VncAuth > /dev/null 2>&1 || true
    else
        # Fallback to passwordless VNC (safe because docker-compose restricts port to 127.0.0.1)
        vncserver :1 -geometry 1920x1080 -depth 24 -SecurityTypes None > /dev/null 2>&1 || true
    fi

    # Start lightweight window manager
    DISPLAY=:1 openbox-session > /dev/null 2>&1 &

    # Start Web-based noVNC proxy on port 6080
    websockify --web=/usr/share/novnc 6080 localhost:5901 > /dev/null 2>&1 &
fi

# Source ROS 2 core environment
if [ -f "/opt/ros/jazzy/setup.bash" ]; then
    source "/opt/ros/jazzy/setup.bash"
fi

# Source local workspace if built
if [ -f "/workspace/ros_ws/install/setup.bash" ]; then
    source "/workspace/ros_ws/install/setup.bash"
fi

# Set Gazebo Sim environment paths
export GZ_SIM_RESOURCE_PATH="/workspace/ros_ws/src/ackbot_description:/workspace/ros_ws/src/ackbot_gazebo/worlds:${GZ_SIM_RESOURCE_PATH}"

# Execute user command
exec "$@"
