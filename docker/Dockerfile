# AckBot ROS 2 Jazzy & Gazebo Harmonic Simulation Container
# 100% Native ARM64 for Apple Silicon (MacBook Pro M4 Pro)
# Built from official multi-arch ros:jazzy-ros-base

FROM --platform=linux/arm64 ros:jazzy-ros-base

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV ROS_DISTRO=jazzy

# 1. Configure APT with Retries and Timeout to prevent connection dropouts on arm64 mirrors
RUN echo 'Acquire::Retries "5";' > /etc/apt/apt.conf.d/80-retries \
    && echo 'Acquire::http::Timeout "60";' >> /etc/apt/apt.conf.d/80-retries \
    && echo 'Acquire::https::Timeout "60";' >> /etc/apt/apt.conf.d/80-retries

# 2. Install Web Desktop stack (TigerVNC + noVNC + Openbox) and build tools
RUN apt-get update && apt-get install -y --fix-missing --no-install-recommends \
    bash-completion \
    build-essential \
    curl \
    git \
    nano \
    net-tools \
    novnc \
    openbox \
    python3-colcon-common-extensions \
    python3-pip \
    python3-rosdep \
    sudo \
    tigervnc-standalone-server \
    tigervnc-common \
    tigervnc-tools \
    websockify \
    wget \
    x11-xserver-utils \
    xterm \
    && rm -rf /var/lib/apt/lists/*

# 3. Install Gazebo Harmonic, RViz2, Xacro, Foxglove Bridge, and Ackermann msgs
RUN apt-get update && apt-get install -y --fix-missing --no-install-recommends \
    ros-jazzy-ackermann-msgs \
    ros-jazzy-foxglove-bridge \
    ros-jazzy-joint-state-publisher \
    ros-jazzy-joint-state-publisher-gui \
    ros-jazzy-robot-state-publisher \
    ros-jazzy-ros-gz \
    ros-jazzy-ros-gz-bridge \
    ros-jazzy-ros-gz-image \
    ros-jazzy-ros-gz-sim \
    ros-jazzy-rviz2 \
    ros-jazzy-teleop-twist-keyboard \
    ros-jazzy-xacro \
    && rm -rf /var/lib/apt/lists/*

# 4. Install Python Dependencies for Neural Policy Inference
# Note: Ubuntu 24.04 / ROS 2 Desktop pre-installs python3-numpy, python3-scipy, and python3-yaml via Debian.
# We install torch, torchvision, and opencv with --ignore-installed so pip never attempts to uninstall Debian packages.
RUN pip3 install --no-cache-dir --break-system-packages --ignore-installed \
    opencv-python-headless==4.10.0.84 \
    torch==2.4.0 \
    torchvision==0.19.0

# 5. Setup noVNC Web Interface
RUN ln -s /usr/share/novnc/vnc.html /usr/share/novnc/index.html

# 6. Environment & Workspace Setup
ENV DISPLAY=:1
ENV VNC_PORT=5901
ENV NO_VNC_PORT=6080
ENV FOXGLOVE_PORT=8765

WORKDIR /workspace

# 7. Setup Entrypoint
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 6080 8765

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
