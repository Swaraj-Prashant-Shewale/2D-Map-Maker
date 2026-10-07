# ROS 2 Humble test environment for kart_map2d (Ubuntu 22.04 = JetPack 6
# userland = the Orin). HUMBLE ONLY - this project targets the Orin and
# nothing else.
#
# Build (PowerShell, from the project root "2d map maker"):
#   docker build -t kart2d:humble -f docker/Dockerfile docker
#
# Run (PowerShell; Git Bash mangles -v paths):
#   docker run --rm -v "S:\Coding_Tools\My_Projects\Maveric\2d map maker:/ws2d" `
#       -v "S:\Coding_Tools\My_Projects\Maveric\Virtual map maker/inputs:/recordings:ro" `
#       --shm-size=1g kart2d:humble bash -lc "cd /ws2d/ros2_ws && colcon build --symlink-install"
#
# The project is NOT copied into the image: it is bind-mounted at /ws2d at
# run time, so edits on Windows are live inside the container.

FROM ros:humble-ros-base-jammy

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8

# rosbag2 sqlite3 storage (the kart's bags are .db3), the message/python
# helper packages both nodes import, tf2_ros for the static imu TF,
# ackermann_msgs for drive commands, and colcon to build the package.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip \
        python3-colcon-common-extensions \
        ros-humble-rosbag2-storage-default-plugins \
        ros-humble-sensor-msgs-py \
        ros-humble-tf2-ros \
        ros-humble-tf2-ros-py \
        ros-humble-ackermann-msgs \
        ros-humble-std-srvs \
    && rm -rf /var/lib/apt/lists/*

# numpy is pinned below 2 on purpose: Humble's rosidl-generated Python
# bindings are compiled against numpy 1.x, and importing them under numpy 2
# fails. The Orin's Humble has the same constraint, so kart_map2d must work
# under this pin (the CI value of this image).
COPY requirements-ros.txt /tmp/requirements-ros.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements-ros.txt

# Source ROS in every interactive shell and every `docker run ... bash -lc`.
RUN echo "source /opt/ros/humble/setup.bash" >> /etc/bash.bashrc \
 && echo "source /opt/ros/humble/setup.bash" >> /root/.profile

WORKDIR /ws2d
CMD ["bash"]
