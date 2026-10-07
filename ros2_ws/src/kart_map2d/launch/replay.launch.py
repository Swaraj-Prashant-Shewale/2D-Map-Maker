"""Replay a recorded bag into the live mapping node (laptop / container test).

    ros2 launch kart_map2d replay.launch.py cfg:=/abs/kart_master.cfg \
        bag:=/recordings/sept3_20260905_130304

Starts the mapping stack exactly like mapping.launch.py but with
``use_sim_time`` FORCED on, and plays the bag with the proven clock
settings: ``--clock 200`` (the node's now() then lags the recording by at
most 5 ms; plain --clock is 40 Hz) ``-r 1.0 --read-ahead-queue-size 1000
--delay 2`` (DDS discovery finishes before the first message). Copy the bag
to a container-local path first when it lives on a Windows bind mount -
reading it through the mount is too slow for 1.0x.

When done:  ros2 service call /kart_map2d/save_map std_srvs/srv/Trigger
"""

from __future__ import annotations

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from kart_map2d.config.master_config import SENSOR_KEYS, MasterConfigError, load_master_config


def _setup(context, *args, **kwargs):
    cfg_path = LaunchConfiguration("cfg").perform(context)
    bag = LaunchConfiguration("bag").perform(context)
    rate = LaunchConfiguration("rate").perform(context)
    overrides = {}
    for key in SENSOR_KEYS:
        value = LaunchConfiguration(key).perform(context).strip()
        if value != "":
            overrides[key] = int(value)
    try:
        cfg = load_master_config(cfg_path, overrides=overrides)
    except MasterConfigError as exc:
        raise RuntimeError("kart_master.cfg refused:\n" + "\n".join(exc.problems)) from None
    set_args: list[str] = []
    for key, value in overrides.items():
        set_args += ["--set", f"sensors.{key}={value}"]
    imu_xyz = cfg["imu"]["mount_xyz"]
    return [
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="base_to_imu_tf",
            arguments=[
                "--x", str(imu_xyz[0]), "--y", str(imu_xyz[1]), "--z", str(imu_xyz[2]),
                "--roll", "0", "--pitch", "0", "--yaw", "0",
                "--frame-id", "base_link", "--child-frame-id", str(cfg["imu"]["frame"]),
            ],
        ),
        Node(
            package="kart_map2d",
            executable="mapping_node",
            output="screen",
            arguments=["--cfg", cfg_path, *set_args, "--ros-args", "--log-level", str(cfg["system"]["log_level"])],
            # A bag replay runs on the bag's clock, full stop.
            parameters=[{"use_sim_time": True}],
        ),
        ExecuteProcess(
            cmd=[
                "ros2", "bag", "play", bag,
                "--clock", "200",
                "-r", rate,
                "--read-ahead-queue-size", "1000",
                "--delay", "2",
            ],
            output="screen",
        ),
    ]


def generate_launch_description() -> LaunchDescription:
    share = Path(get_package_share_directory("kart_map2d"))
    args = [
        DeclareLaunchArgument("cfg", description="absolute path to kart_master.cfg"),
        DeclareLaunchArgument("bag", description="bag directory to play (container-local path)"),
        DeclareLaunchArgument("rate", default_value="1.0", description="playback rate"),
    ]
    args += [
        DeclareLaunchArgument(key, default_value="", description=f"override [sensors] {key} (0/1)")
        for key in SENSOR_KEYS
    ]
    return LaunchDescription(
        args
        + [
            SetEnvironmentVariable(
                "FASTRTPS_DEFAULT_PROFILES_FILE",
                os.environ.get("FASTRTPS_DEFAULT_PROFILES_FILE", str(share / "fastdds_large_shm.xml")),
            ),
            OpaqueFunction(function=_setup),
        ]
    )
