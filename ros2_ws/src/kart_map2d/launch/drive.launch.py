"""Launch the autonomous drive node from kart_master.cfg.

    ros2 launch kart_map2d drive.launch.py cfg:=/abs/path/kart_master.cfg

Starts the ``drive_node`` (reads [drive] / [safety] / the sensor sections
via ``--cfg``) and the static base_link -> imu_link TF (identity rotation,
position from [imu] mount_xyz). The node starts DISENGAGED; first tests on
stands with [drive] max_speed_mps low, then
``ros2 service call /drive/engage std_srvs/srv/SetBool "{data: true}"``.
While driving, only ``max_speed_mps`` and ``speed_scale`` follow
``ros2 param set /drive ...`` live; everything else needs a restart.
"""

from __future__ import annotations

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from kart_map2d.config.master_config import SENSOR_KEYS, MasterConfigError, load_master_config


def _setup(context, *args, **kwargs):
    cfg_path = LaunchConfiguration("cfg").perform(context)
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
    start_pose = LaunchConfiguration("start_pose").perform(context).strip()
    if start_pose:
        set_args += ["--start-pose", start_pose]
    stats_path = LaunchConfiguration("stats_path").perform(context).strip()
    if stats_path:
        set_args += ["--stats-path", stats_path]
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
            executable="drive_node",
            output="screen",
            arguments=["--cfg", cfg_path, *set_args, "--ros-args", "--log-level", str(cfg["system"]["log_level"])],
            parameters=[{"use_sim_time": bool(cfg["system"]["use_sim_time"])}],
        ),
    ]


def generate_launch_description() -> LaunchDescription:
    share = Path(get_package_share_directory("kart_map2d"))
    args = [
        DeclareLaunchArgument("cfg", description="absolute path to kart_master.cfg"),
        DeclareLaunchArgument(
            "start_pose",
            default_value="",
            description="known start pose 'x,y,yaw_deg' in the map frame (the marked grid spot), "
            "or 'global' for global relocalization; empty keeps the node's default",
        ),
        DeclareLaunchArgument("stats_path", default_value="", description="write the drive node's stats JSON here on shutdown"),
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
