"""Launch the live 2D mapping node from kart_master.cfg.

    ros2 launch kart_map2d mapping.launch.py cfg:=/abs/path/kart_master.cfg
    ros2 launch kart_map2d mapping.launch.py cfg:=... lidar:=1        # flip a [sensors] switch
    ros2 launch kart_map2d mapping.launch.py cfg:=... use_sim_time:=1 # bag replay

What it starts:
- the ``mapping_node`` (reads the cfg itself: ``--cfg`` + ``--set`` overrides;
  the [system]/use_sim_time scalars are also passed as ROS parameters, which
  is the launch-file translation of the cfg - a ``-p key:=value`` from the
  command line still outranks both),
- a ``tf2_ros`` static_transform_publisher for base_link -> imu_link
  (identity rotation - verified from data by gyro cross-correlation; the
  position is [imu] mount_xyz) because the kart's own /tf_static lacks
  imu_link,
- optionally a camera driver, when [camera] driver is ``usb_cam`` or ``zed``
  and the camera source is ros2 (driver_fps / driver_width / driver_height
  apply exactly here).

The cfg is validated BEFORE anything starts; a bad file aborts the launch
with every problem listed. The Fast DDS large-SHM profile ships with the
package and is exported for the node (9 MB frames fall back to lossy
fragmented UDP without it).
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

#: Launch arguments that flip [sensors] switches ("" = keep the cfg value).
_SWITCH_ARGS = [k for k in SENSOR_KEYS]


def _overrides_from_context(context) -> dict[str, int]:
    overrides = {}
    for key in _SWITCH_ARGS:
        value = LaunchConfiguration(key).perform(context).strip()
        if value != "":
            overrides[key] = int(value)
    return overrides


def _camera_driver_nodes(cfg) -> list:
    """Driver nodes for ros2-source cameras whose [camera] driver != none."""
    nodes = []
    for section in ("camera", "camera_2"):
        switch = "camera" if section == "camera" else "camera_2"
        if not cfg["sensors"].get(switch):
            continue
        cam = cfg[section]
        if cam["driver"] == "none" or cam["source"] != "ros2":
            continue
        name = str(cam["name"]) or section
        if cam["driver"] == "usb_cam":
            nodes.append(
                Node(
                    package="usb_cam",
                    executable="usb_cam_node_exe",
                    name=f"{name}_driver",
                    parameters=[
                        {
                            "video_device": str(cam["device"]) if str(cam["device"]).startswith("/dev") else f"/dev/video{cam['device']}",
                            "image_width": int(cam["driver_width"]),
                            "image_height": int(cam["driver_height"]),
                            "framerate": float(cam["driver_fps"]),
                            "pixel_format": "yuyv",
                        }
                    ],
                    remappings=[
                        ("image_raw", cam["image_topic"]),
                        ("camera_info", cam["camera_info_topic"] or f"{cam['image_topic']}/camera_info"),
                    ],
                    output="screen",
                )
            )
        elif cam["driver"] == "zed":
            # Preset: include the vendor stack's own launch when it is
            # installed (zed_wrapper's zed_camera.launch.py; the driver then
            # publishes its usual topics - point the [camera] topics at
            # them). Nothing in the MAPPING code knows this preset exists:
            # the camera-agnostic rule holds, this is launch-plumbing only.
            try:
                zed_share = Path(get_package_share_directory("zed_wrapper"))
            except Exception:  # noqa: BLE001 - PackageNotFoundError and kin
                raise MasterConfigError(
                    f"[{section}] driver = zed, but the zed_wrapper package is not installed "
                    "on this machine. Install the ZED ROS 2 wrapper, or start the camera "
                    "driver yourself and set driver = none."
                ) from None
            from launch.actions import IncludeLaunchDescription
            from launch.launch_description_sources import PythonLaunchDescriptionSource

            nodes.append(
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(str(zed_share / "launch" / "zed_camera.launch.py")),
                    launch_arguments={
                        "camera_name": name,
                        "camera_model": "zedxgs",
                        "pub_frame_rate": str(cam["driver_fps"]),
                    }.items(),
                )
            )
    return nodes


def _setup(context, *args, **kwargs):
    cfg_path = LaunchConfiguration("cfg").perform(context)
    try:
        cfg = load_master_config(cfg_path, overrides=_overrides_from_context(context))
    except MasterConfigError as exc:
        raise RuntimeError("kart_master.cfg refused:\n" + "\n".join(exc.problems)) from None

    use_sim_time_arg = LaunchConfiguration("use_sim_time").perform(context).strip()
    use_sim_time = bool(int(use_sim_time_arg)) if use_sim_time_arg else bool(cfg["system"]["use_sim_time"])
    log_level = str(cfg["system"]["log_level"])

    set_args: list[str] = []
    for key, value in _overrides_from_context(context).items():
        set_args += ["--set", f"sensors.{key}={value}"]

    imu_xyz = cfg["imu"]["mount_xyz"]
    actions = [
        # base_link -> imu_link: identity rotation (gyro cross-correlation
        # verified the axes), position from [imu] mount_xyz.
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
            arguments=["--cfg", cfg_path, *set_args, "--ros-args", "--log-level", log_level],
            parameters=[{"use_sim_time": use_sim_time}],
        ),
    ]
    actions += _camera_driver_nodes(cfg)
    return actions


def generate_launch_description() -> LaunchDescription:
    share = Path(get_package_share_directory("kart_map2d"))
    profile = share / "fastdds_large_shm.xml"
    args = [
        DeclareLaunchArgument("cfg", description="absolute path to kart_master.cfg"),
        DeclareLaunchArgument("use_sim_time", default_value="", description="override [system] use_sim_time (0/1)"),
    ]
    args += [
        DeclareLaunchArgument(key, default_value="", description=f"override [sensors] {key} (0/1)")
        for key in _SWITCH_ARGS
    ]
    actions: list = [
        # 9 MB camera frames need large shared-memory segments on every
        # process of the image path; harmless when the transport is not SHM.
        SetEnvironmentVariable(
            "FASTRTPS_DEFAULT_PROFILES_FILE",
            os.environ.get("FASTRTPS_DEFAULT_PROFILES_FILE", str(profile)),
        ),
    ]
    return LaunchDescription(args + actions + [OpaqueFunction(function=_setup)])
