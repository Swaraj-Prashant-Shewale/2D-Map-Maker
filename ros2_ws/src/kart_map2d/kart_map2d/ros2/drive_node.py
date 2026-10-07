"""ROS 2 drive node: load a map package, localize on it, follow its racing line, stay safe.

A SEPARATE executable from the mapping node (:mod:`kart_map2d.ros2.node`):
while racing the kart runs only this. It loads a saved map package (any
method's - the flat grid and the racing line are what it uses) and never runs
segmentation networks, loop closure, cloud building or racing-line
optimisation - it loads their results.

Pipeline per control cycle (``control_rate_hz``, default 50 Hz, on a timer):
the fusion engine is advanced (IMU / visual / wheel odometry), the pose is
extrapolated to "now" with the buffered gyro, pure pursuit computes steering
and speed on the racing line, the safety supervisor limits the speed (e-stop,
deadman, watchdogs, obstacles on the path ahead, speed cap) and one
``ackermann_msgs/AckermannDriveStamped`` goes out on ``drive_topic``
(optionally a ``geometry_msgs/Twist`` too). The command's header stamp is the
newest sensor stamp the pose is based on, so ``receive time - stamp`` is the
sensor-to-command latency. Everything heavy (image decoding, visual
odometry, the ground cue, LiDAR / camera map matching, the global search)
runs on worker threads, one job each, busy -> drop and count: the control
timer never waits for a correction.

Sources are parameters, none is hard-wired: ``imu_topic`` (required: camera
+ IMU is the minimum), ``cameras`` (ANY number, each with its own image /
camera_info topics, raw ``sensor_msgs/Image`` in any 8-bit encoding or
``sensor_msgs/CompressedImage``, mount pose and calibrated height; no camera
is assumed), ``odom_topic`` (wheel or LiDAR odometry, optional),
``lidar_topic`` (optional: scan-to-map fixes + obstacle stops).

Safety inputs: ``estop_topic`` (std_msgs/Bool, latched QoS: True stops and
LATCHES; clear it by publishing False and calling ``~/engage``),
``heartbeat_topic`` + ``deadman_timeout_s`` (std_msgs/Bool True = alive),
``~/engage`` (std_srvs/SetBool: allow / forbid motion; the node starts
disengaged unless ``engage_on_start``), ``~/relocalize`` (std_srvs/Trigger:
restart the global search). Diagnostics: ``drive/diagnostics`` (JSON:
localization quality, safety state + reasons, cross-track error, timing),
``drive/pose`` (PoseStamped, map frame).

Run (a sourced ROS 2 environment, from the project or bundle root)::

    python3 -m kart_map2d.ros2.drive_node --params kart_map2d/ros2/drive_params.yaml

Exit codes: 0 after a clean stop, 2 for a refused configuration (e.g. the
map package does not load / is not drivable, no IMU topic).
"""

from __future__ import annotations

import bisect
import copy
import difflib
import json
import math
import sys
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field, fields
from functools import partial
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from kart_map2d.core.measurements import RelativeMotion
from kart_map2d.core.transforms import make_transform, quat_to_matrix, rpy_matrix
from kart_map2d.drive.controller import (
    KART_MAX_STEER_RAD,
    KART_REAR_AXLE_X_M,
    KART_WHEELBASE_M,
    ControllerConfig,
    PurePursuitController,
    RacingPath,
)
from kart_map2d.drive.localization import (
    CameraGroundCue,
    CameraObservation,
    DriveFusionEngine,
    LocalizerConfig,
    MapLocalizer,
    MapModel,
    ScanMatchConfig,
    extrapolate_pose,
    global_search_camera,
    global_search_lidar,
    lidar_obstacle_points,
    voxel_downsample,
)
from kart_map2d.drive.safety import ObstacleMemory, SafetyConfig, SafetyInputs, SafetySupervisor, corridor_arclength
from kart_map2d.estimation.ekf import BG, EkfConfig
from kart_map2d.mapping.map_package import MapPackageError, load_map_package
from kart_map2d.perception.visual_odometry import VisualOdometry
from kart_map2d.ros2 import converters

__all__ = [
    "DriveCameraParams",
    "DriveConfigError",
    "DriveNode",
    "DriveParams",
    "main",
    "unknown_param_keys",
]

#: A receive time below this is no epoch time (sim time before the first /clock).
_EPOCH_MIN_S = 1e8
#: REP-103: camera optical frame (z forward, x right, y down) -> x-forward body.
_OPTICAL_TO_LINK = rpy_matrix(-np.pi / 2, 0.0, -np.pi / 2)


# ------------------------------------------------------------------ params
@dataclass
class DriveCameraParams:
    """One camera. Nothing about the camera model is assumed: topics, mount,
    height and intrinsics come from here (or its camera_info topic)."""

    image_topic: str
    #: sensor_msgs/CameraInfo topic ("" -> ``intrinsics`` must be given).
    info_topic: str = ""
    #: True: image_topic carries sensor_msgs/CompressedImage (jpeg / png).
    compressed: bool = False
    #: base_link position of the optical centre [m].
    mount_xyz: tuple[float, float, float] = (0.0, 0.0, 0.5)
    #: Camera body rotation (x forward, y left, z up) relative to base_link
    #: [deg] (roll, pitch, yaw); pitch > 0 looks down. The optical-frame
    #: convention (REP-103) is applied on top.
    mount_rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    #: Calibrated camera height over the ground [m] (visual-odometry scale);
    #: None -> mount z minus the base_link ground height (0.10 m).
    height_m: float | None = None
    #: (fx, fy, cx, cy, width, height) at the image's native resolution, used
    #: when there is no info_topic.
    intrinsics: tuple = ()
    #: Resize incoming frames by this factor before VO and the ground cue.
    image_scale: float = 0.5
    vo: bool = True  # visual odometry from this camera
    cue: bool = True  # ground cue: camera map fixes + obstacle memory
    #: Optional image file (white = kart body) masking the kart itself.
    kart_mask_path: str = ""


@dataclass
class DriveParams:
    """Every tunable of the drive node; ``drive_params.yaml`` documents the same keys."""

    # --- the map
    map_package: str = ""
    verify_map_hashes: bool = True
    map_frame: str = "map"
    base_frame: str = "base_link"
    # --- odometry sources ("" disables)
    imu_topic: str = "/imu/data"
    imu_rate_hz: float = 100.0
    imu_rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)  # base_link <- IMU rotation
    imu_accel_in_g: bool = False
    odom_topic: str = ""
    odom_stride: int = 2
    lidar_topic: str = ""
    lidar_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)  # base_link <- LiDAR mount
    lidar_rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    cameras: dict[str, DriveCameraParams] = field(default_factory=dict)
    # --- fusion timing (offline-verified defaults)
    reorder_window_s: float = 0.12
    max_lateness_s: float = 0.25
    imu_hold_s: float = 0.3
    # --- localization
    #: [x_m, y_m, yaw_deg] in the map frame; [] -> global relocalization.
    start_pose: tuple = ()
    start_sigma_xy_m: float = 0.5
    start_sigma_yaw_deg: float = 5.0
    global_relocalization: bool = True
    #: Lost-kart fallback [s]: localization lost (stale / rejected) for this
    #: long with the kart at rest -> restart the global search by itself
    #: (needs a source that can search: a LiDAR or enabled camera fixes);
    #: 0 = only on ~/relocalize.
    auto_relocalize_after_s: float = 3.0
    lidar_match_rate_hz: float = 5.0
    camera_match_rate_hz: float = 1.0
    camera_cue_rate_hz: float = 4.0
    #: Camera patch fixes: auto (only against maps without LiDAR evidence) | on | off.
    camera_fixes: str = "auto"
    converged_sigma_xy_m: float = 0.5
    converged_sigma_yaw_deg: float = 5.0
    max_fix_age_s: float = 8.0
    # --- control
    control_rate_hz: float = 50.0
    drive_topic: str = "/drive"
    twist_topic: str = ""
    speed_scale: float = 1.0
    max_speed_mps: float = 2.0
    max_accel_mps2: float = 1.0
    max_decel_mps2: float = 2.0
    max_lat_accel_mps2: float = 2.5
    lookahead_base_m: float = 0.6
    lookahead_gain_s: float = 0.5
    lookahead_min_m: float = 1.0
    lookahead_max_m: float = 4.0
    wheelbase_m: float = KART_WHEELBASE_M
    rear_axle_x_m: float = KART_REAR_AXLE_X_M
    max_steer_rad: float = KART_MAX_STEER_RAD
    laps: int = 0
    stop_margin_m: float = 0.3
    latency_s: float = 0.0
    # --- safety
    estop_topic: str = "/estop"
    heartbeat_topic: str = ""
    deadman_timeout_s: float = 0.0
    engage_on_start: bool = False
    pose_timeout_s: float = 0.2
    degraded_mode_timeout_s: float = 0.5
    max_consecutive_rejects: int = 3
    max_position_sigma_m: float = 1.0
    stop_decel_mps2: float = 2.0
    estop_decel_mps2: float = 4.0
    resume_hold_s: float = 1.0
    obstacle_stop: bool = True
    kart_half_width_m: float = 0.53
    lateral_margin_m: float = 0.25
    front_overhang_m: float = 0.8
    obstacle_range_m: float = 8.0
    #: Stop this far short of an obstacle on the path [m] ([safety]
    #: stop_distance_m in kart_master.cfg; SafetyConfig.stop_margin_m).
    obstacle_stop_margin_m: float = 0.5
    # --- outputs
    diagnostics_period_s: float = 0.2
    stats_path: str = ""
    #: Debug: record every fusion-engine input (arrival, measurement) and
    #: arrival-clock advance, pickled here at shutdown (replayable offline).
    trace_path: str = ""

    @classmethod
    def from_dict(cls, data: dict | None) -> "DriveParams":
        """From a YAML dict (optionally nested under ``drive``)."""
        data = dict(data or {})
        if "drive" in data and isinstance(data["drive"], dict):
            data = dict(data["drive"])
        cameras = data.pop("cameras", None)
        known = {f.name for f in fields(cls)}
        values = {k: v for k, v in data.items() if k in known}
        for key in ("imu_rpy_deg", "lidar_xyz", "lidar_rpy_deg", "start_pose"):
            if key in values and values[key] is not None:
                values[key] = tuple(float(v) for v in values[key])
        params = cls(**values)
        if cameras:
            cams = {}
            ck = {f.name for f in fields(DriveCameraParams)}
            for name, cfg in cameras.items():
                cfg = {k: v for k, v in dict(cfg).items() if k in ck}
                for key in ("mount_xyz", "mount_rpy_deg", "intrinsics"):
                    if key in cfg and cfg[key] is not None:
                        cfg[key] = tuple(float(v) for v in cfg[key])
                cams[str(name)] = DriveCameraParams(**cfg)
            params.cameras = cams
        return params


def unknown_param_keys(data: dict | None) -> list[str]:
    """Keys of a params dict DriveParams / DriveCameraParams does not know (with hints)."""
    data = dict(data or {})
    if "drive" in data and isinstance(data["drive"], dict):
        data = dict(data["drive"])
    out = []
    known = [f.name for f in fields(DriveParams)]
    for key in data:
        if key not in known:
            close = difflib.get_close_matches(str(key), known, n=1, cutoff=0.6)
            out.append(f"{key!r} (did you mean {close[0]!r}?)" if close else repr(key))
    cam_known = [f.name for f in fields(DriveCameraParams)]
    for name, cfg in dict(data.get("cameras") or {}).items():
        for key in dict(cfg or {}):
            if key not in cam_known:
                close = difflib.get_close_matches(str(key), cam_known, n=1, cutoff=0.6)
                out.append(f"cameras.{name}.{key}" + (f" (did you mean {close[0]!r}?)" if close else ""))
    return out


class DriveConfigError(ValueError):
    """A configuration the drive node refuses; ``problems`` says what to change."""

    def __init__(self, problems: list[str] | str) -> None:
        self.problems = [problems] if isinstance(problems, str) else list(problems)
        super().__init__("; ".join(self.problems))


def config_problems(p: DriveParams) -> list[str]:
    """Everything wrong with ``p`` before anything is started (each with its fix)."""
    problems = []
    if not p.map_package:
        problems.append("map_package is empty: set it to a map package folder (the one with map_meta.json)")
    if not p.imu_topic:
        problems.append("imu_topic is empty: camera + IMU is the minimum configuration (e.g. /imu/data)")
    if not (p.cameras or p.odom_topic or p.lidar_topic):
        problems.append("no odometry source besides the IMU: configure at least one camera (or odom_topic / lidar_topic)")
    if p.start_pose and len(p.start_pose) != 3:
        problems.append("start_pose must be [x_m, y_m, yaw_deg] (or [] for global relocalization)")
    if not p.start_pose and not p.global_relocalization:
        problems.append("no start_pose and global_relocalization is off: the node could never localize")
    if p.control_rate_hz <= 0:
        problems.append("control_rate_hz must be > 0")
    if str(p.camera_fixes).lower() not in ("auto", "on", "off"):
        problems.append(f"camera_fixes must be auto | on | off, got {p.camera_fixes!r}")
    if p.max_speed_mps < 0:
        problems.append("max_speed_mps must be >= 0 (0 = never move)")
    if p.auto_relocalize_after_s < 0:
        problems.append("auto_relocalize_after_s must be >= 0 (0 = only on ~/relocalize)")
    for name, cam in p.cameras.items():
        if not cam.image_topic:
            problems.append(f"cameras.{name}: image_topic is empty")
        if not cam.info_topic and len(cam.intrinsics) != 6:
            problems.append(f"cameras.{name}: give info_topic or intrinsics [fx, fy, cx, cy, width, height]")
    return problems


def start_pose_problems(p: DriveParams, localizer: MapLocalizer) -> list[str]:
    """A start pose the LiDAR map matcher could not verify: its 3-sigma region
    is wider than the tracking match window (+-3 m / +-20 deg). The true pose
    may then lie outside every window the matcher searches and it locks onto
    an alias - measured on the sept3 plaza (loop-closed map): a start 4 m /
    30 deg off with sigma 5 m / 40 deg was declared converged 4.4 m off, one
    3 m / 180 deg off 6 m and 179 deg off; within the window (sigma <= 1 m /
    6.5 deg, errors up to 2.9 m / 19 deg) it converged 0.04-0.06 m off.
    (Camera patch fixes never converged from a wrong start at all, so no
    camera bound is imposed.)"""
    if not p.start_pose or not p.lidar_topic:
        return []
    c = localizer.scan_matcher.config
    k, max_xy, max_yaw = c.window_sigmas, c.max_window_xy_m, math.degrees(c.max_window_yaw_rad)
    if k * p.start_sigma_xy_m <= max_xy + 1e-9 and k * p.start_sigma_yaw_deg <= max_yaw + 1e-9:
        return []
    return [
        f"start_sigma_xy_m {p.start_sigma_xy_m} / start_sigma_yaw_deg {p.start_sigma_yaw_deg}: the LiDAR scan matcher searches at most "
        f"+-{max_xy:.1f} m / +-{max_yaw:.0f} deg ({k:.0f} sigma) around the start pose, so it cannot verify a start this uncertain "
        f"(it may lock onto a wrong match). Use start_sigma_xy_m <= {max_xy / k:.2f} and start_sigma_yaw_deg <= {max_yaw / k:.1f} "
        f"(place the kart more carefully), or start_pose: [] for global relocalization"
    ]


def apply_ros_parameters(node: Any, params: DriveParams, descriptor: Any = None) -> DriveParams:
    """Overlay ``--ros-args -p key:=value`` onto every scalar DriveParams field."""
    declare = getattr(node, "declare_parameter", None)
    if declare is None:
        return params
    defaults = DriveParams()
    out = copy.copy(params)
    problems = []
    for f in fields(DriveParams):
        name, current = f.name, getattr(params, f.name)
        expected = type(getattr(defaults, name))
        if expected not in (bool, int, float, str):
            continue
        value = declare(name, current, descriptor).value if descriptor is not None else declare(name, current).value
        if value is None or (value == current and type(value) is type(current)):
            continue
        if expected is float and isinstance(value, (int, float)) and not isinstance(value, bool):
            setattr(out, name, float(value))
        elif expected is int and isinstance(value, int) and not isinstance(value, bool):
            setattr(out, name, int(value))
        elif isinstance(value, expected):
            setattr(out, name, value)
        else:
            problems.append(f"ROS parameter {name}: expected {expected.__name__}, got {value!r}")
    if problems:
        raise DriveConfigError(problems)
    return out


# ------------------------------------------------------------------ QoS
@dataclass(frozen=True)
class QosSpec:
    reliability: str
    depth: int
    durability: str = "volatile"

    def to_profile(self):
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

        return QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=int(self.depth),
            reliability=ReliabilityPolicy.RELIABLE if self.reliability == "reliable" else ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL if self.durability == "transient_local" else DurabilityPolicy.VOLATILE,
        )


QOS = {
    "imu": QosSpec("reliable", 2000),  # deep: the IMU callback drains its backlog
    "odom": QosSpec("reliable", 50),
    "lidar": QosSpec("reliable", 5),
    "image": QosSpec("best_effort", 1),
    "camera_info": QosSpec("best_effort", 1),
    # Latched e-stop: a late-starting drive node still sees a pressed stop.
    "estop": QosSpec("reliable", 1, "transient_local"),
    "heartbeat": QosSpec("reliable", 5),
    "drive": QosSpec("reliable", 5),
    "diagnostics": QosSpec("reliable", 5),
    "pose": QosSpec("reliable", 5),
}


# ------------------------------------------------------------------ helpers
class _LatencyStats:
    """Per-key timing [ms]: count, mean, p50, p95, max over a bounded window."""

    def __init__(self, window: int = 20000) -> None:
        self._lock = threading.Lock()
        self._data: dict[str, deque] = {}
        self.window = int(window)

    def add(self, key: str, seconds: float) -> None:
        with self._lock:
            buf = self._data.get(key)
            if buf is None:
                buf = self._data[key] = deque(maxlen=self.window)
            buf.append(1e3 * seconds)

    @contextmanager
    def time(self, key: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.add(key, time.perf_counter() - t0)

    def summary(self) -> dict[str, dict[str, float]]:
        with self._lock:
            items = {k: np.asarray(v, dtype=np.float64) for k, v in self._data.items()}
        out = {}
        for key, a in sorted(items.items()):
            if not len(a):
                continue
            out[key] = {
                "n": int(len(a)),
                "mean_ms": round(float(a.mean()), 4),
                "p50_ms": round(float(np.percentile(a, 50)), 4),
                "p95_ms": round(float(np.percentile(a, 95)), 4),
                "max_ms": round(float(a.max()), 4),
            }
        return out


class _GyroBuffer:
    """Recent full-rate gyro z samples: VO yaw prior + control-rate pose extrapolation."""

    def __init__(self, horizon_s: float = 30.0) -> None:
        self.horizon_s = float(horizon_s)
        self._t: deque[float] = deque()
        self._z: deque[float] = deque()
        self._cum: deque[float] = deque()
        self._lock = threading.Lock()

    def push(self, stamp: float, gz: float) -> None:
        with self._lock:
            if self._t and stamp <= self._t[-1]:
                return
            if self._t:
                self._cum.append(self._cum[-1] + 0.5 * (self._z[-1] + gz) * (stamp - self._t[-1]))
            else:
                self._cum.append(0.0)
            self._t.append(stamp)
            self._z.append(gz)
            while self._t and stamp - self._t[0] > self.horizon_s:
                self._t.popleft()
                self._z.popleft()
                self._cum.popleft()

    def prior(self, t0: float, t1: float) -> float:
        """Integrated yaw change [rad] between two stamps (VO yaw prior)."""
        with self._lock:
            if len(self._t) < 2:
                return 0.0
            t, c = np.asarray(self._t), np.asarray(self._cum)
        a, b = np.interp([t0, t1], t, c)
        return float(b - a)

    def since(self, t0: float) -> tuple[list[float], list[float]]:
        """(stamps, gyro z) lists of the samples after ``t0``: walks back from
        the newest sample only (the control loop asks for the last ~0.1-0.2 s
        at 50 Hz; plain lists, as :func:`extrapolate_pose` loops over them)."""
        ts: list[float] = []
        zs: list[float] = []
        with self._lock:
            for t, z in zip(reversed(self._t), reversed(self._z)):
                if t <= t0:
                    break
                ts.append(t)
                zs.append(z)
        ts.reverse()
        zs.reverse()
        return ts, zs

    @property
    def newest(self) -> float | None:
        with self._lock:
            return self._t[-1] if self._t else None


class _LiveVO(VisualOdometry):
    """VisualOdometry fed from memory (the live twin of ``VisualOdometry.process``)."""

    def process_array(self, stamp: float, image_bgr: NDArray[np.uint8]) -> RelativeMotion | None:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
        gray = self._clahe.apply(gray)
        if self._prev is None:
            self._prev = (stamp, gray)
            return None
        t0, prev = self._prev
        dt = stamp - t0
        if 0.0 <= dt < self.config.min_pair_dt_s:
            self._log(t0, stamp, "merged")
            return None
        self._prev = (stamp, gray)
        if not 0.0 < dt <= self.config.max_gap_s:
            self._log(t0, stamp, "gap")
            return None
        return self._solve_pair(prev, gray, t0, stamp, dt)


class _Worker:
    """One background thread, one job in flight: ``submit`` returns False while busy."""

    def __init__(self, name: str, stats: _LatencyStats, warn) -> None:
        self.stats = stats
        self._warn = warn
        self._cv = threading.Condition()
        self._job: tuple[str, Any] | None = None
        self._busy = False
        self._stop = False
        self.errors: list[str] = []
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    @property
    def busy(self) -> bool:
        return self._busy

    def submit(self, key: str, fn) -> bool:
        with self._cv:
            if self._busy or self._stop:
                return False
            self._busy = True
            self._job = (key, fn)
            self._cv.notify()
        return True

    def _run(self) -> None:
        while True:
            with self._cv:
                while self._job is None and not self._stop:
                    self._cv.wait()
                if self._stop:
                    return
                key, fn = self._job
                self._job = None
            try:
                with self.stats.time(key):
                    fn()
            except Exception as exc:  # noqa: BLE001 - a bad frame / scan never kills the node
                if len(self.errors) < 20:
                    self.errors.append(f"{key}: {type(exc).__name__}: {exc}")
                self._warn(f"{key} failed: {type(exc).__name__}: {exc}")
            finally:
                with self._cv:
                    self._busy = False
                    self._cv.notify_all()

    def wait_idle(self, timeout: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout
        with self._cv:
            while self._busy:
                left = deadline - time.monotonic()
                if left <= 0:
                    return False
                self._cv.wait(left)
        return True

    def stop(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        self._thread.join(timeout=5.0)


class _RosMessages:
    """Message classes, imported lazily (the module imports without ROS)."""

    def __init__(self) -> None:
        from ackermann_msgs.msg import AckermannDriveStamped
        from geometry_msgs.msg import PoseStamped, Twist
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import CameraInfo, CompressedImage, Image, Imu, PointCloud2
        from std_msgs.msg import Bool, String

        self.AckermannDriveStamped = AckermannDriveStamped
        self.PoseStamped = PoseStamped
        self.Twist = Twist
        self.Odometry = Odometry
        self.CameraInfo = CameraInfo
        self.CompressedImage = CompressedImage
        self.Image = Image
        self.Imu = Imu
        self.PointCloud2 = PointCloud2
        self.Bool = Bool
        self.String = String
        try:
            from std_srvs.srv import SetBool, Trigger
        except ImportError:
            SetBool = Trigger = None
        self.SetBool = SetBool
        self.Trigger = Trigger


def camera_mount(cam: DriveCameraParams) -> NDArray[np.float64]:
    """base_link <- camera OPTICAL frame from the mount parameters."""
    R_body = rpy_matrix(*np.radians(np.asarray(cam.mount_rpy_deg, dtype=np.float64)))
    return make_transform(R_body @ _OPTICAL_TO_LINK, np.asarray(cam.mount_xyz, dtype=np.float64))


def decode_image(msg, compressed: bool, scale: float) -> tuple[float, NDArray[np.uint8]]:
    """(stamp, BGR uint8) from sensor_msgs/Image (any 8-bit encoding) or CompressedImage."""
    if not compressed:
        return converters.image_to_bgr(msg, scale)
    data = np.frombuffer(bytes(msg.data) if not isinstance(msg.data, (bytes, bytearray)) else msg.data, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot decode a {getattr(msg, 'format', '?')} CompressedImage")
    if scale != 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return converters.stamp_of(msg), img


# ------------------------------------------------------------------ the node
class DriveNode:
    """Wire a (duck-typed) rclpy node to the drive stack (composition, testable without ROS).

    Raises :class:`DriveConfigError` for a configuration it refuses (nothing
    is subscribed or published then).
    """

    def __init__(self, node: Any, params: DriveParams | None = None, service_node: Any = None) -> None:
        self.node = node
        self.service_node = service_node
        self.params = p = params or DriveParams()
        problems = config_problems(p)
        if problems:
            raise DriveConfigError(problems)
        try:
            self.package = load_map_package(p.map_package, require_drivable=True, verify=p.verify_map_hashes)
        except (MapPackageError, OSError) as exc:
            raise DriveConfigError(f"map_package {p.map_package}: {exc}") from None
        self.msgs = _RosMessages()
        self.stats = _LatencyStats()
        t0 = time.perf_counter()
        self.model = MapModel.from_package(self.package)
        self.map_model_s = time.perf_counter() - t0
        self.engine = DriveFusionEngine(
            ekf_config=EkfConfig(max_lateness_s=p.max_lateness_s), reorder_window_s=p.reorder_window_s, output_rate_hz=50.0
        )
        # base_link's sideways slide in turns (the kart's geometry, see MapAwareEkf)
        self.engine.ekf.base_ahead_m = -float(p.rear_axle_x_m)
        start = None
        if p.start_pose:
            start = (float(p.start_pose[0]), float(p.start_pose[1]), math.radians(float(p.start_pose[2])))
        self.localizer = MapLocalizer(
            self.model,
            LocalizerConfig(
                start_pose=start,
                start_sigma_xy_m=p.start_sigma_xy_m,
                start_sigma_yaw_rad=math.radians(p.start_sigma_yaw_deg),
                global_relocalization=p.global_relocalization,
                lidar_rate_hz=p.lidar_match_rate_hz,
                camera_rate_hz=p.camera_match_rate_hz,
                converged_sigma_xy_m=p.converged_sigma_xy_m,
                converged_sigma_yaw_rad=math.radians(p.converged_sigma_yaw_deg),
                max_fix_age_s=p.max_fix_age_s,
                rejected_after=p.max_consecutive_rejects,
                camera_fixes=p.camera_fixes,
            ),
            engine=self.engine,
        )
        problems = start_pose_problems(p, self.localizer)
        if problems:
            raise DriveConfigError(problems)
        self.path = RacingPath.from_racing_line(self.package.racing_line)
        self.controller = PurePursuitController(
            self.path,
            ControllerConfig(
                wheelbase_m=p.wheelbase_m,
                rear_axle_x_m=p.rear_axle_x_m,
                max_steer_rad=p.max_steer_rad,
                lookahead_base_m=p.lookahead_base_m,
                lookahead_gain_s=p.lookahead_gain_s,
                lookahead_min_m=p.lookahead_min_m,
                lookahead_max_m=p.lookahead_max_m,
                speed_scale=p.speed_scale,
                max_speed_mps=p.max_speed_mps,
                max_lat_accel_mps2=p.max_lat_accel_mps2,
                max_accel_mps2=p.max_accel_mps2,
                max_decel_mps2=p.max_decel_mps2,
                stop_margin_m=p.stop_margin_m,
                laps=p.laps,
                latency_s=p.latency_s,
            ),
        )
        self.safety = SafetySupervisor(
            SafetyConfig(
                max_speed_mps=p.max_speed_mps,
                pose_timeout_s=p.pose_timeout_s,
                degraded_mode_timeout_s=p.degraded_mode_timeout_s,
                max_consecutive_rejects=p.max_consecutive_rejects,
                max_position_sigma_m=p.max_position_sigma_m,
                deadman_timeout_s=p.deadman_timeout_s,
                stop_decel_mps2=p.stop_decel_mps2,
                estop_decel_mps2=p.estop_decel_mps2,
                resume_hold_s=p.resume_hold_s,
                obstacle_stop=p.obstacle_stop,
                kart_half_width_m=p.kart_half_width_m,
                lateral_margin_m=p.lateral_margin_m,
                front_overhang_m=p.front_overhang_m,
                stop_margin_m=p.obstacle_stop_margin_m,
            )
        )
        self.obstacles = ObstacleMemory()
        self._scan_cfg = ScanMatchConfig()
        self._engine_lock = threading.Lock()
        self._counter_lock = threading.Lock()
        self._match_refusals: deque = deque(maxlen=40)
        self.counters = {
            "imu": 0,
            "odom": 0,
            "frames": 0,
            "vo_solved": 0,
            "frames_dropped_busy": 0,
            "cue_frames": 0,
            "scans": 0,
            "scans_dropped_busy": 0,
            "lidar_matches": 0,
            "camera_matches": 0,
            "camera_match_refused": 0,
            "fixes_accepted": 0,
            "fixes_rejected": 0,
            "global_attempts": 0,
            "global_restarts": 0,
            "commands": 0,
            "estop_msgs": 0,
            "heartbeats": 0,
            "imu_clock_not_ready": 0,
        }
        self._clock_sync = converters.StreamingClockSync()
        self._decimator = converters.ImuDecimator(p.imu_rate_hz)
        self._R_base_imu = rpy_matrix(*np.radians(np.asarray(p.imu_rpy_deg, dtype=np.float64)))
        self._imu_identity = bool(np.allclose(p.imu_rpy_deg, 0.0))
        self._T_base_lidar = make_transform(rpy_matrix(*np.radians(np.asarray(p.lidar_rpy_deg, dtype=np.float64))), np.asarray(p.lidar_xyz, dtype=np.float64))
        self._odom = converters.OdometryConverter(stride=p.odom_stride)
        self._gyro = _GyroBuffer()
        self._imu_newest: tuple[float, float] | None = None  # (newest IMU stamp, node time received)
        self._imu_checked_at: float | None = None
        self._newest_sensor_stamp: float | None = None
        self._subs: dict[str, Any] = {}
        self._intrinsics: dict[str, tuple[NDArray[np.float64], tuple[int, int]]] = {}
        self._vo: dict[str, _LiveVO] = {}
        self._cues: dict[str, CameraGroundCue] = {}
        self._cam_workers: dict[str, _Worker] = {}
        self._next_cue: dict[str, float] = {}
        self._lidar_worker = _Worker("drive-lidar", self.stats, self._warn)
        self._global_worker = _Worker("drive-global", self.stats, self._warn)
        self._last_cmd_time: float | None = None
        self._last_speed = 0.0
        self._last_steer = 0.0
        self.last_output = None
        self.last_decision = None
        self.last_status = None
        self._e2e: deque[float] = deque(maxlen=20000)
        self._cycle_cpu: deque[float] = deque(maxlen=20000)
        self._cycle_wall: deque[float] = deque(maxlen=20000)
        self._fusion_cpu: deque[float] = deque(maxlen=20000)
        self._fusion_wall: deque[float] = deque(maxlen=20000)
        self._pose_tick = 0
        self._cycle_period: deque[float] = deque(maxlen=20000)
        self._warned: set[str] = set()
        self._was_localized = False
        self._lost_since: float | None = None  # lost-kart fallback timer
        #: Progress reset requested by another thread (global search worker,
        #: ~/relocalize service): only the control thread touches the
        #: controller - a reset landing between step()'s None check and its
        #: read of the progress would raise inside the control timer.
        self._reset_request = False
        self._closed = False
        self.engage_service = self.relocalize_service = None
        for name in p.cameras:
            self.engine.register_camera(name)
        # Map-match priors beyond the filter's stamp integrate the measured gyro.
        self.engine.gyro_since = self._gyro.since
        self._trace: list | None = None
        if p.trace_path:
            self._trace = []
            push, advance = self.engine.push, self.engine.advance

            def traced_push(m, arrival=None):
                self._trace.append(("push", arrival, m))
                return push(m, arrival)

            def traced_advance(arrival):
                self._trace.append(("advance", arrival, None))
                return advance(arrival)

            self.engine.push = traced_push
            self.engine.advance = traced_advance
        self._setup_interfaces()
        if p.engage_on_start:
            self.safety.engage(True, self._now())

    # ------------------------------------------------------------ wiring
    def _setup_interfaces(self) -> None:
        node, p, msgs = self.node, self.params, self.msgs
        self._subs["imu"] = node.create_subscription(msgs.Imu, p.imu_topic, self._on_imu, QOS["imu"].to_profile())
        if p.odom_topic:
            node.create_subscription(msgs.Odometry, p.odom_topic, self._on_odom, QOS["odom"].to_profile())
        if p.lidar_topic:
            node.create_subscription(msgs.PointCloud2, p.lidar_topic, self._on_pointcloud, QOS["lidar"].to_profile())
        for name, cam in p.cameras.items():
            mtype = msgs.CompressedImage if cam.compressed else msgs.Image
            node.create_subscription(mtype, cam.image_topic, partial(self._on_image, name), QOS["image"].to_profile())
            if cam.info_topic:
                node.create_subscription(msgs.CameraInfo, cam.info_topic, partial(self._on_camera_info, name), QOS["camera_info"].to_profile())
            else:
                fx, fy, cx, cy, w, h = cam.intrinsics
                self._set_intrinsics(name, np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]), (int(w), int(h)))
            self._cam_workers[name] = _Worker(f"drive-cam-{name}", self.stats, self._warn)
        if p.estop_topic:
            node.create_subscription(msgs.Bool, p.estop_topic, self._on_estop, QOS["estop"].to_profile())
        if p.heartbeat_topic:
            node.create_subscription(msgs.Bool, p.heartbeat_topic, self._on_heartbeat, QOS["heartbeat"].to_profile())
        self.pub_drive = node.create_publisher(msgs.AckermannDriveStamped, p.drive_topic, QOS["drive"].to_profile())
        self.pub_twist = node.create_publisher(msgs.Twist, p.twist_topic, QOS["drive"].to_profile()) if p.twist_topic else None
        self.pub_diag = node.create_publisher(msgs.String, "drive/diagnostics", QOS["diagnostics"].to_profile())
        self.pub_pose = node.create_publisher(msgs.PoseStamped, "drive/pose", QOS["pose"].to_profile())
        node.create_timer(1.0 / p.control_rate_hz, self._on_control)
        node.create_timer(p.diagnostics_period_s, self._on_diagnostics)
        host = self.service_node if self.service_node is not None else node
        if msgs.SetBool is not None and hasattr(host, "create_service"):
            prefix = "~/" if host is node else getattr(node, "get_fully_qualified_name", lambda: "/drive")().rstrip("/") + "/"
            self.engage_service = host.create_service(msgs.SetBool, prefix + "engage", self._on_engage)
            self.relocalize_service = host.create_service(msgs.Trigger, prefix + "relocalize", self._on_relocalize)

    def set_speed_limits(self, max_speed_mps: float | None = None, speed_scale: float | None = None) -> None:
        """Apply live speed-limit changes (``ros2 param set /drive max_speed_mps ...``).

        Touches only plain floats read by the control cycle (the controller's
        speed cap and the safety supervisor's limit), so it is safe from the
        parameter callback thread; the next control tick uses the new values.
        The safety supervisor still has the last word every cycle.
        """
        if max_speed_mps is not None:
            value = max(float(max_speed_mps), 0.0)
            self.params.max_speed_mps = value
            self.controller.config.max_speed_mps = value
            self.safety.config.max_speed_mps = value
        if speed_scale is not None:
            value = float(np.clip(speed_scale, 0.0, 1.0))
            self.params.speed_scale = value
            self.controller.config.speed_scale = value

    def describe(self) -> list[str]:
        p, pkg = self.params, self.package
        cams = ", ".join(f"{n} ({c.image_topic}{', compressed' if c.compressed else ''})" for n, c in p.cameras.items()) or "none"
        line = pkg.racing_line
        return [
            f"map package: {pkg.path} (method {pkg.method}, grid {self.model.shape[0]}x{self.model.shape[1]} @ {self.model.cell} m, "
            f"{len(self.model.obstacle_xy)} obstacle cells; model built in {self.map_model_s:.2f} s)",
            f"racing line: {len(line.path)} points, {self.path.length:.1f} m, {'closed' if line.closed else 'open'}; laps={p.laps}",
            f"sources: imu={p.imu_topic}, odom={p.odom_topic or 'off'}, lidar={p.lidar_topic or 'off'}, cameras={cams}",
            f"localization: start_pose={list(p.start_pose) or 'none (global relocalization)'}, fixes: "
            f"{'LiDAR scan-to-map ' + str(p.lidar_match_rate_hz) + ' Hz' if p.lidar_topic else 'no LiDAR'}; "
            f"{'camera patches ' + str(p.camera_match_rate_hz) + ' Hz' if (any(c.cue for c in p.cameras.values()) and self.localizer.camera_fixes_enabled) else 'no camera fixes (camera_fixes=' + p.camera_fixes + ('; the map holds LiDAR evidence' if self.model.has_lidar_evidence else '') + ')'}",
            f"map sensors: {', '.join(self.model.sensors) or 'unknown'}",
            f"control: {p.control_rate_hz} Hz -> {p.drive_topic} (AckermannDriveStamped){' + ' + p.twist_topic + ' (Twist)' if p.twist_topic else ''}; "
            f"max_speed {p.max_speed_mps} m/s x speed_scale {p.speed_scale}",
            f"safety: e-stop {p.estop_topic or 'NONE'} (latched), deadman {p.heartbeat_topic or 'off'} "
            f"({p.deadman_timeout_s} s), pose timeout {p.pose_timeout_s} s, obstacle stop {'on' if p.obstacle_stop else 'OFF'}; "
            f"{'ENGAGED at start' if p.engage_on_start else 'disengaged until ~/engage true'}",
        ]

    # ------------------------------------------------------------ basics
    def _now(self) -> float:
        return float(self.node.get_clock().now().nanoseconds) * 1e-9

    def _warn(self, text: str) -> None:
        logger = getattr(self.node, "get_logger", None)
        if logger is not None:
            logger().warning(text)

    def _warn_once(self, key: str, text: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            self._warn(text)

    def _bump(self, key: str, n: int = 1) -> None:
        with self._counter_lock:
            self.counters[key] += n

    def _saw_sensor(self, stamp: float) -> None:
        # Never ahead of the node clock: one sensor stamping in the future
        # (its own clock, not synced) would otherwise hold the newest stamp
        # there and blind the pose_stale watchdog for every other sensor.
        stamp = min(float(stamp), self._now())
        if self._newest_sensor_stamp is None or stamp > self._newest_sensor_stamp:
            self._newest_sensor_stamp = stamp

    def _arrival(self, now: float) -> float:
        """The engine's arrival clock never drains past a received IMU sample
        (see the mapping node's ``_arrival``): capped at the newest IMU stamp
        + reorder window while the IMU delivered within ``imu_hold_s``."""
        if self._imu_newest is None:
            return now
        stamp, seen = self._imu_newest
        ref = now if self._imu_checked_at is None else min(now, self._imu_checked_at)
        if ref - seen > self.params.imu_hold_s:
            return now
        return min(now, max(stamp, ref - self.params.imu_hold_s) + self.params.reorder_window_s)

    # ------------------------------------------------------------ IMU
    _BACKLOG_MAX = 1024

    def _take_queued(self, handler, limit: int) -> int:
        sub = self._subs.get("imu")
        handle = getattr(sub, "handle", None)
        take = getattr(handle, "take_message", None)
        if take is None:
            return 0
        count = 0
        while count < limit:
            try:
                with handle:
                    info = take(sub.msg_type, sub.raw)
            except Exception:  # noqa: BLE001 - shutdown / an rclpy without this API
                self._subs.pop("imu", None)
                return count
            if info is None:
                return count
            handler(info[0], False)
            count += 1
        return count

    def _on_imu(self, msg) -> None:
        t0 = time.perf_counter()
        self._handle_imu(msg, True)
        n = 1 + self._take_queued(self._handle_imu, self._BACKLOG_MAX)
        self._bump("imu", n)
        self.stats.add("cb_imu", time.perf_counter() - t0)

    def _handle_imu(self, msg, fresh: bool = True) -> None:
        now = self._now()
        if now < _EPOCH_MIN_S:
            self._bump("imu_clock_not_ready")
            return
        sample = converters.imu_to_sample(
            msg,
            now,
            channel="imu",
            clock=self._clock_sync,
            accel_in_g=self.params.imu_accel_in_g,
            R_base_imu=None if self._imu_identity else self._R_base_imu,
            observe=fresh,
        )
        newest = self._imu_newest
        self._imu_newest = (sample.stamp if newest is None else max(sample.stamp, newest[0]), now)
        self._gyro.push(sample.stamp, float(sample.gyro[2]))
        self._saw_sensor(sample.stamp)
        out = self._decimator.push(sample)
        if out is not None:
            with self._engine_lock:
                self.localizer.push(out, min(out.stamp, self._arrival(now)))

    # ------------------------------------------------------------ odometry
    def _on_odom(self, msg) -> None:
        t0 = time.perf_counter()
        now = self._now()
        motion = self._odom.push(msg)
        self._bump("odom")
        self._saw_sensor(converters.stamp_of(msg))
        if motion is not None:
            with self._engine_lock:
                self.localizer.push(motion, self._arrival(now))
        self.stats.add("cb_odom", time.perf_counter() - t0)

    # ------------------------------------------------------------ LiDAR
    def _on_pointcloud(self, msg) -> None:
        t0 = time.perf_counter()
        self._bump("scans")
        scan = converters.pointcloud_to_scan(msg)
        if not self._lidar_worker.submit("lidar_step", partial(self._lidar_step, scan)):
            self._bump("scans_dropped_busy")
        self.stats.add("cb_lidar", time.perf_counter() - t0)

    def _lidar_step(self, scan) -> None:
        """Worker: obstacle points -> obstacle memory; map fix (or global search)."""
        R, t = self._T_base_lidar[:3, :3], self._T_base_lidar[:3, 3]
        pts = scan.points.astype(np.float64) @ R.T + t
        xy = lidar_obstacle_points(pts, self._scan_cfg)
        with self._engine_lock:
            initialized = self.localizer.initialized and self.engine.ekf.stamp is not None
            pose, cov = self.localizer.prior_at(scan.stamp) if initialized else (None, None)
            due = self.localizer.lidar_due(scan.stamp)
            # A scan stamped in the future (unsynced clock) is placed at the pose
            # NOW, not extrapolated ahead (that would push its obstacles away).
            now_node = self._now()
            pose_obs = self.engine.pose_at(now_node) if (initialized and scan.stamp > now_node) else pose
        if initialized:
            near = xy[np.hypot(xy[:, 0], xy[:, 1]) <= self.params.obstacle_range_m]
            if len(near):
                near = voxel_downsample(near, 0.1, 400)
                c, s = math.cos(pose_obs[2]), math.sin(pose_obs[2])
                pm = near @ np.array([[c, -s], [s, c]]).T + pose_obs[:2]
                self.obstacles.add(pm, self._obstacle_clock(scan.stamp, "lidar"), "lidar", along=self._corridor_along(pm, "lidar"))
        if not due:
            return
        P = voxel_downsample(xy, self._scan_cfg.voxel_m, self._scan_cfg.max_points)
        if not initialized:
            if self.localizer._pending_start is None and self.params.global_relocalization and not self._global_worker.busy:
                self._global_worker.submit("global_lidar", partial(self._global_lidar, P, scan.stamp))
            return
        with self.stats.time("lidar_match"):
            match = self.localizer.scan_matcher.match(P, pose, cov, stamp=scan.stamp)
        self._bump("lidar_matches")
        if match is None:
            with self._engine_lock:
                self.localizer._next_lidar = scan.stamp + 1.0 / max(self.params.lidar_match_rate_hz, 1e-6)
            return
        with self._engine_lock:
            out = self.localizer.submit(match)
        self._count_outcome(out)

    def _obstacle_clock(self, stamp: float, source: str) -> float:
        """The time an observation's obstacles are aged from: the node clock
        when they were stored, NOT the sensor stamp. ObstacleMemory keeps LiDAR
        points 0.3 s; aged by a sensor stamp that lags the node clock by more
        than that (unsynced sensor clock, or a scan that queued behind a slow
        match) every obstacle was already expired on arrival and the obstacle
        stop was silently off (measured: a wall 3 m ahead never stopped the
        kart with the LiDAR clock 0.5 s behind). Clock disagreement is
        reported once, as the mapping node does."""
        now = self._now()
        if now >= _EPOCH_MIN_S and abs(now - float(stamp)) > 1.0:
            self._warn_once(
                f"obstacle_stamp:{source}",
                f"{source}: message stamps are {abs(now - float(stamp)):.1f} s {'behind' if now > stamp else 'ahead of'} the node clock "
                "(obstacles are aged on the node clock; sync the sensor clocks - chrony / PTP - or map fixes are dropped as late)",
            )
        return now

    def _corridor_along(self, points_map: NDArray[np.float64], source: str) -> NDArray[np.float64] | None:
        """Unwrapped racing-line arclength of the obstacle points inside the
        kart's corridor (None while the controller has no progress).

        Runs once per observation on the sensor thread, so the 50 Hz control
        cycle only compares arclengths (:meth:`ObstacleMemory.ahead`). The
        window starts 0.5 m behind the current progress and reaches as far
        as the kart can get while the observation is remembered, plus the
        obstacle check's reach at the speed cap.
        """
        s0 = self.controller.progress
        if s0 is None:
            return None
        t0 = time.perf_counter()
        c = self.safety.config
        vmax = max(c.max_speed_mps, self._last_speed, 0.0)
        length = self.safety.obstacle_reach(vmax) + vmax * self.obstacles.max_age_s.get(source, 0.5) + 0.5
        back = 0.5
        poly = self.controller.path.points_between(s0 - back, s0 + length, 0.25)
        # The corridor is centred on the racing LINE; a kart that is off it
        # (up to max_start_offset_m after a start / relocalization, or a
        # kidnap) sweeps the band between its own position and the line on its
        # way back, so the corridor widens on the kart's side by the
        # cross-track error: an obstacle at the kart's own edge must count.
        out = self.last_output
        xt = float(out.cross_track) if (out is not None and out.ok and math.isfinite(out.cross_track)) else 0.0
        along = corridor_arclength(
            poly, points_map, c.kart_half_width_m + c.lateral_margin_m, extra_left=max(xt, 0.0), extra_right=max(-xt, 0.0)
        ) + (s0 - back)
        self.stats.add("obstacle_project", time.perf_counter() - t0)
        return along

    def _global_lidar(self, P, stamp) -> None:
        self._bump("global_attempts")
        result = global_search_lidar(self.model, self.localizer.scan_matcher, P, self.localizer.global_config, stamp)
        with self._engine_lock:
            ok = self.localizer.apply_global(result, stamp, "lidar")
        if ok:
            self._reset_request = True
        else:
            self._warn_once(f"global:{result.reason[:20]}", f"global relocalization (LiDAR): {result.reason}")

    def _count_outcome(self, out) -> None:
        """Fix outcomes are booked by the localizer (possibly later, when the
        filter reaches a queued fix); the diagnostics read them from there."""

    # ------------------------------------------------------------ cameras
    def _set_intrinsics(self, name: str, K_full: NDArray[np.float64], size_full: tuple[int, int]) -> None:
        scale = self.params.cameras[name].image_scale
        K = np.asarray(K_full, dtype=np.float64).copy()
        K[:2] *= scale
        self._intrinsics[name] = (K, (int(round(size_full[0] * scale)), int(round(size_full[1] * scale))))

    def _on_camera_info(self, name: str, msg) -> None:
        if name in self._intrinsics:
            return
        K, (w, h) = converters.camera_info_to_intrinsics(msg)
        self._set_intrinsics(name, K, (w, h))

    def _on_image(self, name: str, msg) -> None:
        t0 = time.perf_counter()
        self._bump("frames")
        if name in self._intrinsics:
            if not self._cam_workers[name].submit(f"camera_{name}", partial(self._camera_step, name, msg)):
                self._bump("frames_dropped_busy")
        self.stats.add("cb_image", time.perf_counter() - t0)

    def _ensure_camera(self, name: str) -> None:
        if name in self._vo or name in self._cues:
            return
        cam = self.params.cameras[name]
        K, size = self._intrinsics[name]
        T = camera_mount(cam)
        ground_z = float(T[2, 3]) - float(cam.height_m) if cam.height_m is not None else -0.10
        if cam.vo:
            self._vo[name] = _LiveVO(name, K, size, T, ground_z=ground_z, yaw_prior=self._gyro.prior)
        if cam.cue:
            mask = None
            if cam.kart_mask_path:
                img = cv2.imread(cam.kart_mask_path, cv2.IMREAD_GRAYSCALE)
                mask = None if img is None else cv2.resize(img, size, interpolation=cv2.INTER_NEAREST) > 127
            # The cue projects with the SAME geometry the maps are built with
            # (nominal mount, ground at -0.10 m), so live evidence lines up
            # with a map's camera evidence.
            self._cues[name] = CameraGroundCue(K, T, size, kart_mask=mask)
            self._next_cue[name] = -np.inf

    def _camera_step(self, name: str, msg) -> None:
        """Worker: decode, VO -> EKF, ground cue -> observation + obstacles, camera fix."""
        cam = self.params.cameras[name]
        with self.stats.time("image_decode"):
            stamp, bgr = decode_image(msg, cam.compressed, cam.image_scale)
        self._ensure_camera(name)
        vo = self._vo.get(name)
        if vo is not None:
            with self.stats.time("vo_solve"):
                motion = vo.process_array(stamp, bgr)
            if motion is not None:
                self._bump("vo_solved")
                self._saw_sensor(motion.stamp)
                with self._engine_lock:
                    self.localizer.push(motion, self._arrival(self._now()))
        cue = self._cues.get(name)
        if cue is None or stamp < self._next_cue.get(name, -np.inf):
            return
        self._next_cue[name] = stamp + 1.0 / max(self.params.camera_cue_rate_hz, 1e-6)
        with self._engine_lock:
            moving = self.engine.ekf.stamp is not None and float(self.engine.ekf.x[3]) > 0.3
            initialized = self.localizer.initialized and self.engine.ekf.stamp is not None
            pose = self.engine.pose_at(min(stamp, self._now())) if initialized else None  # never extrapolated ahead (see _lidar_step)
        with self.stats.time("camera_cue"):
            drv, obs, quality = cue.observe(bgr, moving=moving)
        self._bump("cue_frames")
        with self._engine_lock:
            self.localizer.add_camera_observation(CameraObservation(stamp, drv, obs))
        if pose is not None and len(obs):
            near = obs[np.hypot(obs[:, 0], obs[:, 1]) <= self.params.obstacle_range_m]
            if len(near):
                c, s = math.cos(pose[2]), math.sin(pose[2])
                pm = near @ np.array([[c, -s], [s, c]]).T + pose[:2]
                self.obstacles.add(pm, self._obstacle_clock(stamp, "camera"), "camera", along=self._corridor_along(pm, "camera"))
        if not self.localizer.camera_fixes_enabled or not self.localizer.camera_due(stamp):
            return
        if not initialized:
            if self.localizer._pending_start is None and self.params.global_relocalization and not self._global_worker.busy:
                with self._engine_lock:
                    live, _ = self.localizer.camera_patch(stamp)
                if live is not None:
                    self._global_worker.submit("global_camera", partial(self._global_camera, live, stamp))
            return
        with self._engine_lock:
            live, key_pose = self.localizer.camera_patch(stamp)
            _, cov = self.localizer.prior_at(stamp)
            self.localizer._next_camera = stamp + 1.0 / max(self.params.camera_match_rate_hz, 1e-6)
        if live is None:
            return
        with self.stats.time("camera_match"):
            info: dict = {}
            match = self.localizer.patch_matcher.match(live, key_pose, cov, info_out=info)
        self._bump("camera_matches")
        if match is not None:
            with self._engine_lock:
                out = self.localizer.submit(match)
            self._count_outcome(out)
        else:
            # Surface WHY the correlation refused (overlap, obstacle mass,
            # score, ambiguity): without this the camera-fix path fails
            # silently and "no fixes" is undebuggable.
            self._bump("camera_match_refused")
            self._match_refusals.append(
                {"stamp": round(stamp, 2), "reason": str(info.get("reason", "?")), "live_cells": info.get("live_cells")}
            )

    def _global_camera(self, live, stamp) -> None:
        self._bump("global_attempts")
        result = global_search_camera(self.model, self.localizer.patch_matcher, live, self.localizer.global_config)
        with self._engine_lock:
            ok = self.localizer.apply_global(result, stamp, "camera")
        if ok:
            self._reset_request = True
        else:
            self._warn_once(f"globalcam:{result.reason[:20]}", f"global relocalization (camera): {result.reason}")

    # ------------------------------------------------------------ safety inputs
    def _on_estop(self, msg) -> None:
        self._bump("estop_msgs")
        self.safety.on_estop(bool(msg.data), self._now())

    def _on_heartbeat(self, msg) -> None:
        self._bump("heartbeats")
        if bool(getattr(msg, "data", True)):
            self.safety.on_heartbeat(self._now())
        else:
            self.safety.last_heartbeat = None  # the operator let go

    def _on_engage(self, request, response):
        ok, message = self.safety.engage(bool(request.data), self._now())
        response.success = bool(ok)
        response.message = message
        return response

    def _on_relocalize(self, request, response):
        self._restart_global("~/relocalize")
        response.success = True
        response.message = "global relocalization restarted; the kart holds until it converges"
        return response

    def _restart_global(self, reason: str) -> None:
        with self._engine_lock:
            self.localizer.restart_global(self._now(), reason)
        self._reset_request = True
        self.obstacles.clear()  # their corridor arclengths refer to the old progress
        self._lost_since = None
        self._bump("global_restarts")
        self._warn(f"global relocalization restarted: {reason}")

    def _check_lost(self, now: float, status, speed_est: float | None) -> None:
        """Lost-kart fallback: localization lost (stale / rejected) for
        ``auto_relocalize_after_s`` while the kart is at rest -> global search.

        A kart that lost the map stops (safety: not_localized /
        localization_rejected) and, standing still, gathers no new evidence
        for a re-anchor; a unique global match is the way back. Only with a
        source that can search (LiDAR, or camera fixes enabled for this map).
        """
        p = self.params
        can_search = p.global_relocalization and (bool(p.lidar_topic) or self.localizer.camera_fixes_enabled)
        at_rest = self._last_speed <= 1e-6 and speed_est is not None and abs(speed_est) < 0.1
        lost = status.initialized and status.state in ("stale", "rejected")
        if p.auto_relocalize_after_s <= 0.0 or not can_search or not (lost and at_rest):
            self._lost_since = None
            return
        if self._lost_since is None:
            self._lost_since = now
        elif now - self._lost_since >= p.auto_relocalize_after_s:
            self._restart_global(f"localization {status.state} for {now - self._lost_since:.1f} s with the kart at rest")

    # ------------------------------------------------------------ control
    def _on_control(self) -> None:
        """One control cycle: fusion service, then the timed control compute.

        Two parts, timed separately: (1) the FUSION SERVICE - IMU messages
        still queued in DDS are taken and the engine's arrival clock advanced
        (estimation work that belongs to the sensor callbacks; it runs here
        only so a starved executor cannot let the EKF drain past queued IMU
        samples); (2) the CONTROL COMPUTE - status, pose extrapolation, pure
        pursuit, safety, publishing: ``control_cycle_cpu_us`` in the
        diagnostics is this part's thread CPU time.
        """
        fcpu0 = time.thread_time()
        ft0 = time.perf_counter()
        now = self._now()
        if self._last_cmd_time is not None:
            self._cycle_period.append(now - self._last_cmd_time)
        dt = 0.0 if self._last_cmd_time is None else max(now - self._last_cmd_time, 0.0)
        self._last_cmd_time = now
        # Queued IMU first: the drain below must not pass them.
        n = self._take_queued(self._handle_imu, self._BACKLOG_MAX)
        if n:
            self._bump("imu", n)
        self._imu_checked_at = now
        with self._engine_lock:
            self.engine.advance(self._arrival(now))
            if len(self.engine.ekf.updates) > 4096:
                self.localizer.resolve()  # book pending fixes before their log entries are folded
                self.engine.fold_update_log()
                self.engine.trim_history()
        self._fusion_cpu.append(time.thread_time() - fcpu0)
        self._fusion_wall.append(time.perf_counter() - ft0)
        cpu0 = time.thread_time()
        t0 = time.perf_counter()
        with self._engine_lock:
            status = self.localizer.status(now)
            ekf = self.engine.ekf
            have = ekf.stamp is not None
            x = ekf.x.copy() if have else None
            ekf_stamp = float(ekf.stamp) if have else None
            mode = self.engine.mode()
        self.last_status = status
        if self._reset_request:
            self._reset_request = False
            self.controller.reset()
        if status.localized and not self._was_localized:
            self.controller.reset()  # progress restarts from the converged pose
            self.obstacles.clear()  # their corridor arclengths refer to the old progress
        self._was_localized = status.localized
        self._check_lost(now, status, float(x[3]) if x is not None else None)
        out = None
        pose = None
        steer_hold = None
        t1 = time.perf_counter()
        if have and status.initialized:
            gt, gz = self._gyro.since(ekf_stamp)
            pose = extrapolate_pose(x, ekf_stamp, now, gt, gz, float(x[BG]), self.engine.ekf.base_ahead_m)
            t2 = time.perf_counter()
            self.stats.add("ctl_extrapolate", t2 - t1)
            if status.localized:
                out = self.controller.step(pose, max(float(x[3]), 0.0), dt, yaw_rate=float(x[4]))
                self.stats.add("ctl_pure_pursuit", time.perf_counter() - t2)
            elif self._last_speed > 0.0 and self.controller.progress is not None:
                # Localization lost while rolling: the command is 0 m/s (not
                # localized), but snapping the wheels straight would carry the
                # kart out of a bend for the whole braking distance - keep
                # steering along the line on the dead-reckoned pose.
                hold = self.controller.step(pose, max(float(x[3]), 0.0), dt, yaw_rate=float(x[4]))
                steer_hold = float(hold.steer) if hold.ok else None
        t3 = time.perf_counter()
        # Obstacles were placed on the racing line when they arrived: here
        # only their arclengths are compared with the current progress.
        progress = self.controller.progress
        ahead = None
        if out is not None and out.ok and progress is not None:
            ahead = self.obstacles.ahead(now, progress, self.safety.obstacle_reach(self._last_speed))
        decision = self.safety.update(
            SafetyInputs(
                now=now,
                sensor_stamp=self._newest_sensor_stamp,
                fusion_mode=mode,
                localized=bool(status.localized),
                on_path=bool(out is None or out.ok),
                consecutive_rejects=status.consecutive_rejects,
                position_sigma_m=status.sigma_xy_m if status.initialized else math.inf,
                speed=self._last_speed,
                obstacle_ahead=ahead,
            )
        )
        t4 = time.perf_counter()
        self.stats.add("ctl_safety", t4 - t3)
        speed = decision.limit(out.speed if out is not None and out.ok else 0.0)
        steer = float(out.steer) if (out is not None and out.ok) else (steer_hold if steer_hold is not None else 0.0)
        accel = (speed - self._last_speed) / dt if dt > 0 else 0.0
        self.controller.commanded(speed, steer)
        self._last_speed, self._last_steer = speed, steer
        stamp = self._newest_sensor_stamp if self._newest_sensor_stamp is not None else now
        self._publish_command(stamp, steer, speed, accel)
        self._bump("commands")
        # The pose topic is for people (RViz / logs): 10 Hz is plenty, and
        # each rclpy publish costs ~0.5 ms of the control thread.
        self._pose_tick = (self._pose_tick + 1) % max(int(round(self.params.control_rate_hz / 10.0)), 1)
        if pose is not None and self._pose_tick == 0:
            self._publish_pose(now, pose)
        self.stats.add("ctl_publish", time.perf_counter() - t4)
        self.last_output, self.last_decision = out, decision
        if self._newest_sensor_stamp is not None:
            self._e2e.append(self._now() - self._newest_sensor_stamp)
        self._cycle_wall.append(time.perf_counter() - t0)
        self._cycle_cpu.append(time.thread_time() - cpu0)

    def _publish_command(self, stamp: float, steer: float, speed: float, accel: float) -> None:
        msg = self.msgs.AckermannDriveStamped()
        converters.set_stamp(msg.header.stamp, stamp)
        msg.header.frame_id = self.params.base_frame
        msg.drive.steering_angle = float(steer)
        msg.drive.steering_angle_velocity = 0.0
        msg.drive.speed = float(speed)
        msg.drive.acceleration = float(accel)
        msg.drive.jerk = 0.0
        self.pub_drive.publish(msg)
        if self.pub_twist is not None:
            tw = self.msgs.Twist()
            tw.linear.x = float(speed)
            tw.angular.z = float(speed * math.tan(steer) / self.params.wheelbase_m)
            self.pub_twist.publish(tw)

    def _publish_pose(self, now: float, pose: NDArray[np.float64]) -> None:
        msg = self.msgs.PoseStamped()
        converters.set_stamp(msg.header.stamp, now)
        msg.header.frame_id = self.params.map_frame
        msg.pose.position.x = float(pose[0])
        msg.pose.position.y = float(pose[1])
        msg.pose.position.z = 0.0
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = float(math.sin(pose[2] / 2))
        msg.pose.orientation.w = float(math.cos(pose[2] / 2))
        self.pub_pose.publish(msg)

    # ------------------------------------------------------------ diagnostics
    @staticmethod
    def _pct(values, scale: float = 1.0, last: int | None = None) -> dict | None:
        if not values:
            return None
        if last is not None:
            values = list(values)[-last:]
        a = np.asarray(values, dtype=np.float64) * scale
        return {"n": int(len(a)), "p50": round(float(np.percentile(a, 50)), 3), "p95": round(float(np.percentile(a, 95)), 3), "max": round(float(a.max()), 3)}

    def diagnostics(self, window: int | None = None) -> dict:
        """The diagnostics document. ``window``: timing percentiles over only
        the last ``window`` cycles (the periodic topic uses 500 so it stays
        cheap on the executor thread; the stats file uses everything)."""
        now = self._now()
        status = self.last_status
        out, decision = self.last_output, self.last_decision
        with self._counter_lock:
            counters = dict(self.counters)
        with self._engine_lock:
            counters["fixes_accepted"] = self.localizer.fixes_accepted
            counters["fixes_rejected"] = self.localizer.fixes_rejected
            counters["ekf_late_drops"] = int(self.engine.ekf.late_drops)
            counters["engine_queue"] = len(self.engine._queue)
            counters["gyro_blames"] = len(self.engine.yaw_monitor.faults)
            counters["ekf_updates"] = {**{k: list(v) for k, v in self.engine.update_counts.items()}}
            health = self.engine.health.snapshot()
            mode = self.engine.mode()
            events = list(self.localizer.events[-5:])
            glob = list(self.localizer.global_attempts[-3:])
        return {
            "stamp": round(now, 3),
            "localization": status.as_dict() if status is not None else None,
            "global_attempts": glob,
            "localizer_events": [{"stamp": round(t, 3), "kind": k, "text": s} for t, k, s in events],
            "safety": {
                "state": decision.state if decision else "init",
                "reasons": decision.reasons if decision else [],
                "engaged": self.safety.engaged,
                "estop_latched": self.safety.estop_latched,
                "estop_input": self.safety.estop_input,
                "obstacle_distance_m": None if decision is None or decision.obstacle_distance_m is None else round(decision.obstacle_distance_m, 2),
                "speed_cap_mps": None if decision is None else round(decision.speed_cap, 3),
            },
            "control": None
            if out is None
            else {
                "ok": out.ok,
                "reason": out.reason,
                "s_m": round(out.s, 2) if out.ok else None,
                "lap": out.lap,
                "cross_track_m": round(out.cross_track, 3) if out.ok else None,
                "heading_error_deg": round(math.degrees(out.heading_error), 2) if out.ok else None,
                "v_ref_mps": round(out.v_ref, 3),
                "finished": out.finished,
            },
            "command": {"speed_mps": round(self._last_speed, 3), "steer_rad": round(self._last_steer, 4)},
            "fusion": {"mode": mode, "health": health},
            "counters": counters,
            "timing": {
                "control_cycle_cpu_us": self._pct(self._cycle_cpu, 1e6, window),
                "control_cycle_wall_us": self._pct(self._cycle_wall, 1e6, window),
                "fusion_service_cpu_us": self._pct(self._fusion_cpu, 1e6, window),
                "fusion_service_wall_us": self._pct(self._fusion_wall, 1e6, window),
                "control_period_ms": self._pct(self._cycle_period, 1e3, window),
                "sensor_to_command_ms": self._pct(self._e2e, 1e3, window),
                "stages_ms": self.stats.summary() if window is None else None,
            },
        }

    def _on_diagnostics(self) -> None:
        msg = self.msgs.String()
        msg.data = json.dumps(self.diagnostics(window=500))
        self.pub_diag.publish(msg)

    def write_stats(self, path: str | Path) -> Path:
        path = Path(path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            # Windows bind mounts can raise FileExistsError from mkdir even
            # with exist_ok when a signal interrupts the stat; the directory
            # exists, which is all that matters here.
            pass
        doc = self.diagnostics()
        doc["safety_events"] = [{"t": round(t, 3), "trigger": n, "kind": k} for t, n, k in self.safety.events[-500:]]
        with self._engine_lock:
            doc["camera_match_refusals"] = list(self._match_refusals)
            doc["localizer"] = {
                "events": [{"stamp": round(t, 3), "kind": k, "text": s} for t, k, s in self.localizer.events],
                "global_attempts": list(self.localizer.global_attempts),
                "fixes": [
                    {"stamp": round(o.stamp, 3), "source": o.source, "accepted": o.accepted, "score": round(float(o.score), 3), "nis": None if o.nis is None else round(o.nis, 2)}
                    for o in list(self.localizer.outcomes)[-2000:]
                ],
            }
        path.write_text(json.dumps(doc, indent=1), encoding="ascii")
        return path

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._trace is not None and self.params.trace_path:
            import pickle

            with self._engine_lock:
                items = list(self._trace)
            with open(self.params.trace_path, "wb") as f:
                pickle.dump({"start_pose": self.params.start_pose, "items": items}, f)
        for w in [self._lidar_worker, self._global_worker, *self._cam_workers.values()]:
            w.stop()


# ------------------------------------------------------------------ main
def _spin(executor) -> None:
    try:
        executor.spin()
    except Exception:  # noqa: BLE001 - shutdown
        pass


def main(argv: list[str] | None = None) -> int:
    """Entry point: ``python3 -m kart_map2d.ros2.drive_node --params drive_params.yaml``
    or ``--cfg kart_master.cfg`` (the master control file; ``--set
    sensors.lidar=1`` flips a [sensors] switch without editing the file)."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--params", default=None, help="YAML parameter file (see config/drive_params.yaml)")
    parser.add_argument("--cfg", default=None, help="kart_master.cfg (overrides --params)")
    parser.add_argument("--set", action="append", default=[], metavar="sensors.NAME=0|1", help="override a [sensors] switch")
    parser.add_argument(
        "--start-pose",
        default=None,
        metavar="X,Y,YAW_DEG",
        help="the kart's KNOWN start pose in the map frame (the marked grid "
        "spot): 'x,y,yaw_deg', or 'global' for global relocalization. "
        "Overrides --params; kart_master.cfg carries no start pose on "
        "purpose (it changes every session).",
    )
    parser.add_argument(
        "--start-sigma",
        default=None,
        metavar="XY_M,YAW_DEG",
        help="1-sigma uncertainty of --start-pose (default 0.1,1.0: a MARKED "
        "grid spot is a surveyed spot, which arms trusted-start driving; "
        "the kart then drives on odometry until the first map fix must "
        "confirm the pose). Give a looser pair for an eyeballed start.",
    )
    parser.add_argument("--stats-path", default=None, help="write the node's stats JSON here on shutdown")
    args, ros_args = parser.parse_known_args(argv)
    raw = None
    params = DriveParams()
    if args.params:
        import yaml

        with open(args.params, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        params = DriveParams.from_dict(raw)
    cfg_warnings: list[str] = []
    if args.cfg:
        from kart_map2d.config.master_config import MasterConfigError, load_master_config, to_drive_params_dict

        overrides = {}
        for token in args.set:
            key, _, value = token.partition("=")
            key = key.strip().lower().removeprefix("sensors.")
            overrides[key] = int(value.strip() or "1")
        try:
            cfg = load_master_config(args.cfg, overrides=overrides)
            params = DriveParams.from_dict(to_drive_params_dict(cfg))
            cfg_warnings = list(cfg.warnings)
        except MasterConfigError as exc:
            print("configuration refused - the kart will not be driven. Fix:", flush=True)
            for problem in exc.problems:
                print(f"  - {problem}", flush=True)
            return 2
    if args.start_pose is not None:
        text = args.start_pose.strip().lower()
        if text in ("", "global"):
            params.start_pose = ()
        else:
            try:
                x, y, yaw_deg = (float(v) for v in text.replace(",", " ").split())
            except ValueError:
                print(f"--start-pose {args.start_pose!r} is not 'x,y,yaw_deg' or 'global'", flush=True)
                return 2
            params.start_pose = (x, y, yaw_deg)
            # A marked grid spot is a surveyed spot: tight enough to arm
            # trusted-start driving (<= half the converged thresholds), so
            # the kart may move and let map fixes confirm the pose instead
            # of waiting parked for fixes it can only get while moving.
            params.start_sigma_xy_m = 0.1
            params.start_sigma_yaw_deg = 1.0
    if args.start_sigma is not None:
        try:
            sxy, syaw = (float(v) for v in args.start_sigma.replace(",", " ").split())
        except ValueError:
            print(f"--start-sigma {args.start_sigma!r} is not 'xy_m,yaw_deg'", flush=True)
            return 2
        params.start_sigma_xy_m = sxy
        params.start_sigma_yaw_deg = syaw
    if args.stats_path:
        params.stats_path = args.stats_path

    import rclpy
    from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor

    rclpy.init(args=ros_args)
    node = rclpy.create_node("drive")
    log = node.get_logger()
    services = None
    wrapper: DriveNode | None = None
    code = 0
    try:
        for key in unknown_param_keys(raw):
            log.warning(f"{args.params}: unknown parameter {key} is ignored")
        try:
            from rcl_interfaces.msg import ParameterDescriptor

            params = apply_ros_parameters(node, params, ParameterDescriptor(dynamic_typing=True))
            services = rclpy.create_node("drive_services")
            wrapper = DriveNode(node, params, service_node=services)
        except DriveConfigError as exc:
            log.error("configuration refused - the kart will not be driven. Fix:")
            for problem in exc.problems:
                log.error(f"  - {problem}")
            code = 2
            return code
        for line in cfg_warnings:
            log.warning(line)
        for line in wrapper.describe():
            log.info(line)
        # Live-safe knobs stay adjustable while driving:
        #   ros2 param set /drive max_speed_mps 1.0
        #   ros2 param set /drive speed_scale 0.5
        # (kart_master.cfg is read once; these two are the ONLY keys that
        # follow ros2 param set at runtime - everything else needs a restart.)
        try:
            from rcl_interfaces.msg import SetParametersResult

            def _on_params(parameters):
                for p_ in parameters:
                    if p_.name == "max_speed_mps" and isinstance(p_.value, (int, float)):
                        wrapper.set_speed_limits(max_speed_mps=float(p_.value))
                        log.info(f"max_speed_mps -> {float(p_.value)} (live)")
                    elif p_.name == "speed_scale" and isinstance(p_.value, (int, float)):
                        wrapper.set_speed_limits(speed_scale=float(p_.value))
                        log.info(f"speed_scale -> {float(p_.value)} (live)")
                return SetParametersResult(successful=True)

            node.add_on_set_parameters_callback(_on_params)
        except ImportError:
            log.warning("rcl_interfaces unavailable: max_speed_mps / speed_scale are not live-settable")
        executor = SingleThreadedExecutor()
        executor.add_node(services)
        threading.Thread(target=_spin, args=(executor,), name="drive-services", daemon=True).start()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:
        if rclpy.ok():
            raise
    finally:
        if wrapper is not None:
            if wrapper.params.stats_path:
                try:
                    wrapper.write_stats(wrapper.params.stats_path)
                    print(f"[drive] stats written to {wrapper.params.stats_path}", flush=True)
                except Exception as exc:  # noqa: BLE001
                    print(f"[drive] stats not written: {exc}", flush=True)
            wrapper.close()
        for n in (services, node):
            if n is not None:
                try:
                    n.destroy_node()
                except Exception:  # noqa: BLE001
                    pass
        if rclpy.ok():
            rclpy.shutdown()
    return code


if __name__ == "__main__":
    sys.exit(main())
