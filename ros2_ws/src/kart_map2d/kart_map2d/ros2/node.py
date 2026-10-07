"""Live ROS 2 mapping node: subscribe -> convert -> the same FusionEngine and grid.

The 2D (flat) system only. This is a *thin* wrapper: every measurement goes
through :mod:`kart_map2d.ros2.converters` and the sensor adapters
(:mod:`kart_map2d.sensors.adapters`) into the identical
:class:`FusionEngine` / :class:`EvidenceGrid` code that the offline replay
verified, so what was measured offline is what runs on the kart.

The module imports cleanly without ROS: ``rclpy`` and the ``*_msgs``
packages are imported only inside :meth:`KartMapNode.__init__` /
:func:`main`, and the node wraps an ``rclpy`` Node by composition (any object
with ``create_subscription`` / ``create_publisher`` / ``create_timer`` /
``get_clock`` works, which is how the unit tests drive it without ROS).

Inputs (every one optional, gated by kart_master.cfg's [sensors] switches;
the DEFAULT is one camera + the dedicated IMU):

* ``imu_topic`` - the DEDICATED IMU (/imu/data, 400 Hz, frame imu_link,
  device-clock stamps corrected online by StreamingClockSync). The node
  NEVER subscribes to a camera's built-in IMU; /livox/imu exists only
  behind the ``livox_imu_topic`` parameter ([sensors] livox_imu, default 0).
* cameras - 1..N, each fully parameter-described and camera-agnostic
  (requirement: nothing may assume a ZED). Sources per camera: ``ros2``
  (Image or CompressedImage topics), ``device`` (cv2.VideoCapture: V4L2
  index/path or a video file, intrinsics from a calibration YAML) or
  ``gstreamer`` (a pipeline string; needs a GStreamer-enabled OpenCV, see
  kart_map2d/camera/capture.py). All sources feed the same worker path.
* ``lidar_topic`` - 2D grid evidence only (an off-by-default experiment
  switch; there is NO 3D twin in this package).
* ``odom_topic`` - nav_msgs/Odometry increments.
* magnetometer / speedometer / wheel ticks - requirement 13 adapters onto
  the fusion channels ``mag`` / ``speedometer`` / ``wheel``.

``~/save_map`` (std_srvs/Trigger) writes a map package
(:mod:`kart_map2d.mapping.map_package`) to ``output_dir``: loop closure,
corridor and racing line run at save time, never per frame; with
``save_on_shutdown`` the same happens when the node stops.

Latency rules (unchanged from the proven research node): the executor thread
only converts small messages and feeds the EKF - everything heavy runs on
worker threads, each with at most ONE job in flight, and input arriving
while a job runs is *dropped* (counted), never queued: image decoding + VO
(one worker per camera) and segmentation + grid insertion + corridor
extraction (the mapping worker). Why: rclpy runs every callback of this
node on one thread, and a 120-800 ms mapping step there delayed the IMU /
odometry callbacks past the EKF's 0.12 s reorder window, so measurements
were applied out of order or dropped as late. Every callback and stage is
timed (``latency_ms`` in the health JSON: mean / p95 / max).

Segmentation backend (``seg_mode``): ``classical`` (the DEFAULT for this
package) is the zero-network, torch-free path; ``neural`` is SegFormer;
``auto`` picks neural only when torch, transformers and the checkpoint are
all on disk (probed without importing torch). torch is imported only when
neural is selected. Each camera's kart-body mask is loaded from
``mask_file`` or auto-learned from temporally static pixels while driving
(the classical-fix result), with the geometric mask as the fallback.

Run on the kart (with the ROS environment sourced)::

    ros2 launch kart_map2d mapping.launch.py cfg:=/path/to/kart_master.cfg
    # or colcon-less:
    python3 -m kart_map2d.ros2.node --cfg kart_master.cfg
"""

from __future__ import annotations

import bisect
import copy
import datetime
import difflib
import json
import os
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

from kart_map2d.camera.model import CameraModel, load_calibration_yaml
from kart_map2d.core.measurements import LidarScan, RelativeMotion
from kart_map2d.core.transforms import make_transform, rpy_matrix
from kart_map2d.estimation.ekf import EkfConfig
from kart_map2d.estimation.fusion import FusionEngine, channel_role
from kart_map2d.mapping import map_package
from kart_map2d.mapping.corridor import CLASS_DRIVABLE, CLASS_OBSTACLE, extract_corridor
from kart_map2d.mapping.evidence_grid import DYNAMIC_CLASS_IDS, EvidenceGrid
from kart_map2d.perception.segmentation import SEG_MODES, get_segmenter, resolve_seg_mode
from kart_map2d.perception.visual_odometry import VisualOdometry, VoConfig
from kart_map2d.ros2 import converters
from kart_map2d.sensors.adapters import MagnetometerAdapter, SpeedometerAdapter, WheelTickAdapter

__all__ = [
    "NodeParams",
    "CameraParams",
    "NodeConfigError",
    "QosSpec",
    "QOS",
    "KartMapNode",
    "apply_ros_parameters",
    "camera_config_problems",
    "main",
    "unknown_param_keys",
]


# ---------------------------------------------------------------------- QoS
@dataclass(frozen=True)
class QosSpec:
    """Declarative QoS matching the recorded bag's offered profiles."""

    reliability: str  # "reliable" | "best_effort"
    depth: int
    durability: str = "volatile"  # "volatile" | "transient_local"

    def to_profile(self):
        """Build the rclpy QoSProfile (imported lazily; ROS-only path)."""
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

        return QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=int(self.depth),
            reliability=(
                ReliabilityPolicy.RELIABLE
                if self.reliability == "reliable"
                else ReliabilityPolicy.BEST_EFFORT
            ),
            durability=(
                DurabilityPolicy.TRANSIENT_LOCAL
                if self.durability == "transient_local"
                else DurabilityPolicy.VOLATILE
            ),
        )


#: Subscription QoS per stream, copied from the September bags'
#: ``offered_qos_profiles`` (rosbag2 codes: reliability 1 = reliable,
#: 2 = best effort; durability 2 = volatile). Exceptions: the /imu/data and
#: /odom *depths*. The publishers offer 5 and 1, but the executor thread can
#: still be held up (the worker threads share the interpreter lock): with
#: depth 5, when the mapping step still ran on the executor, the node kept
#: only 25 % of /imu/data during a sept3 bag replay (camera + IMU ATE 6.8 m);
#: with 2000 (5 s) nothing is lost and the next IMU callback drains the
#: backlog in one go. /odom (20 Hz, and the lidar3d twin's z / roll / pitch
#: source) keeps 2.5 s for the same reason. Depth is a local history size,
#: not part of QoS compatibility, so these still match the publishers.
QOS = {
    "imu": QosSpec("reliable", 2000),  # /imu/data
    "imu_livox": QosSpec("reliable", 256),  # /livox/imu
    "lidar": QosSpec("reliable", 256),  # /livox/lidar
    "odom": QosSpec("reliable", 50),  # /odom
    "image": QosSpec("best_effort", 1),  # camera image topic
    "camera_info": QosSpec("best_effort", 1),  # camera info topic
    # Requirement-13 inputs. Depths follow the /odom reasoning: enough local
    # history that a starved executor loses nothing (mag can run at 100 Hz).
    "mag": QosSpec("reliable", 200),
    "speed": QosSpec("reliable", 100),
    "wheel": QosSpec("reliable", 200),
}

#: Publisher QoS: latched-style map/corridor for late-joining viewers, plain
#: reliable for the streaming outputs.
PUB_QOS = {
    "pose": QosSpec("reliable", 10),
    "path": QosSpec("reliable", 1),
    "grid": QosSpec("reliable", 1, "transient_local"),
    "corridor": QosSpec("reliable", 1, "transient_local"),
    "health": QosSpec("reliable", 10),
}

#: A receive time below this is no epoch time: the ROS clock has not started
#: (use_sim_time before the first /clock message reads 0).
_EPOCH_MIN_S = 1e8

#: Cityscapes train ids routed to the grid's decaying dynamic layer (as offline).
_DYNAMIC_IDS = np.array(sorted(DYNAMIC_CLASS_IDS), dtype=np.uint8)

#: The kart's planar footprint in base_link (x_min, x_max, half_width) [m]
#: and the minimum camera obstacle-contact range [m]: the classical-fix grid
#: guards, same values as kart_map2d.pipeline (duplicated on purpose - the
#: pipeline module drags offline-only imports; the tests keep them equal).
KART_FOOTPRINT_M = (-1.0, 1.0, 0.6)
CAMERA_MIN_CONTACT_RANGE_M = 2.5


# ----------------------------------------------------------------- params
@dataclass
class CameraParams:
    """One camera, fully described by its parameters. NOTHING here assumes a
    camera make (requirement 6): topics, source, mount, height, calibration
    and mask are all configuration. Keys match [camera] in kart_master.cfg.
    """

    #: Where frames come from: "ros2" (image_topic [+ info_topic or
    #: calibration]), "device" (cv2.VideoCapture: index, path or video file;
    #: calibration_file required) or "gstreamer" (pipeline string;
    #: calibration_file required). See kart_map2d/camera/capture.py.
    source: str = "ros2"
    image_topic: str = ""
    #: sensor_msgs/CameraInfo topic; "" -> calibration_file or ``intrinsics``.
    info_topic: str = ""
    #: base_link position of the camera optical centre [m].
    mount_xyz: tuple[float, float, float] = (0.0, 0.0, 0.5)
    #: Calibrated camera height over ground [m] (monocular VO scale). None
    #: falls back to mount z over the nominal ground plane (see the README's
    #: camera-height calibration procedure: drive a straight known distance).
    height_m: float | None = None
    #: Camera body roll / pitch (+ = looks down) / yaw vs base_link [deg];
    #: (0, 0, 0) = looking straight ahead, level.
    mount_rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    #: True: image_topic carries sensor_msgs/CompressedImage (jpeg / png).
    compressed: bool = False
    #: [fx, fy, cx, cy, width, height] at full resolution, when there is no
    #: info_topic (ignored when info_topic is set). Zero distortion assumed;
    #: give calibration_file instead for a distorted lens.
    intrinsics: tuple = ()
    #: Calibration YAML (kart_map2d/camera/model.py format): intrinsics AND
    #: lens distortion for sources without camera_info, or to override it.
    calibration_file: str = ""
    #: Extrinsics-from-TF switch: "" (default) = use the manual mount below;
    #: "auto" = look the camera up in TF via the image header's frame_id;
    #: any other value = look up that TF frame. A frame whose name contains
    #: "optical" is taken as a REP-103 optical frame (used as-is); anything
    #: else is treated as the camera BODY link (standard forward-optical
    #: convention applied on top). When the lookup fails the node warns once
    #: and uses the manual mount - the fallback is always available.
    frame: str = ""
    #: Kart-body mask image (white/nonzero = kart body) at FULL camera
    #: resolution; "" = learn or use the geometric default.
    mask_file: str = ""
    #: Auto-learn the per-camera kart mask from temporally static pixels
    #: while driving (the classical-fix learner; ~1 s one-time cost on the
    #: mapping worker). Until learned - or if learning refuses - the
    #: geometric default mask applies.
    learn_mask: bool = True
    #: device / gstreamer sources (ignored for ros2):
    device: str = "0"
    gstreamer_pipeline: str = ""
    device_width: int = 0
    device_height: int = 0
    device_fps: float = 0.0
    #: Per-camera processing scale; None = the node-wide ``image_scale``.
    processing_scale: float | None = None
    #: Non-ROS sources: deliver at most this many frames/s to the pipeline
    #: (0 = every frame). ROS sources are paced by the publisher.
    max_process_rate_hz: float = 0.0


def camera_config_problems(cameras: dict[str, CameraParams]) -> list[str]:
    """Every camera entry that cannot work, each with its fix (empty when fine)."""
    problems = []
    for name, cam in cameras.items():
        if cam.source not in ("ros2", "device", "gstreamer"):
            problems.append(f"cameras.{name}: source must be ros2 | device | gstreamer, got {cam.source!r}")
            continue
        if cam.source == "ros2":
            if not cam.image_topic:
                problems.append(f"cameras.{name}: image_topic is empty - give the camera's image topic or remove the entry")
            if not cam.info_topic and len(cam.intrinsics) != 6 and not cam.calibration_file:
                problems.append(
                    f"cameras.{name}: give info_topic, calibration_file or intrinsics [fx, fy, cx, cy, width, height] "
                    "- a camera with no intrinsics is refused (exit 2)"
                )
        else:
            if not cam.calibration_file and len(cam.intrinsics) != 6:
                problems.append(
                    f"cameras.{name}: source={cam.source} has no camera_info topic - set calibration_file "
                    "(tools/dump_camera_info.py writes one from a live camera_info topic)"
                )
            if cam.source == "gstreamer" and not cam.gstreamer_pipeline:
                problems.append(f"cameras.{name}: source=gstreamer needs gstreamer_pipeline")
        if len(cam.mount_xyz) != 3 or len(cam.mount_rpy_deg) != 3:
            problems.append(f"cameras.{name}: mount_xyz and mount_rpy_deg need 3 numbers each")
        if cam.calibration_file and not Path(cam.calibration_file).exists():
            problems.append(f"cameras.{name}: calibration_file {cam.calibration_file!r} does not exist")
        if cam.mask_file and not Path(cam.mask_file).exists():
            problems.append(f"cameras.{name}: mask_file {cam.mask_file!r} does not exist")
    return problems


@dataclass
class NodeParams:
    """All tunables. Normally produced from kart_master.cfg
    (:func:`kart_map2d.config.master_config.to_mapping_params_dict`); a plain
    YAML params file works too (tests, parity runs)."""

    map_frame: str = "map"
    # Sources ("" disables a subscription). The dedicated IMU is the ONLY
    # IMU subscribed by default; /livox/imu exists behind livox_imu_topic
    # ([sensors] livox_imu = 0 by default) and NO camera IMU is ever read.
    imu_topic: str = "/imu/data"
    imu_rate_hz: float = 100.0  # decimation in front of the EKF, as offline
    livox_imu_topic: str = ""  # off by default (user decision)
    livox_accel_in_g: bool = True
    #: base_link <- livox_frame mount (shared by the Livox IMU + LiDAR).
    livox_xyz: tuple[float, float, float] = (0.0, 0.0, 0.52)
    livox_rpy_deg: tuple[float, float, float] = (0.0, 0.0, -90.0)
    odom_topic: str = ""  # off by default ([sensors] odom = 0)
    odom_stride: int = 2
    lidar_topic: str = ""  # 2D grid evidence only ([sensors] lidar, experiment switch)
    cameras: dict[str, CameraParams] = field(default_factory=dict)
    # Requirement-13 inputs ("" disables; see kart_map2d/sensors/adapters.py).
    mag_topic: str = ""
    mag_calibration_file: str = ""
    mag_declination_deg: float = 0.0
    mag_mount_yaw_deg: float = 0.0
    mag_sigma_deg: float = 8.0
    mag_rate_hz: float = 20.0
    speed_topic: str = ""
    speed_type: str = "float32"  # float32 | twist_stamped
    speed_sigma_mps: float = 0.15
    wheel_topic: str = ""
    wheel_type: str = "int64"  # int64 | joint_state
    wheel_joint: str = ""
    wheel_ticks_per_metre: float = 1000.0
    wheel_sigma_floor_mps: float = 0.05
    # Visual odometry ([odometry] in kart_master.cfg).
    vo_orb_features: int = 3000
    vo_max_corners: int = 1500
    vo_feature_spacing_px: int = 10
    vo_min_pair_dt_s: float = 0.09
    # Rates and processing.
    output_rate_hz: float = 50.0  # engine drain + pose publish timer
    #: FusionEngine reorder window [s]: measurements are applied in stamp order
    #: once the arrival clock is this far past them (offline replay: 0.12).
    reorder_window_s: float = 0.12
    #: EKF max lateness [s]: a measurement older than the filter state by more
    #: than this is dropped (offline: 0.25). Live VO arrives after its solve.
    max_lateness_s: float = 0.25
    #: The EKF's arrival clock never runs ahead of the newest sample of an IMU
    #: that delivered within this many seconds (see KartMapNode._arrival).
    imu_hold_s: float = 0.3
    image_scale: float = 0.5  # matches the offline half-res pipeline
    vo_async: bool = True  # decode + solve VO on a worker thread per camera (drop, never queue)
    #: Segmentation, grid inserts, grid / path publishing and the corridor on
    #: the mapping worker thread (one step in flight; a tick that finds it
    #: busy is skipped and counted). False runs them inside the timer
    #: callbacks, blocking the sensor callbacks - only for deterministic tests.
    mapping_async: bool = True
    seg_enabled: bool = True
    #: Segmentation backend: "classical" (zero-network, torch-free),
    #: "neural" (SegFormer; needs torch + transformers + the checkpoint) or
    #: "auto" (neural only when all three are present, else classical).
    seg_mode: str = "classical"  # this package defaults to the zero-network backend
    seg_period_s: float = 0.5  # mapping timer; frames between ticks are skipped
    grid_cell_m: float = 0.15
    grid_publish_period_s: float = 1.0
    corridor_period_s: float = 5.0
    corridor_spacing_m: float = 0.5
    corridor_close_cells: int = 2
    path_min_step_m: float = 0.25
    path_max_poses: int = 2000
    health_period_s: float = 1.0
    # Mapping guards + evidence tuning (classical-fix values; see the
    # module constants for why).
    camera_max_range_m: float = 18.0
    min_contact_range_m: float = CAMERA_MIN_CONTACT_RANGE_M
    kart_footprint: tuple[float, float, float] = KART_FOOTPRINT_M
    dyn_half_life_s: float = 3.0
    #: Scans waiting for the EKF to reach their stamp (~0.1 s); older ones
    #: fall out of this fixed-size ring (counted), so nothing backs up.
    scan_ring_size: int = 8
    # Saving (~/save_map).
    output_dir: str = "outputs/maps_live"
    map_name: str = "live"
    save_on_shutdown: bool = False
    loop_closure: bool = True  # at save time only
    #: Run the save-time loop closure (seconds to tens of seconds of numpy /
    #: Python) in a spawned child process, so it does not compete with the
    #: sensor callbacks and VO for this process's interpreter lock. Falls
    #: back to this process if the child cannot run.
    loop_closure_subprocess: bool = True
    #: Per-frame evidence kept so a save whose loop closure moved the
    #: trajectory re-grids at the corrected poses (as offline). Past this
    #: budget the store is dropped and such a save keeps the live grid.
    rebuild_budget_mb: float = 1024.0

    @classmethod
    def from_dict(cls, data: dict | None) -> "NodeParams":
        """Build params from a dict (a YAML file, or master_config's translation).

        Accepts an optional ``kart_map2d`` nesting level (and ``trackmapper``
        for params files carried over from the research workspace).
        """
        data = dict(data or {})
        for nest in ("kart_map2d", "trackmapper"):
            if nest in data and isinstance(data[nest], dict):
                data = dict(data[nest])
                break
        data.pop("use_sim_time", None)  # a node-level ROS setting, not a NodeParams field
        cameras = data.pop("cameras", None)
        known = {f.name for f in fields(cls)}
        params = cls(**{k: v for k, v in data.items() if k in known})
        params.kart_footprint = tuple(params.kart_footprint) if params.kart_footprint else params.kart_footprint
        if cameras is not None:
            cam_known = {f.name for f in fields(CameraParams)}
            params.cameras = {
                name: CameraParams(
                    **{
                        **{k: v for k, v in dict(cfg).items() if k in cam_known},
                        "mount_xyz": tuple(cfg.get("mount_xyz") or (0.0, 0.0, 0.5)),
                        "mount_rpy_deg": tuple(cfg.get("mount_rpy_deg") or (0.0, 0.0, 0.0)),
                        "intrinsics": tuple(cfg.get("intrinsics") or ()),
                    }
                )
                for name, cfg in cameras.items()
            }
        return params


def unknown_param_keys(data: dict | None) -> list[str]:
    """Keys of a params YAML dict that NodeParams does not know, each with a hint.

    ``from_dict`` ignores them (older / newer params files keep working);
    :func:`main` logs these so a typo such as ``mag_topik`` cannot go unseen.
    """
    data = dict(data or {})
    for nest in ("kart_map2d", "trackmapper"):
        if nest in data and isinstance(data[nest], dict):
            data = dict(data[nest])
            break
    known = [f.name for f in fields(NodeParams)] + ["use_sim_time"]
    out = []
    for key in data:
        if key in known:
            continue
        close = difflib.get_close_matches(str(key), known, n=1, cutoff=0.6)
        out.append(f"{key!r} (did you mean {close[0]!r}?)" if close else repr(key))
    return out


class NodeConfigError(ValueError):
    """A configuration the node must not start with; ``problems`` says what to change."""

    def __init__(self, problems: list[str] | str) -> None:
        self.problems = [problems] if isinstance(problems, str) else list(problems)
        super().__init__("; ".join(self.problems))


def apply_ros_parameters(node: Any, params: NodeParams, descriptor: Any = None) -> NodeParams:
    """Overlay ``--ros-args -p key:=value`` onto every scalar NodeParams field.

    Each scalar field is declared as a ROS parameter whose default is the
    cfg/YAML value, so a command-line override wins and ``ros2 param get``
    shows what the node runs with. Types are checked like the YAML: floats
    accept integers.
    """
    declare = getattr(node, "declare_parameter", None)
    if declare is None:
        return params
    defaults = NodeParams()
    out = copy.copy(params)
    problems = []
    for f in fields(NodeParams):
        name, current = f.name, getattr(params, f.name)
        expected = type(getattr(defaults, name))
        if expected not in (bool, int, float, str):
            continue  # tuples / camera dicts stay cfg-only
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
        raise NodeConfigError(problems)
    return out


# ------------------------------------------------------------- VO plumbing
class _GyroYawBuffer:
    """Rolling trapezoidal yaw integral, the live twin of ``yaw_prior_from_imu``.

    Keeps (stamp, cumulative yaw) over a horizon comfortably longer than the
    VO max frame gap; ``prior(t0, t1)`` interpolates the cumulative integral
    exactly like the offline callback (edge-clamped by ``np.interp``).
    """

    def __init__(self, horizon_s: float = 30.0) -> None:
        self.horizon_s = float(horizon_s)
        self._t: deque[float] = deque()
        self._cum: deque[float] = deque()
        self._last_rate = 0.0
        self._lock = threading.Lock()

    def push(self, stamp: float, yaw_rate: float) -> None:
        with self._lock:
            if self._t and stamp <= self._t[-1]:
                return  # out-of-order duplicate; the integral needs monotonic time
            if not self._t:
                self._t.append(stamp)
                self._cum.append(0.0)
                self._last_rate = yaw_rate
                return
            dt = stamp - self._t[-1]
            self._cum.append(self._cum[-1] + 0.5 * (self._last_rate + yaw_rate) * dt)
            self._t.append(stamp)
            self._last_rate = yaw_rate
            while self._t and stamp - self._t[0] > self.horizon_s:
                self._t.popleft()
                self._cum.popleft()

    def prior(self, t0: float, t1: float) -> float:
        """Integrated base_link yaw change [rad] between two stamps."""
        with self._lock:
            if len(self._t) < 2:
                return 0.0
            t = np.asarray(self._t)
            c = np.asarray(self._cum)
        a, b = np.interp([t0, t1], t, c)
        return float(b - a)


class _LiveVisualOdometry(VisualOdometry):
    """VisualOdometry fed from memory instead of image files.

    ``process_array`` mirrors :meth:`VisualOdometry.process` minus the
    ``cv2.imread``: live frames arrive as arrays and a disk round-trip would
    burn the latency budget. The pair bookkeeping is duplicated (10 lines)
    because the parent method is hard-wired to ``CameraFrame.image_path``.
    """

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


# ------------------------------------------------------------- live stats
class _LatencyStats:
    """Wall time per callback / stage [ms]: count, mean, p95, max. Thread-safe.

    Memory is fixed: the p95 comes from a log-spaced histogram (40 bins per
    decade, 1 us .. 10 s, so within ~6 %) over the WHOLE run, not a window.
    """

    _EDGES = [10.0 ** (k / 40.0) for k in range(-120, 161)]  # ms

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: dict[str, list] = {}  # key -> [count, sum_ms, max_ms, histogram]

    def add(self, key: str, seconds: float) -> None:
        ms = 1e3 * seconds
        slot = bisect.bisect_left(self._EDGES, ms)
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                entry = self._data[key] = [0, 0.0, 0.0, [0] * (len(self._EDGES) + 1)]
            entry[0] += 1
            entry[1] += ms
            entry[2] = max(entry[2], ms)
            entry[3][slot] += 1

    @contextmanager
    def time(self, key: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.add(key, time.perf_counter() - t0)

    def summary(self) -> dict[str, dict[str, float]]:
        with self._lock:
            items = [(k, v[0], v[1], v[2], list(v[3])) for k, v in self._data.items()]
        out = {}
        for key, n, total, peak, hist in sorted(items):
            need, seen, p95 = 0.95 * n, 0, peak
            for slot, count in enumerate(hist):
                seen += count
                if seen >= need:
                    p95 = min(self._EDGES[slot] if slot < len(self._EDGES) else peak, peak)
                    break
            out[key] = {"n": n, "mean_ms": round(total / n, 3), "p95_ms": round(p95, 3), "max_ms": round(peak, 3)}
        return out


def _rss_mb() -> float | None:
    """Resident memory of this process [MB] (Linux /proc; None elsewhere)."""
    try:
        with open("/proc/self/statm", "r", encoding="ascii") as f:
            pages = int(f.read().split()[1])
        return round(pages * os.sysconf("SC_PAGE_SIZE") / 1e6, 1)
    except (OSError, ValueError, AttributeError, IndexError):
        return None


# ------------------------------------------------------ loop-closure evidence
class _GroundObserver:
    """Per-frame ground evidence for loop closure, as ``pipeline.ground_observations``.

    The offline function recomputes every pixel ray per call; live frames
    come one at a time, so the ray geometry is cached per camera and image
    size and only the per-frame part runs (same constants: ground plane
    z = -0.10 m, 12 m range, 0.20 m dedupe cells). Points are kept as int16
    cell indices (``floor(p / cell)``), which is lossless because the offline
    dedupe returns exactly ``index * cell + cell / 2``; tests/test_ros2_node.py
    checks the two agree.
    """

    def __init__(self, camera_model: CameraModel, T_base_cam, shape, ground_z: float = -0.10, max_range: float = 12.0, cell: float = 0.20) -> None:
        T_base_cam = np.asarray(T_base_cam, dtype=np.float64)
        # Distortion-aware ray table (pinhole-exact when D == 0), so the
        # loop-closure evidence uses the same geometry as the grid inserts.
        rays = camera_model.pixel_rays(shape) @ T_base_cam[:3, :3].T
        cam = T_base_cam[:3, 3]
        dz = rays[..., 2]
        self.below = dz < -1e-9
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(self.below, (ground_z - cam[2]) / dz, np.inf)
            self.gx = cam[0] + t * rays[..., 0]
            self.gy = cam[1] + t * rays[..., 1]
            self.rng = np.hypot(self.gx - cam[0], self.gy - cam[1])
        self.shape = tuple(shape)
        self.max_range = float(max_range)
        self.cell = float(cell)

    def _cells(self, xs: NDArray[np.float64], ys: NDArray[np.float64]) -> NDArray[np.int16]:
        if not len(xs):
            return np.zeros((0, 2), dtype=np.int16)
        idx = np.unique(np.floor(np.column_stack([xs, ys]) / self.cell).astype(np.int64), axis=0)
        return idx.astype(np.int16)

    def observe(self, drivable_mask, obstacle_mask) -> tuple[NDArray[np.int16], NDArray[np.int16]]:
        """(drivable cells, obstacle-contact cells) as int16 [ix, iy] in base_link."""
        h, w = self.shape
        obst = obstacle_mask
        col_limit = np.full(w, np.inf)
        contact = (np.empty(0), np.empty(0))
        if obst.any():
            cols = np.flatnonzero(obst.any(axis=0))
            rows = (h - 1) - np.argmax(obst[::-1, :][:, cols], axis=0)
            ok = self.below[rows, cols]
            col_limit[cols[ok]] = self.rng[rows[ok], cols[ok]]
            keep = ok & (self.rng[rows, cols] <= self.max_range)
            contact = (self.gx[rows[keep], cols[keep]], self.gy[rows[keep], cols[keep]])
        sel = drivable_mask & self.below & (self.rng <= self.max_range) & (self.rng <= col_limit[None, :])
        return self._cells(self.gx[sel], self.gy[sel]), self._cells(*contact)


class _JobWorker:
    """One background thread running at most one job at a time - never a queue.

    :meth:`submit` starts ``fn`` when the thread is idle and returns False
    while a job runs (the caller counts the skip). Used for the mapping step:
    it keeps segmentation, grid inserts and corridor extraction (tens to
    hundreds of ms) off the executor thread that feeds the EKF.
    """

    def __init__(self, name: str, stats: _LatencyStats, warn) -> None:
        self.stats = stats
        self._warn = warn
        self.errors: list[str] = []
        self._cv = threading.Condition()
        self._job: tuple[str, Any] | None = None
        self._busy = False
        self._stop = False
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
            except Exception as exc:  # noqa: BLE001 - mapping must never kill localization
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

    def stop(self, timeout: float = 5.0) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        self._thread.join(timeout=timeout)


class _RosMessages:
    """Message classes, imported lazily so the module needs no ROS to import."""

    def __init__(self) -> None:
        from geometry_msgs.msg import PoseStamped
        from nav_msgs.msg import OccupancyGrid, Odometry, Path
        from sensor_msgs.msg import CameraInfo, Image, Imu, PointCloud2
        from std_msgs.msg import String

        self.Imu = Imu
        self.Image = Image
        self.CameraInfo = CameraInfo
        try:  # only for cameras with compressed: true (test stubs may lack it)
            from sensor_msgs.msg import CompressedImage
        except ImportError:
            CompressedImage = None
        self.CompressedImage = CompressedImage
        self.PointCloud2 = PointCloud2
        self.Odometry = Odometry
        self.PoseStamped = PoseStamped
        self.Path = Path
        self.OccupancyGrid = OccupancyGrid
        self.String = String
        # Requirement-13 inputs; each optional so a stripped install (or a
        # test stub) only needs the messages its configuration subscribes.
        try:
            from sensor_msgs.msg import MagneticField
        except ImportError:
            MagneticField = None
        self.MagneticField = MagneticField
        try:
            from sensor_msgs.msg import JointState
        except ImportError:
            JointState = None
        self.JointState = JointState
        try:
            from std_msgs.msg import Float32
        except ImportError:
            Float32 = None
        self.Float32 = Float32
        try:
            from std_msgs.msg import Int64
        except ImportError:
            Int64 = None
        self.Int64 = Int64
        try:
            from geometry_msgs.msg import TwistStamped
        except ImportError:
            TwistStamped = None
        self.TwistStamped = TwistStamped
        # Optional: only the map-saving service and the RViz cloud need these,
        # and a stripped ROS install (or a test stub) may lack them.
        try:
            from sensor_msgs.msg import PointField
        except ImportError:
            PointField = None
        try:
            from std_srvs.srv import Trigger
        except ImportError:
            Trigger = None
        self.PointField = PointField
        self.Trigger = Trigger


# -------------------------------------------------------------------- node
class KartMapNode:
    """Wire a (duck-typed) rclpy node to the kart_map2d 2D mapping stack.

    Parameters
    ----------
    node:
        An ``rclpy.node.Node`` (or a test double exposing
        ``create_subscription`` / ``create_publisher`` / ``create_timer`` /
        ``get_clock``). Composition instead of inheritance keeps this module
        importable and testable without ROS.
    params:
        Tunables (normally translated from kart_master.cfg).
    service_node:
        Where ``save_map`` is served. :func:`main` passes a second node that
        spins on its own thread, so a save (loop closure takes seconds) never
        stalls the sensor callbacks; the service keeps this node's private
        name (``/kart_map2d/save_map``). Default: ``node`` itself.

    Raises :class:`NodeConfigError` for a configuration that cannot run.
    """

    #: This package builds exactly one map method.
    method = "flat"
    method_origin = "kart_map2d (2D-only package)"

    def __init__(self, node: Any, params: NodeParams | None = None, service_node: Any = None) -> None:
        self.node = node
        self.service_node = service_node
        self.params = params or NodeParams()
        problems = camera_config_problems(self.params.cameras)
        problems += self._sensor_config_problems(self.params)
        if problems:
            raise NodeConfigError(problems)
        self.msgs = _RosMessages()
        ros_cams = {n: c for n, c in self.params.cameras.items() if c.source == "ros2"}
        if getattr(self.msgs, "CompressedImage", None) is None and any(c.compressed for c in ros_cams.values()):
            raise NodeConfigError("compressed: true needs sensor_msgs/CompressedImage, which this ROS install lacks")
        self.engine = FusionEngine(
            ekf_config=EkfConfig(max_lateness_s=self.params.max_lateness_s),
            reorder_window_s=self.params.reorder_window_s,
            output_rate_hz=self.params.output_rate_hz,
        )
        # The classical-fix grid guards are ON for the kart rig (forward
        # vehicle-mounted cameras): contacts inside the kart footprint or
        # closer than min_contact_range are never emitted.
        self.grid = EvidenceGrid(
            cell=self.params.grid_cell_m,
            dyn_half_life=self.params.dyn_half_life_s,
            kart_footprint=tuple(self.params.kart_footprint) if self.params.kart_footprint else None,
            min_contact_range_m=self.params.min_contact_range_m,
        )
        self.stats = _LatencyStats()
        # An injected segmenter (tests) serves every camera; otherwise the
        # backend picked by seg_mode is created lazily (_segmenter_for).
        self.segmenter = None
        self.seg_mode = self._resolve_seg_mode()  # "neural" | "classical" | "off"
        self._neural = None  # one shared SegFormer (stateless across cameras)
        self._segmenters: dict[str, Any] = {}  # classical: one per camera
        self.counters = {
            "imu": 0,
            "imu_livox": 0,
            "odom": 0,
            "mag": 0,
            "mag_rejected_field": 0,  # field magnitude outside the calibrated band
            "mag_gyro_inconsistent": 0,  # mag-gyro consistency check verdicts (channel faulted)
            "mag_gated": 0,  # headings withheld pre-validation / post-trip (validation-armed trust)
            "speed": 0,
            "wheel": 0,
            "frames": 0,
            "vo_solved": 0,
            "vo_dropped": 0,
            "seg_inserted": 0,
            "lidar_scans": 0,
            "lidar_inserted": 0,  # grid (one posed scan per mapping tick)
            "lidar_expired": 0,  # fell out of the ring before the EKF reached its stamp
            "lidar_superseded": 0,  # a newer scan became posed in the same timer tick
            "lidar_untrusted": 0,  # reached, but the EKF pose there is not motion-supported
            "lidar_posed": 0,  # handed to the grid
            "imu_clock_not_ready": 0,  # IMU samples before the ROS clock ran (sim time): skipped
            "imu_taken_by_timer": 0,  # queued IMU samples the output timer took before draining
            "stamp_clock_mismatch": 0,  # messages stamped > 5 s away from the node clock
            "mapping_skipped_busy": 0,  # mapping tick while the previous step still ran
            "corridor_deferred": 0,  # corridor tick while mapping ran: done at that step's end
            "capture_frames": 0,  # frames delivered by non-ROS camera sources
            "kart_masks_learned": 0,
            "saves": 0,
        }
        # Counters are bumped from the executor and the worker threads.
        self._counter_lock = threading.Lock()
        # Engine access is shared between rclpy callbacks and VO worker
        # threads, and the engine itself is not thread-safe.
        self._engine_lock = threading.Lock()
        # The grid (+ loop-closure / rebuild evidence) is mutated by the
        # mapping step and snapshotted by a save on the service thread.
        self._map_lock = threading.Lock()
        self._save_lock = threading.Lock()
        self._frame_lock = threading.Lock()  # pending frames: executor, VO workers, mapping worker
        self._scan_lock = threading.Lock()  # the posed scan handed from the output timer to mapping
        #: EKF updates by channel, [accepted, rejected], folded in from the
        #: engine's per-update log, which the node trims (it grows ~500 entries/s).
        self._ekf_updates: dict[str, list[int]] = {}
        # Bit-identical to the offline ClockSync, O(1) per sample instead of
        # O(window) (see converters.StreamingClockSync).
        self._clock_sync: dict[str, converters.StreamingClockSync] = {}
        self._decimators: dict[str, converters.ImuDecimator] = {}
        self._gyro_buffer = _GyroYawBuffer()
        #: IMU channel -> (newest sample stamp, node time it was received): the
        #: arrival clock is held at the newest IMU stamp while that IMU streams.
        self._imu_newest: dict[str, tuple[float, float]] = {}
        #: Node time at which the output timer last emptied the IMU subscription
        #: queues (see _arrival): IMU silence is judged from here, not from now().
        self._imu_checked_at: float | None = None
        #: One-time warnings already given (a misconfigured input is named once, not per message).
        self._warned: set[str] = set()
        self._subs: dict[str, Any] = {}  # key -> subscription (backlog draining)
        self._odom_converter = converters.OdometryConverter(stride=self.params.odom_stride)
        R_livox = rpy_matrix(*np.radians(self.params.livox_rpy_deg))
        self._T_base_livox = make_transform(R_livox, np.asarray(self.params.livox_xyz, dtype=np.float64))
        # Requirement-13 adapters (constructed only for enabled inputs; a
        # magnetometer without a loadable calibration refuses at start).
        self._mag_adapter: MagnetometerAdapter | None = None
        if self.params.mag_topic:
            from kart_map2d.sensors.mag_calibration import MagCalibrationError, load_mag_calibration

            try:
                calibration = load_mag_calibration(self.params.mag_calibration_file)
            except MagCalibrationError as exc:
                raise NodeConfigError(str(exc)) from None
            self._mag_adapter = MagnetometerAdapter(
                calibration,
                declination_deg=self.params.mag_declination_deg,
                mount_yaw_deg=self.params.mag_mount_yaw_deg,
                sigma_deg=self.params.mag_sigma_deg,
                rate_hz=self.params.mag_rate_hz,
            )
        self._speed_adapter: SpeedometerAdapter | None = None
        if self.params.speed_topic:
            self._speed_adapter = SpeedometerAdapter(kind=self.params.speed_type, sigma_mps=self.params.speed_sigma_mps)
        self._wheel_adapter: WheelTickAdapter | None = None
        if self.params.wheel_topic:
            self._wheel_adapter = WheelTickAdapter(
                ticks_per_metre=self.params.wheel_ticks_per_metre,
                kind=self.params.wheel_type,
                joint=self.params.wheel_joint,
                sigma_floor_mps=self.params.wheel_sigma_floor_mps,
            )
        #: Per-camera CameraModel at FULL capture resolution (from a
        #: calibration file, fixed intrinsics or the first CameraInfo).
        self._camera_models: dict[str, CameraModel] = {}
        #: Extrinsics: resolved base_link <- optical mounts (cached once),
        #: the image header frame ids seen, and the TF lookup hook (a real
        #: tf2_ros buffer wired by main(); tests inject a callable).
        self._mount_cache: dict[str, NDArray[np.float64]] = {}
        self._camera_frame_id: dict[str, str] = {}
        self._tf_lookup = None  # callable(frame) -> 4x4 base_link <- frame, or None
        #: Non-ROS capture threads by camera name (started in _setup_interfaces).
        self._captures: dict[str, Any] = {}
        #: Per-camera kart-body masks at working resolution (loaded from
        #: mask_file or learned; None = geometric default), plus the learner
        #: frame stores for cameras still learning.
        self._kart_masks: dict[str, NDArray[np.bool_] | None] = {}
        self._mask_learn_frames: dict[str, list[NDArray[np.uint8]]] = {}
        self._mask_learn_done: set[str] = set()
        self._intrinsics: dict[str, tuple[NDArray[np.float64], tuple[int, int]]] = {}
        self._vo: dict[str, _LiveVisualOdometry] = {}
        self._vo_busy: dict[str, threading.Lock] = {}
        #: Newest frame per camera for the mapping step, (stamp, image message or
        #: decoded array), and the one it replaced (used when the newest has no
        #: pose yet): at most two per camera, older frames are dropped.
        self._pending_frame: dict[str, tuple[float, Any]] = {}
        self._pending_prev: dict[str, tuple[float, Any]] = {}
        self._last_seg_stamp: dict[str, float] = {}
        # LiDAR: scans wait here (fixed size, oldest falls out) until the EKF
        # trajectory reaches their stamp; the output timer then poses the
        # newest one for the grid (_grid_scan).
        self._scan_ring: deque[LidarScan] = deque(maxlen=max(int(self.params.scan_ring_size), 1))
        self._grid_scan: tuple[LidarScan, NDArray[np.float64]] | None = None
        # Loop-closure evidence (per inserted frame) and the rebuild store.
        self._ground_obs: dict[tuple, _GroundObserver] = {}
        self._observations: list[tuple[float, NDArray[np.int16], NDArray[np.int16]]] = []
        self._store: list[tuple] = []
        self._store_bytes = 0
        self._store_complete = True
        self._path_xyyaw: deque[tuple[float, float, float, float]] = deque(
            maxlen=self.params.path_max_poses
        )  # (stamp, x, y, yaw)
        self._path_dirty = False
        self._last_health: tuple | None = None
        self._last_health_pub = -np.inf
        self._last_grid_pub = -np.inf
        self._corridor_centerline: NDArray[np.float64] | None = None
        self._corridor_due = False
        self._queue_max = {"engine": 0, "scan_ring": 0}
        self._rss_first: float | None = None
        self._rss_max: float | None = None
        self.last_save: dict | None = None
        self.save_service = None
        self.save_service_name: str | None = None
        self._map_worker: _JobWorker | None = None
        if self.params.mapping_async:
            self._map_worker = _JobWorker("kart2d-mapping", self.stats, self._warn)
        self._load_camera_calibrations()
        self._setup_interfaces()

    @staticmethod
    def _sensor_config_problems(p: NodeParams) -> list[str]:
        """Non-camera input combinations the node refuses (each with its fix)."""
        problems = []
        if p.speed_topic and p.speed_type not in ("float32", "twist_stamped"):
            problems.append(f"speed_type must be float32 | twist_stamped, got {p.speed_type!r}")
        if p.wheel_topic:
            if p.wheel_type not in ("int64", "joint_state"):
                problems.append(f"wheel_type must be int64 | joint_state, got {p.wheel_type!r}")
            if p.wheel_ticks_per_metre <= 0:
                problems.append("wheel_ticks_per_metre must be > 0 (count ticks over a measured straight)")
        if p.mag_topic and not p.mag_calibration_file:
            from kart_map2d.sensors.mag_calibration import CALIBRATION_PROCEDURE

            problems.append("mag_topic is set but mag_calibration_file is empty.\n" + CALIBRATION_PROCEDURE)
        # The dedicated-IMU-only policy (requirement 4) is structural: there
        # is no parameter that could subscribe a camera IMU, and livox_imu
        # only exists behind livox_imu_topic. Nothing to validate here.
        return problems

    def _load_camera_calibrations(self) -> None:
        """Resolve every camera's CameraModel that is known BEFORE spin.

        ros2-source cameras with an info_topic get theirs from the first
        CameraInfo message instead (in :meth:`_on_camera_info`).
        """
        for name, cam in self.params.cameras.items():
            if cam.calibration_file:
                model = load_calibration_yaml(cam.calibration_file)
            elif len(cam.intrinsics) == 6:
                fx, fy, cx, cy, w, h = (float(v) for v in cam.intrinsics)
                model = CameraModel(converters.intrinsics_matrix(fx, fy, cx, cy), (int(w), int(h)))
            else:
                continue  # CameraInfo topic delivers it later
            self._register_camera_model(name, model)

    def _register_camera_model(self, camera: str, model: CameraModel) -> None:
        """Store the full-resolution model + the working-resolution intrinsics."""
        self._camera_models[camera] = model
        scaled = model.scaled(self._camera_scale(camera))
        self._intrinsics[camera] = (scaled.K, scaled.size)

    def _camera_scale(self, camera: str) -> float:
        cam = self.params.cameras.get(camera)
        if cam is not None and cam.processing_scale is not None:
            return float(cam.processing_scale)
        return float(self.params.image_scale)

    def _camera_model_scaled(self, camera: str) -> CameraModel:
        """The camera's model at the WORKING resolution (rays, undistortion)."""
        return self._camera_models[camera].scaled(self._camera_scale(camera))

    def _bump(self, key: str, n: int = 1) -> None:
        """Thread-safe counter increment (executor and worker threads share counters)."""
        with self._counter_lock:
            self.counters[key] += n

    # -------------------------------------------------------------- wiring
    def _setup_interfaces(self) -> None:
        node, p, msgs = self.node, self.params, self.msgs
        if p.imu_topic:
            # The DEDICATED IMU. This node subscribes no other IMU unless
            # livox_imu_topic is explicitly set, and never a camera IMU.
            self._subs["imu"] = node.create_subscription(msgs.Imu, p.imu_topic, self._on_imu, QOS["imu"].to_profile())
        if p.livox_imu_topic:
            self._subs["imu_livox"] = node.create_subscription(
                msgs.Imu, p.livox_imu_topic, self._on_imu_livox, QOS["imu_livox"].to_profile()
            )
        if p.odom_topic:
            node.create_subscription(msgs.Odometry, p.odom_topic, self._on_odom, QOS["odom"].to_profile())
        if p.lidar_topic:
            node.create_subscription(
                msgs.PointCloud2, p.lidar_topic, self._on_pointcloud, QOS["lidar"].to_profile()
            )
        if p.mag_topic:
            if msgs.MagneticField is None:
                raise NodeConfigError("mag_topic needs sensor_msgs/MagneticField, which this ROS install lacks")
            node.create_subscription(msgs.MagneticField, p.mag_topic, self._on_mag, QOS["mag"].to_profile())
        if p.speed_topic:
            speed_type = msgs.TwistStamped if p.speed_type == "twist_stamped" else msgs.Float32
            if speed_type is None:
                raise NodeConfigError(f"speed_topic needs the {p.speed_type} message type, which this ROS install lacks")
            node.create_subscription(speed_type, p.speed_topic, self._on_speed, QOS["speed"].to_profile())
        if p.wheel_topic:
            wheel_type = msgs.JointState if p.wheel_type == "joint_state" else msgs.Int64
            if wheel_type is None:
                raise NodeConfigError(f"wheel_topic needs the {p.wheel_type} message type, which this ROS install lacks")
            node.create_subscription(wheel_type, p.wheel_topic, self._on_wheel, QOS["wheel"].to_profile())
        for camera, cam in p.cameras.items():
            if cam.source != "ros2":
                continue  # non-ROS sources start in start_captures()
            image_type = msgs.CompressedImage if cam.compressed else msgs.Image
            node.create_subscription(
                image_type, cam.image_topic, partial(self._on_image, camera), QOS["image"].to_profile()
            )
            if cam.info_topic and camera not in self._camera_models:
                node.create_subscription(
                    msgs.CameraInfo,
                    cam.info_topic,
                    partial(self._on_camera_info, camera),
                    QOS["camera_info"].to_profile(),
                )

        self.pub_pose = node.create_publisher(msgs.PoseStamped, "kart_map2d/pose", PUB_QOS["pose"].to_profile())
        self.pub_path = node.create_publisher(msgs.Path, "kart_map2d/path", PUB_QOS["path"].to_profile())
        self.pub_grid = node.create_publisher(
            msgs.OccupancyGrid, "kart_map2d/grid", PUB_QOS["grid"].to_profile()
        )
        self.pub_corridor = node.create_publisher(
            msgs.Path, "kart_map2d/corridor", PUB_QOS["corridor"].to_profile()
        )
        self.pub_health = node.create_publisher(msgs.String, "kart_map2d/health", PUB_QOS["health"].to_profile())

        node.create_timer(1.0 / p.output_rate_hz, self._on_output_timer)
        node.create_timer(p.seg_period_s, self._on_mapping_timer)
        node.create_timer(p.corridor_period_s, self._on_corridor_timer)

        host = self.service_node if self.service_node is not None else node
        if msgs.Trigger is not None and hasattr(host, "create_service"):
            if host is node:
                name = "~/save_map"
            else:
                fqn = getattr(node, "get_fully_qualified_name", lambda: "/trackmapper")()
                name = fqn.rstrip("/") + "/save_map"
            self.save_service = host.create_service(msgs.Trigger, name, self._on_save_map)
            self.save_service_name = name

    def describe(self) -> list[str]:
        """The configuration in effect, one line per aspect (logged by :func:`main`)."""
        p = self.params

        def cam_desc(n: str, c: CameraParams) -> str:
            if c.source == "ros2":
                return f"{n} (ros2{', compressed' if c.compressed else ''})"
            return f"{n} ({c.source}: {c.gstreamer_pipeline[:40] if c.source == 'gstreamer' else c.device})"

        cams = ", ".join(cam_desc(n, c) for n, c in p.cameras.items()) or "none"
        sources = [
            f"imu={p.imu_topic or 'off'}",
            f"imu_livox={p.livox_imu_topic or 'off'}",
            f"odom={p.odom_topic or 'off'}",
            f"lidar={p.lidar_topic or 'off'}",
            f"mag={p.mag_topic or 'off'}",
            f"speed={p.speed_topic or 'off'}",
            f"wheel={p.wheel_topic or 'off'}",
            f"cameras={cams}",
        ]
        lines = [f"method: {self.method} ({self.method_origin})", "sources: " + ", ".join(sources)]
        if p.lidar_topic:
            lines.append(f"flat: {p.lidar_topic} adds 2D LiDAR evidence to the grid (experiment switch)")
        if not (p.imu_topic or p.livox_imu_topic) or not p.cameras:
            lines.append("WARNING: camera + IMU is the baseline; this configuration lacks one of them")
        if not (p.seg_enabled and p.cameras) and not p.lidar_topic:
            lines.append("WARNING: no map evidence (segmentation off, no lidar_topic): localization only, the grid stays empty")
        lines.append(f"segmentation: {self.seg_mode} (seg_mode={p.seg_mode})")
        masks = []
        for name, cam in p.cameras.items():
            masks.append(f"{name}: {'file' if cam.mask_file else ('learn' if cam.learn_mask else 'geometric')}")
        if masks:
            lines.append("kart masks: " + ", ".join(masks) + " (geometric fallback until learned)")
        lines.append(
            f"timing: EKF reorder window {p.reorder_window_s} s, max lateness {p.max_lateness_s} s, held at the newest "
            f"IMU sample (imu_hold_s {p.imu_hold_s}); VO {'on a worker per camera' if p.vo_async else 'inline'}, "
            f"mapping {'on its worker' if p.mapping_async else 'inline'} every {p.seg_period_s} s (busy -> skip)"
        )
        service = self.save_service_name or "unavailable (std_srvs not importable)"
        lines.append(
            f"save_map service: {service} -> {p.output_dir} (loop_closure={p.loop_closure}, "
            f"save_on_shutdown={p.save_on_shutdown})"
        )
        return lines

    def _now(self) -> float:
        """Node receive/arrival time [s]; ROS time is the common epoch clock."""
        return float(self.node.get_clock().now().nanoseconds) * 1e-9

    def _warn(self, text: str) -> None:
        logger = getattr(self.node, "get_logger", None)
        if logger is not None:
            logger().warning(text)

    def _warn_once(self, key: str, text: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            self._warn(text)

    #: IMU samples received before the ROS clock runs until the node says so (~1 s at 400 Hz).
    _CLOCK_NOT_READY_WARN = 400
    #: A message stamped this far [s] from the node clock means the two clocks disagree.
    _CLOCK_MISMATCH_S = 5.0

    def _check_stamp(self, key: str, stamp: float, now: float) -> None:
        """Warn (once per stream) when message stamps and the node clock disagree by seconds.

        Every stream except /imu/data (device clock, mapped online) is fused
        at its header stamp against the node clock. Stamps days off - a bag
        replayed without ``--clock`` / ``use_sim_time``, or a sensor computer
        with an unsynchronized clock - leave the EKF dropping that stream as
        late while it dead-reckons on the IMU, with nothing else to show for it.
        """
        if now < _EPOCH_MIN_S or abs(now - stamp) <= self._CLOCK_MISMATCH_S:
            return
        self._bump("stamp_clock_mismatch")
        where = "behind" if now > stamp else "ahead of"
        self._warn_once(
            f"stamp_clock:{key}",
            f"{key}: message stamps are {abs(now - stamp):.1f} s {where} the node clock; this stream is "
            "dropped as late. Replaying a bag: play it with --clock 200 AND start the node with "
            "-p use_sim_time:=true (both or neither). On the kart: sync the sensor clocks (chrony / PTP).",
        )

    # ------------------------------------------------------------ callbacks
    #: Most queued messages one callback drains from its subscription.
    _BACKLOG_MAX = 256

    def _drain_backlog(self, key: str, msg, handler) -> int:
        """Handle ``msg`` and every message already queued behind it; return the count.

        rclpy's executor takes ONE message per subscription per wait-set
        pass. After any hold-up a 400 Hz IMU backlog therefore drains no
        faster than the 20 Hz odometry, the EKF's arrival clock runs ahead of
        the queued IMU samples, and they reach the filter late - out of order
        or dropped (measured: /imu/data up to 0.6 s late in a bag replay).
        Taking the queued samples right here keeps each IMU stream in step
        with the others and saves the per-message wait-set overhead. Uses
        rclpy's subscription handle the way its executor does (the same call
        on Humble); any other node object (tests) handles one message.

        Only ``msg`` - delivered by the executor, so read right after any
        newer clock message - is ``fresh``. The queued ones waited: their
        receive time is not their arrival, and under sim time the ROS clock
        cannot even advance during this loop (/clock is a callback too), so
        they would look faster than they were and drag the device-clock
        offset down. They are corrected, not observed.
        """
        handler(msg, True)
        return 1 + self._take_queued(key, handler, self._BACKLOG_MAX - 1)

    def _take_queued(self, key: str, handler, limit: int) -> int:
        """Handle up to ``limit`` messages already waiting in ``key``'s subscription queue.

        Executor thread only (the same thread that takes messages for the
        callbacks). Every message taken here waited, so it is handled as not
        ``fresh``. Returns how many were taken; 0 when the subscription has no
        take API (tests, an rclpy without it) or is gone.
        """
        sub = self._subs.get(key)
        handle = getattr(sub, "handle", None)
        take = getattr(handle, "take_message", None)
        if take is None:
            return 0
        count = 0
        while count < limit:
            try:
                with handle:
                    info = take(sub.msg_type, sub.raw)
            except Exception:  # noqa: BLE001 - destroyed at shutdown / an rclpy without this API
                self._subs.pop(key, None)
                return count
            if info is None:
                return count
            handler(info[0], False)
            count += 1
        return count

    #: Most queued IMU messages the output timer takes per IMU and tick (> 2 s at 400 Hz).
    _TIMER_IMU_TAKE_MAX = 1024

    def _pull_imu_queues(self, now: float) -> None:
        """Output timer: take every IMU message already queued, then mark the IMU queues checked.

        Python threads share one interpreter lock, so the executor thread can
        be starved for longer than ``imu_hold_s`` by the VO / mapping
        workers. The IMU samples of that stall sit in the DDS queues, but the
        next timer tick (or odometry / VO push) would see "no IMU for 0.3 s",
        release the arrival cap and drain the EKF past them - they then
        arrive late and are dropped (measured: 204 late drops in one lidar3d
        replay, ~1 s of both IMUs). Taking them here, before the drain, and
        judging IMU silence from the moment the queues were last found empty
        (``_imu_checked_at``) makes a starved executor distinguishable from
        an IMU that really stopped.
        """
        for key, handler in (("imu", self._handle_imu), ("imu_livox", self._handle_imu_livox)):
            n = self._take_queued(key, handler, self._TIMER_IMU_TAKE_MAX)
            if n:
                self._bump(key, n)
                self._bump("imu_taken_by_timer", n)
        self._imu_checked_at = now

    def _on_imu(self, msg) -> None:
        t0 = time.perf_counter()
        n = self._drain_backlog("imu", msg, self._handle_imu)
        self._bump("imu", n)
        self.stats.add("cb_imu", time.perf_counter() - t0)

    def _clock_ready(self, now: float) -> bool:
        """False while the ROS clock reads no epoch time (sim time before the first /clock).

        The IMU device-clock correction takes ``receive - stamp`` as its
        latency floor; a single sample received "at 0 s" would pin that
        floor for good (it may only rise 1e-4 s per sample), so such samples
        are skipped and counted instead.
        """
        if now >= _EPOCH_MIN_S:
            return True
        self._bump("imu_clock_not_ready")
        if self.counters["imu_clock_not_ready"] >= self._CLOCK_NOT_READY_WARN:
            self._warn_once(
                "clock_not_running",
                f"IMU data arrives but the ROS clock reads {now:.1f} s: use_sim_time is true and nothing "
                "publishes /clock, so nothing is fused. Replaying a bag: ros2 bag play <bag> --clock 200. "
                "On the kart: start the node WITHOUT use_sim_time.",
            )
        return False

    def _handle_imu(self, msg, fresh: bool = True) -> None:
        now = self._now()
        if not self._clock_ready(now):
            return
        sync = self._clock_sync.get("imu")
        if sync is None:
            sync = self._clock_sync["imu"] = converters.StreamingClockSync()
        # /imu/data stamps start near 0 s (device clock) and are axis-aligned
        # with base_link; acceleration is already m/s^2.
        sample = converters.imu_to_sample(msg, now, channel="imu", clock=sync, accel_in_g=False, observe=fresh)
        if self._mag_adapter is not None:
            # Magnetometer-gyro consistency: the check integrates the same
            # yaw rate the EKF fuses (clock-corrected stamps, base_link z).
            self._mag_adapter.observe_gyro(sample.stamp, float(sample.gyro[2]))
        self._push_imu(sample, now)

    def _on_imu_livox(self, msg) -> None:
        t0 = time.perf_counter()
        n = self._drain_backlog("imu_livox", msg, self._handle_imu_livox)
        self._bump("imu_livox", n)
        self.stats.add("cb_imu_livox", time.perf_counter() - t0)

    def _handle_imu_livox(self, msg, fresh: bool = True) -> None:
        now = self._now()
        if not self._clock_ready(now):
            return
        sync = self._clock_sync.get("imu_livox")
        if sync is None:
            sync = self._clock_sync["imu_livox"] = converters.StreamingClockSync()
        sample = converters.imu_to_sample(
            msg,
            now,
            channel="imu_livox",
            clock=sync,
            accel_in_g=self.params.livox_accel_in_g,
            R_base_imu=self._T_base_livox[:3, :3],
            observe=fresh,
        )
        self._check_stamp("imu_livox", sample.stamp, now)
        self._push_imu(sample, now)

    def _arrival(self, now: float) -> float:
        """Arrival time for the EKF: ``now``, but never draining past a received IMU sample.

        The engine applies a measurement once its arrival clock is
        ``reorder_window_s`` past the stamp. The offline replay delivers IMU
        samples at their stamps and odometry at its receive time; live, the
        same holds (IMU pushes use their stamp, the rest ``now``) as long as
        the IMU callbacks keep up. If they lag (a busy executor), plain
        ``now`` would push the filter past samples still queued in DDS and
        they would be dropped as late. So the arrival clock is capped at the
        newest sample of every IMU that delivered within ``imu_hold_s`` plus
        one reorder window: the filter never drains past a received IMU
        sample. The wait is bounded both ways: an IMU that stops releases the
        cap after ``imu_hold_s``, and the cap never trails ``now`` by more than
        ``imu_hold_s`` (an IMU whose stamps run behind the ROS clock - an
        unsynchronized sensor clock - must not stall the filter).

        "Delivered within ``imu_hold_s``" is measured at ``_imu_checked_at``,
        the last time the output timer found the IMU queues empty, not at
        ``now``: while the executor thread is starved (nothing reads the
        queues) the IMU cannot be told apart from a dead one, so the cap
        stays; once the timer has emptied the queues and still no sample
        came, the IMU really is silent and the cap is released as before.
        """
        hold = self.params.imu_hold_s
        ref = now if self._imu_checked_at is None else min(now, self._imu_checked_at)
        held = [stamp for stamp, seen in list(self._imu_newest.values()) if ref - seen <= hold]
        return min(now, max(min(held), ref - hold) + self.params.reorder_window_s) if held else now

    def _push_imu(self, sample, now: float) -> None:
        newest = self._imu_newest.get(sample.channel)
        self._imu_newest[sample.channel] = (sample.stamp if newest is None else max(sample.stamp, newest[0]), now)
        if sample.channel == "imu":
            # The VO yaw prior integrates the primary IMU at full rate.
            self._gyro_buffer.push(sample.stamp, float(sample.gyro[2]))
        decimator = self._decimators.setdefault(sample.channel, converters.ImuDecimator(self.params.imu_rate_hz))
        out = decimator.push(sample)
        if out is not None:
            with self._engine_lock:
                # Offline an IMU sample arrives at its own stamp.
                self.engine.push(out, arrival=min(out.stamp, self._arrival(now)))

    def _on_odom(self, msg) -> None:
        t0 = time.perf_counter()
        now = self._now()
        self._check_stamp("odom", converters.stamp_of(msg), now)
        motion = self._odom_converter.push(msg)
        self._bump("odom")
        if motion is not None:
            with self._engine_lock:
                self.engine.push(motion, arrival=self._arrival(now))
        self.stats.add("cb_odom", time.perf_counter() - t0)

    # ------------------------------------------- requirement-13 sensor inputs
    def _on_mag(self, msg) -> None:
        """sensor_msgs/MagneticField -> HeadingMeasurement (channel ``mag``)."""
        t0 = time.perf_counter()
        now = self._now()
        self._check_stamp("mag", converters.stamp_of(msg), now)
        before = self._mag_adapter.rejected_field
        heading = self._mag_adapter.push(msg)
        if self._mag_adapter.rejected_field > before:
            self._bump("mag_rejected_field")
        if heading is not None:
            self._bump("mag")
            if self._mag_adapter.usable:
                with self._engine_lock:
                    self.engine.push(heading, arrival=self._arrival(now))
            else:
                # Validation-armed trust (see MagnetometerAdapter.usable):
                # withheld until the first gyro-consistent turn, and forever
                # once a turn has contradicted the calibration.
                self._bump("mag_gated")
        if self._mag_adapter.tripped:
            # The mag's heading changes disagreed with the gyro's integrated
            # turn: a miscalibration the innovation gate cannot catch once
            # the EKF has converged onto the smoothly-wrong heading
            # (measured live: 573/603 miscalibrated measurements accepted
            # before this check). A calibration is a static property, so one
            # proven contradiction faults the channel FOR THE REST OF THE
            # RUN (re-declared every message, which keeps the cooldown from
            # expiring - otherwise the channel passes probation on the next
            # straight, where a wrong calibration is indistinguishable from
            # a mount offset, and gets re-trusted; measured: 558 accepts and
            # a 40% worse map). A one-off magnetic anomaly big enough to
            # trip the 20-deg gate also drops the mag until restart: losing
            # an assist beats trusting a proven liar.
            if self._mag_adapter.inconsistent:
                self._bump("mag_gyro_inconsistent")
            with self._engine_lock:
                channel = self.engine.health.channels.get("mag")
                if channel is not None:
                    channel.declare_fault(
                        heading.stamp if heading is not None else now,
                        f"mag-gyro inconsistency {self._mag_adapter.inconsistency_deg:.0f} deg "
                        "over a turn (recalibrate: tools/calibrate_magnetometer.py)",
                    )
            self._warn_once(
                "mag_gyro",
                "magnetometer FAULTED for the rest of this run: its heading changes disagree "
                f"with the gyro by {self._mag_adapter.inconsistency_deg:.0f} deg over a turn - "
                "the calibration is wrong or something magnetic moved. "
                "Recalibrate with tools/calibrate_magnetometer.py and restart.",
            )
        self.stats.add("cb_mag", time.perf_counter() - t0)

    def _on_speed(self, msg) -> None:
        """Speed topic (Float32 m/s or TwistStamped) -> SpeedMeasurement."""
        t0 = time.perf_counter()
        now = self._now()
        if hasattr(msg, "header"):
            self._check_stamp("speed", converters.stamp_of(msg), now)
        speed = self._speed_adapter.push(msg, now)
        if speed is not None:
            self._bump("speed")
            with self._engine_lock:
                self.engine.push(speed, arrival=self._arrival(now))
        self.stats.add("cb_speed", time.perf_counter() - t0)

    def _on_wheel(self, msg) -> None:
        """Encoder counts (Int64 or JointState) -> quantization-aware SpeedMeasurement."""
        t0 = time.perf_counter()
        now = self._now()
        if hasattr(msg, "header"):
            self._check_stamp("wheel", converters.stamp_of(msg), now)
        try:
            speed = self._wheel_adapter.push(msg, now)
        except ValueError as exc:  # e.g. the configured joint is missing
            self._warn_once("wheel_convert", f"wheel_ticks: {exc}")
            return
        if speed is not None:
            self._bump("wheel")
            with self._engine_lock:
                self.engine.push(speed, arrival=self._arrival(now))
        self.stats.add("cb_wheel", time.perf_counter() - t0)

    # ----------------------------------------------------- camera inputs
    def _on_camera_info(self, camera: str, msg) -> None:
        if camera in self._camera_models:
            return  # intrinsics are static; first message wins
        K, (w, h) = converters.camera_info_to_intrinsics(msg)
        # Lens distortion from CameraInfo (plumb_bob / rational_polynomial /
        # equidistant): applied to POINTS and ray tables only, never frames.
        D = np.asarray(getattr(msg, "d", []) or [], dtype=np.float64)
        model_name = str(getattr(msg, "distortion_model", "") or "none")
        try:
            model = CameraModel(K, (w, h), D, model_name)
        except ValueError as exc:
            self._warn_once(
                f"distortion:{camera}",
                f"{camera}: {exc}; treating the stream as already rectified",
            )
            model = CameraModel(K, (w, h))
        self._register_camera_model(camera, model)

    def start_captures(self) -> None:
        """Open + start every non-ROS camera source (called by :func:`main`
        after construction; separated so tests can construct without devices).
        Raises NodeConfigError when a device cannot be opened."""
        from kart_map2d.camera.capture import CameraCapture, CaptureConfig

        for camera, cam in self.params.cameras.items():
            if cam.source == "ros2" or camera in self._captures:
                continue
            config = CaptureConfig(
                source=cam.source,
                device=cam.device,
                gstreamer_pipeline=cam.gstreamer_pipeline,
                width=cam.device_width,
                height=cam.device_height,
                fps=cam.device_fps,
                max_rate_hz=cam.max_process_rate_hz,
            )
            capture = CameraCapture(
                camera,
                config,
                on_frame=partial(self._on_capture_frame, camera),
                on_error=lambda text: self._warn(text),
                clock=self._now,
            )
            try:
                capture.start()
            except (RuntimeError, ValueError) as exc:
                raise NodeConfigError(f"cameras.{camera}: {exc}") from None
            self._captures[camera] = capture

    def _on_capture_frame(self, camera: str, stamp: float, bgr: NDArray[np.uint8]) -> None:
        """Capture-thread callback: same cheap path as the ROS image callback.

        The frame arrives at FULL capture resolution; it is downscaled once,
        here on the capture thread (not the executor), so the pending slot
        and the VO worker receive the working-resolution array exactly like
        a decoded ROS frame.
        """
        t0 = time.perf_counter()
        scale = self._camera_scale(camera)
        if scale != 1.0:
            bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        self._bump("capture_frames")
        self._handle_frame(camera, float(stamp), np.ascontiguousarray(bgr))
        self.stats.add("cb_capture", time.perf_counter() - t0)

    def _on_image(self, camera: str, msg) -> None:
        t0 = time.perf_counter()
        try:
            self._handle_image(camera, msg)
        finally:
            self.stats.add("cb_image", time.perf_counter() - t0)

    def _handle_image(self, camera: str, msg) -> None:
        # Executor thread: no pixel work here. The message itself becomes the
        # camera's pending mapping frame (newest wins, older pending frames
        # are dropped) and goes to the camera's VO worker, which decodes and
        # downscales it (~7 ms for a 1920x1200 bgra8 frame) and solves.
        stamp = converters.stamp_of(msg)
        if camera not in self._camera_frame_id:
            # First frame: remember the header frame for extrinsics-from-TF.
            frame_id = str(getattr(getattr(msg, "header", None), "frame_id", "") or "")
            if frame_id:
                self._camera_frame_id[camera] = frame_id
        self._check_stamp(camera, stamp, self._now())
        self._handle_frame(camera, stamp, msg)

    def _handle_frame(self, camera: str, stamp: float, payload) -> None:
        """Shared frame path for every camera source (msg or decoded array)."""
        with self._frame_lock:
            replaced = self._pending_frame.get(camera)
            if replaced is not None and replaced[0] > self._last_seg_stamp.get(camera, -np.inf):
                self._pending_prev[camera] = replaced
            self._pending_frame[camera] = (stamp, payload)
        self._bump("frames")
        if camera not in self._intrinsics:
            return  # VO cannot start before the intrinsics are known
        vo = self._ensure_vo(camera)
        busy = self._vo_busy[camera]
        if not busy.acquire(blocking=False):
            # Latency rule: a frame arriving mid-solve is dropped, never
            # queued. VO is pair-wise and tolerates the gap by design.
            self._bump("vo_dropped")
            return
        if self.params.vo_async:
            threading.Thread(target=self._solve_vo, args=(camera, vo, stamp, payload), daemon=True).start()
        else:
            self._solve_vo(camera, vo, stamp, payload)

    def _working_image(self, camera: str, frame) -> NDArray[np.uint8]:
        """A pending frame as the working-resolution BGR array (decoded once)."""
        if isinstance(frame, np.ndarray):
            return frame
        t0 = time.perf_counter()
        scale = self._camera_scale(camera)
        # sensor_msgs/CompressedImage has `format` and no `encoding`.
        if not hasattr(frame, "encoding") and hasattr(frame, "format"):
            _, bgr = converters.compressed_to_bgr(frame, scale)
        else:
            _, bgr = converters.image_to_bgr(frame, scale)
        self.stats.add("image_decode", time.perf_counter() - t0)
        return bgr

    def _solve_vo(self, camera: str, vo: _LiveVisualOdometry, stamp: float, frame) -> None:
        t0 = time.perf_counter()
        try:
            bgr = self._working_image(camera, frame)
            with self._frame_lock:
                # Hand the decoded pixels to the mapping step too (whichever
                # slot still holds this frame), so it is decoded only once.
                for slots in (self._pending_frame, self._pending_prev):
                    pending = slots.get(camera)
                    if pending is not None and pending[0] == stamp and not isinstance(pending[1], np.ndarray):
                        slots[camera] = (stamp, bgr)
            motion = vo.process_array(stamp, bgr)
            if motion is not None:
                self._bump("vo_solved")
                with self._engine_lock:
                    self.engine.push(motion, arrival=self._arrival(self._now()))
        except Exception as exc:  # noqa: BLE001 - a bad frame must not kill the node
            self._warn(f"VO {camera} failed at {stamp:.3f}: {type(exc).__name__}: {exc}")
        finally:
            self._vo_busy[camera].release()
            self.stats.add("vo_solve", time.perf_counter() - t0)

    def _on_pointcloud(self, msg) -> None:
        # Convert and park the scan; it is posed by the output timer once the
        # EKF has reached its stamp (~0.1 s: reorder window + scan duration).
        t0 = time.perf_counter()
        scan = converters.pointcloud_to_scan(msg)
        self._bump("lidar_scans")
        self._check_stamp("lidar", scan.stamp, self._now())
        if len(self._scan_ring) == self._scan_ring.maxlen:
            self._bump("lidar_expired")  # the oldest falls out below
        self._scan_ring.append(scan)
        self.stats.add("cb_lidar", time.perf_counter() - t0)

    def _vo_config(self) -> VoConfig:
        """VoConfig with the [odometry] tunables applied (rest = proven defaults)."""
        p = self.params
        return VoConfig(
            orb_features=int(p.vo_orb_features),
            max_corners=int(p.vo_max_corners),
            gftt_min_distance=int(p.vo_feature_spacing_px),
            min_pair_dt_s=float(p.vo_min_pair_dt_s),
        )

    def _ensure_vo(self, camera: str) -> _LiveVisualOdometry:
        if camera not in self._vo:
            cam = self.params.cameras[camera]
            K, size = self._intrinsics[camera]
            T_base_cam = self._camera_mount(camera)
            # The calibrated height fixes the monocular scale, expressed as an
            # effective ground height under this mount (same as replay).
            if cam.height_m is not None:
                ground_z = float(T_base_cam[2, 3]) - float(cam.height_m)
            else:
                ground_z = -0.10
            model = self._camera_model_scaled(camera)
            self._vo[camera] = _LiveVisualOdometry(
                camera,
                K,
                size,
                T_base_cam,
                ground_z=ground_z,
                yaw_prior=self._gyro_buffer.prior,
                config=self._vo_config(),
                # Distorted lenses: matched POINTS are undistorted before the
                # pose solve (never the frames); None for rectified streams.
                camera_model=model if model.has_distortion else None,
            )
            self._vo_busy[camera] = threading.Lock()
        return self._vo[camera]

    def set_tf_lookup(self, lookup) -> None:
        """Install the extrinsics TF hook: ``lookup(frame) -> 4x4 base_link <- frame``
        (return None / raise on failure). :func:`main` wires a tf2_ros buffer;
        tests inject a plain callable. Clears cached mounts so a hook
        installed later still takes effect for cameras not yet resolved."""
        self._tf_lookup = lookup
        self._mount_cache.clear()

    def _mount_from_tf(self, camera: str) -> NDArray[np.float64] | None:
        """TF-derived base_link <- OPTICAL mount, or None (manual fallback).

        [cameras] ``frame`` semantics: "" = never use TF; "auto" = the image
        header's frame_id (once a frame arrived); a name = that frame.
        A frame containing "optical" is used as-is (REP-103 optical); any
        other frame is the camera BODY link and gets the standard
        forward-looking optical convention on top - no camera make, no
        flip assumption.
        """
        cam = self.params.cameras[camera]
        if not cam.frame or self._tf_lookup is None:
            return None
        frame = self._camera_frame_id.get(camera, "") if cam.frame == "auto" else cam.frame
        if not frame:
            return None  # auto: no image header seen yet; caller retries later
        try:
            T_link = self._tf_lookup(frame)
        except Exception as exc:  # noqa: BLE001 - lookup failure -> manual fallback
            self._warn_once(f"tf:{camera}", f"{camera}: TF lookup base_link <- {frame!r} failed ({exc}); using the manual mount")
            return None
        if T_link is None:
            self._warn_once(f"tf:{camera}", f"{camera}: TF has no base_link <- {frame!r}; using the manual mount")
            return None
        T_link = np.asarray(T_link, dtype=np.float64)
        if "optical" in frame.lower():
            return T_link
        return T_link @ make_transform(rpy_matrix(-np.pi / 2, 0.0, -np.pi / 2), np.zeros(3))

    def _camera_mount(self, camera: str) -> NDArray[np.float64]:
        """base_link <- camera OPTICAL frame: TF when configured ([cameras]
        ``frame``), else the manual mount (body rotation, then optical) -
        the same convention as the drive node's ``camera_mount``."""
        cached = self._mount_cache.get(camera)
        if cached is not None:
            return cached
        T = self._mount_from_tf(camera)
        if T is None:
            cam = self.params.cameras[camera]
            translation = np.asarray(cam.mount_xyz, dtype=np.float64)
            optical_to_link = rpy_matrix(-np.pi / 2, 0.0, -np.pi / 2)
            rpy = np.radians(np.asarray(cam.mount_rpy_deg, dtype=np.float64))
            if np.any(rpy != 0.0):
                optical_to_link = rpy_matrix(*rpy) @ optical_to_link
            T = make_transform(optical_to_link, translation)
            # "auto" without a seen frame yet: do not cache - a header may
            # still arrive and upgrade the mount before VO/segmentation start.
            cam_frame_pending = self.params.cameras[camera].frame == "auto" and camera not in self._camera_frame_id
            if cam_frame_pending:
                return T
        self._mount_cache[camera] = T
        return T

    # --------------------------------------------------------------- timers
    def _on_output_timer(self) -> None:
        """50 Hz: drain the engine up to the reorder horizon, publish pose, pose scans."""
        t0 = time.perf_counter()
        now = self._now()
        # IMU samples that queued up while this thread was starved go in
        # first: the drain below must not pass them (see _pull_imu_queues).
        self._pull_imu_queues(now)
        with self._engine_lock:
            # The engine drains on push(); between sparse messages this timer
            # advances the arrival clock so queued measurements older than the
            # reorder window still reach the EKF on time. Uses the engine's
            # private drain on purpose: the engine API is frozen and adding a
            # public tick() is not this wrapper's call to make.
            self.engine._latest_arrival = max(self.engine._latest_arrival, self._arrival(now))
            self.engine._drain(self.engine._latest_arrival - self.engine.reorder_window_s)
            have_state = self.engine.ekf.stamp is not None
            snap = self.engine.ekf.snapshot() if have_state else None
            mode, health = self._live_health(now)
            queue = len(self.engine._queue)
            if len(self.engine.ekf.updates) > 4096:
                self._fold_ekf_updates()
        self._queue_max["engine"] = max(self._queue_max["engine"], queue)
        self._queue_max["scan_ring"] = max(self._queue_max["scan_ring"], len(self._scan_ring))
        if snap is not None:
            self._publish_pose(snap)
            self._append_path(snap)
        if self._scan_ring:
            self._dispatch_scans()
        self._publish_health(now, mode, health, queue)
        self.stats.add("timer_output", time.perf_counter() - t0)

    def _live_health(self, now: float) -> tuple[str, dict[str, str]]:
        """(mode, per-channel status) as of the node clock ``now`` (engine lock held).

        The engine judges staleness at measurement stamps (offline
        semantics), so when EVERY sensor goes silent nothing ticks it and it
        keeps reporting the last statuses ("ok") for the whole blackout. For
        the published health a channel whose newest stamp is older than its
        timeout plus the fusion latency (reorder window + max lateness) is
        reported ``stale`` and the mode is derived from the reported statuses.
        Reporting only: the engine's own health (what the EKF uses) is untouched.
        """
        margin = self.params.reorder_window_s + self.params.max_lateness_s
        health: dict[str, str] = {}
        timed: set[str] = set()  # channels that can go stale (ZUPT has no timeout: derived from an IMU)
        for name, ch in self.engine.health.channels.items():
            status = ch.status.value
            if np.isfinite(ch.config.timeout_s):
                timed.add(name)
                if status in ("ok", "degraded") and now - ch.last_stamp > ch.config.timeout_s + margin:
                    status = "stale"
            health[name] = status
        roles = {channel_role(n) for n, st in health.items() if n in timed and st in ("ok", "degraded")}
        parts = [role for role in ("absolute", "heading", "odometric", "inertial") if role in roles]
        if not parts:
            mode = "blind"
        elif parts == ["inertial"]:
            mode = "inertial-only (dead reckoning)"
        else:
            mode = "+".join(parts)
        return mode, health

    def _fold_ekf_updates(self) -> None:
        """Count the EKF's per-update log into ``_ekf_updates`` and empty it (engine lock held).

        The EKF appends one UpdateLog per update (~500/s with two IMUs) and
        nothing in the live path reads it back - only offline evaluation
        does - so the node keeps per-channel accepted / rejected totals and
        bounds the list instead of letting it grow for the whole session.
        """
        log = self.engine.ekf.updates
        for entry in log:
            slot = self._ekf_updates.get(entry.channel)
            if slot is None:
                slot = self._ekf_updates[entry.channel] = [0, 0]
            slot[0 if entry.accepted else 1] += 1
        log.clear()

    def _publish_pose(self, snap) -> None:
        msg = self.msgs.PoseStamped()
        msg.header.frame_id = self.params.map_frame
        converters.set_stamp(msg.header.stamp, snap.stamp)
        x, y, yaw = float(snap.x[0]), float(snap.x[1]), float(snap.x[2])
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = 0.0
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = float(np.sin(0.5 * yaw))
        msg.pose.orientation.w = float(np.cos(0.5 * yaw))
        self.pub_pose.publish(msg)

    def _append_path(self, snap) -> None:
        x, y, yaw = float(snap.x[0]), float(snap.x[1]), float(snap.x[2])
        if self._path_xyyaw:
            _, px, py, _ = self._path_xyyaw[-1]
            if np.hypot(x - px, y - py) < self.params.path_min_step_m:
                return
        self._path_xyyaw.append((float(snap.stamp), x, y, yaw))
        self._path_dirty = True

    def runtime_stats(self, queue: int | None = None) -> dict:
        """Counters, latency (mean / p95 / max ms), queue sizes and memory."""
        rss = _rss_mb()
        if rss is not None:
            self._rss_first = rss if self._rss_first is None else self._rss_first
            self._rss_max = rss if self._rss_max is None else max(self._rss_max, rss)
        with self._engine_lock:
            if queue is None:
                queue = len(self.engine._queue)
            ekf_updates = {k: list(v) for k, v in self._ekf_updates.items()}
            for entry in self.engine.ekf.updates:
                slot = ekf_updates.setdefault(entry.channel, [0, 0])
                slot[0 if entry.accepted else 1] += 1
            late_drops = int(self.engine.ekf.late_drops)
            poses = len(self.engine.trajectory.stamp)
        with self._counter_lock:
            counters = dict(self.counters)
        return {
            "counters": counters,
            "latency_ms": self.stats.summary(),
            # Per channel: [accepted, rejected by the innovation gate]; late
            # drops = measurements older than the EKF's max lateness (0.25 s).
            "ekf": {"updates": ekf_updates, "late_drops": late_drops, "trajectory_poses": poses},
            "queues": {
                "engine": queue,
                "engine_max": self._queue_max["engine"],
                "scan_ring": len(self._scan_ring),
                "scan_ring_max": self._queue_max["scan_ring"],
                "scan_ring_size": self._scan_ring.maxlen,
                "pending_frames": len(self._pending_frame) + len(self._pending_prev),
                "vo_in_flight": sum(1 for lock in list(self._vo_busy.values()) if lock.locked()),
                "mapping_busy": bool(self._map_worker.busy) if self._map_worker else False,
                "rebuild_store_mb": round(self._store_bytes / 1e6, 1),
                "rebuild_store_complete": self._store_complete,
                "loop_observations": len(self._observations),
            },
            "rss_mb": rss,
            "rss_first_mb": self._rss_first,
            "rss_max_mb": self._rss_max,
        }

    def _publish_health(self, now: float, mode: str, health: dict, queue: int | None = None) -> None:
        key = (mode, tuple(sorted(health.items())))
        if key == self._last_health and now - self._last_health_pub < self.params.health_period_s:
            return
        self._last_health = key
        self._last_health_pub = now
        msg = self.msgs.String()
        doc = {"stamp": now, "mode": mode, "health": health, "method": self.method}
        doc.update(self.runtime_stats(queue))
        msg.data = json.dumps(doc, sort_keys=True)
        self.pub_health.publish(msg)

    # ------------------------------------------------------ trajectory access
    def _pose_if_trusted(self, stamp: float) -> NDArray[np.float64] | None:
        """Interpolated [x, y, yaw] at ``stamp``, or None if unfit for mapping.

        Bisects the trajectory lists instead of ``Trajectory.pose_at`` (which
        converts the whole run to arrays on every call - milliseconds under
        the engine lock after a few minutes); same interpolation, yaw taken
        along the short way between the two neighbours.
        """
        with self._engine_lock:
            traj = self.engine.trajectory
            stamps = traj.stamp
            if not stamps or not stamps[0] <= stamp <= stamps[-1]:
                return None
            if not self.engine.pose_trusted(stamp):
                return None
            i = bisect.bisect_left(stamps, stamp)
            if stamps[i] == stamp or i == 0:
                return np.asarray(traj.x[i][:3], dtype=np.float64).copy()
            t_a, t_b = stamps[i - 1], stamps[i]
            a, b = traj.x[i - 1][:3], traj.x[i][:3]
        w = (stamp - t_a) / (t_b - t_a)
        dyaw = (b[2] - a[2] + np.pi) % (2.0 * np.pi) - np.pi
        return np.array([a[0] + w * (b[0] - a[0]), a[1] + w * (b[1] - a[1]), a[2] + w * dyaw])

    def _trajectory_window(self, t_lo: float, t_hi: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """(stamps, xyyaw) of the EKF trajectory within [t_lo, t_hi] (+1 pose each side)."""
        with self._engine_lock:
            traj = self.engine.trajectory
            i = max(bisect.bisect_left(traj.stamp, t_lo) - 1, 0)
            j = min(bisect.bisect_right(traj.stamp, t_hi) + 1, len(traj.stamp))
            stamps = np.asarray(traj.stamp[i:j], dtype=np.float64)
            xyyaw = np.asarray([x[:3] for x in traj.x[i:j]], dtype=np.float64).reshape(-1, 3)
        return stamps, xyyaw

    def _dispatch_scans(self) -> None:
        """Pose the newest scan the EKF has reached; hand it to the grid.

        Runs on the output timer right after the engine drain. Older posed
        scans in the ring are superseded (counted): the grid takes one scan
        per mapping tick by design.
        """
        with self._engine_lock:
            stamps = self.engine.trajectory.stamp
            t_end = stamps[-1] if stamps else None
        if t_end is None:
            return
        ready = [s for s in self._scan_ring if s.stamp <= t_end]
        if not ready:
            return
        newest = max(ready, key=lambda s: s.stamp)
        waiting = [s for s in self._scan_ring if s.stamp > t_end]
        self._scan_ring.clear()
        self._scan_ring.extend(waiting)
        self._bump("lidar_superseded", len(ready) - 1)
        pose = self._pose_if_trusted(newest.stamp)
        if pose is None:
            self._bump("lidar_untrusted")
            return
        self._bump("lidar_posed")
        with self._scan_lock:
            self._grid_scan = (newest, pose)  # the mapping step takes the newest

    # --------------------------------------------------------- segmentation
    def _resolve_seg_mode(self) -> str:
        """Camera-evidence backend: ``"neural"``, ``"classical"`` or ``"off"``.

        Resolved once at construction; ``auto`` probes torch / transformers /
        the checkpoint with ``find_spec`` (resolve_seg_mode), so resolving
        never imports torch. An unknown mode is a configuration error.
        """
        mode = self.params.seg_mode
        if mode not in SEG_MODES:
            raise ValueError(f"seg_mode {mode!r} is not one of {', '.join(SEG_MODES)}")
        if not self.params.seg_enabled:
            return "off"
        return resolve_seg_mode(mode)

    def prepare_segmentation(self) -> str:
        """Create (and for neural, load) the backend now; return the mode in effect.

        :func:`main` calls this before spinning: loading SegFormer takes
        seconds, and inside the mapping timer that stall would also hold up
        the IMU callbacks sharing the executor. When the neural backend
        fails to load under ``seg_mode: auto`` the node warns and continues
        with classical; under an explicit ``neural`` the error propagates so
        a misconfigured kart never maps with a backend nobody chose.
        """
        if self.segmenter is not None or self.seg_mode != "neural" or self._neural is not None:
            return self.seg_mode
        try:
            neural = get_segmenter("neural")  # the only place torch is imported
            neural.load()
        except Exception as exc:  # noqa: BLE001 - auto must degrade, not die
            if self.params.seg_mode != "auto":
                raise
            self._warn(f"neural segmentation unavailable ({type(exc).__name__}: {exc}); using classical")
            self.seg_mode = "classical"
            return self.seg_mode
        self._neural = neural
        return self.seg_mode

    def _segmenter_for(self, camera: str):
        """The segmentation backend for one camera (created on first use)."""
        if self.segmenter is not None:
            return self.segmenter
        if self.seg_mode == "neural":
            self.prepare_segmentation()  # no-op once loaded; auto may fall back here
        if self.seg_mode == "neural":
            return self._neural
        segmenter = self._segmenters.get(camera)
        if segmenter is None:
            # The classical backend keeps a temporal ground model per camera
            # and derives its seed trapezoid / horizon from this camera's K
            # (already scaled to the working image) and mount - exactly what
            # the offline pipeline passes it - so each camera gets its own.
            # The learned/loaded kart mask goes into the CONSTRUCTOR (the
            # classical backend then also neutralises its breathing band);
            # a mask learned later rebuilds the segmenter (_apply_kart_mask).
            K, _ = self._intrinsics[camera]
            mask = self._kart_masks.get(camera)
            segmenter = get_segmenter(
                "classical", K=K, T_base_cam=self._camera_mount(camera),
                kart_mask=None if mask is None else mask,
            )
            self._segmenters[camera] = segmenter
        return segmenter

    # -------------------------------------------------- per-camera kart mask
    def _load_mask_file(self, camera: str, shape: tuple[int, int]) -> NDArray[np.bool_] | None:
        """cameras.<name>.mask_file (white = kart) resized to the working shape."""
        cam = self.params.cameras[camera]
        if not cam.mask_file:
            return None
        raw = cv2.imread(str(cam.mask_file), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            self._warn_once(f"mask:{camera}", f"{camera}: cannot read mask_file {cam.mask_file!r}; using the geometric mask")
            return None
        if raw.shape != shape:
            raw = cv2.resize(raw, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        return raw > 0

    #: Frames sampled for online mask learning (the offline learner's default).
    _MASK_LEARN_FRAMES = 48
    #: Working width for the stored learner frames [px] (the learner's own
    #: internal resolution, so storing more would be wasted memory).
    _MASK_LEARN_WIDTH = 320

    def _collect_mask_frame(self, camera: str, bgr: NDArray[np.uint8]) -> None:
        """Store a downscaled frame for the online kart-mask learner; learn when full.

        Mapping-worker thread only (called from the mapping step, so the
        one-time ~1 s learn cost never touches the executor). Frames are kept
        at the learner's own work resolution (~0.2 MB each at 320 px), and
        the learner's motion filter decides which of them teach.
        """
        cam = self.params.cameras[camera]
        if (
            not cam.learn_mask
            or cam.mask_file
            or camera in self._mask_learn_done
            or self.seg_mode == "off"
        ):
            return
        frames = self._mask_learn_frames.setdefault(camera, [])
        h, w = bgr.shape[:2]
        ww = min(self._MASK_LEARN_WIDTH, w)
        wh = max(int(round(h * ww / w)), 1)
        frames.append(cv2.resize(bgr, (ww, wh), interpolation=cv2.INTER_AREA))
        if len(frames) < self._MASK_LEARN_FRAMES:
            return
        self._mask_learn_done.add(camera)
        try:
            self._learn_kart_mask(camera, frames, (h, w))
        except Exception as exc:  # noqa: BLE001 - learning must never kill mapping
            self._warn(f"{camera}: kart-mask learning failed ({type(exc).__name__}: {exc}); keeping the geometric mask")
        finally:
            self._mask_learn_frames.pop(camera, None)

    def _learn_kart_mask(self, camera: str, frames: list[NDArray[np.uint8]], working_shape: tuple[int, int]) -> None:
        """The offline learner (classical fix), run once online per camera.

        Mirrors ``load_or_learn_kart_mask``: bootstrap drivable/obstacle
        masks from a throwaway classical segmenter at the learner resolution,
        ``learn_kart_mask`` over the collected frames, then the breathing
        margin and the union with the geometric mask. A refused learn (too
        little motion, implausibly large mask) keeps the geometric fallback.
        """
        from kart_map2d.perception.classical_ground import ClassicalGroundSegmenter, kart_body_mask, learn_kart_mask

        t0 = time.perf_counter()
        K, size = self._intrinsics[camera]
        lw = frames[0].shape[1]
        K_small = np.array(K, dtype=np.float64, copy=True)
        K_small[:2] *= lw / float(size[0])
        boot = ClassicalGroundSegmenter(K=K_small, T_base_cam=self._camera_mount(camera))
        results = [boot.segment(f) for f in frames]
        mask_small = learn_kart_mask(frames, [r.drivable_mask for r in results], [r.obstacle_mask for r in results])
        if mask_small is None:
            self._warn(f"{camera}: kart-mask learning refused (not enough scene motion yet or degenerate); geometric mask kept")
            return
        mask = cv2.resize(mask_small.astype(np.uint8), (working_shape[1], working_shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        if mask.any():
            # The silhouette breathes (suspension, steering, anti-aliasing):
            # the offline margin, scaled from its 600-row reference.
            k = 2 * max(int(round(20 * working_shape[0] / 600.0)), 1) + 1
            mask = cv2.dilate(mask.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
        mask = mask | kart_body_mask(*working_shape)
        self._kart_masks[camera] = mask
        self._segmenters.pop(camera, None)  # rebuilt with the learned mask on next use
        self._bump("kart_masks_learned")
        self._warn(
            f"{camera}: kart-body mask learned online ({float(mask.mean()):.1%} of the image, "
            f"{time.perf_counter() - t0:.2f} s); classical segmenter rebuilt with it"
        )

    def _ensure_kart_mask(self, camera: str, shape: tuple[int, int]) -> NDArray[np.bool_] | None:
        """The camera's mask at working shape (file > learned > None=geometric)."""
        if camera in self._kart_masks:
            mask = self._kart_masks[camera]
            return mask if mask is None or mask.shape == shape else None
        mask = self._load_mask_file(camera, shape)
        self._kart_masks[camera] = mask  # may be None: geometric default until learned
        return mask

    def _on_mapping_timer(self) -> None:
        """Slow timer (executor thread): start one mapping step, never queue one.

        With ``mapping_async`` the step runs on the mapping worker; a tick
        that finds the previous step still running is skipped and counted
        (``mapping_skipped_busy``) - the next tick picks up the newest frames
        anyway, so nothing backs up.
        """
        if self._map_worker is None:
            with self.stats.time("timer_mapping"):
                self._mapping_step()
        elif not self._map_worker.submit("timer_mapping", self._mapping_step):
            self._bump("mapping_skipped_busy")

    def _mapping_step(self) -> None:
        """Segment the newest pending frame per camera, grow the grid, publish grid / path."""
        now = self._now()
        try:
            if self.params.seg_enabled:
                self._insert_pending_frames()
            self._insert_pending_scan()
        except Exception as exc:  # noqa: BLE001 - mapping must not kill localization
            self._warn(f"mapping step failed: {type(exc).__name__}: {exc}")
        if self.grid.shape[0] and now - self._last_grid_pub >= self.params.grid_publish_period_s:
            self._last_grid_pub = now
            msg = self.msgs.OccupancyGrid()
            with self._map_lock:
                converters.fill_occupancy_grid(msg, self.grid, now, self.params.map_frame)
            self.pub_grid.publish(msg)
        if self._path_dirty:
            self._path_dirty = False
            self.pub_path.publish(self._path_msg())
        if self._corridor_due:
            # A corridor tick found this worker busy: do it now, same thread.
            self._corridor_due = False
            with self.stats.time("timer_corridor"):
                self._corridor_step()

    #: Longest a mapping stage waits for running VO solves [s] (localization first).
    _VO_YIELD_MAX_S = 0.25

    def _yield_to_vo(self) -> None:
        """Let running VO solves finish before a heavy mapping stage (bounded wait).

        Python threads share one interpreter lock, so a segmentation or grid
        insert running next to a VO solve slows the solve down, and a VO
        increment that reaches the EKF too late is dropped. Localization
        comes first: the mapping worker waits (at most ``_VO_YIELD_MAX_S``)
        while a solve is in flight. Only on the mapping worker - with
        ``mapping_async: false`` the step runs on the executor, which must
        never wait.
        """
        if self._map_worker is None or not self._vo_busy:
            return
        t0 = time.perf_counter()  # not monotonic(): ~15 ms resolution on Windows
        deadline = t0 + self._VO_YIELD_MAX_S
        while any(lock.locked() for lock in list(self._vo_busy.values())) and time.perf_counter() < deadline:
            time.sleep(0.003)
        waited = time.perf_counter() - t0
        if waited > 0.003:
            self.stats.add("mapping_yield_to_vo", waited)

    def _next_mapping_frame(self, camera: str, entries) -> tuple[float, Any, NDArray[np.float64]] | None:
        """The newest pending frame of ``camera`` that is new and has a trusted pose."""
        last = self._last_seg_stamp.get(camera, -np.inf)
        for stamp, frame in entries:  # newest first
            if stamp <= last:
                continue
            pose = self._pose_if_trusted(stamp)
            if pose is not None:
                return stamp, frame, pose
        return None  # keep pending: a pose may become available shortly

    def _insert_pending_frames(self) -> None:
        with self._frame_lock:
            cameras = list(dict.fromkeys([*self._pending_frame, *self._pending_prev]))
            pending = {
                cam: [e for e in (self._pending_frame.get(cam), self._pending_prev.get(cam)) if e is not None]
                for cam in cameras
            }
        for camera, entries in pending.items():
            if camera not in self._intrinsics:
                continue
            picked = self._next_mapping_frame(camera, entries)
            if picked is None:
                continue
            stamp, frame, pose = picked
            self._last_seg_stamp[camera] = stamp
            bgr = self._working_image(camera, frame)
            mask = self._ensure_kart_mask(camera, bgr.shape[:2])
            self._collect_mask_frame(camera, bgr)
            self._yield_to_vo()
            t0 = time.perf_counter()
            seg = self._segmenter_for(camera).segment(bgr)
            self.stats.add("segment", time.perf_counter() - t0)
            class_map = getattr(seg, "class_map", None)
            if self.seg_mode == "neural" and mask is not None and class_map is not None and class_map.shape == mask.shape:
                # The shared neural backend cannot hold one mask per camera:
                # stamp the kart body post hoc (the classical-fix mechanism
                # for cached neural maps), BEFORE the masks are derived.
                from kart_map2d.perception.segmentation import KART_ID

                class_map[mask] = KART_ID
            K, _ = self._intrinsics[camera]
            T_base_cam = self._camera_mount(camera)
            drivable, obstacle = seg.drivable_mask, seg.obstacle_mask
            # Typically-moving classes go to the decaying dynamic layer, as offline.
            dynamic = np.isin(class_map, _DYNAMIC_IDS) if class_map is not None else None
            if dynamic is not None and not dynamic.any():
                dynamic = None
            # Distortion-aware ray table (pinhole-identical when D == 0).
            rays = self._camera_model_scaled(camera).pixel_rays(drivable.shape)
            self._yield_to_vo()
            t0 = time.perf_counter()
            with self._map_lock:
                # The stamp is ALWAYS passed (classical fix): contacts landing
                # on camera-verified-free cells go to the decaying dynamic
                # layer (unknown-class movers, far-facade smear).
                self.grid.insert_camera(
                    drivable,
                    obstacle,
                    K,
                    T_base_cam,
                    pose,
                    max_range=self.params.camera_max_range_m,
                    dynamic_mask=dynamic,
                    stamp=float(stamp),
                    rays=rays,
                )
                self._remember_frame(camera, float(stamp), K, T_base_cam, drivable, obstacle, dynamic)
            self.stats.add("grid_camera_insert", time.perf_counter() - t0)
            self._bump("seg_inserted")

    def _insert_pending_scan(self) -> None:
        with self._scan_lock:
            item, self._grid_scan = self._grid_scan, None
        if item is None:
            return
        scan, pose = item
        t0 = time.perf_counter()
        R, t = self._T_base_livox[:3, :3], self._T_base_livox[:3, 3]
        pts_base = scan.points.astype(np.float64) @ R.T + t
        with self._map_lock:
            self.grid.insert_lidar(pts_base, pose, stamp=float(scan.stamp))
            self._remember(("lidar", float(scan.stamp), scan.points, None), scan.points.nbytes)
        self.stats.add("grid_lidar_insert", time.perf_counter() - t0)
        self._bump("lidar_inserted")

    # ------------------------------------------- loop-closure / rebuild evidence
    def _remember(self, record: tuple, nbytes: int) -> None:
        """Keep one grid insert for a post-loop-closure rebuild, within budget (map lock held)."""
        if not self._store_complete:
            return
        if self._store_bytes + nbytes > self.params.rebuild_budget_mb * 1e6:
            self._store_complete = False
            self._store = []
            self._store_bytes = 0
            self._warn(
                f"rebuild store passed rebuild_budget_mb={self.params.rebuild_budget_mb}: a loop-closed "
                "save will keep the grid at the live poses"
            )
            return
        self._store.append(record)
        self._store_bytes += nbytes

    def _remember_frame(self, camera, stamp, K, T_base_cam, drivable, obstacle, dynamic) -> None:
        key = (camera, drivable.shape)
        observer = self._ground_obs.get(key)
        if observer is None:
            observer = self._ground_obs[key] = _GroundObserver(self._camera_model_scaled(camera), T_base_cam, drivable.shape)
        drv_cells, obs_cells = observer.observe(drivable, obstacle)
        self._observations.append((stamp, drv_cells, obs_cells))
        bits = [np.packbits(drivable), np.packbits(obstacle), None if dynamic is None else np.packbits(dynamic)]
        nbytes = sum(b.nbytes for b in bits if b is not None) + drv_cells.nbytes + obs_cells.nbytes
        self._remember(("camera", stamp, camera, drivable.shape, *bits), nbytes)

    def _rebuild_grid(self, store: list[tuple], stamps, xyyaw) -> EvidenceGrid:
        """Re-insert every remembered frame / scan at the corrected poses (offline order)."""
        from kart_map2d.pipeline import interp_poses

        grid = EvidenceGrid(
            cell=self.params.grid_cell_m,
            dyn_half_life=self.params.dyn_half_life_s,
            kart_footprint=tuple(self.params.kart_footprint) if self.params.kart_footprint else None,
            min_contact_range_m=self.params.min_contact_range_m,
        )
        R, t = self._T_base_livox[:3, :3], self._T_base_livox[:3, 3]
        records = sorted(store, key=lambda r: r[1])
        poses = interp_poses(np.array([r[1] for r in records]), stamps, xyyaw) if records else []
        for record, pose in zip(records, poses):
            if record[0] == "camera":
                _, stamp, camera, shape, drv, obs, dyn = record
                n = int(np.prod(shape))
                unpack = lambda bits: np.unpackbits(bits, count=n).astype(bool).reshape(shape)  # noqa: E731
                K, _ = self._intrinsics[camera]
                rays = self._camera_model_scaled(camera).pixel_rays(shape)
                grid.insert_camera(
                    unpack(drv),
                    unpack(obs),
                    K,
                    self._camera_mount(camera),
                    pose,
                    max_range=self.params.camera_max_range_m,
                    dynamic_mask=None if dyn is None else unpack(dyn),
                    stamp=stamp,
                    rays=rays,
                )
            else:
                _, stamp, points, _ = record
                grid.insert_lidar(points.astype(np.float64) @ R.T + t, pose, stamp=stamp)
        return grid

    # ----------------------------------------------------------------- save
    def _on_save_map(self, request, response):
        ok, message, _ = self.save_map("service")
        response.success = bool(ok)
        response.message = message
        return response

    def _sensors_used(self) -> list[str]:
        """Channels that actually delivered data, named like the offline sources."""
        c, p = self.counters, self.params
        used = [
            name
            for name, n in (
                ("imu", c["imu"]),
                ("imu_livox", c["imu_livox"]),
                ("odom", c["odom"]),
                ("mag", c["mag"]),
                ("speedometer", c["speed"]),
                ("wheel", c["wheel"]),
            )
            if n
        ]
        used += [f"vo_{cam}" for cam in p.cameras if cam in self._vo]
        if c["lidar_scans"]:
            used.append("lidar")
        return used

    def _output_dir(self) -> Path:
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        base = Path(self.params.output_dir) / f"{self.params.map_name}_{self.method}_{stamp}"
        out, k = base, 2
        while out.exists():
            out, k = base.with_name(f"{base.name}_{k}"), k + 1
        return out

    def save_map(self, reason: str = "service") -> tuple[bool, str, str | None]:
        """Write a map package now: ``(ok, message, directory)``.

        Snapshots the fused trajectory and grid under their locks
        (milliseconds), then - without holding them - runs loop closure over
        the per-frame ground evidence, re-grids at the corrected poses when a
        closure moved the trajectory (and the rebuild store is complete),
        extracts the corridor and racing line, and writes the package via
        :func:`kart_map2d.mapping.map_package.save_map_package`. Safe to call
        from another thread while the node keeps mapping; a second call
        while one runs is refused.
        """
        if not self._save_lock.acquire(blocking=False):
            return False, "a map save is already running", None
        t_start = time.perf_counter()
        try:
            with self._engine_lock:
                stamps, states, p_diag, supported = self.engine.trajectory.arrays()
                mode = self._live_health(self._now())[0]
            if len(stamps) < 2:
                return False, "nothing to save yet: the fused trajectory is empty (no sensor data received?)", None
            with self._map_lock:
                grid = copy.deepcopy(self.grid)
                observations = list(self._observations)
                store = list(self._store) if self._store_complete else None
            timing = {"snapshot_s": round(time.perf_counter() - t_start, 3)}
            xyyaw = states[:, :3].copy()

            lc: dict = {"enabled": bool(self.params.loop_closure), "observations": len(observations)}
            accepted = 0
            if self.params.loop_closure and len(observations) >= 3:
                from kart_map2d.mapping.loop_closure import KeyframeObservation, detect_and_close

                t0 = time.perf_counter()
                cell = 0.20  # _GroundObserver's dedupe cell
                kf = [
                    KeyframeObservation(
                        stamp=s,
                        drivable_xy=d.astype(np.float64) * cell + cell / 2,
                        obstacle_xy=o.astype(np.float64) * cell + cell / 2,
                    )
                    for s, d, o in sorted(observations, key=lambda r: r[0])
                ]
                sigma = np.sqrt(p_diag[:, 0] + p_diag[:, 1])
                closed, lc["ran_in"] = _run_isolated(
                    detect_and_close, (stamps, xyyaw, kf), {"position_sigma": sigma}, self.params.loop_closure_subprocess
                )
                accepted = len(closed.accepted)
                lc.update(
                    accepted=accepted,
                    candidates=len(closed.closures),
                    closures=[
                        {"t_a_s": round(c.stamp_a - float(stamps[0]), 1), "t_b_s": round(c.stamp_b - float(stamps[0]), 1),
                         "score": round(float(c.score), 3), "chi2": None if c.chi2 is None else round(float(c.chi2), 1),
                         "accepted": bool(c.accepted)}
                        for c in closed.closures
                    ],
                )
                if accepted:
                    shift = np.linalg.norm(closed.corrected_xyyaw[:, :2] - xyyaw[:, :2], axis=1)
                    lc["max_correction_m"] = round(float(shift.max()), 3)
                    xyyaw = closed.corrected_xyyaw
                    if store is not None:
                        grid = self._rebuild_grid(store, stamps, xyyaw)
                        lc["grid"] = "rebuilt at the loop-closed poses"
                    else:
                        lc["grid"] = "kept at the live poses (rebuild store over rebuild_budget_mb)"
                timing["loop_closure_s"] = round(time.perf_counter() - t0, 3)
            elif self.params.loop_closure:
                lc["skipped"] = "fewer than 3 camera observations"

            out_dir = self._output_dir()
            p = self.params
            source = {
                "kind": "live",
                "node": getattr(self.node, "get_fully_qualified_name", lambda: "/kart_map2d")(),
                "reason": reason,
                "method_origin": self.method_origin,
                "ros_time_first_s": float(stamps[0]),
                "ros_time_last_s": float(stamps[-1]),
                "mode_at_save": mode,
                "seg_mode": self.seg_mode,
                "topics": {
                    "imu": p.imu_topic, "imu_livox": p.livox_imu_topic, "odom": p.odom_topic, "lidar": p.lidar_topic,
                    "mag": p.mag_topic, "speed": p.speed_topic, "wheel": p.wheel_topic,
                    "cameras": {name: (cam.image_topic if cam.source == "ros2" else f"{cam.source}:{cam.device}") for name, cam in p.cameras.items()},
                },
            }
            t0 = time.perf_counter()
            meta = map_package.save_map_package(
                out_dir,
                method=self.method,
                grid=grid,
                stamps=stamps,
                xyyaw=xyyaw,
                supported=supported,
                source=source,
                sensors=self._sensors_used(),
                loop_closed=bool(accepted),
                corridor_spacing_m=p.corridor_spacing_m,
                corridor_close_cells=p.corridor_close_cells,
                build={"loop_closure": lc, "node_timing_s": timing},
            )
            timing["package_s"] = round(time.perf_counter() - t0, 3)
            timing["total_s"] = round(time.perf_counter() - t_start, 3)
            self._bump("saves")
            if accepted:
                # Informative (not a package file): the EKF track before the
                # closures moved it, same format as trajectory.csv.
                map_package.write_trajectory_csv(out_dir / "trajectory_ekf.csv", stamps, states[:, :3], supported)
            stats = self.runtime_stats()
            stats.update(save_timing_s=timing, loop_closure=lc)
            (out_dir / "live_stats.json").write_text(json.dumps(map_package._json_safe(stats), indent=1) + "\n", encoding="ascii")
            corridor = meta.get("corridor") or {}
            summary = (
                f"saved {out_dir}: method {self.method}, {len(stamps)} poses, "
                f"corridor {'ok' if meta.get('corridor') else 'MISSING'}"
                f"{' (%.1f m)' % corridor['length_m'] if 'length_m' in corridor else ''}, "
                f"racing line {'ok' if meta.get('racing_line') else 'MISSING'}, "
                f"loop closures {accepted}, "
                f"{timing['total_s']:.1f} s"
            )
            if meta.get("problems"):
                summary += "; problems: " + "; ".join(meta["problems"])
            self.last_save = {"dir": str(out_dir), "meta": meta, "timing_s": timing, "ok": True}
            return True, summary, str(out_dir)
        except Exception as exc:  # noqa: BLE001 - report through the service, never crash the node
            self.last_save = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            return False, f"save failed: {type(exc).__name__}: {exc}", None
        finally:
            self._save_lock.release()

    def _path_msg(self):
        msg = self.msgs.Path()
        msg.header.frame_id = self.params.map_frame
        rows = list(self._path_xyyaw)
        converters.set_stamp(msg.header.stamp, rows[-1][0] if rows else 0.0)
        poses = []
        for stamp, x, y, yaw in rows:
            pose = self.msgs.PoseStamped()
            pose.header.frame_id = self.params.map_frame
            converters.set_stamp(pose.header.stamp, stamp)
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = float(np.sin(0.5 * yaw))
            pose.pose.orientation.w = float(np.cos(0.5 * yaw))
            poses.append(pose)
        msg.poses = poses
        return msg

    def _on_corridor_timer(self) -> None:
        """Corridor tick (executor thread): extract on the mapping worker, never here.

        A tick that finds the worker busy marks the corridor due; the running
        mapping step then extracts it before it finishes.
        """
        if self._map_worker is None:
            with self.stats.time("timer_corridor"):
                self._corridor_step()
        elif not self._map_worker.submit("timer_corridor", self._corridor_step):
            self._corridor_due = True
            self._bump("corridor_deferred")

    def _corridor_step(self) -> None:
        """Extract the drivable corridor around the driven path; publish on change."""
        rows = list(self._path_xyyaw)  # one C-level copy: the output timer appends concurrently
        if not self.grid.shape[0] or len(rows) < 4:
            return
        t0 = time.perf_counter()
        traj_xy = np.array([[x, y] for _, x, y, _ in rows])
        try:
            # Same hole-closing as pipeline.closed_classes (duplicated: the
            # pipeline module drags offline-only imports such as kiss-icp).
            with self._map_lock:
                classes = np.asarray(self.grid.classify()).T  # (row=y, col=x)
                origin_xy = tuple(np.asarray(self.grid.origin, dtype=float))
            k = 2 * int(self.params.corridor_close_cells) + 1
            if k > 1 and classes.size:
                drivable = (classes == CLASS_DRIVABLE).astype(np.uint8)
                drivable = cv2.morphologyEx(drivable, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
                closed = np.where(drivable > 0, CLASS_DRIVABLE, 0).astype(np.uint8)
                closed[classes == CLASS_OBSTACLE] = CLASS_OBSTACLE
                classes = closed
            corridor = extract_corridor(
                classes,
                traj_xy,
                spacing_m=self.params.corridor_spacing_m,
                origin_xy=origin_xy,
                resolution_m=float(self.grid.cell),
            )
        except Exception as exc:  # noqa: BLE001 - not enough evidence yet is normal
            self._warn(f"corridor extraction skipped: {type(exc).__name__}: {exc}")
            return
        finally:
            self.stats.add("corridor_extract", time.perf_counter() - t0)
        center = np.asarray(corridor.centerline)
        previous = self._corridor_centerline
        if (
            previous is not None
            and previous.shape == center.shape
            and np.allclose(previous, center, atol=0.5 * float(self.grid.cell))
        ):
            return  # unchanged within half a cell: do not re-publish
        self._corridor_centerline = center
        msg = self.msgs.Path()
        msg.header.frame_id = self.params.map_frame
        converters.set_stamp(msg.header.stamp, self._now())
        poses = []
        for x, y in center:
            pose = self.msgs.PoseStamped()
            pose.header.frame_id = self.params.map_frame
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = 0.0
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = 0.0
            pose.pose.orientation.w = 1.0
            poses.append(pose)
        msg.poses = poses
        self.pub_corridor.publish(msg)

    def close(self) -> None:
        """Stop the worker + capture threads; idempotent."""
        for capture in list(self._captures.values()):
            try:
                capture.stop()
            except Exception:  # noqa: BLE001 - best effort at shutdown
                pass
        if self._map_worker is not None:
            self._map_worker.stop()

    def wait_idle(self, timeout: float = 10.0) -> bool:
        """Block until the mapping worker is idle (tests, before a final save)."""
        if self._map_worker is not None:
            return self._map_worker.wait_idle(timeout)
        return True


# ------------------------------------------------------------ isolation
def _run_isolated(fn, args: tuple, kwargs: dict, subprocess_ok: bool = True) -> tuple[Any, str]:
    """``fn(*args, **kwargs)`` in a fresh spawned interpreter; ``(result, where it ran)``.

    Python threads share one interpreter lock: a save's loop closure (11-44 s
    on the 50 s sept3 replay) on the service thread slowed the executor
    thread - IMU callbacks up to 240 ms, VO increments and odometry dropped
    as late by the EKF - while the kart may still be relying on the live
    pose. A child process has its own interpreter (and runs on another core).
    ``spawn`` (not fork: this process runs DDS and worker threads). ``fn``
    must be importable at module level and its arguments picklable. Any
    failure of the child machinery (not of ``fn`` itself) falls back to
    running here, so a save never fails because of the isolation.
    """
    if subprocess_ok:
        import concurrent.futures
        import multiprocessing
        import pickle

        try:
            ctx = multiprocessing.get_context("spawn")
            with concurrent.futures.ProcessPoolExecutor(max_workers=1, mp_context=ctx) as pool:
                return pool.submit(fn, *args, **kwargs).result(), "subprocess"
        except (OSError, RuntimeError, EOFError, ImportError, AttributeError, pickle.PicklingError) as exc:  # BrokenProcessPool is a RuntimeError
            where = f"this process (subprocess failed: {type(exc).__name__}: {exc})"
            return fn(*args, **kwargs), where[:300]
    return fn(*args, **kwargs), "this process"


# -------------------------------------------------------------------- main
def _spin_services(executor) -> None:
    """Spin the service node's executor until the context shuts down."""
    try:
        executor.spin()
    except Exception:  # noqa: BLE001 - ExternalShutdownException / RCLError at shutdown
        pass


def _parse_sensor_overrides(tokens: list[str]) -> dict[str, int]:
    """``--set sensors.lidar=1`` tokens -> {switch: 0|1} (master_config validates)."""
    overrides: dict[str, int] = {}
    for token in tokens:
        key, _, value = token.partition("=")
        key = key.strip().lower()
        if key.startswith("sensors."):
            key = key[len("sensors."):]
        overrides[key] = int(value.strip() or "1")
    return overrides


def main(argv: list[str] | None = None) -> int:
    """Entry point: ``python3 -m kart_map2d.ros2.node --cfg kart_master.cfg``
    (or ``--params <yaml>`` for a plain params file; ``--set sensors.lidar=1``
    flips a [sensors] switch without editing the cfg).

    Exit codes: 0 after a clean stop (SIGINT / SIGTERM), 2 for a
    configuration the node refuses (every problem is logged with its fix).
    """
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cfg", default=None, help="kart_master.cfg (the master control file)")
    parser.add_argument("--params", default=None, help="YAML parameter file (tests / parity runs)")
    parser.add_argument("--set", action="append", default=[], metavar="sensors.NAME=0|1", help="override a [sensors] switch")
    args, ros_args = parser.parse_known_args(argv)

    raw: dict | None = None
    params = NodeParams()
    use_sim_time = False
    cfg_warnings: list[str] = []
    if args.cfg:
        from kart_map2d.config.master_config import MasterConfigError, load_master_config, to_mapping_params_dict

        try:
            cfg = load_master_config(args.cfg, overrides=_parse_sensor_overrides(args.set))
            translated = to_mapping_params_dict(cfg)
            use_sim_time = bool(translated.get("use_sim_time"))
            params = NodeParams.from_dict(translated)
            cfg_warnings = list(cfg.warnings)
        except (MasterConfigError, ValueError) as exc:
            print("configuration refused - nothing was started. Fix:", flush=True)
            for problem in getattr(exc, "problems", [str(exc)]):
                print(f"  - {problem}", flush=True)
            return 2
    elif args.params:
        import yaml

        with open(args.params, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        params = NodeParams.from_dict(raw)

    import rclpy
    from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor

    rclpy.init(args=ros_args)
    node = rclpy.create_node("kart_map2d")
    log = node.get_logger()
    if use_sim_time:
        from rclpy.parameter import Parameter

        node.set_parameters([Parameter("use_sim_time", value=True)])
    services = None
    wrapper: KartMapNode | None = None
    code = 0
    try:
        for line in cfg_warnings:
            log.warning(line)
        for key in unknown_param_keys(raw):
            log.warning(f"{args.params}: unknown parameter {key} is ignored")
        try:
            from rcl_interfaces.msg import ParameterDescriptor

            params = apply_ros_parameters(node, params, ParameterDescriptor(dynamic_typing=True))
            # The service lives on its own node + executor thread: a save
            # (loop closure, seconds) must not stall the sensor callbacks.
            services = rclpy.create_node("kart_map2d_services")
            wrapper = KartMapNode(node, params, service_node=services)
            wrapper.start_captures()  # non-ROS camera sources, if configured
            if any(c.frame for c in params.cameras.values()):
                # Extrinsics-from-TF: a tf2 buffer fed on this node's executor.
                try:
                    from rclpy.time import Time
                    from tf2_ros import Buffer, TransformListener

                    tf_buffer = Buffer()
                    _tf_listener = TransformListener(tf_buffer, node)  # noqa: F841 - keeps subscriptions alive

                    def _lookup(frame: str):
                        t = tf_buffer.lookup_transform("base_link", frame, Time())
                        q = t.transform.rotation
                        v = t.transform.translation
                        from kart_map2d.core.transforms import make_transform as _mk, quat_to_matrix as _qm

                        return _mk(_qm(np.array([q.x, q.y, q.z, q.w])), np.array([v.x, v.y, v.z]))

                    wrapper.set_tf_lookup(_lookup)
                except ImportError:
                    log.warning("tf2_ros is not importable: camera extrinsics fall back to the manual mounts")
        except NodeConfigError as exc:
            log.error("configuration refused - nothing was started. Fix:")
            for problem in exc.problems:
                log.error(f"  - {problem}")
            code = 2
            return code
        # Build / load the segmentation backend before spinning (see
        # prepare_segmentation) and say which configuration is mapping.
        wrapper.prepare_segmentation()
        for line in wrapper.describe():
            log.info(line)
        executor = SingleThreadedExecutor()
        executor.add_node(services)
        threading.Thread(target=_spin_services, args=(executor,), name="kart2d-services", daemon=True).start()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        # SIGINT / SIGTERM: rclpy's signal handler has already shut the
        # context down (spin then raises ExternalShutdownException); a clean
        # stop, not an error.
        pass
    except Exception:
        # A signal landing while the executor builds its wait set surfaces as
        # RCLError ("context is not valid") instead; also a clean stop. Any
        # error while the context is still up is real and propagates.
        if rclpy.ok():
            raise
    finally:
        if wrapper is not None:
            if wrapper.params.save_on_shutdown:
                # The ROS context may already be gone (SIGINT); the save needs
                # none of it, and stdout survives where the ROS logger may not.
                ok, message, _ = wrapper.save_map("shutdown")
                print(f"[kart_map2d] save_on_shutdown: {'OK' if ok else 'FAILED'}: {message}", flush=True)
            wrapper.close()
        for n in (services, node):
            if n is not None:
                try:
                    n.destroy_node()
                except Exception:  # noqa: BLE001 - best effort at shutdown
                    pass
        if rclpy.ok():  # a second shutdown raises RCLError
            rclpy.shutdown()
    return code


if __name__ == "__main__":
    sys.exit(main())
