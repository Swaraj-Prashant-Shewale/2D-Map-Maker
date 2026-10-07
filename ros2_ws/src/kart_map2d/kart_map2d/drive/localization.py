"""Map-relative localization: odometry fusion + low-rate absolute fixes against a saved map.

The mapping stack's :class:`~kart_map2d.estimation.fusion.FusionEngine`
(IMU, visual odometry, wheel / LiDAR odometry - every sensor optional, health
monitored, chi-square gated) keeps doing what it was verified to do offline.
What driving adds is an ABSOLUTE reference: the map package's evidence grid.
Low-rate matches of what the kart senses now against that grid become
:class:`MapPoseFix` measurements (a :class:`PositionFix` that also carries yaw
and a full 3x3 covariance) and go through the same EKF innovation gate and
per-channel health monitor as any sensor - a bad match is gated out, a
channel that keeps producing bad matches is faulted and put on probation.

Sources of fixes (both optional, any combination):

* ``map_lidar`` - LiDAR present: the scan's obstacle points (height band as
  in the grid's own LiDAR insert) are registered against the grid's
  OBSTACLE cells: correlative scan matching over a window sized from the
  EKF covariance (coarse 0.6 m likelihood field, then the fine 0.15 m one),
  refined by point-to-line / point-to-point ICP on a KD-tree of obstacle
  cell centres. The covariance comes from the ICP normal equations, so a
  corridor wall constrains only its normal direction (~5 Hz).
* ``map_camera`` - camera + IMU only: the zero-network classical ground
  classifier (:mod:`kart_map2d.perception.classical_ground`, run on a
  320-px-wide copy of each frame) gives drivable ground and obstacle
  contact points, projected onto the ground plane exactly as the mapping
  pipeline does; the last few seconds of those are rasterized into a local
  signed-occupancy patch (relative poses from the EKF) and correlated with
  the same patch cut from the map at the predicted pose, using the loop
  closure's rotated-patch masked NCC (:func:`kart_map2d.mapping.loop_closure._correlate_pair`)
  with a yaw window sized from the EKF covariance (~1 Hz). This is the only
  image processing besides visual odometry; no neural network runs.

Initialization: ``start_pose`` (operator-given map pose with its
uncertainty), else GLOBAL relocalization - the whole map is searched over
position (known-drivable cells) and a full yaw sweep (LiDAR: FFT
correlation of the scan with the coarse likelihood field; camera: masked
NCC of the local patch with the whole map), the best candidates are refined
with the tracking matcher and a pose is only accepted when it is clearly
better than the best DISTINCT alternative (a repetitive corridor stays
"searching" instead of guessing). Until the localizer reports converged the
safety layer keeps the kart stopped.

Recovery: fixes the EKF keeps refusing but that agree with EACH OTHER
(consistent offset, consistent relative motion) mean the filter, not the
map match, is wrong (kidnapped kart, a wrong start pose): after
``reanchor_after_rejects`` such fixes the pose is re-anchored on the fix and
convergence starts over. The safety layer has stopped the kart long before
(``max_consecutive_rejects``).

Timing: a fix computed for stamp ``t`` is expressed at the EKF's current
stamp before it is applied - carried forward with the EKF's own trajectory
when ``t`` is older, backward with its motion prediction when the filter
has not reached ``t`` yet (at most the 0.12 s reorder window) - so every fix
is applied synchronously and its accept / reject is known immediately.

Frames and units: map frame (the package's planar frame, metres, yaw CCW
from +x, radians); base_link x forward / y left / z up (ground at
``ground_z`` = -0.10 m); covariances in m^2 / rad^2; stamps epoch seconds.
Nothing here names a camera or a topic: callers pass intrinsics and mounts.
"""

from __future__ import annotations

import heapq
import math
import time
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Any, Sequence

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import distance_transform_edt, maximum_filter
from scipy.spatial import cKDTree

from kart_map2d.core.health import ChannelConfig, HealthMonitor, Status
from kart_map2d.core.measurements import ImuSample, Measurement, PositionFix, RelativeMotion
from kart_map2d.drive.controller import KART_REAR_AXLE_X_M
from kart_map2d.estimation.ekf import PX, PY, YAW, EkfConfig, PlanarEkf, wrap
from kart_map2d.estimation.fusion import (
    CHANNEL_ROLES,
    DEFAULT_CHANNEL_CONFIGS,
    VO_CHANNEL_CONFIG,
    FusionEngine,
    YawConsistencyMonitor,
    channel_config,
    channel_role,
)
from kart_map2d.mapping.evidence_grid import CLASS_DRIVABLE, CLASS_OBSTACLE, _pixel_rays
from kart_map2d.mapping.loop_closure import (
    LocalPatch,
    LoopClosureParams,
    _correlate_pair,
    _downsample,
    _masked_blur,
    _masked_ncc,
    _rotated,
    patches_from_grid,
    se2_compose,
    se2_invert,
    se2_relative,
)

__all__ = [
    "DRIVE_CHANNEL_CONFIGS",
    "DRIVE_CHANNEL_ROLES",
    "KART_BASE_AHEAD_OF_REAR_AXLE_M",
    "CameraGroundCue",
    "CameraObservation",
    "DriveFusionEngine",
    "FixOutcome",
    "GlobalConfig",
    "GlobalResult",
    "LocalizerConfig",
    "LocalizerStatus",
    "MapAwareEkf",
    "MapLocalizer",
    "MapMatch",
    "MapModel",
    "MapModelConfig",
    "MapPoseFix",
    "PatchMatchConfig",
    "PatchMatcher",
    "ScanMatchConfig",
    "ScanMatcher",
    "StaticBodyMask",
    "extrapolate_pose",
    "global_search_camera",
    "global_search_lidar",
    "lidar_obstacle_points",
    "predict_state",
    "voxel_downsample",
]

#: Health-monitor channels of the map fixes and what they observe.
DRIVE_CHANNEL_ROLES = {"map_lidar": "absolute", "map_camera": "absolute"}
DRIVE_CHANNEL_CONFIGS = {
    # Fixes are sparse by design (5 Hz / 1 Hz and only where the map has
    # structure), so silence is not a fault; persistent refusals are.
    "map_lidar": ChannelConfig(timeout_s=3.0, window=20, faulty_consecutive_rejects=8, cooldown_s=2.0, probation_accepts=3),
    "map_camera": ChannelConfig(timeout_s=10.0, window=10, faulty_consecutive_rejects=6, cooldown_s=3.0, probation_accepts=2),
}
# VO_CHANNEL_CONFIG (any ``vo_<camera name>``) now lives in
# kart_map2d.estimation.fusion next to the other channel tables; imported
# above so existing consumers keep their name.


# ================================================================ EKF
@dataclass(frozen=True)
class MapPoseFix(PositionFix):
    """Absolute map-frame pose from a map match: position (x, y) + ``yaw`` with
    a full 3x3 covariance ``cov`` over (x [m], y [m], yaw [rad]).

    A :class:`PositionFix` subclass, so the health monitor, fault injection and
    stuck-value detection treat it like one; the drive EKF applies it as a
    joint 3-dof update (a match is anisotropic: a corridor pins the lateral
    position far better than the along-track one).
    """

    yaw: float = 0.0
    cov: tuple = ()
    score: float = 0.0


#: How far base_link sits IN FRONT of the rear axle [m] (kart URDF: axles at
#: x = +-0.489 m, base_link between them).
KART_BASE_AHEAD_OF_REAR_AXLE_M = -KART_REAR_AXLE_X_M


class MapAwareEkf(PlanarEkf):
    """PlanarEkf + the joint (x, y, yaw) update for :class:`MapPoseFix`, and
    the kart's kinematics.

    A kinematic-bicycle kart moves along its heading only AT THE REAR AXLE;
    base_link, ``base_ahead_m`` in front of it, also slides sideways at
    ``yaw_rate * base_ahead_m`` (0.4 m/s in the sept3 hairpin at 2 m/s). The
    mapping EKF moves base_link straight along its heading (right enough at
    the ~1.3 m/s of the mapping runs); driving through the hairpin at 2 m/s
    that unmodelled slide pulled the filter ~0.4 m off between map fixes that
    were themselves within 3 cm (measured in the ROS closed loop, map-fix NIS
    rising to the chi2 gate). So the drive EKF adds the sideways term to its
    prediction and measures speed from the FORWARD component of relative
    motions - equal to the speed along the longitudinal axis whether an
    odometry source reports base_link's or the rear axle's motion (the chord
    length would add the slide to the speed). ``base_ahead_m = 0`` restores
    the mapping model.
    """

    base_ahead_m: float = KART_BASE_AHEAD_OF_REAR_AXLE_M

    def _predict_step(self, dt: float) -> None:
        super()._predict_step(dt)
        lever = self.base_ahead_m
        if lever == 0.0:
            return
        # base_link's sideways slide over the step, along the mid-step normal
        # (composed after the base step: Jacobian G, P <- G P G^T).
        x = self.x
        r = float(x[4])
        yaw_mid = float(x[YAW]) - 0.5 * r * dt
        lat = r * lever * dt
        s, c = math.sin(yaw_mid), math.cos(yaw_mid)
        x[PX] -= lat * s
        x[PY] += lat * c
        G = np.eye(len(x))
        G[PX, YAW] = -lat * c
        G[PY, YAW] = -lat * s
        G[PX, 4] = -lever * dt * s + 0.5 * dt * lat * c
        G[PY, 4] = lever * dt * c + 0.5 * dt * lat * s
        self.P = G @ self.P @ G.T

    def _relative(self, m: RelativeMotion) -> bool:
        if self.base_ahead_m != 0.0 and m.delta_left != 0.0:
            m = replace(m, delta_left=0.0)  # speed = forward component / dt
        return super()._relative(m)

    def process(self, m: Measurement) -> bool:
        if not isinstance(m, MapPoseFix):
            return super().process(m)
        health = self.health.channel(m.channel)
        if self.stamp is not None and m.stamp < self.stamp - self.config.max_lateness_s:
            self.late_drops += 1
            return False
        if not health.usable(m.stamp):
            return False
        if self.stamp is None or m.stamp > self.stamp:
            self.predict(m.stamp)
        H = np.zeros((3, len(self.x)))
        H[0, PX] = H[1, PY] = H[2, YAW] = 1.0
        z = np.array([m.position[0], m.position[1], m.yaw], dtype=np.float64)
        cov = np.asarray(m.cov, dtype=np.float64).reshape(3, 3)
        return self._update(z, self.x[[PX, PY, YAW]], H, cov, m, angular=(2,))


@dataclass
class DriveFusionEngine(FusionEngine):
    """The mapping FusionEngine with map-fix channels and generic camera names.

    Differences, all additive: the EKF understands :class:`MapPoseFix`;
    ``map_*`` channels count as ``absolute`` in :meth:`mode`; any
    ``vo_<name>`` channel registered with :meth:`register_camera` is an
    odometric channel (the stock tables only list the September rig's
    camera names) - for the mode and for the gyro-consistency check.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.roles: dict[str, str] = {**CHANNEL_ROLES, **DRIVE_CHANNEL_ROLES}
        self._health_configs = {**DEFAULT_CHANNEL_CONFIGS, **DRIVE_CHANNEL_CONFIGS}
        self.health = HealthMonitor(self._health_configs, default_factory=channel_config)
        self.ekf = MapAwareEkf(self.health, self.ekf_config)
        self.yaw_monitor = YawConsistencyMonitor(self.health, self.ekf, self.trajectory, self.yaw_check)
        #: EKF update counts per channel [accepted, rejected], folded from the log.
        self.update_counts: dict[str, list[int]] = {}

    def register_camera(self, name: str) -> str:
        """Declare camera ``name``; returns its VO channel ``vo_<name>``."""
        channel = f"vo_{name}"
        self.roles.setdefault(channel, "odometric")
        self._health_configs.setdefault(channel, VO_CHANNEL_CONFIG)
        return channel

    def _drain(self, until: float) -> None:
        # FusionEngine._drain with the role lookup going through self.roles.
        while self._queue and self._queue[0][0] <= until:
            stamp, _, m = heapq.heappop(self._queue)
            self.health.tick(stamp)
            self.ekf.process(m)
            if isinstance(m, ImuSample):
                self.yaw_monitor.add_imu(m)
            elif isinstance(m, RelativeMotion) and (self.roles.get(m.channel) or channel_role(m.channel)) == "odometric":
                self.yaw_monitor.add_relative(m)
            self._record(stamp)

    def mode(self) -> str:
        roles = {self.roles.get(n) or channel_role(n) for n in self.health.channels if self.health.healthy(n)}
        parts = [role for role in ("absolute", "heading", "odometric", "inertial") if role in roles]
        if not parts:
            return "blind"
        if parts == ["inertial"]:
            return "inertial-only (dead reckoning)"
        return "+".join(parts)

    def advance(self, arrival: float) -> None:
        """Move the arrival clock to ``arrival`` and drain what is due (timer tick)."""
        self._latest_arrival = max(self._latest_arrival, arrival)
        self._drain(self._latest_arrival - self.reorder_window_s)

    # ------------------------------------------------------------ poses
    #: Optional ``gyro_since(t0) -> (stamps, gyro_z)`` of raw gyro samples
    #: newer than ``t0`` (the drive node's buffer): pose_at() beyond the
    #: filter's stamp then integrates the MEASURED turn rate instead of
    #: holding the filter's last one - the filter trails real time by the
    #: 0.12 s reorder window, and a constant-rate guess across a steering
    #: transient is off by degrees.
    gyro_since = None

    def pose_at(self, t: float) -> NDArray[np.float64]:
        """EKF pose [x, y, yaw] at ``t``: trajectory interpolation inside it,
        prediction from the current state beyond it (gyro-integrated when
        :attr:`gyro_since` is set, constant rate otherwise)."""
        ekf = self.ekf
        if ekf.stamp is None:
            return np.zeros(3)
        if t > ekf.stamp and self.gyro_since is not None:
            gt, gz = self.gyro_since(float(ekf.stamp))
            if len(gt):
                return extrapolate_pose(ekf.x, float(ekf.stamp), t, gt, gz, float(ekf.x[6]), ekf.base_ahead_m)
        if t >= ekf.stamp or len(self.trajectory.stamp) < 2 or t >= self.trajectory.stamp[-1]:
            return predict_state(ekf.x, t - ekf.stamp, ekf.base_ahead_m)
        stamps = self.trajectory.stamp
        if t <= stamps[0]:
            return np.asarray(self.trajectory.x[0][:3], dtype=np.float64).copy()
        i = int(np.searchsorted(stamps, t))
        t0, t1 = stamps[i - 1], stamps[i]
        a, b = self.trajectory.x[i - 1], self.trajectory.x[i]
        f = (t - t0) / max(t1 - t0, 1e-9)
        return np.array([a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]), wrap(a[2] + f * wrap(b[2] - a[2]))])

    def pose_covariance(self) -> NDArray[np.float64]:
        return np.asarray(self.ekf.P[:3, :3], dtype=np.float64).copy()

    def set_pose(self, xyyaw: NDArray[np.float64], cov: NDArray[np.float64], history_s: float | None = 8.0) -> None:
        """Re-anchor: move the filter (and its recent trajectory, rigidly) to ``xyyaw``.

        The body states (speed, yaw rate, biases) are frame-free and kept;
        the pose block of P becomes ``cov`` and its cross terms are dropped.
        The last ``history_s`` seconds of the recorded trajectory move with
        the same rigid transform (None: all of it - a global initialization,
        where the whole earlier track was in the filter's arbitrary frame),
        so relative motion read from it (camera patches, late-fix
        carry-forward) stays valid; older history is left as it was.
        """
        ekf = self.ekf
        target = np.asarray(xyyaw, dtype=np.float64)
        if ekf.stamp is not None:
            T = se2_compose(target, se2_invert(np.asarray(ekf.x[:3], dtype=np.float64)))
            c, s = math.cos(T[2]), math.sin(T[2])
            cut = -np.inf if history_s is None else float(ekf.stamp) - history_s
            k0 = int(np.searchsorted(self.trajectory.stamp, cut)) if self.trajectory.stamp else 0
            for state in self.trajectory.x[k0:]:
                x, y = float(state[0]), float(state[1])
                state[0] = T[0] + c * x - s * y
                state[1] = T[1] + s * x + c * y
                state[2] = wrap(float(state[2]) + T[2])
        ekf.x[PX], ekf.x[PY], ekf.x[YAW] = target[0], target[1], wrap(float(target[2]))
        ekf.P[:3, :] = 0.0
        ekf.P[:, :3] = 0.0
        ekf.P[:3, :3] = np.asarray(cov, dtype=np.float64).reshape(3, 3)

    def fold_update_log(self, keep: int = 256) -> None:
        """Fold the EKF's per-update log into ``update_counts`` (bounded memory)."""
        log = self.ekf.updates
        if len(log) <= keep:
            return
        for u in log[:-keep]:
            entry = self.update_counts.setdefault(u.channel, [0, 0])
            entry[0 if u.accepted else 1] += 1
        del log[:-keep]

    def trim_history(self, keep_s: float = 60.0) -> None:
        """Drop trajectory samples older than ``keep_s`` (bounded memory on long drives)."""
        traj = self.trajectory
        if not traj.stamp:
            return
        cut = traj.stamp[-1] - keep_s
        k = int(np.searchsorted(traj.stamp, cut))
        if k > 0:
            del traj.stamp[:k], traj.x[:k], traj.p_diag[:k], traj.supported[:k]
        if len(self.health.transitions) > 2000:
            del self.health.transitions[:-1000]
        if len(self.modes) > 2000:
            del self.modes[:-1000]


def predict_state(x: NDArray[np.float64], dt: float, base_ahead_m: float = KART_BASE_AHEAD_OF_REAR_AXLE_M) -> NDArray[np.float64]:
    """Constant speed / yaw-rate prediction of the EKF pose by ``dt`` (either
    sign), with base_link's sideways slide ``yaw_rate * base_ahead_m``
    (see :class:`MapAwareEkf`)."""
    dyaw = x[4] * dt
    yaw_mid = x[YAW] + 0.5 * dyaw
    travel = x[3] * dt
    side = base_ahead_m * dyaw
    c, s = math.cos(yaw_mid), math.sin(yaw_mid)
    return np.array([x[PX] + travel * c - side * s, x[PY] + travel * s + side * c, wrap(x[YAW] + dyaw)])


def extrapolate_pose(
    x: NDArray[np.float64],
    t0: float,
    t1: float,
    gyro_t: NDArray[np.float64] | None = None,
    gyro_z: NDArray[np.float64] | None = None,
    bias: float = 0.0,
    base_ahead_m: float = KART_BASE_AHEAD_OF_REAR_AXLE_M,
) -> NDArray[np.float64]:
    """Pose at ``t1`` from the EKF state ``x`` at ``t0``; yaw integrates the
    measured gyro (bias-corrected) where samples exist after ``t0``, so the
    control loop's heading is as current as the newest IMU sample, not the
    EKF's reorder-window-old state. base_link slides sideways by
    ``base_ahead_m`` per radian turned (see :class:`MapAwareEkf`)."""
    dt = t1 - t0
    if dt <= 0.0:
        return np.asarray(x[:3], dtype=np.float64).copy()
    if gyro_t is None or gyro_z is None or len(gyro_t) == 0:
        return predict_state(x, dt, base_ahead_m)
    # Plain Python: the control loop integrates ~25-50 samples at 50 Hz,
    # where a loop beats a dozen numpy calls. Trapezoidal turn rate between
    # samples (the state's rate at t0, bias-corrected gyro after it, the last
    # rate held to t1); position along the mid-interval heading.
    ts = gyro_t.tolist() if hasattr(gyro_t, "tolist") else list(gyro_t)
    zs = gyro_z.tolist() if hasattr(gyro_z, "tolist") else list(gyro_z)
    v = float(x[3])
    lever = float(base_ahead_m)
    px, py, yaw = float(x[PX]), float(x[PY]), float(x[YAW])
    r_prev = float(x[4])
    t_prev = t0
    cos, sin = math.cos, math.sin
    for t, z in zip(ts, zs):
        if t <= t0 or t > t1:
            continue
        h = t - t_prev
        if h < 0.0:
            h = 0.0
        r = z - bias
        dyaw = 0.5 * (r_prev + r) * h
        ym = yaw + 0.5 * dyaw
        c, s = cos(ym), sin(ym)
        px += v * h * c - lever * dyaw * s
        py += v * h * s + lever * dyaw * c
        yaw += dyaw
        r_prev, t_prev = r, t
    h = t1 - t_prev
    if h > 0.0:
        dyaw = r_prev * h
        ym = yaw + 0.5 * dyaw
        c, s = cos(ym), sin(ym)
        px += v * h * c - lever * dyaw * s
        py += v * h * s + lever * dyaw * c
        yaw += dyaw
    return np.array((px, py, wrap(yaw)))


# ================================================================ the map
@dataclass
class MapModelConfig:
    likelihood_sigma_m: float = 0.20  # scan-matching likelihood field width
    coarse_factor: int = 4  # coarse field cell = factor * grid cell (0.6 m)
    normal_neighbors: int = 9  # obstacle-cell PCA neighbourhood for ICP normals
    linearity_min: float = 0.85  # PCA linearity above which a cell is a wall (point-to-line)
    patch_cell_m: float = 0.20  # camera patch cell (loop-closure default)
    obstacle_weight: float = 3.0  # obstacle vs drivable weight in the camera field


class _ClassesView:
    """Duck-typed grid for loop_closure.patches_from_grid with cached classes."""

    def __init__(self, classes: NDArray[np.uint8], origin: NDArray[np.float64], cell: float) -> None:
        self._classes = classes
        self.origin = origin
        self.cell = cell

    def classify(self) -> NDArray[np.uint8]:
        return self._classes


class MapModel:
    """Everything the matchers need from a map package's grid, computed once at load.

    ``classes`` (nx, ny) static view of the grid (0 unknown / 1 drivable /
    2 obstacle, ix along map x); the LiDAR likelihood field
    ``exp(-d^2 / 2 sigma^2)`` of the distance to the nearest obstacle cell;
    obstacle cell centres with a KD-tree and PCA normals (ICP); a coarse
    max-pooled likelihood field (global search); for the camera the signed
    occupancy field at the patch cell size.
    """

    def __init__(self, grid, config: MapModelConfig | None = None, sensors: Sequence[str] = ()) -> None:
        cfg = config or MapModelConfig()
        self.config = cfg
        #: What built the map (map_meta.json "sensors"): offline sources or
        #: live channels. LiDAR evidence in the grid = "lidar_odom" / "lidar".
        self.sensors = tuple(str(s) for s in sensors)
        self.has_lidar_evidence = any(s in ("lidar", "lidar_odom") for s in self.sensors)
        classes = np.asarray(grid.classify(), dtype=np.uint8)
        self.classes = classes
        self.cell = float(grid.cell)
        self.origin = np.asarray(grid.origin, dtype=np.float64)
        self.shape = classes.shape
        self.obstacle = classes == CLASS_OBSTACLE
        self.drivable = classes == CLASS_DRIVABLE
        if self.obstacle.any():
            dist = distance_transform_edt(~self.obstacle) * self.cell
        else:
            dist = np.full(classes.shape, np.inf)
        like = np.exp(-0.5 * (dist / cfg.likelihood_sigma_m) ** 2).astype(np.float32)
        self.likelihood = like
        # Zero border: clipped out-of-map lookups land on it (score 0).
        self.like_padded = np.pad(like, 1)
        f = max(int(cfg.coarse_factor), 1)
        self.coarse_factor = f
        nx, ny = like.shape
        pad = ((0, (-nx) % f), (0, (-ny) % f))
        pooled = np.pad(like, pad).reshape((nx + pad[0][1]) // f, f, (ny + pad[1][1]) // f, f).max(axis=(1, 3))
        # Max-pooling then a 1-cell dilation keeps the coarse score an upper
        # envelope of the fine one across the coarse cell boundaries.
        self.coarse = maximum_filter(pooled, size=3, mode="constant").astype(np.float32)
        self.coarse_padded = np.pad(self.coarse, 1)
        drv = np.pad(self.drivable, pad).reshape(self.coarse.shape[0], f, self.coarse.shape[1], f).any(axis=(1, 3))
        self.coarse_drivable = maximum_filter(drv, size=3) > 0
        ij = np.argwhere(self.obstacle)
        self.obstacle_xy = self.origin + (ij + 0.5) * self.cell
        self.tree = cKDTree(self.obstacle_xy) if len(self.obstacle_xy) else None
        self.normals, self.linear = self._normals(cfg.normal_neighbors, cfg.linearity_min)
        self.classes_view = _ClassesView(classes, self.origin, self.cell)
        self._camera_field: tuple | None = None

    @classmethod
    def from_package(cls, package, config: MapModelConfig | None = None) -> "MapModel":
        return cls(package.grid, config, sensors=package.meta.get("sensors", ()))

    def _normals(self, k: int, lin_min: float) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
        n = len(self.obstacle_xy)
        normals = np.zeros((n, 2))
        linear = np.zeros(n, dtype=bool)
        if n < 3 or self.tree is None:
            return normals, linear
        k = min(k, n)
        dist, idx = self.tree.query(self.obstacle_xy, k=k, distance_upper_bound=3.5 * self.cell)
        valid = np.isfinite(dist)
        idx = np.where(valid, idx, 0)
        nb = self.obstacle_xy[idx]  # (n, k, 2)
        w = valid.astype(np.float64)
        cnt = np.maximum(w.sum(axis=1), 1.0)
        mean = (nb * w[:, :, None]).sum(axis=1) / cnt[:, None]
        d = (nb - mean[:, None, :]) * w[:, :, None]
        cxx = (d[:, :, 0] ** 2).sum(axis=1) / cnt
        cyy = (d[:, :, 1] ** 2).sum(axis=1) / cnt
        cxy = (d[:, :, 0] * d[:, :, 1]).sum(axis=1) / cnt
        tr, det = cxx + cyy, cxx * cyy - cxy * cxy
        disc = np.sqrt(np.maximum(0.25 * tr * tr - det, 0.0))
        l1, l2 = 0.5 * tr + disc, 0.5 * tr - disc
        # Normal = eigenvector of the small eigenvalue l2.
        nx_, ny_ = cxy, l2 - cxx
        bad = np.hypot(nx_, ny_) < 1e-12
        nx_ = np.where(bad, l2 - cyy, nx_)
        ny_ = np.where(bad, cxy, ny_)
        norm = np.maximum(np.hypot(nx_, ny_), 1e-12)
        normals[:, 0], normals[:, 1] = nx_ / norm, ny_ / norm
        linear = (cnt >= 4) & (l1 > 1e-12) & ((l1 - l2) / np.maximum(l1, 1e-12) >= lin_min)
        return normals, linear

    def camera_field(self, cell: float) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """(signed field, seen mask, obstacle mask, origin) resampled to ``cell``.

        field = obstacle_weight * obstacle - drivable (the loop-closure
        convention: structure localizes, open ground only a little).
        """
        if self._camera_field is not None and abs(self._camera_field[0] - cell) < 1e-9:
            return self._camera_field[1]
        ext = np.array(self.shape) * self.cell
        n = np.maximum(np.ceil(ext / cell).astype(int), 1)
        cx = self.origin[0] + (np.arange(n[0]) + 0.5) * cell
        cy = self.origin[1] + (np.arange(n[1]) + 0.5) * cell
        ix = np.clip(((cx - self.origin[0]) / self.cell).astype(int), 0, self.shape[0] - 1)
        iy = np.clip(((cy - self.origin[1]) / self.cell).astype(int), 0, self.shape[1] - 1)
        cls = self.classes[np.ix_(ix, iy)]
        obst = (cls == CLASS_OBSTACLE).astype(np.float64)
        drv = (cls == CLASS_DRIVABLE).astype(np.float64)
        out = (obst - drv, (cls != 0).astype(np.float64), obst, self.origin.copy())
        self._camera_field = (cell, out)
        return out

    def cells_of(self, xy: NDArray[np.float64]) -> NDArray[np.int64]:
        return np.floor((np.asarray(xy, dtype=np.float64) - self.origin) / self.cell).astype(np.int64)

    def class_at(self, xy: NDArray[np.float64]) -> NDArray[np.uint8]:
        ij = self.cells_of(np.asarray(xy, dtype=np.float64).reshape(-1, 2))
        ok = (ij[:, 0] >= 0) & (ij[:, 0] < self.shape[0]) & (ij[:, 1] >= 0) & (ij[:, 1] < self.shape[1])
        out = np.zeros(len(ij), dtype=np.uint8)
        out[ok] = self.classes[ij[ok, 0], ij[ok, 1]]
        return out


# ================================================================ matches
@dataclass
class MapMatch:
    """One map match: the measured base_link pose at ``stamp`` + covariance.

    ``score`` is source specific (LiDAR: ICP inlier fraction; camera: masked
    NCC peak), ``info`` carries diagnostics (points, iterations, windows...).
    """

    stamp: float
    source: str  # "map_lidar" | "map_camera"
    pose: NDArray[np.float64]
    cov: NDArray[np.float64]
    score: float
    info: dict = field(default_factory=dict)


def _rot(yaw: float) -> NDArray[np.float64]:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]])


def _window(cov: NDArray[np.float64], k: float, xy_lo: float, xy_hi: float, yaw_lo: float, yaw_hi: float) -> tuple[float, float]:
    sxy = math.sqrt(max(float(np.max(np.linalg.eigvalsh(cov[:2, :2]))), 0.0))
    syaw = math.sqrt(max(float(cov[2, 2]), 0.0))
    return float(np.clip(k * sxy, xy_lo, xy_hi)), float(np.clip(k * syaw, yaw_lo, yaw_hi))


# ------------------------------------------------------------------ LiDAR
@dataclass
class ScanMatchConfig:
    ground_z_m: float = -0.10  # ground plane in base_link (as the evidence grid)
    obstacle_min_height_m: float = 0.20  # the grid's ground band
    obstacle_max_height_m: float = 2.5
    min_range_m: float = 0.5
    max_range_m: float = 30.0
    voxel_m: float = 0.15
    max_points: int = 600
    min_points: int = 40
    #: A wide search (initialization, re-acquisition) needs more structure:
    #: a sparse scan scores well almost anywhere in a large window (the
    #: ambiguity test below then refuses what is still ambiguous).
    min_points_wide: int = 80
    #: Wide search: a DISTINCT solution (> ambiguity_xy_m or > ambiguity_yaw
    #: apart) within this ratio of the best inlier fraction = ambiguous -> no fix.
    ambiguity_ratio: float = 0.93
    ambiguity_xy_m: float = 0.5
    ambiguity_yaw_rad: float = math.radians(3.0)
    window_sigmas: float = 3.0
    min_window_xy_m: float = 0.30
    max_window_xy_m: float = 3.0
    min_window_yaw_rad: float = math.radians(2.0)
    max_window_yaw_rad: float = math.radians(20.0)
    yaw_step_rad: float = math.radians(0.5)
    coarse_yaw_step_rad: float = math.radians(2.0)
    coarse_candidates: int = 3
    icp_iterations: int = 10
    icp_max_dist_m: float = 0.5
    icp_huber_m: float = 0.10
    inlier_dist_m: float = 0.25
    min_inlier_fraction: float = 0.45
    #: Correlated points: the normal-equation covariance is scaled by
    #: (points used / effective_points) - neighbouring returns on one wall
    #: are not independent, and the map is only as sharp as its cells.
    effective_points: float = 25.0
    sigma_xy_floor_m: float = 0.08
    sigma_yaw_floor_rad: float = math.radians(0.5)


def lidar_obstacle_points(points_base: NDArray[np.float64], cfg: ScanMatchConfig) -> NDArray[np.float64]:
    """(N, 3) base_link points -> (M, 2) obstacle points (grid height band, range gate)."""
    pts = np.asarray(points_base, dtype=np.float64).reshape(-1, 3)
    h = pts[:, 2] - cfg.ground_z_m
    r = np.hypot(pts[:, 0], pts[:, 1])
    keep = np.isfinite(pts).all(axis=1) & (h > cfg.obstacle_min_height_m) & (h < cfg.obstacle_max_height_m)
    keep &= (r > cfg.min_range_m) & (r < cfg.max_range_m)
    return pts[keep, :2]


#: Lever arm [m] that puts yaw on the scale of metres when judging which
#: pose directions a match constrains.
_YAW_LEVER_M = 10.0


def _constrained_pose(pose: NDArray[np.float64], prior: NDArray[np.float64], H: NDArray[np.float64] | None, rel: float = 1e-3) -> tuple[NDArray[np.float64], int]:
    """``pose`` with its component along unobservable directions reset to ``prior``.

    ``H`` is the ICP information matrix over (x, y, yaw). Eigen-directions
    (yaw scaled by a 10 m lever arm) whose information is below ``rel`` of
    the strongest carry no measurement; returns the pose and their count.
    """
    if H is None:
        return pose, 0
    S = np.diag([1.0, 1.0, _YAW_LEVER_M])
    Si = np.linalg.inv(S)
    w, V = np.linalg.eigh(Si @ H @ Si)
    weak = w < rel * max(float(w.max()), 1e-12)
    if not weak.any():
        return pose, 0
    d = S @ np.array([pose[0] - prior[0], pose[1] - prior[1], wrap(pose[2] - prior[2])])
    Vw = V[:, weak]
    d = d - Vw @ (Vw.T @ d)
    d = Si @ d
    return np.array([prior[0] + d[0], prior[1] + d[1], wrap(prior[2] + d[2])]), int(weak.sum())


def voxel_downsample(xy: NDArray[np.float64], voxel: float, max_points: int, seed: int = 0) -> NDArray[np.float64]:
    """One point (the mean) per ``voxel`` cell, then a deterministic subsample to ``max_points``."""
    if len(xy) == 0:
        return np.zeros((0, 2))
    keys = np.floor(xy / voxel).astype(np.int64)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    sums = np.zeros((len(counts), 2))
    np.add.at(sums, inverse, xy)
    out = sums / counts[:, None]
    if len(out) > max_points:
        rng = np.random.default_rng(seed)
        out = out[np.sort(rng.choice(len(out), max_points, replace=False))]
    return out


class ScanMatcher:
    """Correlative scan matching + ICP of 2D obstacle points against the map."""

    def __init__(self, model: MapModel, config: ScanMatchConfig | None = None) -> None:
        self.model = model
        self.config = config or ScanMatchConfig()

    # ------------------------------------------------------------ CSM
    def _csm(
        self,
        padded: NDArray[np.float32],
        cell: float,
        P: NDArray[np.float64],
        pose0: NDArray[np.float64],
        win_xy: float,
        win_yaw: float,
        yaw_step: float,
        top: int = 1,
    ) -> list[tuple[float, NDArray[np.float64]]]:
        """Exhaustive window search; returns the ``top`` (score, pose) peaks.

        ``padded`` is a likelihood field with a 1-cell zero border (clipped
        out-of-map lookups score 0); scores are the mean likelihood per point.
        """
        o = self.model.origin
        W = max(int(math.ceil(win_xy / cell)), 0)
        K = max(int(math.ceil(win_yaw / yaw_step)), 0)
        offs = np.arange(-W, W + 1)
        hi_x, hi_y = padded.shape[0] - 1, padded.shape[1] - 1
        results: list[tuple[float, float, int, int]] = []
        surfaces = []
        for k in range(-K, K + 1):
            yaw = pose0[2] + k * yaw_step
            c, s = math.cos(yaw), math.sin(yaw)
            qx = P[:, 0] * c - P[:, 1] * s + pose0[0]
            qy = P[:, 0] * s + P[:, 1] * c + pose0[1]
            ix = np.floor((qx - o[0]) / cell).astype(np.int64) + 1
            iy = np.floor((qy - o[1]) / cell).astype(np.int64) + 1
            gx = np.clip(ix[:, None] + offs[None, :], 0, hi_x)
            gy = np.clip(iy[:, None] + offs[None, :], 0, hi_y)
            score = padded[gx[:, :, None], gy[:, None, :]].mean(axis=0)  # (2W+1, 2W+1)
            surfaces.append(score)
            flat = int(np.argmax(score))
            a, b = divmod(flat, 2 * W + 1)
            results.append((float(score[a, b]), yaw, a - W, b - W))
        order = sorted(range(len(results)), key=lambda i: -results[i][0])
        out = []
        for i in order[: max(top, 1)]:
            sc, yaw, dx, dy = results[i]
            pose = np.array([pose0[0] + dx * cell, pose0[1] + dy * cell, yaw])
            # Sub-cell refinement: 1D parabolas through the peak on each axis.
            surf = surfaces[i]
            a, b = dx + W, dy + W
            if 0 < a < 2 * W:
                l, m_, r = surf[a - 1, b], surf[a, b], surf[a + 1, b]
                den = l - 2 * m_ + r
                if den < -1e-9:
                    pose[0] += 0.5 * (l - r) / den * cell
            if 0 < b < 2 * W:
                l, m_, r = surf[a, b - 1], surf[a, b], surf[a, b + 1]
                den = l - 2 * m_ + r
                if den < -1e-9:
                    pose[1] += 0.5 * (l - r) / den * cell
            out.append((sc, pose))
        return out

    # ------------------------------------------------------------ ICP
    def _icp(self, P: NDArray[np.float64], pose: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64], dict]:
        cfg, model = self.config, self.model
        pose = pose.astype(np.float64).copy()
        info: dict = {"iterations": 0}
        H = np.eye(3)
        m_used = 0
        sigma_r = cfg.icp_huber_m
        max_d = cfg.icp_max_dist_m
        for it in range(cfg.icp_iterations):
            R = _rot(pose[2])
            Q = P @ R.T + pose[:2]
            d, idx = model.tree.query(Q, distance_upper_bound=max_d)
            ok = np.isfinite(d)
            if ok.sum() < cfg.min_points:
                info["reason"] = f"only {int(ok.sum())} points near map obstacles"
                info["near_fraction"] = float(ok.sum()) / max(len(P), 1)
                return pose, np.full((3, 3), np.inf), info
            q = model.obstacle_xy[idx[ok]]
            Qo = Q[ok]
            dQ = np.column_stack([-(Qo[:, 1] - pose[1]), Qo[:, 0] - pose[0]])  # d Q / d yaw
            lin = model.linear[idx[ok]]
            nrm = model.normals[idx[ok]]
            diff = Qo - q
            # Point-to-line rows for wall cells, point-to-point rows otherwise.
            r_line = np.sum(nrm * diff, axis=1)[lin]
            J_line = np.column_stack([nrm[lin, 0], nrm[lin, 1], np.sum(nrm[lin] * dQ[lin], axis=1)])
            pp = ~lin
            r_pt = diff[pp].reshape(-1)
            J_pt = np.zeros((2 * int(pp.sum()), 3))
            J_pt[0::2, 0] = 1.0
            J_pt[1::2, 1] = 1.0
            J_pt[0::2, 2] = dQ[pp, 0]
            J_pt[1::2, 2] = dQ[pp, 1]
            r = np.r_[r_line, r_pt]
            J = np.vstack([J_line, J_pt])
            a = np.abs(r)
            w = np.where(a <= cfg.icp_huber_m, 1.0, cfg.icp_huber_m / np.maximum(a, 1e-12))
            JW = J * w[:, None]
            H = J.T @ JW + 1e-9 * np.eye(3)
            g = JW.T @ r
            try:
                delta = -np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                info["reason"] = "singular ICP system"
                return pose, np.full((3, 3), np.inf), info
            pose[:2] += delta[:2]
            pose[2] = wrap(pose[2] + delta[2])
            m_used = len(r)
            dof = max(m_used - 3, 1)
            sigma_r = math.sqrt(float(np.sum(w * r * r)) / dof)
            info["iterations"] = it + 1
            max_d = max(cfg.inlier_dist_m * 1.5, max_d * 0.7)
            if np.hypot(delta[0], delta[1]) < 1e-4 and abs(delta[2]) < 1e-5:
                break
        sigma_r = max(sigma_r, model.cell / math.sqrt(12.0))
        scale = max(m_used / max(cfg.effective_points, 1.0), 1.0)
        try:
            cov = sigma_r * sigma_r * np.linalg.inv(H) * scale
        except np.linalg.LinAlgError:
            return pose, np.full((3, 3), np.inf), {**info, "reason": "singular ICP covariance"}
        cov = 0.5 * (cov + cov.T)
        cov[0, 0] += cfg.sigma_xy_floor_m**2
        cov[1, 1] += cfg.sigma_xy_floor_m**2
        cov[2, 2] += cfg.sigma_yaw_floor_rad**2
        info["residual_m"] = round(sigma_r, 4)
        info["rows"] = int(m_used)
        info["_H"] = H
        return pose, cov, info

    # ------------------------------------------------------------ public
    def prepare(self, points_base: NDArray[np.float64], already_2d: bool = False) -> NDArray[np.float64]:
        cfg = self.config
        xy = np.asarray(points_base, dtype=np.float64).reshape(-1, 2) if already_2d else lidar_obstacle_points(points_base, cfg)
        return voxel_downsample(xy, cfg.voxel_m, cfg.max_points)

    def match(
        self,
        P: NDArray[np.float64],
        prior_pose: NDArray[np.float64],
        prior_cov: NDArray[np.float64],
        stamp: float = 0.0,
        window: tuple[float, float] | None = None,
    ) -> MapMatch | None:
        """Register prepared obstacle points ``P`` (M, 2 base_link) near the prior.

        Returns None (with nothing applied) when there is too little structure,
        the ICP diverges or too few points land on map obstacles.
        """
        cfg, model = self.config, self.model
        t0 = time.perf_counter()
        if model.tree is None or len(P) < cfg.min_points:
            return None
        if window is None:
            win_xy, win_yaw = _window(prior_cov, cfg.window_sigmas, cfg.min_window_xy_m, cfg.max_window_xy_m, cfg.min_window_yaw_rad, cfg.max_window_yaw_rad)
        else:
            win_xy, win_yaw = window
        pose0 = np.asarray(prior_pose, dtype=np.float64)
        seeds: list[NDArray[np.float64]]
        fine_cell = model.cell
        wide = win_xy > 4 * fine_cell or win_yaw > math.radians(4.0)
        if wide and len(P) < cfg.min_points_wide:
            return None
        if wide:
            # Coarse stage over the full window on the max-pooled field.
            coarse_cell = fine_cell * model.coarse_factor
            Pc = voxel_downsample(P, coarse_cell * 0.5, 400)
            peaks = self._csm(model.coarse_padded, coarse_cell, Pc, pose0, win_xy, win_yaw, cfg.coarse_yaw_step_rad, top=cfg.coarse_candidates)
            seeds = [p for _, p in peaks]
            fine_xy, fine_yaw = coarse_cell, cfg.coarse_yaw_step_rad
        else:
            seeds = [pose0]
            fine_xy, fine_yaw = win_xy, win_yaw
        solutions = []
        near_best = 0.0 if seeds else 1.0  # best fraction of points near map obstacles among failed ICPs
        for seed in seeds:
            for sc, pose in self._csm(model.like_padded, fine_cell, P, seed, fine_xy, fine_yaw, cfg.yaw_step_rad, top=1):
                pose, cov, info = self._icp(P, pose)
                if not np.all(np.isfinite(cov)):
                    near_best = max(near_best, float(info.get("near_fraction", 1.0)))
                    continue
                Q = P @ _rot(pose[2]).T + pose[:2]
                d, _ = model.tree.query(Q, distance_upper_bound=cfg.inlier_dist_m)
                solutions.append((float(np.mean(np.isfinite(d))), sc, pose, cov, info))
            if not wide:
                break
        if not solutions:
            if near_best < cfg.min_inlier_fraction:
                # A scan with structure (>= min_points) of which almost nothing
                # lies near ANY map obstacle anywhere in the window is evidence
                # AGAINST the prior pose (kidnapped kart, wrong start pose, off
                # the mapped area), not a lack of evidence: report it as a
                # refused match so the localizer counts it (a None would leave
                # the kart "tracking" on odometry for max_fix_age_s).
                m = MapMatch(stamp, "map_lidar", pose0.copy(), np.diag([1e6, 1e6, 1e3]), near_best, {"reason": f"only {near_best:.0%} of the points near map obstacles", "rejected": True, "mismatch": True})
                return m
            return None
        solutions.sort(key=lambda r: -r[0])
        inlier, csm_score, pose, cov, info = solutions[0]
        if wide:
            # Several coarse basins refined to DIFFERENT poses that explain the
            # scan about equally well: the window holds more than one answer.
            # A rival along a direction the match's own covariance already
            # calls unconstrained (a corridor's along-track) is not a second
            # answer - the fix will not pretend to know that direction anyway.
            try:
                info_best = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                info_best = np.zeros((3, 3))

            def distinct(r) -> bool:
                d = np.array([r[2][0] - pose[0], r[2][1] - pose[1], wrap(r[2][2] - pose[2])])
                far = math.hypot(d[0], d[1]) > cfg.ambiguity_xy_m or abs(d[2]) > cfg.ambiguity_yaw_rad
                return far and float(d @ info_best @ d) > 16.27

            rivals = [r for r in solutions[1:] if distinct(r)]
            if rivals and rivals[0][0] >= cfg.ambiguity_ratio * inlier:
                info["reason"] = f"ambiguous: a distinct pose explains {rivals[0][0]:.2f} vs {inlier:.2f} of the points"
                return None
        # Directions the scan cannot see (a corridor's along-track): keep the
        # prior there instead of wherever the flat score happened to peak.
        pose, free = _constrained_pose(pose, pose0, info.pop("_H", None))
        for r in solutions[1:]:
            r[4].pop("_H", None)
        # Fewer inliers -> less trust (the normal equations cannot know that
        # the outliers were a moving crowd or a changed scene).
        cov = cov / max(inlier, 0.3) ** 2
        info["unconstrained_dirs"] = free
        info.update(
            points=int(len(P)),
            csm_score=round(float(csm_score), 4),
            inliers=round(inlier, 4),
            window_xy_m=round(win_xy, 3),
            window_yaw_deg=round(math.degrees(win_yaw), 2),
            ms=round(1e3 * (time.perf_counter() - t0), 2),
        )
        # A match that had to move out of its own search window is ICP drift.
        shift = pose - pose0
        if math.hypot(shift[0], shift[1]) > win_xy + 0.5 or abs(wrap(shift[2])) > win_yaw + math.radians(3.0):
            # The scan fits the map well only far outside the window the
            # filter's uncertainty allows: the filter is off (measured on
            # sept3 after a 1.5 m kidnap: 1.2 s of these before the window
            # had widened enough for a gate-refused match). A refused match,
            # so it counts toward the "rejected" state - never applied.
            info["reason"] = "ICP left the search window"
            return MapMatch(stamp, "map_lidar", pose, cov, inlier, {**info, "rejected": True, "mismatch": True})
        if inlier < cfg.min_inlier_fraction:
            info["reason"] = f"inlier fraction {inlier:.2f} < {cfg.min_inlier_fraction}"
            m = MapMatch(stamp, "map_lidar", pose, cov, inlier, info)
            m.info["rejected"] = True
            return m
        return MapMatch(stamp, "map_lidar", pose, cov, inlier, info)


# ------------------------------------------------------------------ camera
class StaticBodyMask:
    """Learn which pixels show the kart itself, for ANY camera, from driving.

    The kart's own body (chassis, a wheel, the nose) is fixed in the image
    of a rigidly mounted camera, never looks like ground and never moves,
    while the world streams past. So, over frames taken while the kart
    MOVES: a pixel classified obstacle in >= ``obstacle_frac`` of them whose
    grey level varies by <= ``std_max`` is body, when it belongs to a region
    touching the image's bottom edge (a wall driven alongside also stays put
    in the image, but its texture moves and it does not reach the bottom
    row). Until ``min_frames`` moving frames were seen the mask is empty and
    ``min_contact_range_m`` of :class:`CameraGroundCue` alone keeps body
    contacts out. No camera model, no hand-drawn polygon.
    """

    def __init__(self, shape: tuple[int, int], min_frames: int = 12, obstacle_frac: float = 0.9, std_max: float = 14.0, dilate_px: int = 2) -> None:
        self.shape = tuple(shape)
        self.min_frames = int(min_frames)
        self.obstacle_frac = float(obstacle_frac)
        self.std_max = float(std_max)
        self.dilate_px = int(dilate_px)
        self.n = 0
        self._sum = np.zeros(self.shape)
        self._sumsq = np.zeros(self.shape)
        self._obst = np.zeros(self.shape)
        self.mask: NDArray[np.bool_] | None = None

    def update(self, gray: NDArray[np.uint8], obstacle_mask: NDArray[np.bool_], anchor: NDArray[np.bool_] | None = None) -> None:
        """Add one frame taken while moving. ``anchor``: pixels already known
        to be body (e.g. the classifier's static bottom-band mask) - a body
        region may reach the bottom row through them."""
        g = np.asarray(gray, dtype=np.float64)
        self._sum += g
        self._sumsq += g * g
        self._obst += obstacle_mask
        self.n += 1
        if self.n >= self.min_frames and (self.n - self.min_frames) % 4 == 0:
            mean = self._sum / self.n
            std = np.sqrt(np.maximum(self._sumsq / self.n - mean * mean, 0.0))
            cand = ((self._obst / self.n) >= self.obstacle_frac) & (std <= self.std_max)
            linked = cand if anchor is None else (cand | anchor)
            _, labels = cv2.connectedComponents(linked.astype(np.uint8), connectivity=8)
            bottom = np.unique(labels[-1][linked[-1]])
            mask = np.isin(labels, bottom[bottom > 0]) & cand
            if self.dilate_px > 0 and mask.any():
                k = 2 * self.dilate_px + 1
                mask = cv2.dilate(mask.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
            self.mask = mask


class CameraGroundCue:
    """Per-frame ground evidence of one camera: drivable cells + obstacle contacts.

    The zero-network classical ground classifier runs on a ``work_width``-px
    copy of the frame (its own internal resolution, so no pixel is thrown
    away that it would have used); drivable pixels and each column's lowest
    obstacle pixel (the ground contact) are intersected with the ground plane
    ``z = ground_z`` of base_link through precomputed per-pixel rays - the
    geometry of the mapping pipeline's ``ground_observations`` (range gate
    ``max_range_m``, occluded columns cut at the contact, points deduplicated
    to ``cell_m``). ``K`` must match the images passed in (any resolution).

    The kart's own body must not become an obstacle in front of it: contacts
    closer than ``min_contact_range_m`` are dropped (the body projects there)
    and a :class:`StaticBodyMask` learned while driving removes the body
    pixels themselves (pass ``moving=True`` to :meth:`observe` when the kart
    rolls); an explicit ``kart_mask`` image (True = body) can be given too.
    Contacts beyond ``max_contact_range_m`` are dropped: the far end of the
    drivable region is mostly where the classifier stopped growing it (the
    horizon band), not an obstacle.
    """

    def __init__(
        self,
        K: NDArray[np.float64],
        T_base_cam: NDArray[np.float64],
        image_size: tuple[int, int],
        ground_z: float = -0.10,
        max_range_m: float = 12.0,
        cell_m: float = 0.20,
        work_width: int = 320,
        kart_mask: NDArray[np.bool_] | None = None,
        apply_kart_mask: bool = True,
        min_contact_range_m: float = 2.0,
        max_contact_range_m: float = 10.0,
        learn_body_mask: bool = True,
    ) -> None:
        from kart_map2d.perception.classical_ground import ClassicalGroundSegmenter

        w, h = int(image_size[0]), int(image_size[1])
        scale = min(float(work_width) / w, 1.0)
        self.scale = scale
        self.size = (max(int(round(w * scale)), 1), max(int(round(h * scale)), 1))
        Ks = np.asarray(K, dtype=np.float64).copy()
        Ks[:2] *= scale
        self.K = Ks
        mask = None
        if kart_mask is not None:
            mask = cv2.resize(np.asarray(kart_mask, dtype=np.uint8), self.size, interpolation=cv2.INTER_NEAREST) > 0
        self.segmenter = ClassicalGroundSegmenter(
            K=Ks, T_base_cam=np.asarray(T_base_cam, dtype=np.float64), ground_z=ground_z, work_width=self.size[0],
            kart_mask=mask, apply_kart_mask=apply_kart_mask,
        )
        T = np.asarray(T_base_cam, dtype=np.float64)
        rays = _pixel_rays(Ks, (self.size[1], self.size[0])) @ T[:3, :3].T
        cam = T[:3, 3]
        dz = rays[..., 2]
        self.below = dz < -1e-9
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(self.below, (ground_z - cam[2]) / dz, np.inf)
            self.gx = cam[0] + t * rays[..., 0]
            self.gy = cam[1] + t * rays[..., 1]
            self.rng = np.hypot(self.gx - cam[0], self.gy - cam[1])
        self.max_range = float(max_range_m)
        self.min_contact = float(min_contact_range_m)
        self.max_contact = float(max_contact_range_m)
        self.cell = float(cell_m)
        self.body = StaticBodyMask((self.size[1], self.size[0])) if learn_body_mask else None
        self.frames = 0
        self.seconds = 0.0

    def _dedupe(self, xs: NDArray[np.float64], ys: NDArray[np.float64]) -> NDArray[np.float64]:
        if not len(xs):
            return np.zeros((0, 2))
        idx = np.unique(np.floor(np.column_stack([xs, ys]) / self.cell).astype(np.int64), axis=0)
        return idx * self.cell + 0.5 * self.cell

    def observe(self, image_bgr: NDArray[np.uint8], moving: bool = False) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
        """(drivable_xy, obstacle_xy, quality) in base_link [m] for one BGR frame.

        ``moving``: the kart is rolling (feeds the body-mask learner).
        """
        t0 = time.perf_counter()
        img = image_bgr
        if (img.shape[1], img.shape[0]) != self.size:
            img = cv2.resize(img, self.size, interpolation=cv2.INTER_AREA)
        seg = self.segmenter.segment(img)
        h, w = self.below.shape
        obst = seg.obstacle_mask
        drivable = seg.drivable_mask
        if self.body is not None:
            if moving:
                gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                self.body.update(gray, obst, seg.kart_mask)
            if self.body.mask is not None:
                obst = obst & ~self.body.mask
                drivable = drivable & ~self.body.mask
        # Obstacle pixels nearer than min_contact are the kart's own body (or
        # not resolvable from it): they neither count nor hide the ground behind.
        obst = obst & ~(self.below & (self.rng < self.min_contact))
        col_limit = np.full(w, np.inf)
        cx, cy = np.empty(0), np.empty(0)
        if obst.any():
            cols = np.flatnonzero(obst.any(axis=0))
            rows = (h - 1) - np.argmax(obst[::-1, :][:, cols], axis=0)
            ok = self.below[rows, cols]
            col_limit[cols[ok]] = self.rng[rows[ok], cols[ok]]
            r = self.rng[rows, cols]
            keep = ok & (r <= min(self.max_range, self.max_contact))
            cx, cy = self.gx[rows[keep], cols[keep]], self.gy[rows[keep], cols[keep]]
        sel = drivable & self.below & (self.rng <= self.max_range) & (self.rng <= col_limit[None, :])
        drv = self._dedupe(self.gx[sel], self.gy[sel])
        obs = self._dedupe(cx, cy)
        self.frames += 1
        self.seconds += time.perf_counter() - t0
        return drv, obs, float(seg.quality)

    @property
    def ms_per_frame(self) -> float:
        return 1e3 * self.seconds / self.frames if self.frames else 0.0


@dataclass
class PatchMatchConfig:
    """Camera patch matching (loop-closure correlation, tracking-sized windows)."""

    cell_m: float = 0.20
    half_size_m: float = 10.0
    center_forward_m: float = 4.0
    window_s: float = 4.0  # observations from the last window_s seconds ...
    window_m: float = 8.0  # ... and at most this much travel back
    min_frames: int = 2
    saturate_hits: int = 3
    window_sigmas: float = 3.0
    min_window_xy_m: float = 0.6
    max_window_xy_m: float = 3.0
    min_window_yaw_deg: float = 4.0
    #: IMU + visual odometry hold yaw to a degree or two; a camera patch that
    #: needs to turn the kart by more is a false match (measured on sept3:
    #: the camera matches that slipped past a wide window were 20-30 deg off).
    max_window_yaw_deg: float = 10.0
    coarse_step_deg: float = 3.0
    fine_step_deg: float = 1.0
    score_min: float = 0.45
    min_overlap_m2: float = 12.0
    min_obstacle_m2: float = 1.0
    blur_m: float = 0.4
    amb_margin: float = 0.05
    #: Forward-axis (range) error of a monocular ground projection: camera
    #: height / pitch errors scale ranges, added in quadrature on the
    #: keyframe's x axis (loop closure uses 1.5 m for opposed views).
    range_bias_m: float = 0.5
    sigma_xy_floor_m: float = 0.25
    #: Camera fixes barely steer the heading: the gyro and VO measure it better.
    sigma_yaw_floor_deg: float = 5.0
    sigma_xy_max_m: float = 2.5
    sigma_yaw_max_deg: float = 12.0
    #: ALONG-TRACK sigma floor [m]. A monocular ground patch in a corridor
    #: has no reliable along-corridor anchor: the drivable band is
    #: translation-invariant and the sparse edge blobs repeat, so a locally
    #: peaked correlation ridge still ratchets the along belief fix after
    #: fix (closed-loop measurement: 147/147 fixes accepted while the true
    #: along error grew to 1.8 m and the kart obstacle-stopped 3 m in).
    #: Camera fixes therefore constrain LATERAL + yaw; along-track stays
    #: with the odometry (VO), whose corridor-scale drift is percent-level.
    sigma_along_floor_m: float = 1.5


@dataclass
class CameraObservation:
    stamp: float
    drivable_xy: NDArray[np.float64]  # (N, 2) base_link at stamp
    obstacle_xy: NDArray[np.float64]


class PatchMatcher:
    """Correlate a local camera evidence patch with the map (loop-closure NCC)."""

    def __init__(self, model: MapModel, config: PatchMatchConfig | None = None) -> None:
        self.model = model
        self.config = config or PatchMatchConfig()

    def params(self, win_xy: float | None = None, win_yaw_deg: float | None = None) -> LoopClosureParams:
        c = self.config
        return LoopClosureParams(
            half_size_m=c.half_size_m,
            center_forward_m=c.center_forward_m,
            cell_m=c.cell_m,
            saturate_hits=c.saturate_hits,
            yaw_gate_deg=win_yaw_deg if win_yaw_deg is not None else c.max_window_yaw_deg,
            coarse_step_deg=c.coarse_step_deg,
            fine_step_deg=c.fine_step_deg,
            coarse_factor=2,
            blur_m=c.blur_m,
            min_overlap_m2=c.min_overlap_m2,
            min_obstacle_m2=c.min_obstacle_m2,
            score_min=c.score_min,
            amb_margin=c.amb_margin,
            obstacle_weight=self.model.config.obstacle_weight,
            range_bias_m=c.range_bias_m,
            sigma_xy_closure_m=c.sigma_xy_floor_m,
            sigma_xy_max_m=c.sigma_xy_max_m,
            sigma_yaw_closure_deg=c.sigma_yaw_floor_deg,
            sigma_yaw_max_deg=c.sigma_yaw_max_deg,
        )

    def live_patch(self, observations: Sequence[CameraObservation], poses: NDArray[np.float64], key_pose: NDArray[np.float64], key_stamp: float) -> LocalPatch:
        """Rasterize observations (their EKF poses ``poses``) into ``key_pose``'s frame.

        Hit counts are one per frame and cell and saturate at
        ``saturate_hits`` (a standstill cannot outvote moving coverage), as in
        :func:`kart_map2d.mapping.loop_closure.build_local_patches`.
        """
        c = self.config
        half = int(round(c.half_size_m / c.cell_m))
        size = 2 * half + 1
        center = np.array([c.center_forward_m, 0.0])
        R_key_inv = _rot(float(key_pose[2])).T
        counts = {}
        for key in ("drv", "obs"):
            keys = []
            for oi, (o, pose) in enumerate(zip(observations, poses)):
                pts = o.drivable_xy if key == "drv" else o.obstacle_xy
                if not len(pts):
                    continue
                world = np.asarray(pts, dtype=np.float64) @ _rot(float(pose[2])).T + pose[:2]
                local = (world - key_pose[:2]) @ R_key_inv.T
                ij = np.rint((local - center) / c.cell_m).astype(np.int64) + half
                ok = (ij >= 0).all(axis=1) & (ij < size).all(axis=1)
                if ok.any():
                    keys.append(oi * size * size + ij[ok, 0] * size + ij[ok, 1])
            grid = np.zeros(size * size, dtype=np.int32)
            if keys:
                uniq = np.unique(np.concatenate(keys))
                grid = np.bincount(uniq % (size * size), minlength=size * size).astype(np.int32)
            counts[key] = grid.reshape(size, size)
        drv = np.minimum(counts["drv"], c.saturate_hits) / c.saturate_hits
        obs = np.minimum(counts["obs"], c.saturate_hits) / c.saturate_hits
        return LocalPatch(
            stamp=float(key_stamp),
            xyyaw=np.asarray(key_pose, dtype=np.float64).copy(),
            grid=(obs - drv).astype(np.float32),
            seen=(counts["drv"] + counts["obs"]) > 0,
            cell=c.cell_m,
            center_forward_m=c.center_forward_m,
            path_m=0.0,
        )

    def match(
        self,
        live: LocalPatch,
        prior_pose: NDArray[np.float64],
        prior_cov: NDArray[np.float64],
        window: tuple[float, float] | None = None,
        info_out: dict | None = None,
    ) -> MapMatch | None:
        c = self.config
        t0 = time.perf_counter()
        if window is None:
            win_xy, win_yaw = _window(
                prior_cov, c.window_sigmas, c.min_window_xy_m, c.max_window_xy_m, math.radians(c.min_window_yaw_deg), math.radians(c.max_window_yaw_deg)
            )
        else:
            win_xy, win_yaw = window
        p = self.params(win_xy, math.degrees(win_yaw))
        map_patch = patches_from_grid(self.model.classes_view, np.array([live.stamp]), np.asarray(prior_pose, dtype=np.float64)[None, :], p)[0]
        debug: dict = {}
        m = _correlate_pair(map_patch, live, np.zeros(3), win_xy, p, debug)
        info = {
            "ms": round(1e3 * (time.perf_counter() - t0), 1),
            "window_xy_m": round(win_xy, 2),
            "window_yaw_deg": round(math.degrees(win_yaw), 1),
            "live_cells": int(live.seen.sum()),
        }
        if info_out is not None:
            info_out.update(info)  # refusal reasons must be visible to callers
        if m is None:
            info["reason"] = debug.get("reason", "no correlation peak")
            if info_out is not None:
                info_out["reason"] = info["reason"]
            return None
        if m.score < c.score_min:
            if info_out is not None:
                info_out["reason"] = f"score {m.score:.2f} < {c.score_min}"
            return None
        pose = se2_compose(np.asarray(prior_pose, dtype=np.float64), m.xyyaw_ab)
        R = _rot(float(prior_pose[2]))
        # Axis 0 of the correlation sigma is the key frame's heading = the
        # along-corridor direction: floored to sigma_along_floor_m so a
        # camera fix never claims along-track knowledge it does not have
        # (see the config note - the repetitive corridor ratchet).
        sig_along = max(float(m.sigma_xyyaw[0]), c.sigma_along_floor_m)
        sig_lat = float(m.sigma_xyyaw[1])
        cov = np.zeros((3, 3))
        cov[:2, :2] = R @ np.diag([sig_along**2, sig_lat**2]) @ R.T
        cov[2, 2] = float(m.sigma_xyyaw[2]) ** 2
        info.update(score=round(float(m.score), 3), overlap_m2=round(float(m.overlap_m2), 1), sigma=[round(float(v), 3) for v in m.sigma_xyyaw])
        return MapMatch(live.stamp, "map_camera", pose, cov, float(m.score), info)


# ================================================================ global search
@dataclass
class GlobalConfig:
    lidar_yaw_step_deg: float = 3.0
    camera_yaw_step_deg: float = 6.0
    candidates: int = 8
    distinct_xy_m: float = 2.0
    distinct_yaw_deg: float = 20.0
    lidar_min_inliers: float = 0.55
    lidar_ambiguity_ratio: float = 0.85  # second-best distinct / best must stay below
    camera_min_score: float = 0.40
    camera_ambiguity_margin: float = 0.08  # best - second-best distinct NCC must exceed
    camera_cell_m: float = 0.40
    refine_window_xy_m: float = 1.2
    refine_window_yaw_deg: float = 8.0


@dataclass
class GlobalResult:
    ok: bool
    pose: NDArray[np.float64] | None
    cov: NDArray[np.float64] | None
    score: float
    second: float
    reason: str
    candidates: list = field(default_factory=list)
    ms: float = 0.0


def _distinct(cands: list[tuple[float, NDArray[np.float64]]], dxy: float, dyaw: float) -> list[tuple[float, NDArray[np.float64]]]:
    out: list[tuple[float, NDArray[np.float64]]] = []
    for sc, pose in sorted(cands, key=lambda c: -c[0]):
        if all(math.hypot(pose[0] - q[0], pose[1] - q[1]) > dxy or abs(wrap(pose[2] - q[2])) > dyaw for _, q in out):
            out.append((sc, pose))
    return out


def global_search_lidar(model: MapModel, matcher: ScanMatcher, P: NDArray[np.float64], cfg: GlobalConfig | None = None, stamp: float = 0.0) -> GlobalResult:
    """Search the whole map for the scan: FFT correlation per yaw on the coarse field.

    Candidate positions are known-drivable coarse cells (the kart stands on
    ground it could drive); the ``candidates`` best distinct peaks are
    refined with the tracking matcher, and a pose is only returned when its
    ICP inlier fraction clears ``lidar_min_inliers`` AND the best DISTINCT
    alternative stays below ``lidar_ambiguity_ratio`` of it.
    """
    cfg = cfg or GlobalConfig()
    t0 = time.perf_counter()
    if model.tree is None or len(P) < matcher.config.min_points:
        return GlobalResult(False, None, None, 0.0, 0.0, "too few obstacle points or an empty map")
    cc = model.cell * model.coarse_factor
    Pc = voxel_downsample(P, cc, 600)
    R = int(math.ceil(np.max(np.linalg.norm(Pc, axis=1)) / cc)) + 1
    field_c = model.coarse.astype(np.float32)
    cand_mask = model.coarse_drivable if model.coarse_drivable.any() else np.ones_like(model.coarse_drivable)
    peaks: list[tuple[float, NDArray[np.float64]]] = []
    for yaw in np.radians(np.arange(0.0, 360.0, cfg.lidar_yaw_step_deg)):
        Q = Pc @ _rot(yaw).T
        k = np.rint(Q / cc).astype(np.int64) + R
        kernel = np.zeros((2 * R + 1, 2 * R + 1), dtype=np.float32)
        kernel[k[:, 0], k[:, 1]] = 1.0
        n = float(kernel.sum())
        score = cv2.filter2D(field_c, -1, kernel, anchor=(R, R), borderType=cv2.BORDER_CONSTANT) / max(n, 1.0)
        score = np.where(cand_mask, score, -1.0)
        # Keep a few local maxima per yaw.
        mx = maximum_filter(score, size=5, mode="constant", cval=-1.0)
        cand = np.argwhere((score == mx) & (score > 0.0))
        if not len(cand):
            continue
        vals = score[cand[:, 0], cand[:, 1]]
        for j in np.argsort(-vals)[:3]:
            ix, iy = cand[j]
            xy = model.origin + (np.array([ix, iy]) + 0.5) * cc
            peaks.append((float(vals[j]), np.array([xy[0], xy[1], wrap(yaw)])))
    coarse = _distinct(peaks, cfg.distinct_xy_m, math.radians(cfg.distinct_yaw_deg))[: cfg.candidates]
    refined = []
    win = (cfg.refine_window_xy_m, math.radians(cfg.refine_window_yaw_deg))
    for sc, pose in coarse:
        m = matcher.match(P, pose, np.diag([1.0, 1.0, 0.1]), stamp=stamp, window=win)
        if m is not None and not m.info.get("mismatch"):
            refined.append((m.score if not m.info.get("rejected") else 0.5 * m.score, m.pose, m.cov, sc))
    ms = 1e3 * (time.perf_counter() - t0)
    cands = [{"xyyaw": [round(float(v), 3) for v in p], "inliers": round(s, 3), "coarse": round(c, 3)} for s, p, _, c in refined]
    if not refined:
        return GlobalResult(False, None, None, 0.0, 0.0, "no candidate survived refinement", cands, ms)
    refined.sort(key=lambda r: -r[0])
    best = refined[0]
    others = [r for r in refined[1:] if math.hypot(r[1][0] - best[1][0], r[1][1] - best[1][1]) > cfg.distinct_xy_m or abs(wrap(r[1][2] - best[1][2])) > math.radians(cfg.distinct_yaw_deg)]
    second = others[0][0] if others else 0.0
    if best[0] < cfg.lidar_min_inliers:
        return GlobalResult(False, None, None, best[0], second, f"best inlier fraction {best[0]:.2f} < {cfg.lidar_min_inliers}", cands, ms)
    if second > cfg.lidar_ambiguity_ratio * best[0]:
        return GlobalResult(False, None, None, best[0], second, f"ambiguous: a distinct pose scores {second:.2f} vs {best[0]:.2f}", cands, ms)
    return GlobalResult(True, best[1], best[2], best[0], second, "unique", cands, ms)


def global_search_camera(model: MapModel, matcher: PatchMatcher, live: LocalPatch, cfg: GlobalConfig | None = None) -> GlobalResult:
    """Search the whole map for the camera patch (masked NCC per yaw, coarse cells).

    The same acceptance logic as the LiDAR search, on NCC scores: absolute
    floor ``camera_min_score`` and a margin over the best distinct pose.
    """
    cfg = cfg or GlobalConfig()
    t0 = time.perf_counter()
    pc = matcher.config
    cell = cfg.camera_cell_m
    fmap, seen, obst, origin = model.camera_field(cell)
    w = model.config.obstacle_weight
    fmap_w = np.where(fmap > 0, fmap * w, fmap)
    blur_cells = pc.blur_m / cell
    fmap_b = _masked_blur(fmap_w, seen, blur_cells)
    factor = max(int(round(cell / live.cell)), 1)
    patch_f = _downsample(live.grid, factor)
    patch_m = (_downsample(live.seen.astype(np.float64), factor) >= 0.5).astype(np.float64)
    patch_p = _downsample((live.grid > 0).astype(np.float64), factor)
    patch_f = np.where(patch_f > 0, patch_f * w, patch_f)
    patch_f = _masked_blur(patch_f, patch_m, blur_cells)
    if patch_m.sum() * cell * cell < pc.min_overlap_m2:
        return GlobalResult(False, None, None, 0.0, 0.0, "too little camera evidence for a global search")
    H = patch_f.shape[0]
    h = (H - 1) / 2.0
    min_ov = pc.min_overlap_m2 / (cell * cell)
    min_pos = pc.min_obstacle_m2 / (cell * cell)
    peaks: list[tuple[float, NDArray[np.float64]]] = []
    center_fwd = live.center_forward_m
    for yaw in np.radians(np.arange(0.0, 360.0, cfg.camera_yaw_step_deg)):
        rf, rm, rp = _rotated([patch_f, patch_m, patch_p], float(yaw))
        ncc, _ = _masked_ncc(fmap_b, seen, obst, rf, (rm >= 0.5).astype(np.float64), rp, min_ov, min_pos)
        if not np.isfinite(ncc).any():
            continue
        mx = maximum_filter(np.where(np.isfinite(ncc), ncc, -1.0), size=5, mode="constant", cval=-1.0)
        cand = np.argwhere((ncc == mx) & np.isfinite(ncc))
        vals = ncc[cand[:, 0], cand[:, 1]]
        for j in np.argsort(-vals)[:3]:
            kx, ky = cand[j]
            # Patch centre lands on map cell (k - (H - 1) + h); the keyframe
            # sits center_forward behind it along yaw.
            cxy = origin + (np.array([kx, ky]) - (H - 1) + h + 0.5) * cell
            xy = cxy - _rot(float(yaw)) @ np.array([center_fwd, 0.0])
            peaks.append((float(vals[j]), np.array([xy[0], xy[1], wrap(float(yaw))])))
    coarse = _distinct(peaks, cfg.distinct_xy_m, math.radians(cfg.distinct_yaw_deg))[: cfg.candidates]
    refined = []
    win = (cfg.refine_window_xy_m, math.radians(cfg.refine_window_yaw_deg))
    for sc, pose in coarse:
        m = matcher.match(replace(live, xyyaw=pose), pose, np.diag([1.0, 1.0, 0.1]), window=win)
        if m is not None:
            refined.append((m.score, m.pose, m.cov, sc))
    ms = 1e3 * (time.perf_counter() - t0)
    cands = [{"xyyaw": [round(float(v), 3) for v in p], "ncc": round(s, 3), "coarse": round(c, 3)} for s, p, _, c in refined]
    if not refined:
        return GlobalResult(False, None, None, 0.0, 0.0, "no candidate survived refinement", cands, ms)
    refined.sort(key=lambda r: -r[0])
    best = refined[0]
    others = [r for r in refined[1:] if math.hypot(r[1][0] - best[1][0], r[1][1] - best[1][1]) > cfg.distinct_xy_m or abs(wrap(r[1][2] - best[1][2])) > math.radians(cfg.distinct_yaw_deg)]
    second = others[0][0] if others else -1.0
    if best[0] < cfg.camera_min_score:
        return GlobalResult(False, None, None, best[0], second, f"best NCC {best[0]:.2f} < {cfg.camera_min_score}", cands, ms)
    if best[0] - second < cfg.camera_ambiguity_margin:
        return GlobalResult(False, None, None, best[0], second, f"ambiguous: a distinct pose scores {second:.2f} vs {best[0]:.2f}", cands, ms)
    return GlobalResult(True, best[1], best[2], best[0], second, "unique", cands, ms)


# ================================================================ the localizer
@dataclass
class LocalizerConfig:
    """Initialization, fix rates, convergence and recovery tunables."""

    #: Operator-given start pose in the map frame (x [m], y [m], yaw [rad]);
    #: None -> global relocalization.
    start_pose: tuple[float, float, float] | None = None
    start_sigma_xy_m: float = 0.5
    start_sigma_yaw_rad: float = math.radians(5.0)
    global_relocalization: bool = True
    #: Retry the global search at most this often [s] while searching.
    global_retry_s: float = 1.0
    lidar_rate_hz: float = 5.0
    camera_rate_hz: float = 1.0
    #: Camera patch fixes: "auto" = only against maps WITHOUT LiDAR evidence.
    #: A LiDAR-built grid calls flat grass / kerbs / steps drivable (ground
    #: height) where the zero-network camera classifier sees non-drivable, and
    #: its obstacles are mostly beyond the camera's ground range: measured on
    #: sept3, 97-99 % of the camera's obstacle contacts land on the LiDAR map's
    #: DRIVABLE cells at the true pose, and its fixes turned the heading by
    #: 20-30 deg. "on" / "off" force it.
    camera_fixes: str = "auto"
    #: Converged = initialized, >= converged_min_fixes accepted fixes, the
    #: EKF pose 1-sigma below these, and a fix accepted within max_fix_age_s.
    converged_min_fixes: int = 3
    #: ... in a row that AGREE with the filter (NIS below chi2(3 dof, 95 %)):
    #: the first fix after a wrong start pulls the state, the next ones must
    #: confirm it before the kart may move.
    agree_nis: float = 7.81
    converged_sigma_xy_m: float = 0.5
    converged_sigma_yaw_rad: float = math.radians(5.0)
    max_fix_age_s: float = 8.0
    #: This many consecutive refused fixes put the localizer in state
    #: "rejected" (not localized: the safety layer stops the kart).
    rejected_after: int = 3
    #: Recovery: this many consecutive refused fixes that agree with each
    #: other re-anchor the pose on them.
    reanchor_after_rejects: int = 4
    reanchor_consistency_m: float = 0.6
    reanchor_consistency_rad: float = math.radians(6.0)
    #: A fix older than the EKF by more than this is dropped, not carried.
    max_fix_delay_s: float = 1.0
    #: While no fix is accepted the match window widens with the distance
    #: travelled since the last accepted one (re-acquisition after losing
    #: track; the matchers clamp it to their maximum windows). By DISTANCE:
    #: a parked kart cannot drift, and widening it by time would only invite
    #: an alias into the window.
    widen_xy_per_m: float = 0.10
    widen_yaw_rad_per_m: float = math.radians(0.5)
    #: Re-anchoring needs the refused fixes to come from this much travel
    #: (different viewpoints agreeing): a parked kart re-matching one scan
    #: repeats the same alias, a moving one does not.
    reanchor_min_travel_m: float = 2.0


@dataclass
class FixOutcome:
    stamp: float
    source: str
    accepted: bool | None  # None = dropped before the gate (late / channel faulty)
    pose: NDArray[np.float64]
    score: float
    nis: float | None = None
    note: str = ""
    #: Largest 1-sigma of the fix's position covariance [m] (its weakest axis).
    cov_max_sigma: float = 0.0


@dataclass
class LocalizerStatus:
    state: str  # uninitialized | searching | converging | tracking | rejected | stale
    localized: bool
    initialized: bool
    sigma_xy_m: float
    sigma_yaw_rad: float
    consecutive_rejects: int
    fixes_accepted: int
    fixes_rejected: int
    last_fix_age_s: float | None
    reanchors: int
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "state": self.state,
            "localized": self.localized,
            "initialized": self.initialized,
            "sigma_xy_m": round(self.sigma_xy_m, 3) if math.isfinite(self.sigma_xy_m) else None,
            "sigma_yaw_deg": round(math.degrees(self.sigma_yaw_rad), 2) if math.isfinite(self.sigma_yaw_rad) else None,
            "consecutive_rejects": self.consecutive_rejects,
            "fixes_accepted": self.fixes_accepted,
            "fixes_rejected": self.fixes_rejected,
            "last_fix_age_s": None if self.last_fix_age_s is None else round(self.last_fix_age_s, 2),
            "reanchors": self.reanchors,
            "note": self.note,
        }


class MapLocalizer:
    """Odometry fusion + map fixes + initialization/recovery; see the module docstring.

    NOT thread-safe: callers serialize access (the drive node holds its
    engine lock around every call that touches ``engine``; the matchers'
    heavy work - :meth:`ScanMatcher.match`, :meth:`PatchMatcher.match`,
    the global searches - runs outside it).
    """

    def __init__(
        self,
        model: MapModel,
        config: LocalizerConfig | None = None,
        engine: DriveFusionEngine | None = None,
        scan_config: ScanMatchConfig | None = None,
        patch_config: PatchMatchConfig | None = None,
        global_config: GlobalConfig | None = None,
    ) -> None:
        self.model = model
        self.config = config or LocalizerConfig()
        self.engine = engine or DriveFusionEngine()
        self.scan_matcher = ScanMatcher(model, scan_config)
        self.patch_matcher = PatchMatcher(model, patch_config)
        self.global_config = global_config or GlobalConfig()
        self.initialized = False
        self.fixes_accepted = 0
        self.fixes_rejected = 0
        self.accepted_since_init = 0
        self.agree_streak = 0
        self.converged_latch = False
        self.consecutive_rejects = 0
        self.last_accept_stamp: float | None = None
        self._accept_pose: NDArray[np.float64] | None = None  # EKF pose at the last accepted fix
        self.reanchors = 0
        self.outcomes: deque[FixOutcome] = deque(maxlen=5000)
        self._rejected_streak: list[tuple[FixOutcome, NDArray[np.float64]]] = []
        #: Fixes queued in the engine, not yet processed: (stamp, source, pose, cov, score).
        self._pending: list[tuple] = []
        self._next_lidar = -np.inf
        self._next_camera = -np.inf
        self._next_global = -np.inf
        self._observations: deque[CameraObservation] = deque(maxlen=64)
        mode = str(self.config.camera_fixes).lower()
        if mode not in ("auto", "on", "off"):
            raise ValueError(f"camera_fixes must be auto | on | off, got {self.config.camera_fixes!r}")
        #: Whether camera patches may produce fixes against THIS map (see LocalizerConfig.camera_fixes).
        self.camera_fixes_enabled = mode == "on" or (mode == "auto" and not model.has_lidar_evidence)
        self.global_attempts: list[dict] = []
        self.events: list[tuple[float, str, str]] = []
        if self.config.start_pose is not None:
            sp = self.config.start_pose
            self._pending_start = np.array([float(sp[0]), float(sp[1]), float(sp[2])])
        else:
            self._pending_start = None

    # ------------------------------------------------------------ odometry
    def push(self, m: Measurement, arrival: float | None = None) -> None:
        """An odometry / IMU measurement for the fusion engine."""
        self.engine.push(m, arrival)
        if self._pending:
            self.resolve()
        if self._pending_start is not None and self.engine.ekf.stamp is not None:
            self.initialize(self._pending_start, self.config.start_sigma_xy_m, self.config.start_sigma_yaw_rad, "start_pose")
            self._pending_start = None

    def restart_global(self, stamp: float | None = None, reason: str = "") -> None:
        """Forget where the kart is and search the whole map again.

        ``~/relocalize`` and the drive node's lost-kart fallback call this:
        the next LiDAR scan / camera patch starts a global search (at most
        every ``global_retry_s``) and only a clearly unique answer
        re-initializes the filter, which must then converge again (3
        agreeing fixes) before the kart may move.
        """
        self.initialized = False
        self._pending_start = None
        self._next_global = -np.inf
        self.agree_streak = 0
        self.converged_latch = False
        self.consecutive_rejects = 0
        self._rejected_streak.clear()
        t = stamp if stamp is not None else float(self.engine.ekf.stamp or 0.0)
        self.events.append((t, "relocalize", reason or "global search restarted"))

    def initialize(self, pose: NDArray[np.float64], sigma_xy: float, sigma_yaw: float, why: str) -> None:
        """Place the filter at ``pose`` (map frame) with the given 1-sigma."""
        cov = np.diag([sigma_xy**2, sigma_xy**2, sigma_yaw**2])
        # Everything before initialization was in the filter's own frame.
        self.engine.set_pose(np.asarray(pose, dtype=np.float64), cov, history_s=None)
        self.initialized = True
        self.accepted_since_init = 0
        self.agree_streak = 0
        self.converged_latch = False
        self.consecutive_rejects = 0
        self._rejected_streak.clear()
        stamp = float(self.engine.ekf.stamp or 0.0)
        c = self.config
        if why == "start_pose" and sigma_xy <= 0.5 * c.converged_sigma_xy_m and sigma_yaw <= 0.5 * c.converged_sigma_yaw_rad:
            # A TRUSTED start (the operator put the kart on a surveyed spot,
            # e.g. where the mapping run began, and says so with a tight
            # sigma) counts as the first fix: the kart may drive on odometry
            # for up to max_fix_age_s before a map fix must confirm it, and
            # map fixes that disagree are refused -> rejected -> stop.
            self.converged_latch = True
            self.last_accept_stamp = stamp
        self._accept_pose = np.asarray(pose, dtype=np.float64).copy()
        self.events.append((stamp, "initialize", f"{why}: x={pose[0]:.2f} y={pose[1]:.2f} yaw={math.degrees(pose[2]):.1f} deg"))

    # ------------------------------------------------------------ priors
    def prior_at(self, stamp: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """EKF pose + the covariance that sizes the match window at ``stamp``.

        The EKF covariance, widened while no fix has been accepted for a
        while (``widen_*``): after losing track the true pose may sit
        outside a window sized from an over-confident filter.
        """
        cov = self.engine.pose_covariance()
        pose = self.engine.pose_at(stamp)
        if self._accept_pose is not None:
            dist = float(np.hypot(*(pose[:2] - self._accept_pose[:2])))
            cov = cov + np.diag([(self.config.widen_xy_per_m * dist) ** 2] * 2 + [(self.config.widen_yaw_rad_per_m * dist) ** 2])
        return pose, cov

    def lidar_due(self, stamp: float) -> bool:
        return stamp >= self._next_lidar

    def camera_due(self, stamp: float) -> bool:
        return stamp >= self._next_camera

    # ------------------------------------------------------------ fixes
    def submit(self, match: MapMatch) -> FixOutcome | None:
        """Hand a finished match to the filter as a :class:`MapPoseFix`.

        A fix for a stamp the filter has already PASSED (its computation took
        longer than the reorder window) is carried forward to the filter's
        stamp with the filter's own recorded motion and applied at once. A
        fix for a stamp the filter has NOT reached yet is queued at its own
        stamp: the reorder buffer applies it in stamp order against the
        filter's real state then - no model extrapolation in between (a
        constant-turn-rate guess across a steering transient is off by
        degrees, which is exactly what an innovation gate punishes).
        Accept / reject is booked when the filter has processed the fix
        (:meth:`resolve`); the outcome is returned when already known.
        """
        ekf = self.engine.ekf
        if match.source == "map_lidar":
            self._next_lidar = match.stamp + 1.0 / max(self.config.lidar_rate_hz, 1e-6)
        else:
            self._next_camera = match.stamp + 1.0 / max(self.config.camera_rate_hz, 1e-6)
        if ekf.stamp is None or not self.initialized:
            return FixOutcome(match.stamp, match.source, None, match.pose, match.score, note="not initialized")
        if match.info.get("rejected"):
            out = FixOutcome(match.stamp, match.source, False, match.pose, match.score, note=match.info.get("reason", "matcher rejected"))
            self.outcomes.append(out)
            # The scan does not fit the map at the filter's pose (too few
            # inliers / nothing near the map's obstacles): evidence AGAINST
            # the pose, so it counts toward the "rejected" state (the safety
            # layer stops the kart) - without it a kidnapped kart or a wrong
            # trusted start pose stayed "tracking" on odometry until
            # max_fix_age_s (measured on sept3: 19 s with a wrong start
            # pose). Not a re-anchor candidate: its pose is not a measurement.
            self.fixes_rejected += 1
            self.consecutive_rejects += 1
            return out
        stamp = float(match.stamp)
        pose = np.asarray(match.pose, dtype=np.float64)
        cov = np.asarray(match.cov, dtype=np.float64)
        now = float(ekf.stamp)
        if now - stamp > self.config.max_fix_delay_s:
            out = FixOutcome(stamp, match.source, None, pose, match.score, note=f"{now - stamp:.2f} s late")
            self.outcomes.append(out)
            return out
        if stamp <= now:
            # Late: carry the fix forward with the filter's measured motion.
            rel = se2_relative(self.engine.pose_at(stamp), self.engine.pose_at(now))
            pose = se2_compose(pose, rel)
            grow = 0.02 * (now - stamp)
            cov = cov + np.diag([grow**2, grow**2, (0.2 * grow) ** 2])
            fix_stamp = now
            # Arrival one reorder window past it: applied in this very call.
            arrival = max(self.engine._latest_arrival, now + self.engine.reorder_window_s)
        else:
            fix_stamp = stamp  # queued: applied when the filter gets there
            arrival = self.engine._latest_arrival if np.isfinite(self.engine._latest_arrival) else stamp
        fix = MapPoseFix(fix_stamp, match.source, np.asarray(pose[:2], dtype=np.float64), 1.0, "map", float(pose[2]), tuple(cov.reshape(-1)), float(match.score))
        self._pending.append((fix_stamp, match.source, pose, cov, float(match.score)))
        self.engine.push(fix, arrival)
        resolved = self.resolve()
        for out in resolved:
            if out.stamp == fix_stamp and out.source == match.source:
                return out
        return None

    def resolve(self) -> list[FixOutcome]:
        """Book the fixes the filter has processed since the last call.

        A pending fix is resolved once the filter's stamp has reached it: its
        update-log entry says accepted / refused (with the NIS); no entry
        means it never reached the gate (channel faulty, late, frozen).
        """
        ekf = self.engine.ekf
        if not self._pending or ekf.stamp is None:
            return []
        now = float(ekf.stamp)
        done: list[FixOutcome] = []
        keep = []
        for fix_stamp, source, pose, cov, score in self._pending:
            if fix_stamp > now:
                keep.append((fix_stamp, source, pose, cov, score))
                continue
            entry = None
            for u in reversed(ekf.updates[-3000:]):
                if u.channel == source and u.stamp == fix_stamp:
                    entry = u
                    break
            if entry is None and fix_stamp == now and any(
                isinstance(item[2], MapPoseFix) and item[2].stamp == fix_stamp and item[2].channel == source for item in self.engine._queue
            ):
                keep.append((fix_stamp, source, pose, cov, score))  # a same-stamp tie still queued
                continue
            accepted = None if entry is None else bool(entry.accepted)
            nis = None if entry is None else float(entry.nis)
            out = FixOutcome(fix_stamp, source, accepted, pose, score, nis, cov_max_sigma=float(np.sqrt(max(np.linalg.eigvalsh(cov[:2, :2]).max(), 0.0))))
            self._book(out, self.engine.pose_at(fix_stamp), cov)
            done.append(out)
        self._pending = keep
        return done

    def _book(self, out: FixOutcome, ekf_pose: NDArray[np.float64], cov: NDArray[np.float64]) -> None:
        self.outcomes.append(out)
        if out.accepted:
            self.fixes_accepted += 1
            self.accepted_since_init += 1
            self.agree_streak = self.agree_streak + 1 if (out.nis is not None and out.nis <= self.config.agree_nis) else 0
            self.consecutive_rejects = 0
            self._rejected_streak.clear()
            self.last_accept_stamp = out.stamp
            self._accept_pose = np.asarray(ekf_pose, dtype=np.float64).copy()
        else:
            self.fixes_rejected += 1
            self.consecutive_rejects += 1
            self._rejected_streak.append((out, np.asarray(ekf_pose, dtype=np.float64).copy()))
            self._maybe_reanchor(out, cov)

    def _maybe_reanchor(self, latest: FixOutcome, cov: NDArray[np.float64]) -> None:
        c = self.config
        # The whole current streak of refused fixes (at most the last 12).
        streak = self._rejected_streak[-12:]
        if len(streak) < c.reanchor_after_rejects:
            return
        # Different viewpoints must agree: the streak spans real travel ...
        travel = float(np.hypot(*(streak[-1][1][:2] - streak[0][1][:2])))
        if travel < c.reanchor_min_travel_m:
            return
        # ... and each refused fix pins the pose in EVERY direction (a fix
        # along an unobservable corridor axis is no evidence of where the kart is).
        if any(o.cov_max_sigma > c.reanchor_consistency_m for o, _ in streak):
            return
        # The fixes must agree with each other: the map <- filter correction
        # C_i = fix_i (+) filter_i^-1 is CONSTANT when the filter's frame is
        # rigidly off (wrong start pose, kidnapped kart), whatever the kart
        # did in between; scattered matches give scattered corrections.
        corr = np.array([se2_compose(o.pose, se2_invert(ekf_pose)) for o, ekf_pose in streak])
        spread_xy = float(np.max(np.linalg.norm(corr[:, :2] - corr[:, :2].mean(axis=0), axis=1)))
        spread_yaw = float(np.max(np.abs(wrap(corr[:, 2] - corr[0, 2]))))
        if spread_xy > c.reanchor_consistency_m or spread_yaw > c.reanchor_consistency_rad:
            return
        offsets = np.array([se2_relative(ekf_pose, o.pose) for o, ekf_pose in streak])
        cov2 = np.asarray(cov, dtype=np.float64) * 2.0
        target = np.asarray(latest.pose, dtype=np.float64)
        ekf_stamp = float(self.engine.ekf.stamp)
        if latest.stamp < ekf_stamp:
            # the fix was for an earlier stamp: move it along the filter's motion since
            target = se2_compose(target, se2_relative(self.engine.pose_at(latest.stamp), self.engine.pose_at(ekf_stamp)))
        self.engine.set_pose(target, cov2)
        # The fix channel was being refused because the FILTER was wrong:
        # forget that history so it is heard again at once.
        h = self.engine.health.channel(latest.source)
        h.status = Status.OK
        h.consecutive_rejects = 0
        h.faulty_until = -np.inf
        h.reason = ""
        self.reanchors += 1
        self.accepted_since_init = 1
        self.agree_streak = 0
        self.converged_latch = False
        self.consecutive_rejects = 0
        self._rejected_streak.clear()
        self.last_accept_stamp = latest.stamp
        self._accept_pose = np.asarray(latest.pose, dtype=np.float64).copy()
        self.events.append((latest.stamp, "reanchor", f"{len(streak)} consistent refused fixes; offset {np.linalg.norm(offsets[-1, :2]):.2f} m / {math.degrees(offsets[-1, 2]):.1f} deg"))

    # ------------------------------------------------------------ LiDAR
    def lidar_match(self, P: NDArray[np.float64], stamp: float) -> MapMatch | None:
        """Tracking match of prepared points ``P`` at ``stamp`` (compute only)."""
        pose, cov = self.prior_at(stamp)
        return self.scan_matcher.match(P, pose, cov, stamp=stamp)

    def process_scan(self, points_base: NDArray[np.float64], stamp: float) -> FixOutcome | GlobalResult | None:
        """Synchronous LiDAR step (offline replay / tests): match + submit, or
        global search while uninitialized. Rate-limited to ``lidar_rate_hz``."""
        if not self.lidar_due(stamp):
            return None
        P = self.scan_matcher.prepare(points_base)
        if not self.initialized:
            if self._pending_start is not None or not self.config.global_relocalization:
                return None
            if stamp < self._next_global or self.engine.ekf.stamp is None:
                return None
            self._next_global = stamp + self.config.global_retry_s
            result = global_search_lidar(self.model, self.scan_matcher, P, self.global_config, stamp)
            self.apply_global(result, stamp, "lidar")
            return result
        self._next_lidar = stamp + 1.0 / max(self.config.lidar_rate_hz, 1e-6)
        m = self.lidar_match(P, stamp)
        return self.submit(m) if m is not None else None

    def apply_global(self, result: GlobalResult, stamp: float, source: str) -> bool:
        """Initialize from a successful global search (the pose at ``stamp``)."""
        self.global_attempts.append({"stamp": stamp, "source": source, "ok": result.ok, "reason": result.reason, "score": round(result.score, 3), "second": round(result.second, 3), "ms": round(result.ms, 1)})
        if not result.ok or self.initialized:
            return False
        ekf = self.engine.ekf
        pose = np.asarray(result.pose, dtype=np.float64)
        if ekf.stamp is not None and stamp != ekf.stamp:
            # The filter ran in its own frame until now: carry the global pose
            # to the filter's stamp with the filter's relative motion.
            pose = se2_compose(pose, se2_relative(self.engine.pose_at(stamp), self.engine.pose_at(float(ekf.stamp))))
        cov = np.asarray(result.cov, dtype=np.float64) + np.diag([0.05**2, 0.05**2, math.radians(0.5) ** 2])
        sig_xy = math.sqrt(max(cov[0, 0], cov[1, 1]))
        sig_yaw = math.sqrt(cov[2, 2])
        self.initialize(pose, max(sig_xy, 0.1), max(sig_yaw, math.radians(1.0)), f"global ({source}, score {result.score:.2f} vs {result.second:.2f})")
        return True

    # ------------------------------------------------------------ camera
    def add_camera_observation(self, obs: CameraObservation) -> None:
        self._observations.append(obs)

    def camera_window(self, key_stamp: float) -> list[CameraObservation]:
        c = self.patch_matcher.config
        obs = [o for o in self._observations if key_stamp - c.window_s <= o.stamp <= key_stamp]
        return obs

    def camera_patch(self, key_stamp: float) -> tuple[LocalPatch | None, NDArray[np.float64]]:
        """(live patch at ``key_stamp`` from the recent observations, key pose)."""
        c = self.patch_matcher.config
        obs = self.camera_window(key_stamp)
        if len(obs) < c.min_frames:
            return None, np.zeros(3)
        poses = np.array([self.engine.pose_at(o.stamp) for o in obs])
        key_pose = self.engine.pose_at(key_stamp)
        # Only observations within window_m of travel (relative motion error grows with it).
        travel = np.r_[np.cumsum(np.linalg.norm(np.diff(poses[::-1, :2], axis=0), axis=1))[::-1], 0.0]
        keep = travel <= c.window_m
        obs = [o for o, k in zip(obs, keep) if k]
        poses = poses[keep]
        if len(obs) < c.min_frames:
            return None, key_pose
        return self.patch_matcher.live_patch(obs, poses, key_pose, key_stamp), key_pose

    def process_camera(self, stamp: float) -> FixOutcome | GlobalResult | None:
        """Synchronous camera step (offline replay / tests) at ``camera_rate_hz``."""
        if not self.camera_fixes_enabled or not self.camera_due(stamp):
            return None
        if not self.initialized:
            if self._pending_start is not None or not self.config.global_relocalization or self.engine.ekf.stamp is None:
                return None
            if stamp < self._next_global:
                return None
            self._next_global = stamp + self.config.global_retry_s
            live, _ = self.camera_patch(stamp)
            if live is None:
                return None
            result = global_search_camera(self.model, self.patch_matcher, live, self.global_config)
            self.apply_global(result, stamp, "camera")
            return result
        self._next_camera = stamp + 1.0 / max(self.config.camera_rate_hz, 1e-6)
        live, key_pose = self.camera_patch(stamp)
        if live is None:
            return None
        _, cov = self.prior_at(stamp)
        m = self.patch_matcher.match(live, key_pose, cov)
        return self.submit(m) if m is not None else None

    # ------------------------------------------------------------ status
    def status(self, now: float | None = None) -> LocalizerStatus:
        c = self.config
        if self._pending:
            self.resolve()
        P = self.engine.pose_covariance()
        sxy = math.sqrt(max(float(np.max(np.linalg.eigvalsh(P[:2, :2]))), 0.0))
        syaw = math.sqrt(max(float(P[2, 2]), 0.0))
        ref = now if now is not None else (float(self.engine.ekf.stamp) if self.engine.ekf.stamp is not None else None)
        age = None if (self.last_accept_stamp is None or ref is None) else max(ref - self.last_accept_stamp, 0.0)
        if not self.initialized:
            state = "searching" if (self.config.global_relocalization and self._pending_start is None) else "uninitialized"
            return LocalizerStatus(state, False, False, math.inf, math.inf, self.consecutive_rejects, self.fixes_accepted, self.fixes_rejected, age, self.reanchors)
        if not self.converged_latch and self.agree_streak >= c.converged_min_fixes:
            self.converged_latch = True
        if self.consecutive_rejects >= max(self.config.rejected_after, 1):
            state = "rejected"
        elif not self.converged_latch or sxy > c.converged_sigma_xy_m or syaw > c.converged_sigma_yaw_rad:
            state = "converging"
        elif age is None or age > c.max_fix_age_s:
            state = "stale"
        else:
            state = "tracking"
        return LocalizerStatus(state, state == "tracking", True, sxy, syaw, self.consecutive_rejects, self.fixes_accepted, self.fixes_rejected, age, self.reanchors)
