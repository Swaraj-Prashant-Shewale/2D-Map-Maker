"""Replay an exported recording through the fusion engine.

Measurements are fed in *arrival* order (receive time), exactly as a live node
would see them, so the reorder buffer and lateness handling are exercised too.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kart_map2d.core.measurements import Measurement, RelativeMotion
from kart_map2d.estimation.fusion import FusionEngine
from kart_map2d.eval.faults import Fault, FaultInjector
from kart_map2d.io.dataset import ExportedRun, poses_to_relative
from kart_map2d.perception.lidar_odometry import LidarOdometryResult, run_lidar_odometry

ALL_SOURCES = ("imu", "imu_livox", "lidar_odom", "odom")

#: A ``vo_<camera>`` source runs VisualOdometry over that camera's frames
#: (with the IMU yaw prior when an IMU is present). ANY camera name works
#: (requirement 6): the name is whatever folder the bag exporter wrote.

#: A VO increment for the pair (t0, t1) can only exist once the second frame
#: has arrived and the pair is solved (~50 ms compute + transport), so replay
#: delivers it this long after its stamp, like the LiDAR-odometry delay.
VO_ARRIVAL_DELAY_S = 0.15

#: RUNTIME per-camera heights [m] over ground (the monocular scale anchor),
#: filled by callers (``vo_motions(..., height_m=...)`` /
#: :func:`set_camera_heights` from kart_master.cfg [camera] height_m). No
#: camera make is assumed anywhere (requirement 6): a camera without an
#: entry falls back to the RECORDING's own rig data
#: (kart_map2d/io/recorded_rigs.json - the September calibrations, keyed by
#: recording folder name), and finally to the mount height over the nominal
#: ground. To calibrate a NEW camera: README.md, "Calibrating the camera
#: height" (drive a straight, measured distance and scale height by
#: true/estimated distance; repeat once).
VO_CAMERA_HEIGHT_M: dict[str, float] = {}


def _recorded_rig() -> dict:
    """kart_map2d/io/recorded_rigs.json: rig DATA for the existing exports."""
    global _RECORDED_RIG
    if _RECORDED_RIG is None:
        import json

        path = Path(__file__).resolve().parent / "io" / "recorded_rigs.json"
        try:
            _RECORDED_RIG = json.loads(path.read_text(encoding="ascii"))
        except (OSError, ValueError):
            _RECORDED_RIG = {}
    return _RECORDED_RIG


_RECORDED_RIG: dict | None = None


def camera_height_m(camera: str) -> float | None:
    """Effective calibrated height for one camera: runtime table first, then
    the recording-rig data file, else None (mount-height fallback)."""
    height = VO_CAMERA_HEIGHT_M.get(camera)
    if height is None:
        height = (_recorded_rig().get("camera_heights_m") or {}).get(camera)
    return None if height is None else float(height)


def _effective_heights() -> dict[str, float]:
    merged = dict(_recorded_rig().get("camera_heights_m") or {})
    merged.update(VO_CAMERA_HEIGHT_M)
    return merged


def set_camera_heights(heights: dict[str, float]) -> None:
    """Replace/extend the per-camera height table (from [camera] height_m).

    Affects subsequent :func:`vo_motions` calls and the cache fingerprint,
    so a changed height can never be masked by a stale VO cache.
    """
    for camera, height in (heights or {}).items():
        if height is not None:
            VO_CAMERA_HEIGHT_M[str(camera)] = float(height)


def run_name(run: ExportedRun) -> str:
    """Recording directory name (ExportedRun may point at ``<run>/export``)."""
    root = Path(run.root)
    return root.parent.name if root.name == "export" else root.name


# ------------------------------------------------------------ visual odometry
_VO_CACHE_COLUMNS = (
    "stamp",
    "start_stamp",
    "delta_forward",
    "delta_left",
    "delta_yaw",
    "sigma_translation",
    "sigma_yaw",
    "sigma_position",
    "quality",
)


def vo_cache_fingerprint() -> str:
    """What a VO cache depends on: the VO source code and the camera heights.

    Stored in every cache written by :func:`save_vo_cache`; :func:`vo_motions`
    recomputes a cache whose fingerprint is missing or different, so a change
    to visual_odometry.py (or to VO_CAMERA_HEIGHT_M) can never be masked by
    increments an older version computed. Hashing the ~50 KB source is
    sub-millisecond.
    """
    import hashlib

    from kart_map2d.perception import visual_odometry

    digest = hashlib.sha256(Path(visual_odometry.__file__).read_bytes())
    # The EFFECTIVE per-camera heights (runtime table over the rig data
    # file): with nothing overridden this reprs identically to the historic
    # in-code dict, so existing caches stay valid.
    digest.update(repr(sorted(_effective_heights().items())).encode("ascii"))
    return digest.hexdigest()[:16]


def _cached_fingerprint(path: Path) -> str | None:
    with np.load(path) as data:
        return str(data["fingerprint"]) if "fingerprint" in data.files else None


def save_vo_cache(path: Path, motions: list[RelativeMotion]) -> None:
    rows = np.array(
        [[getattr(m, c) for c in _VO_CACHE_COLUMNS] for m in motions], dtype=np.float64
    ).reshape(-1, len(_VO_CACHE_COLUMNS))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, rows=rows, columns=np.array(_VO_CACHE_COLUMNS), fingerprint=np.array(vo_cache_fingerprint())
    )


def load_vo_cache(path: Path, channel: str) -> list[RelativeMotion]:
    with np.load(path) as data:
        rows = data["rows"]
    return [
        RelativeMotion(channel=channel, **dict(zip(_VO_CACHE_COLUMNS, map(float, row))))
        for row in rows
    ]


def vo_motions(
    run: ExportedRun,
    camera: str,
    cache_dir: Path = Path("outputs/cache"),
    height_m: float | None = None,
) -> list[RelativeMotion]:
    """VisualOdometry increments for one camera, cached like the KISS cache.

    The first call per run/camera solves every frame pair (~50 ms each) and
    writes ``<cache_dir>/<run>_vo_<camera>.npz``; later replays load the
    increments directly so fault studies stay fast. A cache written by other
    VO code or camera heights (:func:`vo_cache_fingerprint`) is recomputed.

    ``height_m`` overrides the height table for this call (per-camera
    [camera] height_m from kart_master.cfg); None consults the table.
    """
    from kart_map2d.perception.visual_odometry import (
        VisualOdometry,
        run_visual_odometry,
        yaw_prior_from_imu,
    )

    if height_m is not None:
        VO_CAMERA_HEIGHT_M[camera] = float(height_m)
    channel = f"vo_{camera}"
    cache = Path(cache_dir) / f"{run_name(run)}_vo_{camera}.npz"
    if cache.exists() and _cached_fingerprint(cache) == vo_cache_fingerprint():
        return load_vo_cache(cache, channel)
    prior = None
    if (imu := run.imu("imu_data")) is not None:
        imu = imu.decimated(100.0)
        prior = yaw_prior_from_imu(imu.stamp, imu.gyro[:, 2])
    K, size = run.camera_intrinsics(camera)
    T_base_cam = run.camera_mount(camera)
    # The calibrated height fixes the one monocularly unobservable scale
    # degree of freedom; expressed as an effective ground_z under this mount.
    height = camera_height_m(camera)
    ground_z = float(T_base_cam[2, 3]) - height if height is not None else -0.10
    vo = VisualOdometry(camera, K, size, T_base_cam, ground_z=ground_z, yaw_prior=prior)
    motions = run_visual_odometry(vo, run.camera_frames(camera))
    save_vo_cache(cache, motions)
    return motions


@dataclass
class ReplayInputs:
    run: ExportedRun
    lidar: LidarOdometryResult
    t0: float


def prepare(run_dir: Path, cache_dir: Path = Path("outputs/cache")) -> ReplayInputs:
    run = ExportedRun(run_dir)
    name = Path(run_dir).name
    lidar = run_lidar_odometry(run.lidar_scans(), cache=cache_dir / f"{name}_kiss.npz")
    return ReplayInputs(run, lidar, float(lidar.stamp[0]))


def build_measurements(inputs: ReplayInputs, sources=ALL_SOURCES, imu_rate_hz: float = 100.0) -> list[tuple[float, Measurement]]:
    run = inputs.run
    items: list[tuple[float, Measurement]] = []
    if "imu" in sources and (imu := run.imu("imu_data")) is not None:
        items += [(m.stamp, m) for m in imu.decimated(imu_rate_hz).samples()]
    if "imu_livox" in sources and (imu := run.imu("livox_imu")) is not None:
        items += [(m.stamp, m) for m in imu.decimated(imu_rate_hz).samples()]
    if "odom" in sources:
        motions, arrival = run.odom_relative("odom")
        items += list(zip(arrival.tolist(), motions))
    if "lidar_odom" in sources:
        T_base_lidar = run.tf.lookup("base_link", run.lidar_frame())
        # KISS-ICP poses are sensor poses in a frame anchored at the first
        # sweep; conjugating gives base_link motion.
        poses = np.array([T_base_lidar @ P for P in inputs.lidar.pose])
        motions = poses_to_relative(
            inputs.lidar.stamp,
            poses,
            "lidar_odom",
            T_base_sensor=T_base_lidar,
            sigma_translation=0.03,
            sigma_yaw=0.006,
            sigma_position=0.012,
            quality=inputs.lidar.quality,
            stride=4,  # 200 ms increments: pose jitter shrinks relative to distance
        )
        # A sweep can be registered once it has fully arrived (+ ~40 ms ICP).
        items += [(m.stamp + 0.09, m) for m in motions]
    for source in sources:
        if source.startswith("vo_"):
            motions = vo_motions(inputs.run, source[len("vo_") :])
            items += [(m.stamp + VO_ARRIVAL_DELAY_S, m) for m in motions]
    items.sort(key=lambda item: item[0])
    return items


def replay(
    inputs: ReplayInputs,
    sources=ALL_SOURCES,
    faults: list[Fault | dict] | None = None,
    engine: FusionEngine | None = None,
) -> FusionEngine:
    engine = engine or FusionEngine()
    injector = FaultInjector([f if isinstance(f, Fault) else Fault.from_dict(f) for f in faults or []], inputs.t0)
    for arrival, m in build_measurements(inputs, sources):
        corrupted = injector.apply(m)
        if corrupted is not None:
            engine.push(corrupted, arrival)
    engine.flush()
    return engine
