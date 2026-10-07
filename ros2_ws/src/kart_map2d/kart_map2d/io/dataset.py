"""Load an exported recording and turn it into time-ordered measurements.

This is the only module that knows about the kart's specific topics, units and
mounting. Everything downstream consumes ``kart_map2d.core.measurements``.
A live ROS 2 node performs the same conversions per message.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from kart_map2d.core.measurements import CameraFrame, ImuSample, LidarScan, RelativeMotion
from kart_map2d.core.timesync import batch_correct
from kart_map2d.core.transforms import StaticTfTree, invert, make_transform, quat_to_matrix, rpy_matrix

GRAVITY = 9.80665


@dataclass
class ImuStream:
    channel: str
    stamp: NDArray[np.float64]
    gyro: NDArray[np.float64]  # base_link axes
    accel: NDArray[np.float64]  # base_link axes, m/s^2

    def decimated(self, rate_hz: float = 100.0) -> "ImuStream":
        """Block-average to ``rate_hz``; this is also the anti-alias filter."""
        if len(self.stamp) < 2:
            return self
        bins = np.floor((self.stamp - self.stamp[0]) * rate_hz).astype(np.int64)
        edges = np.flatnonzero(np.diff(bins)) + 1
        starts = np.r_[0, edges]
        counts = np.diff(np.r_[starts, len(bins)])

        def mean(values: NDArray) -> NDArray:
            return np.add.reduceat(values, starts, axis=0) / counts.reshape((-1,) + (1,) * (values.ndim - 1))

        return ImuStream(self.channel, mean(self.stamp), mean(self.gyro), mean(self.accel))

    def samples(self) -> list[ImuSample]:
        return [ImuSample(float(t), self.channel, g, a) for t, g, a in zip(self.stamp, self.gyro, self.accel)]


class ExportedRun:
    """Accessor for a directory written by ``kart_map2d.io.bag_export``."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        if not (self.root / "summary.json").exists() and (self.root / "export" / "summary.json").exists():
            self.root = self.root / "export"
        self.summary = json.loads((self.root / "summary.json").read_text(encoding="utf-8"))
        self.tf = StaticTfTree.load(self.root / "tf_static.json")
        # /imu/data is published in "imu_link", which is missing from /tf_static.
        # Cross-correlation with the LiDAR IMU (see docs/DATA_AUDIT.md) shows it
        # is axis-aligned with base_link; its position is irrelevant for yaw
        # rate and small for acceleration at kart speeds.
        self.tf.add("base_link", "imu_link", np.eye(4))
        self.camera_info = json.loads((self.root / "camera_info.json").read_text(encoding="utf-8"))

    # ------------------------------------------------------------------ IMUs
    def imu(self, name: str = "imu_data") -> ImuStream | None:
        path = self.root / f"imu_{name}.npz"
        if not path.exists():
            return None
        data = np.load(path)
        stamp = data["stamp"]
        gyro = data["gyro"].astype(np.float64)
        accel = data["accel"].astype(np.float64)
        topic = "/" + name.replace("_", "/", 1) if name != "livox_imu" else "/livox/imu"
        frame = self.summary["topics"].get(topic, {}).get("frame_id", "imu_link")
        if np.median(np.linalg.norm(accel, axis=1)) < 2.0:
            accel = accel * GRAVITY  # the Livox driver reports acceleration in g
        R = self.tf.lookup("base_link", frame)[:3, :3]
        stamp = batch_correct(stamp, data["receive"])  # no-op for epoch stamps
        order = np.argsort(stamp, kind="stable")
        channel = "imu" if name == "imu_data" else "imu_" + name.split("_")[0]
        return ImuStream(channel, stamp[order], gyro[order] @ R.T, accel[order] @ R.T)

    # ------------------------------------------------------------------ odometry
    def odom_relative(self, channel: str = "odom") -> tuple[list[RelativeMotion], NDArray[np.float64]]:
        """Increments from the kart's own /odom, with their receive times."""
        path = self.root / "odom.npz"
        if not path.exists():
            return [], np.empty(0)
        data = np.load(path)
        poses = np.array([make_transform(quat_to_matrix(q), p) for p, q in zip(data["position"], data["quat_xyzw"])])
        motions = poses_to_relative(
            data["stamp"], poses, channel, sigma_translation=0.05, sigma_yaw=0.008, sigma_position=0.015, stride=2
        )
        arrival = dict(zip(data["stamp"].tolist(), data["receive"].tolist()))
        return motions, np.array([arrival[m.stamp] for m in motions])

    # ------------------------------------------------------------------ LiDAR
    def lidar_frame(self) -> str:
        return self.summary["topics"].get("/livox/lidar", {}).get("frame_id", "livox_frame")

    def lidar_scans(self, start: float | None = None, stop: float | None = None):
        for chunk in sorted((self.root / "lidar").glob("chunk_*.npz")):
            data = np.load(chunk)
            stamps = data["stamp"]
            if stop is not None and stamps[0] > stop:
                break
            if start is not None and stamps[-1] < start:
                continue
            offsets = np.r_[0, np.cumsum(data["count"])]
            xyz, intensity, dt = data["xyz"], data["intensity"], data["dt"]
            for i, stamp in enumerate(stamps):
                if (start is not None and stamp < start) or (stop is not None and stamp > stop):
                    continue
                s, e = offsets[i], offsets[i + 1]
                yield LidarScan(float(stamp), xyz[s:e].astype(np.float32), intensity[s:e], dt[s:e].astype(np.float32))

    def lidar_stamps(self) -> NDArray[np.float64]:
        return np.concatenate([np.load(c)["stamp"] for c in sorted((self.root / "lidar").glob("chunk_*.npz"))])

    # ------------------------------------------------------------------ cameras
    def cameras(self) -> list[str]:
        return sorted(p.parent.name for p in self.root.glob("*/index.npz"))

    def camera_frames(self, camera: str) -> list[CameraFrame]:
        data = np.load(self.root / camera / "index.npz")
        return [
            CameraFrame(float(t), str(self.root / camera / str(f)), camera)
            for t, f in zip(data["stamp"], data["filename"])
        ]

    def camera_intrinsics(self, camera: str) -> tuple[NDArray[np.float64], tuple[int, int]]:
        info = self.camera_info[camera]
        scale = float(info.get("export_scale", 1.0))
        K = np.asarray(info["K"], dtype=np.float64)
        K[:2] *= scale
        return K, (int(round(info["width"] * scale)), int(round(info["height"] * scale)))

    def camera_link(self, camera: str) -> str:
        """The TF link of one exported camera, resolved camera-agnostically.

        Priority: an explicit ``link`` entry in camera_info.json (new
        exports), then a frame that literally contains the camera name, then
        the recording-rig DATA file (kart_map2d/io/recorded_rigs.json: the
        September rigs' link names) - a compatibility fallback for the
        existing recordings, not an assumption about future cameras.
        """
        info = self.camera_info.get(camera, {})
        if info.get("link"):
            return str(info["link"])
        frames = getattr(self.tf, "frames", lambda: [])()
        for frame in frames:
            if camera in str(frame):
                return str(frame)
        import json as _json

        try:
            rig = _json.loads((Path(__file__).resolve().parent / "recorded_rigs.json").read_text(encoding="ascii"))
        except (OSError, ValueError):
            rig = {}
        legacy = rig.get("camera_links") or {}
        if camera in legacy:
            return str(legacy[camera])
        raise KeyError(
            f"no TF link found for camera {camera!r}: add a 'link' entry to camera_info.json "
            "or record the camera frame in tf_static"
        )

    def camera_mount(self, camera: str) -> NDArray[np.float64]:
        """base_link <- camera optical frame (z forward, x right, y down).

        Uses only the link POSITION plus the standard forward-looking optical
        convention: the September ZED X GS units are mounted upside down
        (URDF roll = pi) with the driver flipping the image, so the published
        image is an upright, forward-looking view from the link position -
        and any other upright forward camera gets the same treatment.
        """
        T_link = self.tf.lookup("base_link", self.camera_link(camera))
        optical_to_link = rpy_matrix(-np.pi / 2, 0.0, -np.pi / 2)  # REP-103 optical -> x-forward body
        return make_transform(optical_to_link, T_link[:3, 3])

    @property
    def start_stamp(self) -> float:
        return float(self.lidar_stamps()[0])


def poses_to_relative(
    stamps: NDArray[np.float64],
    poses: NDArray[np.float64],
    channel: str,
    T_base_sensor: NDArray[np.float64] | None = None,
    sigma_translation: float = 0.03,
    sigma_yaw: float = 0.005,
    sigma_position: float = 0.01,
    max_gap_s: float = 0.5,
    quality: NDArray[np.float64] | None = None,
    stride: int = 1,
) -> list[RelativeMotion]:
    """Turn an odometry pose sequence (N, 4, 4) into body-frame increments.

    ``stride`` forms increments over several front-end steps: per-pose jitter
    (``sigma_position``) is independent of interval length, so longer
    increments carry proportionally less speed noise. Increments spanning a
    gap longer than ``max_gap_s`` are dropped; the health monitor sees the gap
    as a timeout rather than one huge jump.
    """
    if T_base_sensor is not None:
        poses = np.einsum("nij,jk->nik", poses, invert(T_base_sensor))
    out: list[RelativeMotion] = []
    for i in range(stride, len(stamps), stride):
        j = i - stride
        if not 0 < stamps[i] - stamps[j] <= max_gap_s * stride:
            continue
        delta = invert(poses[j]) @ poses[i]
        window_quality = float(np.min(quality[j : i + 1])) if quality is not None else 1.0
        out.append(
            RelativeMotion(
                stamp=float(stamps[i]),
                channel=channel,
                start_stamp=float(stamps[j]),
                delta_forward=float(delta[0, 3]),
                delta_left=float(delta[1, 3]),
                delta_yaw=float(np.arctan2(delta[1, 0], delta[0, 0])),
                sigma_translation=sigma_translation,
                sigma_yaw=sigma_yaw,
                sigma_position=sigma_position,
                quality=window_quality,
            )
        )
    return out
