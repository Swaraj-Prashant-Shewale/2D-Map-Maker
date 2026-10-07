"""LiDAR odometry front-end built on KISS-ICP.

KISS-ICP (Vizzo et al., RA-L 2023) needs no per-sensor tuning and runs a
C++ core, so it keeps up with a 20 Hz MID-360 in real time on a laptop. We
wrap it to (a) feed it de-skew timestamps, (b) report a quality score that the
estimator uses to inflate the measurement noise when registration is weak,
and (c) express the resulting poses for base_link rather than the sensor.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from kart_map2d.core.measurements import LidarScan


@dataclass
class LidarOdometryResult:
    stamp: NDArray[np.float64]
    pose: NDArray[np.float64]  # (N, 4, 4) sensor poses in the odometry frame
    quality: NDArray[np.float64]  # (N,) 0..1

    def save(self, path: Path) -> None:
        np.savez(path, stamp=self.stamp, pose=self.pose, quality=self.quality)

    @classmethod
    def load(cls, path: Path) -> "LidarOdometryResult":
        data = np.load(path)
        return cls(data["stamp"], data["pose"], data["quality"])


class LidarOdometry:
    def __init__(self, voxel_size: float = 0.5, max_range: float = 60.0, min_range: float = 0.8) -> None:
        from kiss_icp.config import KISSConfig
        from kiss_icp.kiss_icp import KissICP

        config = KISSConfig()
        config.data.max_range = max_range
        config.data.min_range = min_range
        config.data.deskew = True
        config.mapping.voxel_size = voxel_size
        self._icp = KissICP(config)
        self._frames = 0
        self._tree: cKDTree | None = None
        self._inlier_distance = max(0.3, 0.6 * voxel_size)

    def register(self, scan: LidarScan) -> tuple[NDArray[np.float64], float]:
        points = scan.points.astype(np.float64)
        dt = scan.point_dt.astype(np.float64)
        span = float(dt.max() - dt.min()) if len(dt) else 0.0
        # KISS-ICP expects per-point times normalised to [0, 1] within a sweep.
        timestamps = (dt - dt.min()) / span if span > 1e-6 else np.zeros(len(points))
        _, source = self._icp.register_frame(points, timestamps)
        self._frames += 1
        quality = self._quality(np.asarray(source))
        return self._icp.last_pose.copy(), quality

    def _quality(self, source: NDArray[np.float64]) -> float:
        """Share of the registered sweep lying on the local map.

        Low values mean the sweep saw mostly unmapped or moving structure
        (open space, crowds) and the registration deserves less trust. The
        KD-tree over the local map is rebuilt every few sweeps; the map
        changes slowly, so the amortised cost is ~1 ms per sweep.
        """
        if self._frames < 3 or len(source) == 0:
            return 1.0
        if self._tree is None or self._frames % 10 == 0:
            cloud = self._icp.local_map.point_cloud()
            self._tree = cKDTree(cloud) if len(cloud) else None
        if self._tree is None:
            return 1.0
        pose = self._icp.last_pose
        world = source @ pose[:3, :3].T + pose[:3, 3]
        distances, _ = self._tree.query(world, k=1, distance_upper_bound=self._inlier_distance)
        return float(np.mean(np.isfinite(distances)))


def run_lidar_odometry(scans, cache: Path | None = None, **kwargs) -> LidarOdometryResult:
    if cache is not None and cache.exists():
        return LidarOdometryResult.load(cache)
    odometry = LidarOdometry(**kwargs)
    stamps, poses, quality = [], [], []
    for scan in scans:
        pose, q = odometry.register(scan)
        stamps.append(scan.stamp)
        poses.append(pose)
        quality.append(q)
    result = LidarOdometryResult(np.asarray(stamps), np.asarray(poses), np.asarray(quality))
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        result.save(cache)
    return result
