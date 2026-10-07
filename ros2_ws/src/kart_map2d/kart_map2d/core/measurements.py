"""Sensor-agnostic measurement types.

Every sensor, live or replayed, is reduced to one of these small records before
it reaches the estimator. The estimator never knows which physical device or
ROS topic produced a measurement, which is what lets any device drop out.

Frames: all body quantities are in ``base_link`` (x forward, y left, z up).
Map quantities are in a planar map frame (x east/forward-at-start, y
north/left-at-start) whose origin is the first pose.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Measurement:
    stamp: float  # sensor capture time on the common clock [s]
    channel: str  # health-monitor channel, e.g. "imu", "wheel", "lidar_odom"


@dataclass(frozen=True)
class ImuSample(Measurement):
    gyro: NDArray[np.float64]  # rad/s, base_link axes
    accel: NDArray[np.float64]  # m/s^2, base_link axes (includes gravity)


@dataclass(frozen=True)
class SpeedMeasurement(Measurement):
    """Forward speed from wheel encoders, a speedometer or a drive motor."""

    speed: float  # m/s
    sigma: float = 0.15


@dataclass(frozen=True)
class RelativeMotion(Measurement):
    """Body-frame motion between two epochs (LiDAR, visual or /odom odometry).

    ``stamp`` is the end of the interval and ``start_stamp`` its beginning.
    """

    start_stamp: float
    delta_forward: float  # m, along base_link x at the start pose
    delta_left: float  # m, along base_link y at the start pose
    delta_yaw: float  # rad
    sigma_translation: float = 0.05  # fraction of distance travelled (scale error)
    sigma_yaw: float = 0.01  # rad per interval
    quality: float = 1.0  # 0..1 front-end confidence (e.g. ICP inlier ratio)
    sigma_position: float = 0.01  # m, pose jitter at each end of the interval


@dataclass(frozen=True)
class PositionFix(Measurement):
    """Absolute planar position, already projected into the map frame."""

    position: NDArray[np.float64]  # [x, y] m
    sigma: float = 1.0  # m, 1-sigma horizontal
    fix_type: str = "single"  # "single", "float", "fixed" (RTK)


@dataclass(frozen=True)
class VelocityFix(Measurement):
    """Absolute planar velocity over ground, e.g. GNSS Doppler."""

    velocity: NDArray[np.float64]  # [vx, vy] m/s in the map frame
    sigma: float = 0.1


@dataclass(frozen=True)
class ZeroVelocity(Measurement):
    """The kart is standing still (from IMU stillness detection)."""

    sigma_speed: float = 0.02
    sigma_yaw_rate: float = 0.002


@dataclass(frozen=True)
class HeadingMeasurement(Measurement):
    """Absolute yaw in the map frame, e.g. calibrated magnetometer."""

    yaw: float
    sigma: float = np.radians(5.0)


@dataclass
class LidarScan:
    """One LiDAR sweep in the sensor frame, with per-point relative times."""

    stamp: float
    points: NDArray[np.float32]  # (N, 3) in the sensor frame
    intensity: NDArray[np.uint8]
    point_dt: NDArray[np.float32]  # (N,) seconds relative to ``stamp``
    channel: str = "lidar"


@dataclass
class CameraFrame:
    stamp: float
    image_path: str
    camera: str
    channel: str = field(default="")

    def __post_init__(self) -> None:
        if not self.channel:
            self.channel = self.camera
