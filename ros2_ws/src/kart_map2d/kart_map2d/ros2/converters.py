"""Pure converters: ROS 2 message objects -> trackmapper measurements.

These functions mirror, message for message, what ``kart_map2d.io.dataset``
does to the exported arrays, so a measurement produced live on the kart is
bit-identical to the one the offline replay would have produced from the same
bag message. They operate on *duck-typed* message objects (anything with the
right attributes: real rclpy messages, rosbags records, or plain namedtuples
in tests) and import neither ``rclpy`` nor any ``*_msgs`` package.

Frames and units follow trackmapper conventions: base_link x forward / y left
/ z up, stamps in epoch seconds (float64) on the common clock, gyro rad/s,
accel m/s^2 (gravity included).
"""

from __future__ import annotations

import array
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np
from numpy.typing import NDArray

from kart_map2d.core.measurements import ImuSample, LidarScan, RelativeMotion
from kart_map2d.core.timesync import ClockSync
from kart_map2d.core.transforms import make_transform, quat_to_matrix
from kart_map2d.io.dataset import GRAVITY, poses_to_relative
from kart_map2d.mapping.corridor import CLASS_DRIVABLE, CLASS_OBSTACLE

__all__ = [
    "ImuDecimator",
    "OdometryConverter",
    "StreamingClockSync",
    "camera_info_to_intrinsics",
    "fill_occupancy_grid",
    "image_to_bgr",
    "imu_to_sample",
    "pointcloud_to_scan",
    "set_stamp",
    "stamp_of",
]


def _bytes_like(data):
    """A message's byte payload without copying it when that is possible.

    rclpy delivers ``uint8[]`` fields as ``array.array('B')`` (buffer protocol:
    numpy can view it in place); ``bytes`` / ``bytearray`` / ``memoryview``
    work the same way. Only a plain sequence (test doubles) is copied. The
    old ``bytes(msg.data)`` copied every 9 MB camera frame on the executor
    thread for nothing.
    """
    try:
        memoryview(data)
        return data
    except TypeError:
        return bytes(data)


# -------------------------------------------------------------- clock sync
class StreamingClockSync:
    """:class:`~kart_map2d.core.timesync.ClockSync` with an O(1) window minimum.

    Same algorithm, same arithmetic, bit-identical output (tests/test_ros2_node.py
    checks it sample by sample): the device clock's offset is the lower
    envelope of ``receive - device_stamp`` over the last ``window_s`` of
    device time, allowed to rise by at most 1e-4 s per sample and to drop
    instantly. ClockSync recomputes that minimum over the whole window for
    every sample - 4000 samples at /imu/data's 400 Hz, ~0.5 ms per message
    on the node's executor thread. Here a monotonic deque keeps the running
    minimum: the window is always a suffix of the arrival sequence (ClockSync
    only ever evicts from the front), so a sample whose delta is >= a later
    sample's can never be the minimum again and is discarded on arrival.
    """

    def __init__(self, window_s: float = 10.0, epoch_threshold_s: float = 1e8) -> None:
        self.window_s = window_s
        self.epoch_threshold_s = epoch_threshold_s
        self._window: deque[tuple[int, float]] = deque()  # (arrival index, device stamp)
        self._mins: deque[tuple[int, float]] = deque()  # (arrival index, delta), deltas increasing
        self._count = 0
        self._offset: float | None = None

    def correct(self, device_stamp: float, receive_stamp: float, observe: bool = True) -> float:
        """Return ``device_stamp`` expressed on the common clock.

        ``observe=False`` applies the current offset without letting this
        sample shape the envelope: for a message known to have waited in a
        queue, whose ``receive_stamp`` says nothing about the latency floor
        (see TrackmapperNode._drain_backlog). Before the first observed
        sample there is no offset yet, so the sample is observed anyway.
        """
        if device_stamp > self.epoch_threshold_s:
            return device_stamp  # already on an epoch clock
        if not observe and self._offset is not None:
            return device_stamp + self._offset
        delta = receive_stamp - device_stamp
        index = self._count
        self._count += 1
        self._window.append((index, device_stamp))
        while self._mins and self._mins[-1][1] >= delta:
            self._mins.pop()
        self._mins.append((index, delta))
        while self._window and device_stamp - self._window[0][1] > self.window_s:
            self._window.popleft()
        start = self._window[0][0]  # never empty: the newest sample is always inside
        while self._mins[0][0] < start:
            self._mins.popleft()
        floor = self._mins[0][1]
        # Allow the envelope to rise slowly (clock drift) but drop instantly.
        if self._offset is None or floor < self._offset:
            self._offset = floor
        else:
            self._offset += min(floor - self._offset, 1e-4)
        return device_stamp + self._offset


# ------------------------------------------------------------------ stamps
def stamp_of(msg) -> float:
    """Header stamp -> float epoch seconds (``header.stamp.sec/nanosec``)."""
    stamp = msg.header.stamp
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def set_stamp(time_msg, stamp: float) -> None:
    """Write float seconds into a ``builtin_interfaces/Time``-shaped object."""
    sec = int(np.floor(stamp))
    nanosec = int(round((stamp - sec) * 1e9))
    if nanosec >= 1_000_000_000:  # fraction rounded up to a full second
        sec += 1
        nanosec -= 1_000_000_000
    time_msg.sec = sec
    time_msg.nanosec = nanosec


# --------------------------------------------------------------------- IMU
def imu_to_sample(
    msg,
    receive_stamp: float,
    channel: str = "imu",
    clock: ClockSync | StreamingClockSync | None = None,
    R_base_imu: NDArray[np.float64] | None = None,
    accel_in_g: bool | None = None,
    observe: bool = True,
) -> ImuSample:
    """One ``sensor_msgs/Imu`` -> :class:`ImuSample` on the common clock.

    ``clock`` applies the same device-clock correction as the offline loader
    (``batch_correct`` is a per-sample loop over one :class:`ClockSync`): the
    kart's ``/imu/data`` stamps start near 0 s and are mapped onto the epoch
    clock against ``receive_stamp``; stamps already on an epoch clock pass
    through unchanged. Pass one ClockSync (or the bit-identical, O(1)
    :class:`StreamingClockSync` the node uses) per topic and keep it across
    calls, or ``None`` to trust the stamp as-is.

    ``accel_in_g`` mirrors the offline unit fix for ``/livox/imu`` (reported
    in g, not m/s^2); ``None`` auto-detects per sample with the same < 2.0
    magnitude test the offline loader applies to the stream median.
    ``R_base_imu`` rotates gyro and accel into base_link axes (identity when
    omitted, as assumed for ``/imu/data``). ``observe=False`` (a message that
    waited in a queue; :class:`StreamingClockSync` only) corrects the stamp
    without letting its receive time shape the offset.
    """
    stamp = stamp_of(msg)
    if clock is not None:
        stamp = clock.correct(stamp, receive_stamp) if observe else clock.correct(stamp, receive_stamp, observe=False)
    gyro = np.array(
        [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z], dtype=np.float64
    )
    accel = np.array(
        [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z],
        dtype=np.float64,
    )
    if accel_in_g is None:
        accel_in_g = bool(np.linalg.norm(accel) < 2.0)
    if accel_in_g:
        accel = accel * GRAVITY
    if R_base_imu is not None:
        R = np.asarray(R_base_imu, dtype=np.float64)
        gyro = R @ gyro
        accel = R @ accel
    return ImuSample(float(stamp), channel, gyro, accel)


class ImuDecimator:
    """Streaming twin of ``ImuStream.decimated``: block-average to ``rate_hz``.

    The offline loader bins samples by ``floor((t - t0) * rate)`` and emits
    the per-bin mean, which is also the anti-alias filter in front of the
    100 Hz EKF. This class reproduces those bins one sample at a time; a bin
    is emitted when the first sample of the next bin arrives (the trailing
    partial bin is never emitted, which offline only affects the last
    ~10 ms of a recording).
    """

    def __init__(self, rate_hz: float = 100.0) -> None:
        self.rate_hz = float(rate_hz)
        self._t0: float | None = None
        self._bin: int | None = None
        self._acc: list[ImuSample] = []

    @staticmethod
    def _mean(values: NDArray[np.float64]) -> NDArray[np.float64]:
        # Same reduction as ImuStream.decimated (np.add.reduceat), whose
        # accumulation order differs from np.mean by an ulp; using it keeps
        # live bins bit-identical to offline ones.
        return np.add.reduceat(values, [0], axis=0)[0] / len(values)

    def push(self, sample: ImuSample) -> ImuSample | None:
        """Add one sample; returns the completed bin's mean sample or None."""
        if self._t0 is None:
            self._t0 = sample.stamp
        index = int(np.floor((sample.stamp - self._t0) * self.rate_hz))
        out: ImuSample | None = None
        if self._bin is not None and index != self._bin and self._acc:
            out = ImuSample(
                float(self._mean(np.array([s.stamp for s in self._acc]))),
                self._acc[0].channel,
                self._mean(np.array([s.gyro for s in self._acc])),
                self._mean(np.array([s.accel for s in self._acc])),
            )
            self._acc = []
        self._bin = index
        self._acc.append(sample)
        return out


# ---------------------------------------------------------------- odometry
@dataclass
class OdometryConverter:
    """Streaming ``nav_msgs/Odometry`` -> :class:`RelativeMotion` increments.

    Buffers poses and forms each increment through the *same*
    :func:`kart_map2d.io.dataset.poses_to_relative` call the offline loader
    uses (same stride, sigmas and gap rule as ``ExportedRun.odom_relative``),
    so live increments match the replayed ones exactly. An increment spanning
    more than ``max_gap_s * stride`` is dropped and the converter re-anchors;
    the health monitor then sees the gap as a timeout, not a jump.
    """

    channel: str = "odom"
    stride: int = 2
    sigma_translation: float = 0.05
    sigma_yaw: float = 0.008
    sigma_position: float = 0.015
    max_gap_s: float = 0.5
    _stamps: list[float] = field(default_factory=list, repr=False)
    _poses: list[NDArray[np.float64]] = field(default_factory=list, repr=False)

    def push(self, msg) -> RelativeMotion | None:
        """Add one Odometry message; returns an increment every ``stride`` msgs."""
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        pose = make_transform(
            quat_to_matrix(np.array([q.x, q.y, q.z, q.w], dtype=np.float64)),
            np.array([p.x, p.y, p.z], dtype=np.float64),
        )
        self._stamps.append(stamp_of(msg))
        self._poses.append(pose)
        if len(self._stamps) <= self.stride:
            return None
        motions = poses_to_relative(
            np.asarray(self._stamps),
            np.asarray(self._poses),
            self.channel,
            sigma_translation=self.sigma_translation,
            sigma_yaw=self.sigma_yaw,
            sigma_position=self.sigma_position,
            max_gap_s=self.max_gap_s,
            stride=self.stride,
        )
        # The newest pose anchors the next increment either way; a gap-dropped
        # pair simply yields nothing (offline marches the same fixed strides).
        self._stamps = self._stamps[-1:]
        self._poses = self._poses[-1:]
        return motions[0] if motions else None

    def reset(self) -> None:
        self._stamps.clear()
        self._poses.clear()


# ------------------------------------------------------------------ camera
#: uint8 channel count per ROS image encoding (the ones this rig can publish).
_ENCODING_CHANNELS = {"bgra8": 4, "rgba8": 4, "bgr8": 3, "rgb8": 3, "mono8": 1, "8uc1": 1}


def image_to_bgr(msg, scale: float = 1.0) -> tuple[float, NDArray[np.uint8]]:
    """``sensor_msgs/Image`` -> (stamp, BGR uint8 array), for the VO handoff.

    Mirrors the decoding in ``kart_map2d.io.bag_export``: the kart publishes
    ``bgra8`` (alpha dropped); ``rgb8``/``rgba8`` are swapped to BGR and
    ``mono8`` passes through 2-D. Row padding (``step`` > width * channels)
    is stripped.

    ``scale`` != 1 also resizes with ``cv2.INTER_AREA`` exactly like the
    exporter (bit-identical to resizing the full-size BGR result: INTER_AREA
    treats channels independently, so resizing first and dropping alpha /
    swapping channels afterwards gives the same bytes). Resizing first is
    what makes it cheap: the full-resolution 1920x1200 frame is never copied,
    only viewed in the message buffer.
    """
    h, w = int(msg.height), int(msg.width)
    data = np.frombuffer(_bytes_like(msg.data), dtype=np.uint8)
    encoding = str(msg.encoding).lower()
    # The channel count must come from the encoding, not from len(data):
    # padded rows would inflate a length-based guess (the fallback for
    # unlisted encodings is only correct when rows are dense).
    channels = _ENCODING_CHANNELS.get(encoding, max(len(data) // (h * w), 1))
    step = int(getattr(msg, "step", w * channels))
    if step > w * channels:
        image = data.reshape(h, step)[:, : w * channels]
    else:
        image = data
    image = image.reshape(h, w, channels) if channels > 1 else image.reshape(h, w)
    if scale != 1.0:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if encoding.startswith("rgb") and channels >= 3:
        image = image[:, :, 2::-1]
    elif channels == 4:
        image = image[:, :, :3]
    return stamp_of(msg), np.ascontiguousarray(image)


def compressed_to_bgr(msg, scale: float = 1.0) -> tuple[float, NDArray[np.uint8]]:
    """``sensor_msgs/CompressedImage`` (jpeg / png) -> (stamp, BGR uint8 array).

    Same downscale as :func:`image_to_bgr` (``cv2.INTER_AREA``), applied after
    decoding because a compressed frame cannot be viewed before it is decoded.
    """
    data = np.frombuffer(_bytes_like(msg.data), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot decode a {getattr(msg, 'format', '?')!s} CompressedImage")
    if scale != 1.0:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return stamp_of(msg), np.ascontiguousarray(image)


def intrinsics_matrix(fx: float, fy: float, cx: float, cy: float) -> NDArray[np.float64]:
    """Pinhole K (3x3) from focal lengths and principal point [px]."""
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def camera_info_to_intrinsics(msg) -> tuple[NDArray[np.float64], tuple[int, int]]:
    """``sensor_msgs/CameraInfo`` -> (K 3x3, (width, height)) at full resolution."""
    K = np.asarray(msg.k, dtype=np.float64).reshape(3, 3)
    return K, (int(msg.width), int(msg.height))


# ------------------------------------------------------------------- LiDAR
# Mirrors kart_map2d.io.bag_export._POINT_DTYPES / _cloud_dtype. Duplicated
# (10 lines) rather than imported: bag_export pulls in the offline-only
# `rosbags` and `cv2` dependencies, which do not belong on the kart.
_POINT_DTYPES = {1: "<i1", 2: "<u1", 3: "<i2", 4: "<u2", 5: "<i4", 6: "<u4", 7: "<f4", 8: "<f8"}


def _cloud_dtype(msg) -> np.dtype:
    return np.dtype(
        {
            "names": [f.name for f in msg.fields],
            "formats": [_POINT_DTYPES[f.datatype] for f in msg.fields],
            "offsets": [f.offset for f in msg.fields],
            "itemsize": msg.point_step,
        }
    )


def pointcloud_to_scan(
    msg, channel: str = "lidar", min_range: float = 0.3, max_range: float = 60.0
) -> LidarScan:
    """``sensor_msgs/PointCloud2`` -> :class:`LidarScan` in the sensor frame.

    Applies the same point filter as the offline export (finite, and range in
    (0.3, 60) m: the MID-360 returns points on the kart itself inside ~0.3 m)
    and the same per-point relative times from the Livox ``timestamp`` field.
    Points stay float32 (the offline cache additionally rounds to float16 for
    disk size, which live operation does not need).
    """
    stamp = stamp_of(msg)
    cloud = np.frombuffer(_bytes_like(msg.data), dtype=_cloud_dtype(msg))
    xyz = np.column_stack([cloud["x"], cloud["y"], cloud["z"]]).astype(np.float32)
    finite = np.all(np.isfinite(xyz), axis=1)
    rng = np.linalg.norm(xyz, axis=1)
    keep = finite & (rng > min_range) & (rng < max_range)
    cloud, xyz = cloud[keep], xyz[keep]
    names = cloud.dtype.names or ()
    if "timestamp" in names and len(cloud):
        dt = (cloud["timestamp"].astype(np.float64) * 1e-9 - stamp).astype(np.float32)
    else:
        dt = np.zeros(len(cloud), dtype=np.float32)
    if "intensity" in names:
        intensity = np.clip(cloud["intensity"], 0, 255).astype(np.uint8)
    else:
        intensity = np.zeros(len(cloud), dtype=np.uint8)
    return LidarScan(float(stamp), xyz, intensity, dt, channel=channel)


# ---------------------------------------------------------- occupancy grid
def fill_occupancy_grid(msg, grid, stamp: float, frame_id: str = "map"):
    """Write an EvidenceGrid's static ``classify()`` view into a ``nav_msgs/OccupancyGrid``.

    Metadata follows the grid exactly: ``resolution`` is the cell size [m]
    and ``origin`` the world position of the outer corner of cell (0, 0), so
    cell centres land at ``origin + (index + 0.5) * resolution`` on both
    sides. Data is row-major with x varying fastest (``data[iy * width + ix]``
    for the grid's ``classes[ix, iy]``); unknown -> -1, drivable -> 0,
    obstacle -> 100.
    """
    classes = np.asarray(grid.classify())
    nx, ny = classes.shape
    msg.header.frame_id = frame_id
    set_stamp(msg.header.stamp, stamp)
    set_stamp(msg.info.map_load_time, stamp)
    msg.info.resolution = float(grid.cell)
    msg.info.width = int(nx)
    msg.info.height = int(ny)
    origin = np.asarray(grid.origin, dtype=np.float64)
    msg.info.origin.position.x = float(origin[0])
    msg.info.origin.position.y = float(origin[1])
    msg.info.origin.position.z = 0.0
    msg.info.origin.orientation.x = 0.0
    msg.info.origin.orientation.y = 0.0
    msg.info.origin.orientation.z = 0.0
    msg.info.origin.orientation.w = 1.0
    occ = np.full((nx, ny), -1, dtype=np.int8)
    occ[classes == CLASS_DRIVABLE] = 0
    occ[classes == CLASS_OBSTACLE] = 100
    # array.array('b') is the rclpy fast path for an int8[] field: a list (or
    # bytes) is checked element by element in Python by the generated setter,
    # ~0.1 s for a few hundred thousand cells.
    msg.data = array.array("b", np.ascontiguousarray(occ.T).tobytes())
    return msg
