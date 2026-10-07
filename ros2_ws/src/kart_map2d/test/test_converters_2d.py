"""ROS-message converters (2D package): byte-accurate against the offline loaders.

Ported from the research workspace's test_ros2_node.py converter section
(values copied verbatim from the September bag exports), plus the
requirement-6 additions: EVERY accepted encoding and CompressedImage.
No rclpy anywhere: duck-typed message objects only.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from kart_map2d.core.timesync import ClockSync, batch_correct
from kart_map2d.core.transforms import make_transform, quat_to_matrix, rpy_matrix
from kart_map2d.io.dataset import GRAVITY, ImuStream, poses_to_relative
from kart_map2d.mapping.evidence_grid import EvidenceGrid
from kart_map2d.ros2 import converters

# ------------------------------------------------------------ msg fixtures
def _time(stamp: float) -> SimpleNamespace:
    sec = int(np.floor(stamp))
    return SimpleNamespace(sec=sec, nanosec=int(round((stamp - sec) * 1e9)))


def _header(stamp: float, frame_id: str = "") -> SimpleNamespace:
    return SimpleNamespace(stamp=_time(stamp), frame_id=frame_id)


def _vec3(x: float, y: float, z: float) -> SimpleNamespace:
    return SimpleNamespace(x=x, y=y, z=z)


def _imu_msg(stamp: float, gyro, accel, frame_id: str = "imu_link") -> SimpleNamespace:
    return SimpleNamespace(
        header=_header(stamp, frame_id),
        angular_velocity=_vec3(*gyro),
        linear_acceleration=_vec3(*accel),
    )


def _odom_msg(stamp: float, position, quat_xyzw) -> SimpleNamespace:
    return SimpleNamespace(
        header=_header(stamp, "odom"),
        child_frame_id="base_link",
        pose=SimpleNamespace(
            pose=SimpleNamespace(
                position=_vec3(*position),
                orientation=SimpleNamespace(
                    x=quat_xyzw[0], y=quat_xyzw[1], z=quat_xyzw[2], w=quat_xyzw[3]
                ),
            )
        ),
    )


#: First /imu/data sample of sept3 (device-clock stamp near 0, base_link axes,
#: m/s^2) with its bag receive time, copied from export/imu_imu_data.npz.
IMU_DATA_FIX = {
    "stamp": 0.087884,
    "receive": 1788593584.9049785,
    "gyro": (0.00032857063342817133, -0.00022649766469839963, -0.001093745231628418),
    "accel": (0.10784168541431427, 0.2130830138921738, 9.786248207092287),
}

#: First /livox/imu sample of sept3 (epoch stamp, livox_frame axes, g units).
LIVOX_IMU_FIX = {
    "stamp": 1788593587.3673744,
    "gyro": (0.01781933754682541, -0.007494945079088211, -0.0014754440635442734),
    "accel": (-0.021309729665517807, 0.02640550211071968, 0.9954469799995422),
}

#: First seven /odom rows of sept3 (stamp, position, quat_xyzw). The early
#: gaps (0.15 s, 0.95 s, 1.9 s) exercise the same max-gap drop rule offline
#: replay applies; only the last stride-2 pair survives it.
ODOM_FIX = [
    (1788593584.2988224, (0.0, 0.0, 0.0), (0.0, 0.0, -1.1863515016145732e-17, 1.0)),
    (1788593584.449891, (0.009464526131609025, -0.024562784608313813, -0.08314908312548014),
     (-0.002277768260875962, -0.0013481724254056108, 0.0005684477707546907, 0.9999963355282824)),
    (1788593585.3999312, (0.022507610472700057, -0.008054390927037922, -0.017296977585177564),
     (-0.0007403588464831206, 0.0001334572079769139, 0.0002519035267464235, 0.9999996853012332)),
    (1788593587.3001542, (0.027713485665564582, -0.03813087246215305, -0.08930724613981322),
     (-0.0033365026220465565, -0.0021367816731827265, 0.0005116552351926438, 0.9999920200297874)),
    (1788593587.3503344, (0.014067737914797346, 0.007980004108840776, 0.0007737670316698697),
     (0.0008664617322279886, -3.0509543185374632e-05, 0.00013874204295007733, 0.9999996145318657)),
    (1788593587.400174, (0.0030018652215952385, -0.005154169479264873, -0.07076481526597578),
     (-0.0010065488583909693, -0.000225748505952709, 0.0003738006939557744, 0.9999993980848434)),
    (1788593587.4502718, (0.011664804328839613, 0.0046993909592877746, -0.01329343292629459),
     (1.9803784001530898e-05, -8.443715331144459e-05, 0.00037526441609086, 0.9999999258273949)),
]

#: zed_right CameraInfo (full resolution, already undistorted).
CAMERA_INFO_K = [675.8616943359375, 0.0, 890.8087158203125,
                 0.0, 722.2686767578125, 578.778564453125,
                 0.0, 0.0, 1.0]


def _camera_info_msg() -> SimpleNamespace:
    return SimpleNamespace(
        header=_header(1788593584.5, "zed_right_optical_frame"),
        width=1920,
        height=1200,
        k=list(CAMERA_INFO_K),
        d=[0.0] * 8,
        distortion_model="rational_polynomial",
    )




# ------------------------------------------------------------- converters
def test_imu_data_uses_offline_clock_correction():
    """/imu/data device stamps map to the epoch exactly like batch_correct."""
    fix = IMU_DATA_FIX
    device = [fix["stamp"], fix["stamp"] + 0.0025, fix["stamp"] + 0.005]
    receive = [fix["receive"], fix["receive"] + 0.0026, fix["receive"] + 0.0049]
    msgs = [_imu_msg(t, fix["gyro"], fix["accel"]) for t in device]
    device_seen = np.array([converters.stamp_of(m) for m in msgs])

    clock = ClockSync()
    samples = [
        converters.imu_to_sample(m, r, channel="imu", clock=clock, accel_in_g=False)
        for m, r in zip(msgs, receive)
    ]
    expected = batch_correct(device_seen, np.array(receive))
    assert np.array_equal([s.stamp for s in samples], expected)
    # The first corrected stamp is the receive time itself (zero-latency floor).
    assert samples[0].stamp == receive[0]
    assert samples[0].channel == "imu"
    assert np.array_equal(samples[0].gyro, np.array(fix["gyro"]))
    assert np.array_equal(samples[0].accel, np.array(fix["accel"]))


def test_livox_imu_epoch_stamp_units_and_rotation():
    fix = LIVOX_IMU_FIX
    msg = _imu_msg(fix["stamp"], fix["gyro"], fix["accel"], frame_id="livox_frame")
    stamp_seen = converters.stamp_of(msg)
    R = rpy_matrix(0.0, 0.0, np.radians(-90.0))  # base_link <- livox_frame

    clock = ClockSync()
    sample = converters.imu_to_sample(
        msg, fix["stamp"] + 0.002, channel="imu_livox", clock=clock, accel_in_g=True, R_base_imu=R
    )
    # Epoch stamps pass through the ClockSync untouched.
    assert sample.stamp == stamp_seen
    assert sample.channel == "imu_livox"
    assert np.array_equal(sample.gyro, R @ np.array(fix["gyro"]))
    assert np.array_equal(sample.accel, R @ (np.array(fix["accel"]) * GRAVITY))


def test_imu_g_autodetection_matches_offline_heuristic():
    livox = _imu_msg(LIVOX_IMU_FIX["stamp"], LIVOX_IMU_FIX["gyro"], LIVOX_IMU_FIX["accel"])
    ms2 = _imu_msg(IMU_DATA_FIX["stamp"], IMU_DATA_FIX["gyro"], IMU_DATA_FIX["accel"])
    scaled = converters.imu_to_sample(livox, 0.0, accel_in_g=None)
    kept = converters.imu_to_sample(ms2, 0.0, accel_in_g=None)
    assert np.array_equal(scaled.accel, np.array(LIVOX_IMU_FIX["accel"]) * GRAVITY)
    assert np.array_equal(kept.accel, np.array(IMU_DATA_FIX["accel"]))


def test_odometry_converter_matches_offline_pairs():
    """Streaming increments equal ExportedRun.odom_relative's, drops included."""
    msgs = [_odom_msg(t, p, q) for t, p, q in ODOM_FIX]
    stamps = np.array([converters.stamp_of(m) for m in msgs])
    poses = np.array(
        [make_transform(quat_to_matrix(np.array(q)), np.array(p)) for _, p, q in ODOM_FIX]
    )
    expected = poses_to_relative(
        stamps, poses, "odom",
        sigma_translation=0.05, sigma_yaw=0.008, sigma_position=0.015, stride=2,
    )
    assert len(expected) == 1  # the 0.95 s and 1.9 s gaps drop the first two pairs

    conv = converters.OdometryConverter(stride=2)
    got = [m for m in map(conv.push, msgs) if m is not None]
    assert len(got) == len(expected)
    for a, b in zip(got, expected):
        assert a == b  # frozen dataclasses: field-exact equality


def test_imu_decimator_matches_offline_block_average():
    rng = np.random.default_rng(0)
    n = 400
    stamp = 5.0 + np.arange(n) / 400.0 + rng.uniform(0, 2e-4, n)
    stream = ImuStream("imu", np.sort(stamp), rng.normal(0, 0.1, (n, 3)), rng.normal(0, 1.0, (n, 3)))
    offline = stream.decimated(100.0)

    dec = converters.ImuDecimator(100.0)
    out = [s for s in map(dec.push, stream.samples()) if s is not None]
    # Streaming emits a bin when the next one starts; the trailing bin stays open.
    assert len(out) == len(offline.stamp) - 1
    assert np.array_equal([s.stamp for s in out], offline.stamp[:-1])
    assert np.array_equal([s.gyro for s in out], offline.gyro[:-1])
    assert np.array_equal([s.accel for s in out], offline.accel[:-1])


def test_image_converter_bgra_and_rgb():
    h, w = 2, 3
    bgra = np.arange(h * w * 4, dtype=np.uint8).reshape(h, w, 4)
    msg = SimpleNamespace(
        header=_header(1788593590.25), height=h, width=w, encoding="bgra8",
        is_bigendian=0, step=w * 4, data=bgra.tobytes(),
    )
    stamp, bgr = converters.image_to_bgr(msg)
    assert stamp == converters.stamp_of(msg)
    assert bgr.shape == (h, w, 3)
    assert np.array_equal(bgr, bgra[:, :, :3])

    rgb = np.arange(h * w * 3, dtype=np.uint8).reshape(h, w, 3)
    msg = SimpleNamespace(
        header=_header(1.5), height=h, width=w, encoding="rgb8",
        is_bigendian=0, step=w * 3, data=rgb.tobytes(),
    )
    _, bgr = converters.image_to_bgr(msg)
    assert np.array_equal(bgr, rgb[:, :, ::-1])


def test_image_converter_strips_row_padding():
    """Row-padded images decode correctly (channels come from the encoding).

    Regression: inferring channels from len(data) counts padding bytes and
    produced a garbage (h, w, 5) array for a padded bgra8 image.
    """
    h, w, step = 4, 6, 32  # bgra8 rows padded from 24 to 32 bytes
    bgra = np.arange(h * w * 4, dtype=np.uint8).reshape(h, w, 4)
    rows = np.zeros((h, step), dtype=np.uint8)
    rows[:, : w * 4] = bgra.reshape(h, w * 4)
    msg = SimpleNamespace(
        header=_header(2.0), height=h, width=w, encoding="bgra8",
        is_bigendian=0, step=step, data=rows.tobytes(),
    )
    _, bgr = converters.image_to_bgr(msg)
    assert bgr.shape == (h, w, 3)
    assert np.array_equal(bgr, bgra[:, :, :3])

    mono = np.arange(h * w, dtype=np.uint8).reshape(h, w)
    rows = np.zeros((h, 8), dtype=np.uint8)
    rows[:, :w] = mono
    msg = SimpleNamespace(
        header=_header(2.5), height=h, width=w, encoding="mono8",
        is_bigendian=0, step=8, data=rows.tobytes(),
    )
    _, out = converters.image_to_bgr(msg)
    assert out.shape == (h, w)
    assert np.array_equal(out, mono)


def test_streaming_clock_sync_is_bit_identical_to_clock_sync():
    """The node's O(1) clock sync must give exactly the offline ClockSync's stamps.

    Non-monotonic device stamps, latency jumps down (the envelope drops
    instantly) and up (it may only creep 1e-4 s per sample), several window
    lengths, epoch stamps passing through - sample for sample, bit for bit.
    """
    rng = np.random.default_rng(11)
    for trial in range(12):
        n = 3000
        device = np.cumsum(rng.uniform(0.0, 0.006, n))
        if trial % 2:
            device = device + rng.normal(0.0, 0.004, n)  # out-of-order stamps
        latency = rng.exponential(0.01, n) + np.where(np.arange(n) > n // 2, rng.choice([-0.03, 0.04]), 0.0)
        receive = device + 1788593584.0 + latency
        window = [0.5, 2.0, 10.0][trial % 3]
        slow, fast = ClockSync(window_s=window), converters.StreamingClockSync(window_s=window)
        expected = [slow.correct(float(d), float(r)) for d, r in zip(device, receive)]
        got = [fast.correct(float(d), float(r)) for d, r in zip(device, receive)]
        assert got == expected, trial
    epoch = converters.StreamingClockSync()
    assert epoch.correct(1788593587.25, 1788593587.3) == 1788593587.25


def test_image_decode_with_scale_matches_decode_then_resize():
    """Resizing in the message buffer (the node's path) == decode, then INTER_AREA resize.

    INTER_AREA works per channel, so resizing BGRA / RGB first and dropping
    alpha / swapping afterwards must give the same bytes; the fast path only
    avoids copying the full-resolution frame.
    """
    import cv2

    rng = np.random.default_rng(4)
    for encoding, channels in (("bgra8", 4), ("bgr8", 3), ("rgb8", 3), ("rgba8", 4), ("mono8", 1)):
        for h, w, pad in ((60, 96, 0), (61, 97, 5)):
            rows = rng.integers(0, 256, (h, w * channels + pad), dtype=np.uint8)
            msg = SimpleNamespace(header=_header(3.0), height=h, width=w, encoding=encoding,
                                  is_bigendian=0, step=w * channels + pad, data=rows.tobytes())
            _, full = converters.image_to_bgr(msg)
            for scale in (0.5, 0.37):
                _, got = converters.image_to_bgr(msg, scale)
                want = cv2.resize(full, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                assert got.shape == want.shape and np.array_equal(got, want), (encoding, pad, scale)
                assert got.flags.c_contiguous


def test_occupancy_grid_data_uses_the_rclpy_fast_path():
    """int8[] data as array.array('b'): rclpy's setter validates lists element by element."""
    import array

    grid = EvidenceGrid(cell=0.2)
    grid.insert_lidar(np.array([[x, 0.0, -0.10] for x in np.linspace(0.5, 3.0, 20)]), np.zeros(3))
    msg = AutoMsg()
    converters.fill_occupancy_grid(msg, grid, 1.0)
    assert isinstance(msg.data, array.array) and msg.data.typecode == "b"
    assert len(msg.data) == msg.info.width * msg.info.height


def test_camera_info_intrinsics():
    K, size = converters.camera_info_to_intrinsics(_camera_info_msg())
    assert size == (1920, 1200)
    assert np.array_equal(K, np.array(CAMERA_INFO_K).reshape(3, 3))
    assert K[0, 0] == 675.8616943359375 and K[1, 2] == 578.778564453125


def test_pointcloud_converter_livox_layout():
    """PointCloud2 decoding mirrors the export's dtype and range filter."""
    fields = [
        SimpleNamespace(name="x", offset=0, datatype=7, count=1),
        SimpleNamespace(name="y", offset=4, datatype=7, count=1),
        SimpleNamespace(name="z", offset=8, datatype=7, count=1),
        SimpleNamespace(name="intensity", offset=12, datatype=7, count=1),
        SimpleNamespace(name="tag", offset=16, datatype=2, count=1),
        SimpleNamespace(name="line", offset=17, datatype=2, count=1),
        SimpleNamespace(name="timestamp", offset=18, datatype=8, count=1),
    ]
    stamp = 1788593587.4
    dtype = converters._cloud_dtype(SimpleNamespace(fields=fields, point_step=26))
    cloud = np.zeros(4, dtype=dtype)
    cloud["x"] = [0.1, 3.0, np.nan, 70.0]  # self-return, keeper, invalid, too far
    cloud["y"] = [0.0, 4.0, 1.0, 0.0]
    cloud["z"] = [0.0, 0.5, 0.0, 0.0]
    cloud["intensity"] = [10.0, 300.7, 5.0, 5.0]
    cloud["timestamp"] = (stamp + np.array([0.0, 0.05, 0.06, 0.07])) * 1e9

    msg = SimpleNamespace(
        header=_header(stamp), fields=fields, point_step=26,
        height=1, width=4, data=cloud.tobytes(),
    )
    scan = converters.pointcloud_to_scan(msg)
    assert scan.channel == "lidar"
    assert scan.stamp == converters.stamp_of(msg)
    assert scan.points.shape == (1, 3) and scan.points.dtype == np.float32
    assert np.array_equal(scan.points[0], np.array([3.0, 4.0, 0.5], dtype=np.float32))
    assert scan.intensity.dtype == np.uint8 and scan.intensity[0] == 255  # clipped
    expected_dt = (cloud["timestamp"][1:2].astype(np.float64) * 1e-9 - scan.stamp).astype(np.float32)
    assert np.array_equal(scan.point_dt, expected_dt)


# --------------------------------------------------------- occupancy grid
class AutoMsg:
    """Auto-vivifying stand-in for a ROS message: nested field assignment works."""

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        child = AutoMsg()
        object.__setattr__(self, name, child)
        return child


def test_occupancy_grid_metadata_roundtrips_evidence_grid():
    grid = EvidenceGrid(cell=0.2)
    ground = np.array([[x, y, -0.10] for x in np.linspace(0.5, 2.0, 8) for y in np.linspace(-1, 1, 8)])
    wall = np.array([[2.6, y, z] for y in np.linspace(-1, 1, 5) for z in np.linspace(0.2, 0.8, 4)])
    grid.insert_lidar(np.vstack([ground, wall]), np.array([0.0, 0.0, 0.0]))
    classes = grid.classify()
    assert (classes == 1).any() and (classes == 2).any()  # meaningful fixture

    msg = AutoMsg()
    converters.fill_occupancy_grid(msg, grid, stamp=1788593590.5, frame_id="map")
    nx, ny = classes.shape
    assert msg.info.width == nx and msg.info.height == ny
    assert msg.info.resolution == float(grid.cell)
    assert (msg.info.origin.position.x, msg.info.origin.position.y) == tuple(grid.origin)
    assert msg.info.origin.orientation.w == 1.0
    assert msg.header.frame_id == "map"
    assert msg.header.stamp.sec == 1788593590 and msg.header.stamp.nanosec == 500000000

    # Row-major, x fastest: data[iy * width + ix] -> classes[ix, iy].
    data = np.array(msg.data, dtype=np.int8)
    assert len(data) == nx * ny and set(np.unique(data)) <= {-1, 0, 100}
    recon = np.zeros((nx, ny), dtype=np.uint8)
    grid_view = data.reshape(ny, nx).T
    recon[grid_view == 0] = 1
    recon[grid_view == 100] = 2
    assert np.array_equal(recon, classes)

    # Cell centres agree between the grid and the published metadata.
    centre = grid.cell_to_world(np.array([[3, 4]]))[0]
    meta = np.array([msg.info.origin.position.x, msg.info.origin.position.y])
    assert np.allclose(centre, meta + (np.array([3, 4]) + 0.5) * msg.info.resolution)




# ------------------------------------------- requirement 6: all encodings
def test_every_accepted_encoding_decodes():
    """bgr8, rgb8, bgra8, rgba8 and mono8 all decode to working BGR/mono."""
    import cv2

    rng = np.random.default_rng(9)
    h, w = 8, 10
    for encoding, channels in (("bgr8", 3), ("rgb8", 3), ("bgra8", 4), ("rgba8", 4), ("mono8", 1)):
        raw = rng.integers(0, 256, (h, w, channels) if channels > 1 else (h, w), dtype=np.uint8)
        msg = SimpleNamespace(header=_header(4.0), height=h, width=w, encoding=encoding,
                              is_bigendian=0, step=w * channels, data=raw.tobytes())
        stamp, out = converters.image_to_bgr(msg)
        assert stamp == converters.stamp_of(msg)
        if channels == 1:
            assert out.shape == (h, w)
            assert np.array_equal(out, raw)
        else:
            assert out.shape == (h, w, 3)
            if encoding.startswith("rgb"):
                assert np.array_equal(out, raw[:, :, 2::-1])
            else:
                assert np.array_equal(out, raw[:, :, :3])
    _ = cv2


def _jpeg_bytes(image) -> bytes:
    import cv2

    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    assert ok
    return buf.tobytes()


def test_compressed_image_jpeg_and_png_decode():
    import cv2

    rng = np.random.default_rng(10)
    image = rng.integers(0, 256, (24, 32, 3), dtype=np.uint8)
    for fmt, encode in (("jpeg", ".jpg"), ("png", ".png")):
        ok, buf = cv2.imencode(encode, image)
        assert ok
        msg = SimpleNamespace(header=_header(6.5), format=fmt, data=buf.tobytes())
        stamp, out = converters.compressed_to_bgr(msg)
        assert stamp == pytest.approx(6.5)
        assert out.shape == image.shape and out.dtype == np.uint8
        if fmt == "png":  # lossless: bytes must round-trip exactly
            assert np.array_equal(out, image)
    # scale applies after decoding
    msg = SimpleNamespace(header=_header(7.0), format="png", data=cv2.imencode(".png", image)[1].tobytes())
    _, half = converters.compressed_to_bgr(msg, 0.5)
    want = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    assert np.array_equal(half, want)


def test_compressed_image_garbage_raises():
    msg = SimpleNamespace(header=_header(1.0), format="jpeg", data=b"not an image")
    with pytest.raises(ValueError, match="CompressedImage"):
        converters.compressed_to_bgr(msg)
