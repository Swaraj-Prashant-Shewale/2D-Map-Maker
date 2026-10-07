"""The live 2D mapping node, driven through a fake rclpy (no ROS installed).

New-requirement coverage on top of the ported behaviours:
- single camera + dedicated IMU is the DEFAULT configuration
- the node never subscribes another IMU unless livox_imu is switched on,
  and never any camera IMU (structural + observed)
- requirement-13 inputs (mag / speed / wheel) reach the fusion engine
- camera source abstraction: a device camera creates NO ROS image
  subscription and feeds the same worker path
- per-camera kart mask: file-loaded and learned wiring
- extrinsics from TF with the manual fallback
- lens distortion wired as points/ray tables (never frames)
- the classical-fix grid guards are ON in the live node
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from kart_map2d.core.transforms import make_transform, rpy_matrix
from kart_map2d.ros2 import node as node_mod

PKG_ROOT = Path(__file__).resolve().parents[1]


def _find_project_root() -> Path:
    """Directory holding kart_master.cfg: works on the laptop layout
    (<project>/ros2_ws/src/kart_map2d) and the kart layout
    (~/kart_ws/src/kart_map2d with the cfg staged at the package top)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "kart_master.cfg").exists():
            return parent
    return here.parents[4]


PROJECT_ROOT = _find_project_root()


# ------------------------------------------------------------ msg fixtures
def _time(stamp: float) -> SimpleNamespace:
    sec = int(np.floor(stamp))
    return SimpleNamespace(sec=sec, nanosec=int(round((stamp - sec) * 1e9)))


def _header(stamp: float, frame_id: str = "") -> SimpleNamespace:
    return SimpleNamespace(stamp=_time(stamp), frame_id=frame_id)


def _vec3(x, y, z) -> SimpleNamespace:
    return SimpleNamespace(x=x, y=y, z=z)


def _imu_msg(stamp: float, gyro=None, accel=None) -> SimpleNamespace:
    """A realistic IMU sample: MEMS noise rides on rest values, because a
    byte-identical repeated payload trips the stuck-sensor detector (as it
    must on the kart)."""
    rng = np.random.default_rng(int(stamp * 1e4) % 2**31)
    g = np.asarray(gyro if gyro is not None else rng.normal(0.0, 8e-4, 3))
    a = np.asarray(accel if accel is not None else [0.0, 0.0, 9.81] + rng.normal(0.0, 0.02, 3))
    return SimpleNamespace(header=_header(stamp, "imu_link"), angular_velocity=_vec3(*g), linear_acceleration=_vec3(*a))


def _odom_msg(stamp: float, x: float) -> SimpleNamespace:
    return SimpleNamespace(
        header=_header(stamp, "odom"),
        pose=SimpleNamespace(pose=SimpleNamespace(position=_vec3(x, 0.0, 0.0), orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0))),
    )


def _mag_msg(stamp: float, field=(5e-5, 0.0, 0.0)) -> SimpleNamespace:
    return SimpleNamespace(header=_header(stamp), magnetic_field=_vec3(*field))


def _image_msg(stamp: float, h=24, w=32, frame_id="cam_frame") -> SimpleNamespace:
    rng = np.random.default_rng(int(stamp * 1000) % 2**31)
    data = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    return SimpleNamespace(header=_header(stamp, frame_id), height=h, width=w, encoding="bgr8",
                           is_bigendian=0, step=w * 3, data=data.tobytes())


# -------------------------------------------------------------- node stubs
class AutoMsg:
    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        child = AutoMsg()
        object.__setattr__(self, name, child)
        return child


class FakeQoSProfile:
    def __init__(self, history=None, depth=None, reliability=None, durability=None):
        self.depth = depth
        self.reliability = reliability
        self.durability = durability


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def now(self):
        return SimpleNamespace(nanoseconds=int(round(self.t * 1e9)))


class FakePublisher:
    def __init__(self, msg_type, topic, qos):
        self.topic = topic
        self.msgs = []

    def publish(self, msg):
        self.msgs.append(msg)


class FakeNode:
    def __init__(self) -> None:
        self.clock = FakeClock()
        self.subs: dict[str, SimpleNamespace] = {}
        self.pubs: dict[str, FakePublisher] = {}
        self.timers = []
        self.services = {}
        self.warnings = []

    def create_subscription(self, msg_type, topic, callback, qos):
        sub = SimpleNamespace(msg_type=msg_type, topic=topic, callback=callback, qos=qos)
        self.subs[topic] = sub
        return sub

    def create_publisher(self, msg_type, topic, qos):
        pub = FakePublisher(msg_type, topic, qos)
        self.pubs[topic] = pub
        return pub

    def create_timer(self, period, callback):
        timer = SimpleNamespace(period=period, callback=callback)
        self.timers.append(timer)
        return timer

    def create_service(self, srv_type, name, callback):
        service = SimpleNamespace(name=name, callback=callback)
        self.services[name] = service
        return service

    def get_clock(self):
        return self.clock

    def get_logger(self):
        return SimpleNamespace(warning=self.warnings.append)

    def get_fully_qualified_name(self):
        return "/kart_map2d"

    def timer(self, period: float):
        matches = [t for t in self.timers if abs(t.period - period) < 1e-9]
        assert len(matches) == 1, (period, [t.period for t in self.timers])
        return matches[0]


@pytest.fixture()
def ros_stubs(monkeypatch):
    """Install fake rclpy / *_msgs modules in sys.modules, as on the kart."""
    rclpy = types.ModuleType("rclpy")
    qos = types.ModuleType("rclpy.qos")
    qos.QoSProfile = FakeQoSProfile
    qos.ReliabilityPolicy = SimpleNamespace(RELIABLE="reliable", BEST_EFFORT="best_effort")
    qos.DurabilityPolicy = SimpleNamespace(VOLATILE="volatile", TRANSIENT_LOCAL="transient_local")
    qos.HistoryPolicy = SimpleNamespace(KEEP_LAST="keep_last")
    rclpy.qos = qos
    node_pkg = types.ModuleType("rclpy.node")
    node_pkg.Node = object
    monkeypatch.setitem(sys.modules, "rclpy", rclpy)
    monkeypatch.setitem(sys.modules, "rclpy.qos", qos)
    monkeypatch.setitem(sys.modules, "rclpy.node", node_pkg)

    msg_types = {}
    packages = {
        "sensor_msgs": ("Imu", "Image", "CompressedImage", "CameraInfo", "PointCloud2", "PointField",
                        "MagneticField", "JointState"),
        "nav_msgs": ("Odometry", "Path", "OccupancyGrid"),
        "geometry_msgs": ("PoseStamped", "TwistStamped"),
        "std_msgs": ("String", "Float32", "Int64"),
    }
    for package, names in packages.items():
        pkg = types.ModuleType(package)
        msg = types.ModuleType(package + ".msg")
        for name in names:
            msg_types[name] = type(name, (AutoMsg,), {})
            setattr(msg, name, msg_types[name])
        pkg.msg = msg
        monkeypatch.setitem(sys.modules, package, pkg)
        monkeypatch.setitem(sys.modules, package + ".msg", msg)
    std_srvs = types.ModuleType("std_srvs")
    srv = types.ModuleType("std_srvs.srv")
    msg_types["Trigger"] = srv.Trigger = type("Trigger", (), {})
    std_srvs.srv = srv
    monkeypatch.setitem(sys.modules, "std_srvs", std_srvs)
    monkeypatch.setitem(sys.modules, "std_srvs.srv", srv)
    return SimpleNamespace(types=msg_types)


def one_camera_params(**overrides) -> node_mod.NodeParams:
    """The kart-target shape: one generic ros2 camera + the dedicated IMU."""
    params = node_mod.NodeParams(
        cameras={
            "camera": node_mod.CameraParams(
                image_topic="/cam/image",
                intrinsics=(60.0, 60.0, 16.0, 12.0, 32, 24),
                mount_xyz=(0.0, 0.0, 0.5),
                height_m=0.5,
                learn_mask=False,
            )
        },
        mapping_async=False,
        vo_async=False,
        seg_mode="classical",
    )
    for key, value in overrides.items():
        setattr(params, key, value)
    return params


# ----------------------------------------------------------- import purity
def test_module_import_pulls_no_ros(monkeypatch):
    """GATE: importing the ros2 package must not import rclpy or any *_msgs."""
    ros_names = ("rclpy", "sensor_msgs", "nav_msgs", "geometry_msgs", "std_msgs")
    for name in list(sys.modules):
        if name.startswith("kart_map2d.ros2") or name.split(".")[0] in ros_names:
            monkeypatch.delitem(sys.modules, name, raising=False)
    importlib.import_module("kart_map2d.ros2.converters")
    importlib.import_module("kart_map2d.ros2.node")
    leaked = [m for m in sys.modules if m.split(".")[0] in ros_names]
    assert leaked == []


# ------------------------------------------------------------------- QoS
def test_qos_covers_every_input_kind():
    expected = {
        "imu": ("reliable", 2000, "volatile"),
        "imu_livox": ("reliable", 256, "volatile"),
        "lidar": ("reliable", 256, "volatile"),
        "odom": ("reliable", 50, "volatile"),
        "image": ("best_effort", 1, "volatile"),
        "camera_info": ("best_effort", 1, "volatile"),
        "mag": ("reliable", 200, "volatile"),
        "speed": ("reliable", 100, "volatile"),
        "wheel": ("reliable", 200, "volatile"),
    }
    assert set(node_mod.QOS) == set(expected)
    for key, (reliability, depth, durability) in expected.items():
        spec = node_mod.QOS[key]
        assert (spec.reliability, spec.depth, spec.durability) == (reliability, depth, durability)


def test_grid_guards_match_pipeline_constants():
    """The classical-fix guards are live-node defaults AND equal the pipeline's."""
    from kart_map2d import pipeline

    assert node_mod.KART_FOOTPRINT_M == pipeline.KART_FOOTPRINT_M
    assert node_mod.CAMERA_MIN_CONTACT_RANGE_M == pipeline.CAMERA_MIN_CONTACT_RANGE_M
    p = node_mod.NodeParams()
    assert tuple(p.kart_footprint) == pipeline.KART_FOOTPRINT_M
    assert p.min_contact_range_m == pipeline.CAMERA_MIN_CONTACT_RANGE_M


def test_live_grid_is_built_with_the_guards(ros_stubs):
    wrapper = node_mod.KartMapNode(FakeNode(), one_camera_params())
    assert wrapper.grid.kart_footprint == node_mod.KART_FOOTPRINT_M
    assert wrapper.grid.min_contact_range == node_mod.CAMERA_MIN_CONTACT_RANGE_M


# ------------------------------------------------ single camera + IMU only
def test_default_params_have_no_hardwired_camera():
    """No hidden ZED (or any) camera in the bare defaults - configuration
    (kart_master.cfg) is the only place cameras come from."""
    assert node_mod.NodeParams().cameras == {}


def test_shipped_cfg_gives_single_camera_and_dedicated_imu(ros_stubs):
    from kart_map2d.config.master_config import load_master_config, to_mapping_params_dict

    cfg = load_master_config(PROJECT_ROOT / "kart_master.cfg")
    params = node_mod.NodeParams.from_dict(to_mapping_params_dict(cfg))
    params.mapping_async = False
    fake = FakeNode()
    wrapper = node_mod.KartMapNode(fake, params)
    image_subs = [t for t in fake.subs if "image" in t]
    assert len(image_subs) == 1  # exactly ONE camera
    imu_subs = [t for t in fake.subs if "imu" in t.lower()]
    assert imu_subs == ["/imu/data"]  # the dedicated IMU and NOTHING else
    wrapper.close()


def test_never_subscribes_another_imu_unless_livox_switched(ros_stubs):
    """Requirement 4: /livox/imu only behind the switch; never a camera IMU.

    The camera topics here LOOK like a vendor stack that also offers
    /cam/imu/data - the node must not touch it: subscriptions are exactly
    the configured imu_topic, the camera image topic and nothing IMU-ish else.
    """
    params = one_camera_params()
    params.cameras["camera"].image_topic = "/cam/rgb/image"  # vendor-style namespace
    fake = FakeNode()
    wrapper = node_mod.KartMapNode(fake, params)
    topics = set(fake.subs)
    assert topics == {"/imu/data", "/cam/rgb/image"}
    assert not any("cam" in t and "imu" in t for t in topics)
    wrapper.close()

    # livox_imu switched on -> exactly one more IMU subscription appears.
    params2 = one_camera_params(livox_imu_topic="/livox/imu")
    fake2 = FakeNode()
    wrapper2 = node_mod.KartMapNode(fake2, params2)
    assert set(fake2.subs) == {"/imu/data", "/cam/rgb/image".replace("/rgb", ""), "/livox/imu"} or "/livox/imu" in fake2.subs
    imu_topics = sorted(t for t in fake2.subs if "imu" in t.lower())
    assert imu_topics == ["/imu/data", "/livox/imu"]
    wrapper2.close()


def test_structurally_no_camera_imu_parameter_exists():
    """There is no parameter that could name a camera IMU: every IMU-ish
    NodeParams field is exactly imu_topic / livox_imu_topic."""
    from dataclasses import fields

    imu_fields = [f.name for f in fields(node_mod.NodeParams) if "imu" in f.name and f.name.endswith("_topic")]
    assert sorted(imu_fields) == ["imu_topic", "livox_imu_topic"]
    cam_fields = [f.name for f in fields(node_mod.CameraParams)]
    assert not any("imu" in name for name in cam_fields)


# ------------------------------------------------- requirement-13 inputs
T0 = 1788593585.0  # epoch-like: the node's clock-ready guard needs real time


def _drive_imu(wrapper, fake, t0=T0, seconds=1.0, rate=100):
    for k in range(int(seconds * rate)):
        t = t0 + k / rate
        fake.clock.t = t
        wrapper._handle_imu(_imu_msg(t))


def test_mag_speed_wheel_reach_the_engine(ros_stubs):
    from kart_map2d.sensors.mag_calibration import MagCalibration, save_mag_calibration
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        cal_path = Path(tmp) / "mag.yaml"
        save_mag_calibration(cal_path, MagCalibration(np.zeros(3), np.eye(3), field_norm=5e-5))
        params = one_camera_params(
            mag_topic="/mag", mag_calibration_file=str(cal_path), mag_rate_hz=1000.0,
            speed_topic="/speed", speed_type="float32",
            wheel_topic="/ticks", wheel_type="int64", wheel_ticks_per_metre=100.0,
        )
        fake = FakeNode()
        wrapper = node_mod.KartMapNode(fake, params)
        assert {"/mag", "/speed", "/ticks"} <= set(fake.subs)
        fake.clock.t = T0
        _drive_imu(wrapper, fake, seconds=0.2)
        fake.clock.t = T0 + 0.2
        fake.subs["/mag"].callback(_mag_msg(T0 + 0.2))
        fake.subs["/speed"].callback(SimpleNamespace(data=1.0))
        fake.subs["/ticks"].callback(SimpleNamespace(data=100))
        fake.clock.t = T0 + 0.3
        fake.subs["/ticks"].callback(SimpleNamespace(data=110))
        assert wrapper.counters["mag"] == 1
        assert wrapper.counters["speed"] == 1
        assert wrapper.counters["wheel"] == 1  # first tick anchors, second measures
        for channel in ("speedometer", "wheel"):
            assert channel in wrapper.engine.health.channels
        # Validation-armed trust: with gyro data flowing but no turn seen
        # yet, magnetic headings are WITHHELD (a straight cannot distinguish
        # a wrong calibration from a mount offset), so the mag channel has
        # not reached the engine - only the gated counter moves.
        assert wrapper.counters["mag_gated"] == 1
        assert "mag" not in wrapper.engine.health.channels
        assert not wrapper._mag_adapter.usable and not wrapper._mag_adapter.tripped
        wrapper.close()


def test_mag_without_calibration_is_refused_with_procedure(ros_stubs):
    params = one_camera_params(mag_topic="/mag")
    with pytest.raises(node_mod.NodeConfigError, match="calibrate_magnetometer"):
        node_mod.KartMapNode(FakeNode(), params)


def test_sensors_used_names_the_new_channels(ros_stubs):
    from kart_map2d.sensors.mag_calibration import MagCalibration, save_mag_calibration
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        cal_path = Path(tmp) / "mag.yaml"
        save_mag_calibration(cal_path, MagCalibration(np.zeros(3), np.eye(3), field_norm=5e-5))
        params = one_camera_params(mag_topic="/mag", mag_calibration_file=str(cal_path), speed_topic="/speed")
        fake = FakeNode()
        wrapper = node_mod.KartMapNode(fake, params)
        fake.clock.t = T0
        fake.subs["/imu/data"].callback(_imu_msg(T0))
        fake.subs["/mag"].callback(_mag_msg(T0 + 0.05))
        fake.subs["/speed"].callback(SimpleNamespace(data=0.5))
        used = wrapper._sensors_used()
        assert {"imu", "mag", "speedometer"} <= set(used)
        wrapper.close()


# ------------------------------------------------- camera source: device
def test_device_camera_creates_no_ros_image_subscription(ros_stubs, tmp_path):
    from kart_map2d.camera.model import CameraModel, save_calibration_yaml

    cal = tmp_path / "cal.yaml"
    save_calibration_yaml(cal, CameraModel(np.array([[60.0, 0, 16], [0, 60.0, 12], [0, 0, 1]]), (32, 24)))
    params = node_mod.NodeParams(
        cameras={
            "camera": node_mod.CameraParams(
                source="device", device="0", calibration_file=str(cal), learn_mask=False,
                mount_xyz=(0.0, 0.0, 0.5), height_m=0.5, processing_scale=1.0,
            )
        },
        mapping_async=False,
        vo_async=False,
    )
    fake = FakeNode()
    wrapper = node_mod.KartMapNode(fake, params)
    # ZERO ROS image topics subscribed (requirement 14's proof shape).
    assert set(fake.subs) == {"/imu/data"}
    # Intrinsics came from the calibration file, before any frame.
    assert "camera" in wrapper._intrinsics
    # Frames from the capture callback flow down the same path.
    rng = np.random.default_rng(0)
    fake.clock.t = T0
    wrapper._on_capture_frame("camera", T0, rng.integers(0, 256, (24, 32, 3), dtype=np.uint8))
    wrapper._on_capture_frame("camera", T0 + 0.2, rng.integers(0, 256, (24, 32, 3), dtype=np.uint8))
    assert wrapper.counters["capture_frames"] == 2
    assert wrapper.counters["frames"] == 2
    assert wrapper._pending_frame["camera"][0] == T0 + 0.2
    wrapper.close()


def test_gstreamer_without_pipeline_refused(ros_stubs, tmp_path):
    from kart_map2d.camera.model import CameraModel, save_calibration_yaml

    cal = tmp_path / "cal.yaml"
    save_calibration_yaml(cal, CameraModel(np.eye(3) * 60.0, (32, 24)))
    params = node_mod.NodeParams(
        cameras={"camera": node_mod.CameraParams(source="gstreamer", calibration_file=str(cal))},
    )
    with pytest.raises(node_mod.NodeConfigError, match="gstreamer_pipeline"):
        node_mod.KartMapNode(FakeNode(), params)


def test_camera_without_any_intrinsics_is_refused(ros_stubs):
    params = node_mod.NodeParams(cameras={"camera": node_mod.CameraParams(image_topic="/cam/image")})
    with pytest.raises(node_mod.NodeConfigError, match="intrinsics"):
        node_mod.KartMapNode(FakeNode(), params)


# ---------------------------------------------------------- kart masks
def test_mask_file_is_loaded_and_reaches_the_segmenter(ros_stubs, tmp_path):
    import cv2

    mask = np.zeros((24, 32), np.uint8)
    mask[18:, :10] = 255
    mask_path = tmp_path / "mask.png"
    cv2.imwrite(str(mask_path), mask)
    params = one_camera_params()
    params.cameras["camera"].mask_file = str(mask_path)
    fake = FakeNode()
    wrapper = node_mod.KartMapNode(fake, params)
    got = wrapper._ensure_kart_mask("camera", (24, 32))
    assert got is not None and got.shape == (24, 32)
    assert bool(got[20, 5]) and not bool(got[0, 0])
    segmenter = wrapper._segmenter_for("camera")
    assert segmenter._custom_kart_mask is not None
    wrapper.close()


def test_learned_mask_rebuilds_the_segmenter(ros_stubs, monkeypatch):
    """Wiring test: when the online learner produces a mask, the camera's
    classical segmenter is rebuilt with it (the learner itself is covered by
    test_classical_ground.py)."""
    params = one_camera_params()
    params.cameras["camera"].learn_mask = True
    fake = FakeNode()
    wrapper = node_mod.KartMapNode(fake, params)
    first = wrapper._segmenter_for("camera")
    assert first._custom_kart_mask is None

    def fake_learner(images, drv, obs, **kw):
        h, w = images[0].shape[:2]
        mask = np.zeros((h, w), bool)
        mask[2 * h // 3 :, :] = True  # bottom band = "kart body"
        return mask

    monkeypatch.setattr("kart_map2d.perception.classical_ground.learn_kart_mask", fake_learner)
    rng = np.random.default_rng(1)
    wrapper._mask_learn_frames["camera"] = [
        rng.integers(0, 256, (24, 32, 3), dtype=np.uint8) for _ in range(wrapper._MASK_LEARN_FRAMES - 1)
    ]
    wrapper._collect_mask_frame("camera", rng.integers(0, 256, (24, 32, 3), dtype=np.uint8))
    assert wrapper.counters["kart_masks_learned"] == 1
    mask = wrapper._kart_masks["camera"]
    assert mask is not None and mask[20, 16]
    rebuilt = wrapper._segmenter_for("camera")
    assert rebuilt is not first and rebuilt._custom_kart_mask is not None
    wrapper.close()


# ------------------------------------------------------- TF vs manual mounts
def test_manual_mount_is_the_default_extrinsics(ros_stubs):
    wrapper = node_mod.KartMapNode(FakeNode(), one_camera_params())
    T = wrapper._camera_mount("camera")
    want = make_transform(rpy_matrix(-np.pi / 2, 0.0, -np.pi / 2), np.array([0.0, 0.0, 0.5]))
    np.testing.assert_allclose(T, want)
    wrapper.close()


def test_tf_extrinsics_optical_frame_used_as_is(ros_stubs):
    params = one_camera_params()
    params.cameras["camera"].frame = "auto"
    fake = FakeNode()
    wrapper = node_mod.KartMapNode(fake, params)
    T_tf = make_transform(rpy_matrix(-np.pi / 2, 0.05, -np.pi / 2), np.array([0.3, -0.1, 0.7]))
    wrapper.set_tf_lookup(lambda frame: T_tf if frame == "front_optical_frame" else None)
    # Before any image arrives, "auto" has no frame: manual fallback, uncached.
    manual = wrapper._camera_mount("camera")
    assert manual[2, 3] == pytest.approx(0.5)
    fake.subs["/cam/image"].callback(_image_msg(T0, frame_id="front_optical_frame"))
    np.testing.assert_allclose(wrapper._camera_mount("camera"), T_tf)
    wrapper.close()


def test_tf_extrinsics_body_frame_gets_optical_convention(ros_stubs):
    params = one_camera_params()
    params.cameras["camera"].frame = "camera_link"
    wrapper = node_mod.KartMapNode(FakeNode(), params)
    T_link = make_transform(np.eye(3), np.array([0.2, 0.0, 0.6]))
    wrapper.set_tf_lookup(lambda frame: T_link if frame == "camera_link" else None)
    T = wrapper._camera_mount("camera")
    want = T_link @ make_transform(rpy_matrix(-np.pi / 2, 0.0, -np.pi / 2), np.zeros(3))
    np.testing.assert_allclose(T, want)
    wrapper.close()


def test_tf_lookup_failure_falls_back_to_manual_with_warning(ros_stubs):
    params = one_camera_params()
    params.cameras["camera"].frame = "missing_frame"
    fake = FakeNode()
    wrapper = node_mod.KartMapNode(fake, params)
    wrapper.set_tf_lookup(lambda frame: (_ for _ in ()).throw(KeyError("no such frame")))
    T = wrapper._camera_mount("camera")
    assert T[2, 3] == pytest.approx(0.5)  # the manual mount
    assert any("manual mount" in w for w in fake.warnings)
    wrapper.close()


# ---------------------------------------------------------- distortion wiring
def test_distorted_calibration_reaches_vo_and_ray_tables(ros_stubs, tmp_path):
    from kart_map2d.camera.model import CameraModel, save_calibration_yaml

    K = np.array([[60.0, 0, 16.0], [0, 60.0, 12.0], [0, 0, 1.0]])
    D = np.array([-0.25, 0.05, 0.0, 0.0, 0.0])
    cal = tmp_path / "cal.yaml"
    save_calibration_yaml(cal, CameraModel(K, (32, 24), D, "plumb_bob"))
    params = one_camera_params()
    params.cameras["camera"].calibration_file = str(cal)
    params.cameras["camera"].intrinsics = ()
    params.cameras["camera"].processing_scale = 1.0
    fake = FakeNode()
    wrapper = node_mod.KartMapNode(fake, params)
    vo = wrapper._ensure_vo("camera")
    assert vo.camera_model is not None and vo.camera_model.has_distortion
    rays = wrapper._camera_model_scaled("camera").pixel_rays((24, 32))
    from kart_map2d.mapping.evidence_grid import _pixel_rays

    assert not np.allclose(rays, _pixel_rays(K, (24, 32)))  # the table IS distortion-aware
    wrapper.close()


def test_rectified_camera_gets_no_camera_model_in_vo(ros_stubs):
    wrapper = node_mod.KartMapNode(FakeNode(), one_camera_params())
    vo = wrapper._ensure_vo("camera")
    assert vo.camera_model is None  # bit-identical proven path
    wrapper.close()


# ----------------------------------------------------- mapping + save_map
class FakeSeg:
    def __init__(self, h=24, w=32):
        class_map = np.zeros((h, w), np.uint8)
        class_map[: h // 3, :] = 2  # far band: obstacle-ish class (building)
        self.class_map = class_map
        self.drivable_mask = np.zeros((h, w), bool)
        self.drivable_mask[h // 2 :, :] = True
        self.obstacle_mask = np.zeros((h, w), bool)
        self.obstacle_mask[h // 3 : h // 2, :] = True

    def segment(self, image):
        return self


def test_camera_insert_always_passes_stamp_and_rays(ros_stubs, monkeypatch):
    """The classical-fix insert contract: stamp ALWAYS given (history
    routing), rays always given (distortion-aware table)."""
    params = one_camera_params()
    fake = FakeNode()
    wrapper = node_mod.KartMapNode(fake, params)
    wrapper.segmenter = FakeSeg()
    calls = []
    original = wrapper.grid.insert_camera

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(wrapper.grid, "insert_camera", spy)
    _drive_imu(wrapper, fake, seconds=0.5)
    for k in range(6):  # odometric support so the pose is trusted
        t = T0 + k * 0.1
        fake.clock.t = t
        wrapper._on_odom(_odom_msg(t, 0.3 * k))
    fake.clock.t = T0 + 0.45
    fake.subs["/cam/image"].callback(_image_msg(T0 + 0.3))
    fake.timer(1.0 / params.output_rate_hz).callback()
    fake.timer(params.seg_period_s).callback()
    assert wrapper.counters["seg_inserted"] == 1
    assert len(calls) == 1
    assert calls[0]["stamp"] == pytest.approx(T0 + 0.3)
    assert calls[0]["rays"] is not None
    wrapper.close()


def test_save_map_writes_a_loadable_flat_package(ros_stubs, tmp_path):
    from kart_map2d.mapping.map_package import load_map_package

    params = one_camera_params(output_dir=str(tmp_path / "maps"), loop_closure=False)
    fake = FakeNode()
    wrapper = node_mod.KartMapNode(fake, params)
    wrapper.segmenter = FakeSeg()
    _drive_imu(wrapper, fake, seconds=1.0)
    for k in range(11):
        t = T0 + k * 0.1
        fake.clock.t = t
        wrapper._on_odom(_odom_msg(t, 0.5 * k))
    for stamp in (T0 + 0.3, T0 + 0.6, T0 + 0.9):
        fake.clock.t = stamp + 0.05
        fake.subs["/cam/image"].callback(_image_msg(stamp))
        fake.timer(1.0 / params.output_rate_hz).callback()
        fake.timer(params.seg_period_s).callback()
    assert wrapper.counters["seg_inserted"] >= 2
    ok, message, out_dir = wrapper.save_map("test")
    assert ok, message
    pkg = load_map_package(out_dir, require_drivable=False)
    assert pkg.method == "flat"
    assert (pkg.grid.classify() > 0).any()
    assert "_flat_" in Path(out_dir).name
    wrapper.close()


def test_save_service_is_registered_under_the_node_name(ros_stubs):
    fake = FakeNode()
    wrapper = node_mod.KartMapNode(fake, one_camera_params())
    assert wrapper.save_service_name == "~/save_map"
    assert "~/save_map" in fake.services
    wrapper.close()
