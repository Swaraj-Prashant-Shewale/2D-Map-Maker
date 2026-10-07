"""The ROS 2 drive node, driven without ROS through a duck-typed fake rclpy node.

Stub ``rclpy`` / ``*_msgs`` / ``ackermann_msgs`` / ``std_srvs`` modules are put
into ``sys.modules`` (as the mapping node's tests do), a synthetic map
package is written with the real ``save_map_package``, and the node's
callbacks are called exactly as rclpy's executor would call them.
"""

from __future__ import annotations

import json
import math
import re
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from kart_map2d.mapping.evidence_grid import EvidenceGrid
from kart_map2d.mapping.map_package import save_map_package
from kart_map2d.ros2 import drive_node as dn

ROOT = Path(__file__).resolve().parents[1]
T0 = 1.7e9  # an epoch-like ROS time (the node skips IMU before the clock runs)


# ------------------------------------------------------------------ stubs
class AutoMsg:
    """Auto-vivifying stand-in for a ROS message: nested field assignment works."""

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        child = AutoMsg()
        object.__setattr__(self, name, child)
        return child


class FakeQoS:
    def __init__(self, history=None, depth=None, reliability=None, durability=None):
        self.history, self.depth, self.reliability, self.durability = history, depth, reliability, durability


class FakeClock:
    def __init__(self) -> None:
        self.t = T0

    def now(self):
        return SimpleNamespace(nanoseconds=int(round(self.t * 1e9)))


class FakePublisher:
    def __init__(self, msg_type, topic, qos):
        self.msg_type, self.topic, self.qos, self.msgs = msg_type, topic, qos, []

    def publish(self, msg):
        self.msgs.append(msg)


class FakeNode:
    def __init__(self) -> None:
        self.clock = FakeClock()
        self.subs, self.pubs, self.timers, self.services, self.warnings = {}, {}, [], {}, []

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
        service = SimpleNamespace(srv_type=srv_type, name=name, callback=callback)
        self.services[name] = service
        return service

    def get_clock(self):
        return self.clock

    def get_logger(self):
        return SimpleNamespace(warning=self.warnings.append, info=lambda *a: None, error=self.warnings.append)


@pytest.fixture()
def ros_stubs(monkeypatch):
    rclpy = types.ModuleType("rclpy")
    qos = types.ModuleType("rclpy.qos")
    qos.QoSProfile = FakeQoS
    qos.ReliabilityPolicy = SimpleNamespace(RELIABLE="reliable", BEST_EFFORT="best_effort")
    qos.DurabilityPolicy = SimpleNamespace(VOLATILE="volatile", TRANSIENT_LOCAL="transient_local")
    qos.HistoryPolicy = SimpleNamespace(KEEP_LAST="keep_last")
    rclpy.qos = qos
    monkeypatch.setitem(sys.modules, "rclpy", rclpy)
    monkeypatch.setitem(sys.modules, "rclpy.qos", qos)
    types_ = {}
    for package, names in {
        "sensor_msgs": ("Imu", "Image", "CompressedImage", "CameraInfo", "PointCloud2"),
        "nav_msgs": ("Odometry",),
        "geometry_msgs": ("PoseStamped", "Twist"),
        "std_msgs": ("String", "Bool"),
        "ackermann_msgs": ("AckermannDriveStamped",),
    }.items():
        pkg, msg = types.ModuleType(package), types.ModuleType(package + ".msg")
        for name in names:
            types_[name] = type(name, (AutoMsg,), {})
            setattr(msg, name, types_[name])
        pkg.msg = msg
        monkeypatch.setitem(sys.modules, package, pkg)
        monkeypatch.setitem(sys.modules, package + ".msg", msg)
    std_srvs, srv = types.ModuleType("std_srvs"), types.ModuleType("std_srvs.srv")
    types_["SetBool"] = srv.SetBool = type("SetBool", (), {})
    types_["Trigger"] = srv.Trigger = type("Trigger", (), {})
    std_srvs.srv = srv
    monkeypatch.setitem(sys.modules, "std_srvs", std_srvs)
    monkeypatch.setitem(sys.modules, "std_srvs.srv", srv)
    return SimpleNamespace(types=types_)


# ------------------------------------------------------------------ a map package
@pytest.fixture(scope="module")
def package_dir(tmp_path_factory) -> Path:
    """A straight 30 m walled lane with a trajectory along its centre."""
    out = tmp_path_factory.mktemp("drive_pkg") / "lane"
    g = EvidenceGrid(cell=0.15)
    xs, ys = np.meshgrid(np.arange(-2.0, 32.0, 0.1), np.arange(-2.4, 2.4, 0.1), indexing="ij")
    floor = np.column_stack([xs.ravel(), ys.ravel(), np.full(xs.size, -0.10)])
    wx = np.arange(-2.0, 32.0, 0.05)
    walls = np.vstack([np.column_stack([wx, np.full(len(wx), 2.6), np.full(len(wx), 0.5)]), np.column_stack([wx, np.full(len(wx), -2.6), np.full(len(wx), 0.5)])])
    for _ in range(4):
        g.insert_lidar(np.vstack([floor, walls]), np.zeros(3), carve=False)
    t = np.linspace(0.0, 30.0, 301)
    xyyaw = np.column_stack([t, np.zeros_like(t), np.zeros_like(t)])
    meta = save_map_package(out, method="flat", grid=g, stamps=T0 - 100.0 + t, xyyaw=xyyaw, supported=np.ones(len(t), bool), source={"kind": "test"}, sensors=["imu", "vo_front"])
    assert meta["drivable"], meta["problems"]
    return out


def params(package_dir: Path, **kw) -> dn.DriveParams:
    base = dict(
        map_package=str(package_dir),
        cameras={"front": dn.DriveCameraParams(image_topic="/cam/image", info_topic="/cam/info", mount_xyz=(0.3, 0.0, 0.6))},
        start_pose=(2.0, 0.0, 0.0),
        start_sigma_xy_m=0.1,
        start_sigma_yaw_deg=1.0,
        odom_topic="/wheel_odom",
    )
    base.update(kw)
    return dn.DriveParams(**base)


# ------------------------------------------------------------------ messages
def _time(stamp: float) -> SimpleNamespace:
    sec = int(math.floor(stamp))
    return SimpleNamespace(sec=sec, nanosec=int(round((stamp - sec) * 1e9)))


def imu_msg(stamp: float, gz: float = 0.0, ax: float = 0.0, k: int = 0) -> SimpleNamespace:
    jitter = 1e-4 * math.sin(0.7 * k)  # never bit-identical (the frozen-sensor detector)
    return SimpleNamespace(
        header=SimpleNamespace(stamp=_time(stamp), frame_id="imu"),
        angular_velocity=SimpleNamespace(x=jitter, y=-jitter, z=gz + jitter),
        linear_acceleration=SimpleNamespace(x=ax + jitter, y=jitter, z=9.81 + jitter),
    )


def odom_msg(stamp: float, x: float) -> SimpleNamespace:
    return SimpleNamespace(
        header=SimpleNamespace(stamp=_time(stamp), frame_id="odom"),
        pose=SimpleNamespace(pose=SimpleNamespace(position=SimpleNamespace(x=x, y=0.0, z=0.0), orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0))),
    )


def run(fake: FakeNode, wrapper: dn.DriveNode, seconds: float, speed: float = 1.0, x0: float = 2.0, imu: bool = True, odom: bool = True):
    """Feed IMU (100 Hz) + odometry (20 Hz) of a kart rolling at ``speed`` and tick the 50 Hz control timer."""
    control = next(t for t in fake.timers if abs(t.period - 0.02) < 1e-9)
    n = int(round(seconds / 0.01))
    for k in range(n):
        fake.clock.t += 0.01
        now = fake.clock.t
        if imu:
            fake.subs[wrapper.params.imu_topic].callback(imu_msg(now - 0.002, k=k))
        if odom and k % 5 == 0:
            run.x = getattr(run, "x", x0)
            fake.subs[wrapper.params.odom_topic].callback(odom_msg(now - 0.005, run.x))
            run.x += speed * 0.05
        if k % 2 == 0:
            control.callback()


def last_cmd(fake: FakeNode, wrapper: dn.DriveNode):
    return fake.pubs[wrapper.params.drive_topic].msgs[-1]


# ------------------------------------------------------------------ construction
def test_node_constructs_with_only_the_configured_topics(ros_stubs, package_dir):
    fake = FakeNode()
    w = dn.DriveNode(fake, params(package_dir))
    assert set(fake.subs) == {"/imu/data", "/wheel_odom", "/cam/image", "/cam/info", "/estop"}
    assert fake.subs["/cam/image"].msg_type is ros_stubs.types["Image"]
    assert fake.subs["/estop"].qos.durability == "transient_local"  # latched e-stop
    assert set(fake.pubs) == {"/drive", "drive/diagnostics", "drive/pose"}
    assert fake.pubs["/drive"].msg_type is ros_stubs.types["AckermannDriveStamped"]
    assert sorted(t.period for t in fake.timers) == [pytest.approx(0.02), pytest.approx(0.2)]
    assert set(fake.services) == {"~/engage", "~/relocalize"}
    assert fake.services["~/engage"].srv_type is ros_stubs.types["SetBool"]
    assert w.engine.roles["vo_front"] == "odometric"
    text = "\n".join(w.describe())
    assert "disengaged" in text and "/drive" in text
    w.close()


def test_compressed_camera_and_twist_output(ros_stubs, package_dir):
    fake = FakeNode()
    p = params(package_dir, twist_topic="/cmd_vel")
    p.cameras = {"cam0": dn.DriveCameraParams(image_topic="/c/compressed", compressed=True, intrinsics=(300.0, 300.0, 160.0, 100.0, 320, 200))}
    w = dn.DriveNode(fake, p)
    assert fake.subs["/c/compressed"].msg_type is ros_stubs.types["CompressedImage"]
    assert "cam0" in w._intrinsics and w._intrinsics["cam0"][1] == (160, 100)  # image_scale 0.5
    assert "/cmd_vel" in fake.pubs
    w.close()


@pytest.mark.parametrize(
    "change, needle",
    [
        ({"map_package": ""}, "map_package is empty"),
        ({"imu_topic": ""}, "camera + IMU is the minimum"),
        ({"cameras": {}, "odom_topic": ""}, "no odometry source"),
        ({"start_pose": (), "global_relocalization": False}, "could never localize"),
        ({"cameras": {"c": dn.DriveCameraParams(image_topic="/x")}}, "give info_topic or intrinsics"),
    ],
)
def test_refused_configurations(ros_stubs, package_dir, change, needle):
    with pytest.raises(dn.DriveConfigError) as exc:
        dn.DriveNode(FakeNode(), params(package_dir, **change))
    assert any(needle in p for p in exc.value.problems)


def test_refuses_a_missing_or_undrivable_map_package(ros_stubs, package_dir, tmp_path):
    with pytest.raises(dn.DriveConfigError, match="not a map package|no map_meta"):
        dn.DriveNode(FakeNode(), params(tmp_path / "nothing_here"))
    # a copy whose racing line was edited: the sha256 check refuses it
    import shutil

    bad = tmp_path / "edited"
    shutil.copytree(package_dir, bad)
    with (bad / "racing_line.csv").open("a", encoding="ascii") as f:
        f.write("99.0,99.0,0.0,0.0,1.0\n")
    with pytest.raises(dn.DriveConfigError, match="sha256|bytes"):
        dn.DriveNode(FakeNode(), params(bad))


def test_drive_code_names_no_zed_topics_frames_or_encodings():
    """The kart's target is ONE camera of any make: nothing ZED-specific in the drive code."""
    files = sorted((ROOT / "kart_map2d" / "drive").glob("*.py")) + [ROOT / "kart_map2d" / "ros2" / "drive_node.py"]
    hits = [f"{f.name}:{i + 1}" for f in files for i, line in enumerate(f.read_text(encoding="utf-8").splitlines()) if re.search(r"\bzed", line, re.I)]
    assert hits == []
    assert dn.DriveParams().cameras == {}  # no default camera at all


def test_params_yaml_example_parses_with_no_unknown_keys():
    raw = yaml.safe_load((ROOT / "config" / "drive_params.example.yaml").read_text(encoding="utf-8"))
    assert dn.unknown_param_keys(raw) == []
    p = dn.DriveParams.from_dict(raw)
    assert p.max_speed_mps == 2.0 and p.drive_topic == "/drive" and p.control_rate_hz == 50.0
    assert set(p.cameras) == {"front"} and p.cameras["front"].image_topic == "/camera/image_raw"
    assert not p.engage_on_start and p.estop_topic == "/estop"
    assert dn.unknown_param_keys({"drive": {"max_sped_mps": 1}}) == ["'max_sped_mps' (did you mean 'max_speed_mps'?)"]


# ------------------------------------------------------------------ control loop
def test_holds_until_engaged_then_drives_and_stamps_commands(ros_stubs, package_dir):
    fake = FakeNode()
    w = dn.DriveNode(fake, params(package_dir))
    run.x = 2.0
    run(fake, w, 1.0)
    cmd = last_cmd(fake, w)
    assert cmd.drive.speed == 0.0 and "not_engaged" in w.last_decision.reasons
    resp = fake.services["~/engage"].callback(SimpleNamespace(data=True), SimpleNamespace())
    assert resp.success
    run(fake, w, 3.0)
    cmd = last_cmd(fake, w)
    assert w.last_status.localized, w.last_status
    assert 0.0 < cmd.drive.speed <= 2.0 and abs(cmd.drive.steering_angle) < 0.2
    stamp = cmd.header.stamp.sec + cmd.header.stamp.nanosec * 1e-9
    assert fake.clock.t - stamp < 0.05  # the newest sensor stamp the pose is based on
    diag = json.loads(fake.pubs["drive/diagnostics"].msgs[-1].data) if fake.pubs["drive/diagnostics"].msgs else w.diagnostics()
    assert diag["safety"]["engaged"] is True
    assert diag["timing"]["control_cycle_cpu_us"]["n"] > 50
    w.close()


def test_estop_latches_until_released_and_reengaged(ros_stubs, package_dir):
    fake = FakeNode()
    w = dn.DriveNode(fake, params(package_dir, engage_on_start=True))
    run.x = 2.0
    run(fake, w, 3.0)
    assert last_cmd(fake, w).drive.speed > 0.3
    fake.subs["/estop"].callback(SimpleNamespace(data=True))
    run(fake, w, 1.0)
    assert last_cmd(fake, w).drive.speed == 0.0 and w.safety.estop_latched
    fake.subs["/estop"].callback(SimpleNamespace(data=False))  # released: still latched
    run(fake, w, 1.0)
    assert last_cmd(fake, w).drive.speed == 0.0
    assert fake.services["~/engage"].callback(SimpleNamespace(data=True), SimpleNamespace()).success
    run(fake, w, 2.0)
    assert last_cmd(fake, w).drive.speed > 0.0
    w.close()


def test_sensor_silence_stops_the_kart(ros_stubs, package_dir):
    fake = FakeNode()
    w = dn.DriveNode(fake, params(package_dir, engage_on_start=True))
    run.x = 2.0
    run(fake, w, 3.0)
    v0 = last_cmd(fake, w).drive.speed
    assert v0 > 0.3
    run(fake, w, 0.3, imu=False, odom=False)  # all sensors silent
    assert "pose_stale" in w.last_decision.reasons
    run(fake, w, 1.5, imu=False, odom=False)
    assert last_cmd(fake, w).drive.speed == 0.0
    w.close()


def test_max_speed_zero_never_moves(ros_stubs, package_dir):
    fake = FakeNode()
    w = dn.DriveNode(fake, params(package_dir, engage_on_start=True, max_speed_mps=0.0))
    run.x = 2.0
    run(fake, w, 3.0)
    assert all(m.drive.speed == 0.0 for m in fake.pubs["/drive"].msgs)
    w.close()


def test_deadman_heartbeat(ros_stubs, package_dir):
    fake = FakeNode()
    w = dn.DriveNode(fake, params(package_dir, engage_on_start=True, heartbeat_topic="/deadman", deadman_timeout_s=0.3))
    run.x = 2.0
    run(fake, w, 2.0)
    assert last_cmd(fake, w).drive.speed == 0.0 and "deadman" in w.last_decision.reasons
    w.close()


def test_lost_kart_at_rest_restarts_the_global_search(ros_stubs, package_dir):
    """No map fix for max_fix_age_s (stale) with the kart at rest: after
    auto_relocalize_after_s the node searches the whole map again by itself -
    but only with a source that can search (here a LiDAR topic)."""
    fake = FakeNode()
    w = dn.DriveNode(fake, params(package_dir, lidar_topic="/scan", max_fix_age_s=2.0, auto_relocalize_after_s=1.0))
    run.x = 2.0
    run(fake, w, 2.5, speed=0.0, odom=False)
    assert w.last_status.state == "stale" and w.counters["global_restarts"] == 0
    run(fake, w, 1.5, speed=0.0, odom=False)
    assert w.counters["global_restarts"] == 1 and w.last_status.state == "searching"
    assert last_cmd(fake, w).drive.speed == 0.0
    w.close()
    # no LiDAR and camera fixes switched off: nothing can search, the kart just holds
    fake2 = FakeNode()
    w2 = dn.DriveNode(fake2, params(package_dir, camera_fixes="off", max_fix_age_s=2.0, auto_relocalize_after_s=1.0))
    run.x = 2.0
    run(fake2, w2, 4.0, speed=0.0, odom=False)
    assert w2.last_status.state == "stale" and w2.counters["global_restarts"] == 0
    w2.close()


def test_relocalize_service_restarts_the_search(ros_stubs, package_dir):
    fake = FakeNode()
    w = dn.DriveNode(fake, params(package_dir, lidar_topic="/scan"))
    run.x = 2.0
    run(fake, w, 1.0, speed=0.0, odom=False)
    assert w.last_status.state == "tracking"  # the trusted start pose
    resp = fake.services["~/relocalize"].callback(SimpleNamespace(), SimpleNamespace())
    assert resp.success
    run(fake, w, 0.2, speed=0.0, odom=False)
    assert w.last_status.state == "searching" and not w.last_status.localized
    assert last_cmd(fake, w).drive.speed == 0.0 and w.counters["global_restarts"] == 1
    w.close()



@pytest.mark.parametrize("change, ok", [
    ({"lidar_topic": "/scan", "start_sigma_xy_m": 2.5, "start_sigma_yaw_deg": 15.0}, False),
    ({"lidar_topic": "/scan", "start_sigma_xy_m": 1.0, "start_sigma_yaw_deg": 6.5}, True),
    ({"lidar_topic": "/scan", "start_sigma_xy_m": 0.5, "start_sigma_yaw_deg": 5.0}, True),
    ({"lidar_topic": "/scan", "start_pose": (), "start_sigma_xy_m": 5.0}, True),  # global: sigma unused
])
def test_start_pose_too_uncertain_to_verify_is_refused(ros_stubs, package_dir, change, ok):
    """3 sigma of the start pose must fit the map matcher's window, or a wrong
    local match can be declared converged (sept3: 4.4 m off)."""
    if ok:
        dn.DriveNode(FakeNode(), params(package_dir, **change)).close()
    else:
        with pytest.raises(dn.DriveConfigError) as exc:
            dn.DriveNode(FakeNode(), params(package_dir, **change))
        assert any("cannot verify a start this uncertain" in p for p in exc.value.problems)


def test_a_sensor_stamping_in_the_future_does_not_blind_the_pose_watchdog(ros_stubs, package_dir):
    fake = FakeNode()
    w = dn.DriveNode(fake, params(package_dir, engage_on_start=True))
    run.x = 2.0
    run(fake, w, 3.0)
    assert last_cmd(fake, w).drive.speed > 0.3
    # one odometry message stamped 100 s ahead (an unsynced clock), then silence
    fake.subs["/wheel_odom"].callback(odom_msg(fake.clock.t + 100.0, run.x))
    run(fake, w, 0.5, imu=False, odom=False)
    assert "pose_stale" in w.last_decision.reasons
    run(fake, w, 1.5, imu=False, odom=False)
    assert last_cmd(fake, w).drive.speed == 0.0
    w.close()


def test_global_search_worker_reset_is_applied_on_the_control_thread(ros_stubs, package_dir):
    fake = FakeNode()
    w = dn.DriveNode(fake, params(package_dir, engage_on_start=True))
    run.x = 2.0
    run(fake, w, 3.0)
    assert w.controller.progress is not None
    w._reset_request = True  # what a worker / service thread does now
    assert w.controller.progress is not None  # untouched until the control thread runs
    control = next(t for t in fake.timers if abs(t.period - 0.02) < 1e-9)
    control.callback()
    assert not w._reset_request
    w.close()


def test_obstacle_corridor_widens_with_the_cross_track_error(ros_stubs, package_dir):
    """An obstacle at the kart's own edge counts even when the kart is off the line."""
    fake = FakeNode()
    w = dn.DriveNode(fake, params(package_dir, engage_on_start=True))
    run.x = 2.0
    run(fake, w, 3.0)
    s0 = w.controller.progress
    x_ahead = float(w.controller.path.point_at(s0 + 3.0)[0])
    edge = np.array([[x_ahead, 0.95]])  # 0.95 m left of the line: outside the 0.78 m corridor
    assert len(w._corridor_along(edge, "lidar")) == 0
    w.last_output = SimpleNamespace(ok=True, cross_track=0.4)  # kart 0.4 m left of the line
    assert len(w._corridor_along(edge, "lidar")) == 1
    # ... but only on the kart's side: the mirror point right of the line is not in its way
    assert len(w._corridor_along(edge * np.array([1.0, -1.0]), "lidar")) == 0
    w.close()



@pytest.mark.parametrize("lag", [0.0, 0.5, 1.0, -0.8])
def test_obstacle_stop_does_not_depend_on_the_lidar_clock(ros_stubs, package_dir, lag):
    """Obstacles are aged on the node clock: a LiDAR stamping 0.5 s behind it
    (unsynced clock, or a scan that queued behind a slow match) used to
    expire every obstacle on arrival - the kart drove on toward a wall."""
    from kart_map2d.core.measurements import LidarScan

    fake = FakeNode()
    w = dn.DriveNode(fake, params(package_dir, engage_on_start=True, lidar_topic="/scan", lidar_match_rate_hz=0.001))
    run.x = 2.0
    run(fake, w, 3.0)
    assert last_cmd(fake, w).drive.speed > 0.3
    seen = False
    for _ in range(40):  # 10 Hz scans of an obstacle 3 m ahead
        pts = np.column_stack([np.full(20, 3.0), np.linspace(-0.4, 0.4, 20), np.full(20, 0.3)]).astype(np.float32)
        w._lidar_step(LidarScan(fake.clock.t - lag, pts, np.zeros(20, np.uint8), np.zeros(20, np.float32)))
        run(fake, w, 0.05)
        if "obstacle" in w.last_decision.reasons:
            seen = True
            break
    w.close()
    assert seen


def test_camera_mount_is_an_optical_frame_looking_forward():
    T = dn.camera_mount(dn.DriveCameraParams(image_topic="/x", mount_xyz=(0.2, 0.0, 0.5), mount_rpy_deg=(0.0, 10.0, 0.0)))
    z_optical_in_base = T[:3, :3] @ np.array([0.0, 0.0, 1.0])
    assert z_optical_in_base[0] > 0.9 and z_optical_in_base[2] < -0.1  # forward, pitched down
    assert np.allclose(T[:3, 3], [0.2, 0.0, 0.5])
