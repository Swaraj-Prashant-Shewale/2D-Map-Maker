"""Map-relative localization: map fixes through the EKF gate, scan / patch matching,
global relocalization (incl. refusing an ambiguous map), recovery, the camera cue.

Synthetic worlds only: an L-shaped walled yard rasterized into an
EvidenceGrid exactly as the pipeline's LiDAR insert does, a 2D LiDAR
ray-cast by kart_map2d.drive.sim, and synthetic odometry / IMU streams.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from kart_map2d.core.health import Status
from kart_map2d.core.measurements import ImuSample, RelativeMotion
from kart_map2d.drive.localization import (
    CameraGroundCue,
    CameraObservation,
    DriveFusionEngine,
    LocalizerConfig,
    MapLocalizer,
    MapModel,
    MapPoseFix,
    PatchMatcher,
    ScanMatcher,
    StaticBodyMask,
    extrapolate_pose,
    global_search_lidar,
    predict_state,
    se2_compose,
)
from kart_map2d.drive.sim import KartSim, planar_scan
from kart_map2d.mapping.evidence_grid import EvidenceGrid


def _grid(walls: list[np.ndarray], floor_box=(-9.5, -5.5, 19.5, 7.5)) -> EvidenceGrid:
    g = EvidenceGrid(cell=0.15)
    obs = np.vstack(walls)
    x0, y0, x1, y1 = floor_box
    # Dense floor (every 0.15 m cell sees ground, as a real LiDAR grid does).
    xs, ys = np.meshgrid(np.arange(x0, x1, 0.1), np.arange(y0, y1, 0.1), indexing="ij")
    floor = np.column_stack([xs.ravel(), ys.ravel()])
    for _ in range(4):
        g.insert_lidar(np.column_stack([obs, np.full(len(obs), 0.5)]), np.zeros(3), carve=False)
        g.insert_lidar(np.column_stack([floor, np.full(len(floor), -0.10)]), np.zeros(3), carve=False)
    return g


def _line(a, b, step=0.05) -> np.ndarray:
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = max(int(np.linalg.norm(b - a) / step), 2)
    return a + np.linspace(0, 1, n)[:, None] * (b - a)


@pytest.fixture(scope="module")
def yard() -> MapModel:
    """Asymmetric walled yard: outer walls with a notch, pillars, a planter, wall stubs."""
    walls = [
        _line((-10, 8), (20, 8)),
        _line((-10, -6), (6, -6)),
        _line((6, -6), (6, -4)),
        _line((6, -4), (12, -4)),
        _line((12, -4), (12, -6)),
        _line((12, -6), (20, -6)),
        _line((-10, -6), (-10, 8)),
        _line((20, -6), (20, 8)),
        _line((14, -1), (17, -1)),
        _line((14, -2), (17, -2)),
        _line((-4, 8), (-4, 4)),
        _line((-7, -6), (-7, -3)),
        np.column_stack([5 + 0.6 * np.cos(np.linspace(0, 2 * np.pi, 60)), 2 + 0.6 * np.sin(np.linspace(0, 2 * np.pi, 60))]),
        np.column_stack([-2 + 0.3 * np.cos(np.linspace(0, 2 * np.pi, 30)), -3 + 0.3 * np.sin(np.linspace(0, 2 * np.pi, 30))]),
    ]
    return MapModel(_grid(walls))


@pytest.fixture(scope="module")
def corridor() -> MapModel:
    """Two long parallel walls and nothing else: laterally tight, along-track blind."""
    walls = [_line((-30, 2.0), (30, 2.0)), _line((-30, -2.0), (30, -2.0))]
    return MapModel(_grid(walls, floor_box=(-29, -1.8, 29, 1.8)))


@pytest.fixture(scope="module")
def box_room() -> MapModel:
    """A plain rectangle: 180-degree symmetric (two equally good global answers)."""
    walls = [_line((-8, 5), (8, 5)), _line((-8, -5), (8, -5)), _line((-8, -5), (-8, 5)), _line((8, -5), (8, 5))]
    return MapModel(_grid(walls, floor_box=(-7.5, -4.5, 7.5, 4.5)))


def scan_at(model: MapModel, pose, seed=0, boxes=None) -> np.ndarray:
    return planar_scan(np.asarray(pose, float), model.obstacle_xy, model.cell, rng=np.random.default_rng(seed), extra_boxes=boxes)


# ------------------------------------------------------------------ EKF
def test_map_pose_fix_is_a_joint_gated_update_reported_to_health():
    eng = DriveFusionEngine()
    rng = np.random.default_rng(0)  # constant samples would trip the frozen-sensor detector
    for k in range(50):
        eng.push(ImuSample(0.01 * k, "imu", rng.normal(0, 1e-3, 3), np.array([0.0, 0.0, 9.81]) + rng.normal(0, 1e-2, 3)), 0.01 * k)
    eng.flush()
    eng.set_pose(np.array([1.0, 1.0, 0.1]), np.diag([1.0, 1.0, 0.05]))
    t = float(eng.ekf.stamp)
    fix = MapPoseFix(t, "map_lidar", np.array([1.5, 0.8]), 1.0, "map", 0.05, tuple(np.diag([0.01, 0.01, 1e-4]).reshape(-1)), 0.9)
    eng.push(fix, t + eng.reorder_window_s)
    x = eng.ekf.x
    assert x[0] == pytest.approx(1.5, abs=0.02) and x[1] == pytest.approx(0.8, abs=0.02) and x[2] == pytest.approx(0.05, abs=0.01)
    h = eng.health.channel("map_lidar")
    assert h.accepted == 1 and h.status == Status.OK
    assert eng.mode().startswith("absolute")
    # A confident fix 5 m away is gated out and reported as a rejection.
    bad = MapPoseFix(t, "map_lidar", np.array([6.5, 0.8]), 1.0, "map", 0.05, tuple(np.diag([0.01, 0.01, 1e-4]).reshape(-1)), 0.9)
    eng.push(bad, t + eng.reorder_window_s)
    assert eng.ekf.x[0] == pytest.approx(1.5, abs=0.05) and h.rejected == 1


def test_generic_camera_names_are_odometric_channels():
    eng = DriveFusionEngine()
    ch = eng.register_camera("front")
    assert ch == "vo_front" and eng.roles[ch] == "odometric"
    rng = np.random.default_rng(1)
    for k in range(60):
        t = 0.1 * k
        eng.push(ImuSample(t, "imu", rng.normal(0, 1e-3, 3), np.array([0.0, 0.0, 9.81]) + rng.normal(0, 1e-2, 3)), t)
        if k and k % 3 == 0:
            eng.push(RelativeMotion(t, ch, t - 0.3, 0.3, 0.0, 0.0, sigma_translation=0.1), t)
    assert "odometric" in eng.mode()


def test_set_pose_moves_only_recent_history_unless_global():
    eng = DriveFusionEngine()
    rng = np.random.default_rng(2)
    for k in range(1000):
        t = 0.02 * k
        eng.push(ImuSample(t, "imu", rng.normal(0, 1e-3, 3), np.array([0.0, 0.0, 9.81]) + rng.normal(0, 1e-2, 3)), t)
    old = np.array(eng.trajectory.x[0][:3])
    eng.set_pose(np.array([5.0, 0.0, 0.0]), np.eye(3) * 0.01, history_s=2.0)
    assert np.allclose(eng.trajectory.x[0][:3], old)  # 20 s ago: untouched
    assert eng.trajectory.x[-1][0] == pytest.approx(5.0, abs=1e-6)
    eng.set_pose(np.array([7.0, 0.0, 0.0]), np.eye(3) * 0.01, history_s=None)
    assert eng.trajectory.x[0][0] == pytest.approx(old[0] + 2.0, abs=1e-6)


def test_extrapolation_uses_the_measured_gyro():
    x = np.zeros(8)
    x[3] = 2.0  # m/s, yaw rate state 0
    t = np.linspace(0.001, 0.1, 40)
    pose = extrapolate_pose(x, 0.0, 0.1, t, np.full(40, 1.0))
    assert pose[2] == pytest.approx(0.1, abs=5e-3)  # 1 rad/s measured for 0.1 s
    assert pose[0] == pytest.approx(0.2, abs=5e-3)


def _extrapolate_reference(x, t0, t1, gyro_t, gyro_z, bias):
    """The vectorized formulation (trapezoidal rate, mid-interval heading)."""
    sel = (gyro_t > t0) & (gyro_t <= t1)
    ts = np.r_[t0, gyro_t[sel], t1]
    rates = np.r_[x[4], gyro_z[sel] - bias]
    rates = np.r_[rates, rates[-1]]
    h = np.maximum(np.diff(ts), 0.0)
    dyaw = 0.5 * (rates[:-1] + rates[1:]) * h
    ym = x[2] + np.r_[0.0, np.cumsum(dyaw)[:-1]] + 0.5 * dyaw
    px = x[0] + np.sum(x[3] * h * np.cos(ym))
    py = x[1] + np.sum(x[3] * h * np.sin(ym))
    return np.array([px, py, (x[2] + np.sum(dyaw) + np.pi) % (2 * np.pi) - np.pi])


def test_prediction_follows_a_kinematic_bicycle_through_a_hairpin():
    """base_link (0.489 m ahead of the rear axle) slides sideways in turns:
    the drive EKF's prediction and the control-loop extrapolation follow the
    kart simulator's rear-axle bicycle; the mapping model (no slide) does not."""
    sim = KartSim(0.0, 0.0, 0.0, v=2.0)
    for _ in range(50):
        sim.step(0.45, 2.0, 0.01)  # settle into the turn (steering lag)
    x = np.zeros(8)
    x[:5] = [sim.x, sim.y, sim.yaw, sim.v, sim.yaw_rate]
    t, z = [], []
    for k in range(40):  # 0.4 s of 100 Hz gyro
        sim.step(0.45, 2.0, 0.01)
        t.append(0.01 * (k + 1))
        z.append(sim.yaw_rate)
    truth = sim.pose
    assert abs(sim.yaw_rate) > 0.8  # a hairpin-like turn at 2 m/s
    kart = extrapolate_pose(x, 0.0, 0.4, t, z)
    naive = extrapolate_pose(x, 0.0, 0.4, t, z, base_ahead_m=0.0)
    assert np.hypot(*(kart[:2] - truth[:2])) < 0.01
    assert np.hypot(*(naive[:2] - truth[:2])) > 0.1
    const = predict_state(x, 0.4)
    assert np.hypot(*(const[:2] - truth[:2])) < 0.01


def test_extrapolation_matches_the_vectorized_reference_for_arrays_and_lists():
    rng = np.random.default_rng(0)
    x = np.zeros(8)
    x[:5] = [1.0, -2.0, 3.0, 1.7, 0.4]
    t = np.sort(rng.uniform(-0.05, 0.2, 60))
    z = rng.normal(0.3, 0.5, 60)
    for t1 in (0.03, 0.12, 0.25):
        ref = _extrapolate_reference(x, 0.0, t1, t, z, 0.01)
        for gt, gz in ((t, z), (list(t), list(z))):
            got = extrapolate_pose(x, 0.0, t1, gt, gz, 0.01, base_ahead_m=0.0)
            assert np.allclose(got[:2], ref[:2], atol=1e-12)
            assert abs(math.remainder(got[2] - ref[2], 2 * math.pi)) < 1e-12


# ------------------------------------------------------------------ LiDAR
@pytest.mark.parametrize("offset", [(0.1, -0.1, 1.0), (1.5, 1.0, 8.0), (-2.0, 1.5, -12.0)])
def test_scan_matcher_recovers_the_pose(yard, offset):
    m = ScanMatcher(yard)
    true = np.array([2.0, 1.0, math.radians(20)])
    P = m.prepare(scan_at(yard, true), already_2d=True)
    prior = true + np.array([offset[0], offset[1], math.radians(offset[2])])
    cov = np.diag([max(abs(offset[0]), 0.2) ** 2, max(abs(offset[1]), 0.2) ** 2, math.radians(max(abs(offset[2]), 2)) ** 2])
    res = m.match(P, prior, cov)
    assert res is not None and not res.info.get("rejected")
    assert np.hypot(*(res.pose[:2] - true[:2])) < 0.08 and abs(res.pose[2] - true[2]) < math.radians(0.5)
    assert res.score > 0.9 and res.info["ms"] < 1000


def test_corridor_match_is_confident_only_across_the_corridor(corridor):
    m = ScanMatcher(corridor)
    true = np.array([0.0, 0.3, 0.0])
    P = m.prepare(scan_at(corridor, true, seed=2), already_2d=True)
    res = m.match(P, true + np.array([0.0, 0.1, 0.0]), np.diag([0.04, 0.04, 1e-3]))
    assert res is not None
    sx, sy = math.sqrt(res.cov[0, 0]), math.sqrt(res.cov[1, 1])
    assert sy < 0.2 and sx > 3 * sy  # along-track (x) is the weak direction
    assert abs(res.pose[1] - true[1]) < 0.08


def test_global_lidar_search_finds_the_pose_in_an_asymmetric_map(yard):
    m = ScanMatcher(yard)
    true = np.array([-3.0, 1.5, math.radians(-70)])
    P = m.prepare(scan_at(yard, true, seed=4), already_2d=True)
    res = global_search_lidar(yard, m, P)
    assert res.ok, res.reason
    assert np.hypot(*(res.pose[:2] - true[:2])) < 0.15 and abs(math.remainder(res.pose[2] - true[2], 2 * math.pi)) < math.radians(1.0)


def test_global_lidar_search_refuses_a_symmetric_map(box_room):
    m = ScanMatcher(box_room)
    true = np.array([2.0, 1.0, math.radians(10)])
    P = m.prepare(scan_at(box_room, true, seed=5), already_2d=True)
    res = global_search_lidar(box_room, m, P)
    assert not res.ok and "ambiguous" in res.reason  # never guesses between the two rotations


# ------------------------------------------------------------------ localizer
def _drive(loc: MapLocalizer, model: MapModel, sim: KartSim, seconds: float, steer: float = 0.05, speed: float = 1.5, kidnap_at: float | None = None):
    """Kinematic kart + synthetic IMU / odometry + scans through the localizer."""
    rng = np.random.default_rng(11)
    dt = 0.01
    last_odo = (sim.t, sim.pose.copy())
    errors = []
    for k in range(int(seconds / dt)):
        sim.step(steer, speed, dt)
        t = 100.0 + sim.t
        if kidnap_at is not None and abs(sim.t - kidnap_at) < dt / 2:
            sim.y += 1.2  # someone moves the kart sideways; odometry never sees it
            last_odo = (sim.t, sim.pose.copy())
        g, a = sim.imu(rng)
        loc.push(ImuSample(t, "imu", g, a), t)
        if k % 10 == 0:
            t0p, p0 = last_odo
            c, s = math.cos(p0[2]), math.sin(p0[2])
            d = sim.pose[:2] - p0[:2]
            fwd, left = c * d[0] + s * d[1], -s * d[0] + c * d[1]
            dyaw = math.remainder(sim.yaw - p0[2], 2 * math.pi)
            if sim.t - t0p > 0.05:
                loc.push(RelativeMotion(t, "odom", 100.0 + t0p, fwd * 1.01, left, dyaw, sigma_translation=0.05, sigma_yaw=0.01), t)
            last_odo = (sim.t, sim.pose.copy())
        if k % 20 == 0 and loc.engine.ekf.stamp is not None:
            loc.engine.advance(t)
            pts = scan_at(model, sim.pose, seed=k)
            loc.process_scan(np.column_stack([pts, np.full(len(pts), 0.5)]), t)
        if loc.engine.ekf.stamp is not None and loc.initialized:
            est = loc.engine.ekf.x[:3]
            errors.append((sim.t, float(np.hypot(*(est[:2] - sim.pose[:2]))), loc.status(t).localized))
    return np.array(errors)


def test_wrong_start_pose_converges_with_lidar_fixes(yard):
    sim = KartSim(-5.0, -2.0, 0.3)
    start = se2_compose(sim.pose, np.array([1.2, -1.6, math.radians(10)]))
    loc = MapLocalizer(yard, LocalizerConfig(start_pose=tuple(start), start_sigma_xy_m=2.5, start_sigma_yaw_rad=math.radians(15)))
    err = _drive(loc, yard, sim, 8.0)
    assert err[0, 1] > 1.5  # really started 2 m off
    localized = err[err[:, 2] > 0]
    assert len(localized) and localized[0, 0] < 3.0  # declared within 3 s
    assert localized[:, 1].max() < 0.3  # and never wrong while declared localized


def test_global_relocalization_then_tracking(yard):
    sim = KartSim(-5.0, -2.0, 0.3)
    loc = MapLocalizer(yard, LocalizerConfig(start_pose=None, global_relocalization=True))
    err = _drive(loc, yard, sim, 6.0)
    assert loc.initialized and any(a["ok"] for a in loc.global_attempts)
    localized = err[err[:, 2] > 0]
    assert len(localized) and localized[:, 1].max() < 0.3


def test_kidnapped_kart_is_refused_then_reanchored(yard):
    sim = KartSim(-5.0, -2.0, 0.3)
    loc = MapLocalizer(yard, LocalizerConfig(start_pose=tuple(sim.pose), start_sigma_xy_m=0.1, start_sigma_yaw_rad=math.radians(1)))
    err = _drive(loc, yard, sim, 10.0, kidnap_at=4.0)
    after = err[err[:, 0] > 4.0]
    # right after the kidnap the localizer must stop claiming it is localized ...
    assert (~after[:40, 2].astype(bool)).any() or loc.reanchors >= 1
    # ... and it recovers by re-anchoring on the consistent map fixes.
    assert loc.reanchors >= 1
    assert err[-1, 1] < 0.3 and err[-1, 2]



@pytest.mark.parametrize("offset", [(0.0, 2.0, 0.0), (0.0, 1.2, 0.0)])
def test_scans_that_do_not_fit_at_a_wrong_trusted_start_stop_tracking_fast(yard, offset):
    """A TRUSTED start pose that is wrong (operator error, kidnap before start):
    the scans fit the map only far outside the tight match window (or hardly
    at all). The matcher's refusals must count - the localizer goes
    "rejected" within a second instead of "tracking" on odometry until
    max_fix_age_s (8 s)."""
    sim = KartSim(-5.0, -2.0, 0.3)
    wrong = se2_compose(sim.pose, np.array([*offset[:2], math.radians(offset[2])]))
    loc = MapLocalizer(yard, LocalizerConfig(start_pose=tuple(wrong), start_sigma_xy_m=0.1, start_sigma_yaw_rad=math.radians(1)))
    err = _drive(loc, yard, sim, 2.0)
    assert err[0, 1] > 1.0 and err[0, 2]  # really wrong, and trusted at first
    unloc = err[~err[:, 2].astype(bool)]
    assert len(unloc) and unloc[0, 0] < 1.2, "still claiming to be localized with a wrong pose"
    assert loc.consecutive_rejects >= 3 or loc.reanchors >= 1


def test_mismatch_matches_are_refused_not_applied(yard):
    m = ScanMatcher(yard)
    true = np.array([2.0, 1.0, math.radians(20)])
    P = m.prepare(scan_at(yard, true), already_2d=True)
    # prior 1.5 m off with a tight covariance: the fit lies outside the window
    res = m.match(P, true + np.array([0.0, 1.5, 0.0]), np.diag([0.05**2, 0.05**2, math.radians(0.5) ** 2]))
    assert res is not None and res.info.get("rejected")  # low inliers or a fit outside the window
    loc = MapLocalizer(yard, LocalizerConfig(start_pose=tuple(true), start_sigma_xy_m=0.1, start_sigma_yaw_rad=math.radians(1)))
    for k in range(40):
        t = 100.0 + 0.01 * k
        loc.push(ImuSample(t, "imu", np.array([0.0, 0.0, 1e-4 * k]), np.array([0.0, 0.0, 9.81])), t)
    loc.engine.advance(101.0)
    loc.push(ImuSample(100.5, "imu", np.zeros(3), np.array([0.0, 0.0, 9.81])), 101.0)
    assert loc.initialized and loc.engine.ekf.stamp is not None
    before = loc.engine.ekf.x[:3].copy()
    from dataclasses import replace

    out = loc.submit(replace(res, stamp=float(loc.engine.ekf.stamp)))
    assert out.accepted is False and loc.consecutive_rejects == 1
    assert np.allclose(loc.engine.ekf.x[:3], before)  # never applied to the filter


# ------------------------------------------------------------------ camera
def test_patch_matcher_recovers_an_offset_from_camera_like_evidence(yard):
    """Camera-like evidence: obstacle contacts + drivable ground in a forward cone."""
    pm = PatchMatcher(yard)
    true = np.array([0.0, 0.0, math.radians(15)])
    rng = np.random.default_rng(1)
    obs = []
    for k in range(4):
        pose_k = se2_compose(true, np.array([-0.5 * (3 - k), 0.0, 0.0]))
        pts_w = yard.obstacle_xy
        c, s = math.cos(pose_k[2]), math.sin(pose_k[2])
        rel = (pts_w - pose_k[:2]) @ np.array([[c, s], [-s, c]]).T
        cone = (rel[:, 0] > 1.0) & (rel[:, 0] < 11.0) & (np.abs(rel[:, 1]) < rel[:, 0])
        drv = np.array([(x, y) for x in np.arange(1.5, 9.0, 0.2) for y in np.arange(-3.0, 3.0, 0.2) if abs(y) < x])
        drv_w = drv @ np.array([[c, -s], [s, c]]).T + pose_k[:2]
        drv_ok = yard.class_at(drv_w) == 1
        obs.append(CameraObservation(100.0 + k, drv[drv_ok], rel[cone] + rng.normal(0, 0.03, (cone.sum(), 2))))
    poses = np.array([se2_compose(true, np.array([-0.5 * (3 - k), 0.0, 0.0])) for k in range(4)])
    live = pm.live_patch(obs, poses, true, 103.0)
    prior = true + np.array([0.6, -0.4, math.radians(4)])
    m = pm.match(live, prior, np.diag([0.5**2, 0.5**2, math.radians(5) ** 2]))
    assert m is not None
    assert np.hypot(*(m.pose[:2] - true[:2])) < 0.35 and abs(m.pose[2] - true[2]) < math.radians(2.5)


def test_static_body_mask_learns_the_kart_from_motion():
    rng = np.random.default_rng(0)
    h, w = 100, 160
    mask = StaticBodyMask((h, w), min_frames=8)
    body = np.zeros((h, w), bool)
    body[70:90, 60:100] = True  # the kart's nose, above the classifier's own bottom band
    anchor = np.zeros((h, w), bool)
    anchor[90:, :] = True
    for k in range(12):
        gray = rng.integers(0, 255, (h, w)).astype(np.uint8)  # the world streams past
        gray[body] = 40  # the body does not change
        obst = body.copy()
        obst[30:40, :] = True  # the horizon band: obstacle every frame, but not touching the bottom
        mask.update(gray, obst, anchor)
    assert mask.mask is not None
    assert mask.mask[body].mean() > 0.95 and not mask.mask[30:40, :].any()


def test_camera_cue_projects_ground_and_ignores_near_body_contacts():
    from kart_map2d.core.transforms import make_transform, rpy_matrix

    K = np.array([[300.0, 0, 160.0], [0, 300.0, 100.0], [0, 0, 1]])
    T = make_transform(rpy_matrix(0.0, math.radians(8.0), 0.0) @ rpy_matrix(-np.pi / 2, 0.0, -np.pi / 2), np.array([0.0, 0.0, 0.46]))
    cue = CameraGroundCue(K, T, (320, 200), learn_body_mask=False, apply_kart_mask=False)
    img = np.full((200, 320, 3), (150, 155, 160), np.uint8)
    img[:95] = (235, 190, 120)  # sky
    rng = np.random.default_rng(3)
    img[95:] = np.clip(img[95:] + rng.normal(0, 2, img[95:].shape), 0, 255).astype(np.uint8)
    drv, obs, q = cue.observe(img)
    assert len(drv) > 50
    assert (drv[:, 0] > 0).all() and np.hypot(drv[:, 0], drv[:, 1]).max() <= 12.0 + 0.2
    assert len(obs) == 0 or np.hypot(obs[:, 0], obs[:, 1]).min() >= 2.0 - 0.2
