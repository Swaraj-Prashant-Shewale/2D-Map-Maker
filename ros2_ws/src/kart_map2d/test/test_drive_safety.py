"""Safety supervisor: every trigger stops the kart within its bound; e-stop latches.

Unit tests drive SafetySupervisor directly at a 50 Hz control rate; the
fault-injection tests close the loop through the controller and the
kinematic kart simulator. All synthetic.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from kart_map2d.drive.controller import ControllerConfig, PurePursuitController, RacingPath
from kart_map2d.drive.safety import (
    DEGRADED_MODES,
    ObstacleMemory,
    SafetyConfig,
    SafetyInputs,
    SafetySupervisor,
    corridor_arclength,
    obstacles_on_path,
)
from kart_map2d.drive.sim import KartSim, planar_scan

DT = 0.02
OK_MODE = "absolute+odometric+inertial"


def engaged(cfg: SafetyConfig | None = None) -> SafetySupervisor:
    s = SafetySupervisor(cfg or SafetyConfig())
    ok, _ = s.engage(True, 0.0)
    assert ok
    return s


def inputs(now, speed, **kw) -> SafetyInputs:
    base = dict(now=now, sensor_stamp=now, fusion_mode=OK_MODE, localized=True, on_path=True, consecutive_rejects=0, position_sigma_m=0.1, speed=speed)
    base.update(kw)
    return SafetyInputs(**base)


def drive_until_stop(sup: SafetySupervisor, fault_at: float, make, v0: float = 2.0, seconds: float = 5.0):
    """Run 50 Hz cycles at v0 until a fault (``make(now) -> SafetyInputs``); return
    (detection time after the fault, time the command reached 0 after it)."""
    speed = v0
    detected = zero = None
    for k in range(int(seconds / DT)):
        now = k * DT
        dec = sup.update(make(now, speed))
        speed = dec.limit(v0)
        if now >= fault_at - 1e-9:
            if dec.stopping and detected is None:
                detected = now - fault_at
            if detected is not None and zero is None and speed <= 1e-12:
                zero = now - fault_at
    return detected, zero


# ------------------------------------------------------------------ triggers
def test_all_clear_allows_the_max_speed_cap_only():
    sup = engaged(SafetyConfig(max_speed_mps=2.0))
    dec = sup.update(inputs(0.0, 0.0))
    assert not dec.stopping and dec.state == "driving" and dec.limit(5.0) == 2.0 and dec.limit(1.2) == 1.2


def test_not_engaged_holds_the_kart():
    sup = SafetySupervisor()
    dec = sup.update(inputs(0.0, 0.0))
    assert dec.stopping and "not_engaged" in dec.reasons and dec.limit(2.0) == 0.0 and dec.state == "holding"


def test_max_speed_zero_never_moves():
    sup = engaged(SafetyConfig(max_speed_mps=0.0))
    for k in range(50):
        assert sup.update(inputs(k * DT, 0.0)).limit(2.0) == 0.0


def test_estop_latches_and_stops_at_estop_decel():
    cfg = SafetyConfig(estop_decel_mps2=4.0)
    sup = engaged(cfg)

    def make(now, speed):
        if abs(now - 1.0) < 1e-9:
            sup.on_estop(True, now)
        if abs(now - 1.1) < 1e-9:
            sup.on_estop(False, now)  # released: must stay latched
        return inputs(now, speed)

    detected, zero = drive_until_stop(sup, 1.0, make)
    assert detected == pytest.approx(0.0)  # the very cycle it arrives
    assert zero <= 2.0 / 4.0 + DT + 1e-9
    assert sup.estop_latched and not sup.engaged
    # Still latched after release: nothing moves until engage().
    for k in range(10):
        assert sup.update(inputs(10.0 + k * DT, 0.0)).limit(2.0) == 0.0
    assert sup.engage(True, 11.0)[0]
    for k in range(60):
        dec = sup.update(inputs(11.0 + k * DT, 0.0))
    assert dec.limit(2.0) == 2.0


def test_engage_refused_while_estop_pressed():
    sup = engaged()
    sup.on_estop(True, 0.0)
    ok, why = sup.engage(True, 0.1)
    assert not ok and "released" in why or "release" in why


@pytest.mark.parametrize("timeout", [0.2])
def test_pose_stream_stops_within_its_timeout(timeout):
    sup = engaged(SafetyConfig(pose_timeout_s=timeout, stop_decel_mps2=2.0))
    last = {"t": 0.0}

    def make(now, speed):
        if now < 1.0:
            last["t"] = now
        return inputs(now, speed, sensor_stamp=last["t"])

    detected, zero = drive_until_stop(sup, 1.0, make)
    assert detected <= timeout + DT + 1e-9
    assert zero <= timeout + 2.0 / 2.0 + 2 * DT + 1e-9


@pytest.mark.parametrize("mode", DEGRADED_MODES)
def test_degraded_fusion_mode_stops_after_its_timeout(mode):
    sup = engaged(SafetyConfig(degraded_mode_timeout_s=0.5))
    detected, zero = drive_until_stop(sup, 1.0, lambda now, speed: inputs(now, speed, fusion_mode=mode if now >= 1.0 else OK_MODE))
    assert 0.5 - 1e-9 <= detected <= 0.5 + DT + 1e-9
    assert zero <= 0.5 + 1.0 + 2 * DT + 1e-9


def test_localization_rejections_and_uncertainty_stop():
    sup = engaged(SafetyConfig(max_consecutive_rejects=3))
    detected, _ = drive_until_stop(sup, 1.0, lambda now, speed: inputs(now, speed, consecutive_rejects=3 if now >= 1.0 else 2))
    assert detected == pytest.approx(0.0)
    sup = engaged(SafetyConfig(max_position_sigma_m=1.0))
    detected, _ = drive_until_stop(sup, 1.0, lambda now, speed: inputs(now, speed, position_sigma_m=1.5 if now >= 1.0 else 0.2))
    assert detected == pytest.approx(0.0)
    sup = engaged()
    detected, _ = drive_until_stop(sup, 1.0, lambda now, speed: inputs(now, speed, localized=now < 1.0))
    assert detected == pytest.approx(0.0) and "not_localized" in sup.active


def test_deadman():
    sup = engaged(SafetyConfig(deadman_timeout_s=0.3))
    assert "deadman" in sup.update(inputs(0.0, 0.0)).reasons  # never seen a heartbeat

    def make(now, speed):
        if now < 1.0:
            sup.on_heartbeat(now)
        return inputs(now, speed)

    sup = engaged(SafetyConfig(deadman_timeout_s=0.3, resume_hold_s=0.0))
    for k in range(20):  # clear the startup trigger
        sup.on_heartbeat(-1.0 + k * DT)
        sup.update(inputs(-1.0 + k * DT, 0.0))
    detected, zero = drive_until_stop(sup, 1.0, make)
    assert detected <= 0.3 + DT + 1e-9 and zero is not None


def test_resume_needs_the_condition_clear_for_the_hold_time():
    sup = engaged(SafetyConfig(resume_hold_s=1.0))
    sup.update(inputs(0.0, 0.0, sensor_stamp=-5.0))  # stale
    for k in range(1, 40):  # data back, but only 0.78 s
        assert sup.update(inputs(k * DT, 0.0)).stopping
    for k in range(40, 60):
        dec = sup.update(inputs(k * DT, 0.0))
    assert not dec.stopping


def test_controller_cannot_override_a_stop_in_the_same_cycle():
    sup = engaged()
    sup.on_estop(True, 0.0)
    dec = sup.update(inputs(0.0, 2.0))
    assert dec.limit(15.0) == pytest.approx(2.0 * 0.0 + dec.speed_cap) and dec.limit(15.0) <= 2.0
    dec = sup.update(inputs(0.02, dec.limit(15.0)))
    assert dec.limit(15.0) < 2.0  # ramps down whatever the controller asks


# ------------------------------------------------------------------ obstacles
def test_obstacle_geometry_uses_the_kart_width():
    path = np.column_stack([np.linspace(0, 10, 41), np.zeros(41)])
    half = 0.53 + 0.25
    for lateral, hit in ((0.0, True), (0.5, True), (0.75, True), (0.85, False), (1.5, False)):
        pts = np.array([[4.0, lateral], [4.0, lateral + 0.01], [4.05, lateral]])
        d, n = obstacles_on_path(path, pts, half, 6.0)
        assert (d is not None) == hit, lateral
    d, n = obstacles_on_path(path, np.array([[7.0, 0.0]] * 3), half, 6.0)
    assert d is None  # beyond the check distance
    d, n = obstacles_on_path(path, np.array([[4.0, 0.0], [2.5, 0.1], [9.0, 0.0]]), half, 6.0)
    assert d == pytest.approx(2.5)
    d, n = obstacles_on_path(path, np.array([[-0.5, 0.0], [-0.2, 0.3], [-1.0, 0.1]]), half, 6.0)
    assert d is None  # behind the kart


def test_obstacle_memory_ages_out_per_source():
    mem = ObstacleMemory({"lidar": 0.3, "camera": 1.5})
    mem.add(np.array([[1.0, 0.0]]), 0.0, "lidar")
    mem.add(np.array([[2.0, 0.0]]), 0.0, "camera")
    assert len(mem.points(0.2)) == 2
    assert len(mem.points(1.0)) == 1
    assert len(mem.points(2.0)) == 0


def ring_path(radius=10.0) -> RacingPath:
    th = np.linspace(0.0, 2 * np.pi, 126, endpoint=False)
    return RacingPath(np.column_stack([radius * np.cos(th), radius * np.sin(th)]), np.full(126, 5.0), closed=True)


def test_closed_loop_obstacle_on_the_line_stops_short_of_it():
    """A box appears on the racing line 5 m ahead of a kart doing 2 m/s."""
    path = ring_path()
    ctrl = PurePursuitController(path, ControllerConfig(max_speed_mps=2.0))
    sup = engaged(SafetyConfig())
    mem = ObstacleMemory()
    sim = KartSim(*path.point_at(0.0), path.heading_at(0.0))
    speed, box = 0.0, None
    for k in range(int(12 / DT)):
        now = k * DT
        if box is None and now >= 5.0:
            c = path.point_at((ctrl.progress or 0.0) + 5.0)
            box = (c[0] - 0.3, c[1] - 0.3, c[0] + 0.3, c[1] + 0.3)
        if box is not None:
            pts = planar_scan(sim.pose, np.zeros((0, 2)), 0.15, extra_boxes=[box], rays=360)
            if len(pts):
                c_, s_ = math.cos(sim.yaw), math.sin(sim.yaw)
                mem.add(pts @ np.array([[c_, -s_], [s_, c_]]).T + sim.pose[:2], now, "lidar")
        out = ctrl.step(sim.pose, sim.v, DT)
        reach = 0.8 + 0.5 + speed * 0.3 + speed * speed / 4.0 + 1.0
        dec = sup.update(inputs(now, speed, path_ahead=ctrl.path_ahead(sim.pose, reach), obstacles=mem.points(now)))
        speed = dec.limit(out.speed)
        ctrl.commanded(speed)
        sim.step(out.steer, speed, DT)
    assert "obstacle" in sup.active and sim.v < 0.05
    cx, cy = 0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])
    bumper = sim.pose[:2] + 0.8 * np.array([math.cos(sim.yaw), math.sin(sim.yaw)])
    assert np.hypot(bumper[0] - cx, bumper[1] - cy) - 0.3 > 0.2  # stopped short, with room


def test_arrival_projection_matches_the_per_cycle_check():
    """ObstacleMemory.ahead over arclengths projected once on arrival gives the
    same nearest distance / count as obstacles_on_path on the path ahead."""
    path = ring_path()
    rng = np.random.default_rng(3)
    half = 0.53 + 0.25
    for trial in range(20):
        s_arr = float(rng.uniform(0.0, 3 * path.length))
        s_now = s_arr + float(rng.uniform(0.0, 0.6))  # the kart moved on since the arrival
        ahead_pts = path.points_between(s_now, s_now + 4.0, 0.25)
        pts = ahead_pts[rng.integers(0, len(ahead_pts), 6)] + rng.normal(0.0, 0.5, (6, 2))
        mem = ObstacleMemory()
        window = path.points_between(s_arr - 0.5, s_arr + 8.0, 0.25)
        mem.add(pts, 0.0, "lidar", along=corridor_arclength(window, pts, half) + (s_arr - 0.5))
        d_mem, n_mem = mem.ahead(0.1, s_now, 3.0)
        d_ref, n_ref = obstacles_on_path(ahead_pts, pts, half, 3.0)
        assert n_mem == n_ref, trial
        if d_ref is None:
            assert d_mem is None
        else:
            assert d_mem == pytest.approx(d_ref, abs=0.02), trial
    # aged-out batches and batches without arclengths are ignored
    assert mem.ahead(1.0, s_now, 3.0) == (None, 0)
    mem2 = ObstacleMemory()
    mem2.add(np.array([[10.0, 0.0]]), 0.0, "lidar")
    assert mem2.ahead(0.1, 0.0, 5.0) == (None, 0)


def test_closed_loop_obstacle_stop_with_arrival_projection():
    """The drive node's path: each scan is placed on the racing line once
    (corridor_arclength) and the 50 Hz cycle compares arclengths only."""
    path = ring_path()
    ctrl = PurePursuitController(path, ControllerConfig(max_speed_mps=2.0))
    sup = engaged(SafetyConfig())
    c = sup.config
    mem = ObstacleMemory()
    sim = KartSim(*path.point_at(0.0), path.heading_at(0.0))
    speed, box = 0.0, None
    for k in range(int(12 / DT)):
        now = k * DT
        if box is None and now >= 5.0:
            p = path.point_at((ctrl.progress or 0.0) + 5.0)
            box = (p[0] - 0.3, p[1] - 0.3, p[0] + 0.3, p[1] + 0.3)
        if box is not None and k % 5 == 0 and ctrl.progress is not None:  # 10 Hz LiDAR
            pts = planar_scan(sim.pose, np.zeros((0, 2)), 0.15, extra_boxes=[box], rays=360)
            if len(pts):
                c_, s_ = math.cos(sim.yaw), math.sin(sim.yaw)
                pm = pts @ np.array([[c_, -s_], [s_, c_]]).T + sim.pose[:2]
                s0 = ctrl.progress
                window = path.points_between(s0 - 0.5, s0 + sup.obstacle_reach(2.0) + 2.0 * 0.3 + 0.5, 0.25)
                along = corridor_arclength(window, pm, c.kart_half_width_m + c.lateral_margin_m) + (s0 - 0.5)
                mem.add(pm, now, "lidar", along=along)
        out = ctrl.step(sim.pose, sim.v, DT)
        ahead = mem.ahead(now, ctrl.progress, sup.obstacle_reach(speed)) if out.ok else None
        dec = sup.update(inputs(now, speed, obstacle_ahead=ahead))
        speed = dec.limit(out.speed)
        ctrl.commanded(speed)
        sim.step(out.steer, speed, DT)
    assert "obstacle" in sup.active and sim.v < 0.05
    cx, cy = 0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])
    bumper = sim.pose[:2] + 0.8 * np.array([math.cos(sim.yaw), math.sin(sim.yaw)])
    assert np.hypot(bumper[0] - cx, bumper[1] - cy) - 0.3 > 0.2  # stopped short, with room


def test_obstacle_beside_the_path_does_not_stop_the_kart():
    path = np.column_stack([np.linspace(0, 10, 41), np.zeros(41)])
    sup = engaged()
    pts = np.array([[4.0, 1.4], [4.1, 1.4], [4.2, 1.4], [4.0, -1.4], [4.1, -1.4]])  # a corridor wall 1.4 m off
    for k in range(20):
        dec = sup.update(inputs(k * DT, 2.0, path_ahead=path, obstacles=pts))
    assert not dec.stopping


def test_obstacle_memory_survives_a_sensor_thread_adding_while_the_control_loop_reads():
    """The drive node's sensor workers add observations while the control
    timer iterates them: that must never raise (it would kill the timer)."""
    import sys
    import threading
    import time

    mem = ObstacleMemory()
    stop = threading.Event()

    def worker():
        k = 0
        while not stop.is_set():
            mem.add(np.random.rand(10, 2), time.monotonic(), "lidar", along=np.random.rand(3) * 5.0)
            k += 1
            if k % 40 == 0:
                mem.clear()

    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-5)  # force frequent thread switches
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    try:
        t_end = time.monotonic() + 1.0
        while time.monotonic() < t_end:
            mem.ahead(time.monotonic(), 0.0, 5.0)
            mem.points(time.monotonic())
    finally:
        stop.set()
        t.join()
        sys.setswitchinterval(old)


def test_corridor_widens_on_one_side_only():
    path = np.column_stack([np.linspace(0.0, 10.0, 41), np.zeros(41)])
    pts = np.array([[5.0, 1.2], [5.0, -1.2], [5.0, 0.5], [5.0, -0.5]])
    assert len(corridor_arclength(path, pts, 0.78)) == 2  # only the two near the line
    along = corridor_arclength(path, pts, 0.78, extra_left=0.6)  # kart 0.6 m left of the line
    assert len(along) == 3 and np.allclose(along, 5.0)
    assert len(corridor_arclength(path, pts, 0.78, extra_right=0.6)) == 3
