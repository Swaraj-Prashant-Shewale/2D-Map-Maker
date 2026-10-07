"""Pure pursuit on the racing line: geometry, sign conventions, progress, speed profile.

Synthetic paths only (no recordings). The closed-loop cases drive the
kinematic kart simulator (kart_map2d.drive.sim) with the steering lag.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from kart_map2d.drive.controller import (
    KART_MAX_STEER_RAD,
    ControllerConfig,
    PurePursuitController,
    RacingPath,
    wrap_angle,
)
from kart_map2d.drive.sim import GaussMarkovNoise, KartSim, run_closed_loop
from kart_map2d.mapping.map_package import RacingLineData


def straight(length=30.0, speed=5.0, spacing=0.5) -> RacingPath:
    x = np.arange(0.0, length + 1e-9, spacing)
    return RacingPath(np.column_stack([x, np.zeros_like(x)]), np.full(len(x), speed))


def ring(radius=10.0, speed=5.0, n=126) -> RacingPath:
    th = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    return RacingPath(np.column_stack([radius * np.cos(th), radius * np.sin(th)]), np.full(n, speed), closed=True)


def out_and_back(leg=20.0, gap=2.0, speed=5.0) -> RacingPath:
    """Out along y=0, a U-turn of radius gap/2 + 1, back along y=-gap (like sept3)."""
    x = np.arange(0.0, leg + 1e-9, 0.5)
    out = np.column_stack([x, np.zeros_like(x)])
    r = gap / 2.0
    th = np.linspace(np.pi / 2, -np.pi / 2, 13)[1:-1]
    turn = np.column_stack([leg + r * np.cos(th) + 0.0, -r + r * np.sin(th)])
    back = np.column_stack([x[::-1], np.full(len(x), -gap)])
    return RacingPath(np.vstack([out, turn, back]), np.full(len(out) + len(turn) + len(back), speed))


# ------------------------------------------------------------------ path
def test_path_projection_signs_and_wrap():
    p = straight()
    proj = p.project(np.array([5.2, 0.4]))
    assert proj.s == pytest.approx(5.2) and proj.lateral == pytest.approx(0.4)  # left of the path = +
    assert p.project(np.array([5.2, -0.3])).lateral == pytest.approx(-0.3)
    r = ring()
    assert r.closed and r.length == pytest.approx(2 * np.pi * 10.0, rel=2e-3)
    # the closing segment wraps: a point just before the start projects near s = length
    q = r.project(np.array([10.0 * np.cos(-0.01), 10.0 * np.sin(-0.01)]))
    assert q.s > r.length - 0.2
    assert np.allclose(r.point_at(r.length + 1.0), r.point_at(1.0))
    # open paths extrapolate along the end tangent (lookahead past the finish)
    assert np.allclose(p.point_at(p.length + 2.0), [32.0, 0.0])
    assert np.allclose(p.point_at(-1.0), [-1.0, 0.0])
    # the tuple accessor and the vectorized one agree with point_at
    for path in (r, p):
        for s in (-1.0, 0.0, 0.3, 7.7, path.length - 0.01, path.length + 2.5):
            assert np.allclose(path.xy_at(s), path.point_at(s))
            assert np.allclose(path.points_at(np.array([s]))[0], path.point_at(s))


def test_racing_path_from_map_package_line():
    s = np.arange(0.0, 10.0, 0.5)
    line = RacingLineData(s, np.column_stack([s, np.zeros_like(s)]), np.zeros_like(s), np.full(len(s), 3.0), False, 3.3)
    p = RacingPath.from_racing_line(line)
    assert p.length == pytest.approx(9.5) and not p.closed and p.speed_at(4.0) == pytest.approx(3.0)


# ------------------------------------------------------------------ steering
def test_steering_sign_left_is_positive():
    """REP-103 / ackermann_msgs: positive steering angle turns LEFT (CCW yaw)."""
    ctrl = PurePursuitController(straight(), ControllerConfig())
    # Kart to the RIGHT of the line (y < 0), heading along it: steer left (+).
    out = ctrl.step(np.array([2.0, -0.8, 0.0]), 1.0, 0.02)
    assert out.ok and out.steer > 0.05 and out.cross_track < 0
    ctrl.reset()
    out = ctrl.step(np.array([2.0, 0.8, 0.0]), 1.0, 0.02)
    assert out.steer < -0.05 and out.cross_track > 0
    # A left-hand ring (CCW) needs positive steering in steady state.
    ctrl = PurePursuitController(ring(), ControllerConfig())
    out = ctrl.step(np.array([10.0, 0.0, np.pi / 2]), 1.5, 0.02)
    assert out.steer > 0.0
    kart = KartSim(10.0, 0.0, np.pi / 2)
    kart.step(out.steer, 1.5, 0.5)
    assert kart.yaw_rate > 0.0  # the model agrees: positive steer -> CCW yaw


def test_steering_is_clamped_to_the_kart_limit():
    ctrl = PurePursuitController(straight(), ControllerConfig())
    assert ctrl.step(np.array([2.0, 0.0, 0.0]), 0.5, 0.02).ok  # progress established
    out = ctrl.step(np.array([2.0, -2.5, -1.2]), 0.5, 0.02)  # then knocked far off, pointing away
    assert out.ok and out.steer == pytest.approx(KART_MAX_STEER_RAD)
    out = ctrl.step(np.array([2.0, 2.5, 1.2]), 0.5, 0.02)
    assert out.steer == pytest.approx(-KART_MAX_STEER_RAD)


# ------------------------------------------------------------------ progress
def test_progress_never_jumps_to_the_other_leg_of_an_out_and_back():
    path = out_and_back()
    ctrl = PurePursuitController(path, ControllerConfig(max_speed_mps=2.0))
    # Drive the outbound leg: the return leg is only 2 m to the right.
    s_prev = -1.0
    for x in np.arange(0.0, 18.0, 0.2):
        out = ctrl.step(np.array([x, -0.9, 0.0]), 2.0, 0.02)  # 0.9 m right of the outbound line
        assert out.s >= s_prev - 1e-9
        assert out.s < 20.0  # still on the outbound leg
        s_prev = out.s
    # Starting on the return leg (heading -x) picks the return leg, not the nearer outbound one.
    ctrl2 = PurePursuitController(path, ControllerConfig())
    out = ctrl2.step(np.array([10.0, -1.1, np.pi]), 1.0, 0.02)
    assert out.ok and out.s > path.length / 2


def test_start_refused_far_from_the_line_or_pointing_away():
    ctrl = PurePursuitController(straight(), ControllerConfig(max_start_offset_m=3.0))
    out = ctrl.step(np.array([5.0, 4.0, 0.0]), 0.0, 0.02)
    assert not out.ok and out.speed == 0.0 and "from the racing line" in out.reason
    out = ctrl.step(np.array([5.0, 0.0, np.pi]), 0.0, 0.02)
    assert not out.ok and out.speed == 0.0 and "heading" in out.reason


# ------------------------------------------------------------------ speed
def test_speed_profile_caps_scale_and_rate_limits():
    cfg = ControllerConfig(max_speed_mps=2.0, speed_scale=0.5, max_accel_mps2=1.0, max_decel_mps2=2.0)
    ctrl = PurePursuitController(straight(speed=3.0), cfg)
    speeds = [ctrl.step(np.array([0.1 * k, 0.0, 0.0]), 0.0, 0.02).speed for k in range(10)]
    assert speeds[-1] == pytest.approx(10 * 0.02 * 1.0)  # accel-limited ramp
    ctrl = PurePursuitController(straight(speed=3.0), cfg)
    for k in range(400):
        out = ctrl.step(np.array([5.0 + 0.001 * k, 0.0, 0.0]), 1.5, 0.02)
    assert out.v_ref == pytest.approx(1.5)  # 3.0 * 0.5, under the 2.0 cap
    cfg2 = ControllerConfig(max_speed_mps=2.0, speed_scale=1.0)
    ctrl = PurePursuitController(straight(speed=3.0), cfg2)
    for k in range(400):
        out = ctrl.step(np.array([5.0 + 0.001 * k, 0.0, 0.0]), 1.5, 0.02)
    assert out.speed <= 2.0 + 1e-9 and out.v_ref == pytest.approx(2.0)


def test_max_speed_zero_never_moves():
    ctrl = PurePursuitController(straight(), ControllerConfig(max_speed_mps=0.0))
    for k in range(100):
        out = ctrl.step(np.array([1.0, 0.0, 0.0]), 0.0, 0.02)
        assert out.speed == 0.0


def test_commanded_feedback_restarts_the_ramp_from_what_was_sent():
    cfg = ControllerConfig(max_speed_mps=2.0, max_accel_mps2=1.0)
    ctrl = PurePursuitController(straight(), cfg)
    for k in range(200):
        out = ctrl.step(np.array([1.0 + 0.01 * k, 0.0, 0.0]), 1.0, 0.02)
    assert out.speed == pytest.approx(2.0)
    ctrl.commanded(0.0)  # the safety layer stopped the kart
    out = ctrl.step(np.array([3.0, 0.0, 0.0]), 0.0, 0.02)
    assert out.speed == pytest.approx(0.02)  # ramps up again from zero, not from 2.0


def test_open_route_brakes_to_a_stop_before_its_end():
    res = run_closed_loop(straight(length=20.0, speed=5.0), ControllerConfig(max_speed_mps=2.0, stop_margin_m=0.3))
    assert res.finished
    end_x = res.xyyaw[-1, 0]
    assert 19.0 < end_x < 20.0 + 1e-6  # stopped before the end, not past it
    assert res.v[-1] < 0.05


def test_curvature_limits_speed():
    tight = ring(radius=3.0, speed=10.0)
    ctrl = PurePursuitController(tight, ControllerConfig(max_speed_mps=5.0, max_lat_accel_mps2=2.5))
    assert ctrl.profile_speed_at(1.0) == pytest.approx(math.sqrt(2.5 * 3.0), rel=0.03)


# ------------------------------------------------------------------ closed loop
@pytest.mark.parametrize("speed", [1.0, 2.0])
def test_closed_loop_ring_tracks_within_centimetres(speed):
    res = run_closed_loop(ring(), ControllerConfig(max_speed_mps=speed, laps=2), settle_s=1.0)
    m = res.metrics
    assert res.finished and m["distance_m"] == pytest.approx(2 * ring().length, rel=0.03)
    assert m["cross_track_rms_m"] < 0.05 and m["steer_saturation_fraction"] == 0.0


def test_closed_loop_out_and_back_with_noise():
    path = out_and_back(leg=20.0, gap=4.0)
    res = run_closed_loop(path, ControllerConfig(max_speed_mps=2.0), pose_noise=GaussMarkovNoise((0.05, 0.05, math.radians(0.5)), 0.5, 3), settle_s=1.0)
    m = res.metrics
    assert res.finished
    assert m["cross_track_rms_m"] < 0.3  # the drive gate, at 2 m/s
    # the kart really came back along the return leg
    assert res.xyyaw[-1, 1] < -3.0 and res.xyyaw[-1, 0] < 2.0


def test_step_compute_is_microseconds():
    ctrl = PurePursuitController(ring(), ControllerConfig())
    import time

    t0 = time.perf_counter()
    for k in range(2000):
        a = 0.001 * k
        ctrl.step(np.array([10 * math.cos(a), 10 * math.sin(a), wrap_angle(a + math.pi / 2)]), 2.0, 0.02)
    per_step_us = (time.perf_counter() - t0) / 2000 * 1e6
    assert per_step_us < 1000.0  # well inside a 20 ms control period
