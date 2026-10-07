"""Tests for the sensor-agnostic fusion core: health, timesync, EKF, faults."""

from __future__ import annotations

import numpy as np
import pytest

from kart_map2d.core.health import ChannelConfig, HealthMonitor, Status
from kart_map2d.core.measurements import (
    ImuSample,
    PositionFix,
    RelativeMotion,
    SpeedMeasurement,
    ZeroVelocity,
)
from kart_map2d.core.timesync import ClockSync, cross_correlation_lag
from kart_map2d.estimation.ekf import EkfConfig, PlanarEkf
from kart_map2d.estimation.fusion import FusionEngine, StillnessDetector
from kart_map2d.eval.faults import Fault, FaultInjector
from kart_map2d.eval.metrics import score_trajectory


# --------------------------------------------------------------------- health
def test_stale_then_recover():
    mon = HealthMonitor({"wheel": ChannelConfig(timeout_s=0.5)})
    ch = mon.channel("wheel")
    ch.on_receive(0.0, 1.0)
    mon.tick(0.1)
    assert ch.status == Status.OK
    mon.tick(1.0)  # nothing since 0.0
    assert ch.status == Status.STALE
    assert not ch.usable(1.0)
    ch.on_receive(1.2, 2.0)  # data resumes -> immediately usable again
    assert ch.usable(1.2)


def test_frozen_value_faults_and_cooldown():
    cfg = ChannelConfig(stuck_repeats=5, cooldown_s=2.0)
    mon = HealthMonitor({"wheel": cfg})
    ch = mon.channel("wheel")
    for i in range(20):
        ok = ch.on_receive(i * 0.1, 3.14)
    assert not ok and ch.status == Status.FAULTY
    # The cooldown refreshes while the value stays frozen (last fault ~t=1.9).
    assert not ch.usable(2.5)
    ch.on_receive(2.6, 2.71)  # value finally changes
    assert ch.usable(4.5)  # cooldown elapsed -> probation
    assert ch.status == Status.PROBATION


def test_persistent_rejections_fault_then_probation_recovers():
    cfg = ChannelConfig(faulty_consecutive_rejects=5, probation_accepts=3, cooldown_s=0.5)
    mon = HealthMonitor({"gnss": cfg})
    ch = mon.channel("gnss")
    for i in range(5):
        ch.on_receive(i * 0.1, float(i))
        ch.report(False)
    assert ch.status == Status.FAULTY
    ch.on_receive(1.0, 9.9)
    assert ch.usable(1.0)  # past cooldown -> probation
    for _ in range(3):
        ch.report(True)
    assert ch.status == Status.OK


def test_nonfinite_rejected():
    mon = HealthMonitor()
    assert not mon.channel("gnss").on_receive(0.0, np.array([np.nan, 1.0]))
    assert mon.channel("gnss").status == Status.FAULTY


# ------------------------------------------------------------------- timesync
def test_clock_sync_recovers_offset():
    rng = np.random.default_rng(0)
    device = np.arange(0.0, 30.0, 0.01)
    offset = 1000.0
    latency = rng.exponential(0.004, len(device))
    receive = device + offset + latency
    sync = ClockSync()
    corrected = np.array([sync.correct(d, r) for d, r in zip(device, receive)])
    # After warm-up the corrected stamps should sit within a few ms of truth.
    err = corrected[500:] - (device[500:] + offset)
    assert np.max(np.abs(err)) < 0.005


def test_clock_sync_epoch_passthrough():
    sync = ClockSync()
    assert sync.correct(1.7e9, 1.7e9 + 0.1) == 1.7e9


def test_cross_correlation_lag_finds_shift():
    t = np.arange(0.0, 20.0, 0.005)
    x = np.sin(t * 1.3) + 0.5 * np.sin(t * 3.1)
    lag, corr = cross_correlation_lag(t - 0.12, x, t, x)  # signal a delayed by 120 ms
    assert corr > 0.99
    assert abs(lag - 0.12) < 0.01


# ------------------------------------------------------------------------ EKF
def _drive_circle(engine: FusionEngine, radius=20.0, speed=5.0, duration=30.0, dt=0.01, imu=True, wheel=True):
    """Feed noisy circular-motion IMU + wheel data; return true final pose.

    Realistic sensor noise is required: bit-identical samples would (rightly)
    trip the frozen-value detector.
    """
    rng = np.random.default_rng(3)
    omega = speed / radius
    for k in range(int(duration / dt)):
        t = k * dt
        if imu:
            gyro = np.array([0.0, 0.0, omega]) + rng.normal(0.0, 0.005, 3)
            accel = np.array([0.0, speed * omega, 9.81]) + rng.normal(0.0, 0.05, 3)
            engine.push(ImuSample(t, "imu", gyro, accel), t)
        if wheel and k % 5 == 0:
            engine.push(SpeedMeasurement(t, "wheel", speed + rng.normal(0.0, 0.05)), t)
    engine.flush()
    theta = omega * duration
    true_xy = radius * np.array([np.sin(theta), 1.0 - np.cos(theta)])
    return true_xy, theta


def test_ekf_circle_with_imu_and_wheel():
    engine = FusionEngine()
    true_xy, theta = _drive_circle(engine)
    st, X, _, sup = engine.trajectory.arrays()
    assert np.hypot(*(X[-1, :2] - true_xy)) < 3.0
    # ~1% of the 7.5 rad integrated: the early bias-convergence transient.
    assert abs((X[-1, 2] - theta + np.pi) % (2 * np.pi) - np.pi) < 0.12
    assert sup[-1]


def test_ekf_survives_missing_imu():
    """Wheel + odometry increments alone must still localize (IMU optional)."""
    engine = FusionEngine()
    radius, speed, dt = 20.0, 5.0, 0.1
    omega = speed / radius
    for k in range(1, int(30.0 / dt)):
        t = k * dt
        engine.push(
            RelativeMotion(t, "odom", t - dt, speed * dt, 0.0, omega * dt, sigma_position=0.005), t
        )
    engine.flush()
    st, X, _, _ = engine.trajectory.arrays()
    theta = omega * (30.0 - dt)
    true_xy = radius * np.array([np.sin(theta), 1.0 - np.cos(theta)])
    assert np.hypot(*(X[-1, :2] - true_xy)) < 4.0


def test_ekf_reanchors_after_speed_blackout():
    """After a long gap in speed evidence the next source must not be gated out."""
    engine = FusionEngine()
    # 5 s of healthy driving at 5 m/s.
    for k in range(500):
        t = k * 0.01
        engine.push(ImuSample(t, "imu", np.zeros(3), np.array([0.0, 0.0, 9.81])), t)
        if k % 5 == 0:
            engine.push(SpeedMeasurement(t, "wheel", 5.0), t)
    # 4 s with IMU only (accel noise-free here, speed coasts at 5)...
    for k in range(500, 900):
        t = k * 0.01
        engine.push(ImuSample(t, "imu", np.zeros(3), np.array([0.0, 0.0, 9.81])), t)
    # ...then the kart has actually stopped: wheel returns reporting 0.
    for k in range(900, 1000):
        t = k * 0.01
        engine.push(ImuSample(t, "imu", np.zeros(3), np.array([0.0, 0.0, 9.81])), t)
        if k % 5 == 0:
            engine.push(SpeedMeasurement(t, "wheel", 0.0), t)
    engine.flush()
    assert engine.ekf.reanchors, "expected a re-anchor after the speed blackout"
    assert abs(engine.ekf.x[3]) < 0.3  # speed pulled back to ~0


def test_zupt_pins_speed():
    ekf = PlanarEkf(HealthMonitor(), EkfConfig())
    ekf.process(SpeedMeasurement(0.0, "wheel", 2.0, sigma=1.0))
    for k in range(1, 50):
        ekf.process(ZeroVelocity(k * 0.1, "zupt"))
    assert abs(ekf.x[3]) < 0.05


def test_stillness_detector_fires_only_at_rest():
    rng = np.random.default_rng(1)
    det = StillnessDetector()
    fired_still = fired_moving = 0
    for k in range(200):
        t = k * 0.01
        z = det.update(ImuSample(t, "imu", rng.normal(0, 0.001, 3), np.array([0, 0, 9.81]) + rng.normal(0, 0.02, 3)))
        fired_still += z is not None
    for k in range(200, 400):
        t = k * 0.01
        z = det.update(ImuSample(t, "imu", rng.normal(0, 0.05, 3), np.array([0.3, 0.5, 9.81]) + rng.normal(0, 0.5, 3)))
        fired_moving += z is not None
    assert fired_still > 5
    assert fired_moving == 0


# ---------------------------------------------------------------------- faults
def test_fault_injector_dropout_and_stuck():
    inj = FaultInjector([Fault("wheel", "dropout", start=1.0, stop=2.0), Fault("imu", "stuck", start=0.5)], t0=0.0)
    assert inj.apply(SpeedMeasurement(1.5, "wheel", 3.0)) is None
    assert inj.apply(SpeedMeasurement(2.5, "wheel", 3.0)) is not None
    first = ImuSample(0.6, "imu", np.array([0.0, 0.0, 0.7]), np.zeros(3))
    inj.apply(first)
    later = inj.apply(ImuSample(0.9, "imu", np.array([0.0, 0.0, -0.3]), np.zeros(3)))
    assert later.stamp == 0.9 and later.gyro[2] == pytest.approx(0.7)  # frozen payload, fresh stamp


def test_fault_injector_bias():
    inj = FaultInjector([Fault("imu", "bias", value=0.2)], t0=0.0)
    out = inj.apply(ImuSample(1.0, "imu", np.zeros(3), np.zeros(3)))
    assert out.gyro[2] == pytest.approx(0.2)


# --------------------------------------------------------------------- metrics
def test_score_trajectory_alignment_invariance():
    t = np.arange(0.0, 60.0, 0.1)
    xy = np.column_stack([20 * np.sin(t * 0.2), 20 * (1 - np.cos(t * 0.2))])
    yaw = t * 0.2 + np.pi / 2
    ref = np.column_stack([xy, yaw])
    # Estimate = reference rigidly moved: ATE must be ~0 regardless.
    ang = 1.1
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    est = np.column_stack([xy @ R.T + [100, -50], yaw + ang])
    s = score_trajectory(t, est, t, ref)
    assert s.ate_rmse_m < 1e-6
    assert s.rpe_trans_pct < 1e-6


def test_score_trajectory_detects_scale_error():
    t = np.arange(0.0, 60.0, 0.1)
    xy = np.column_stack([t * 2.0, np.zeros_like(t)])
    ref = np.column_stack([xy, np.zeros_like(t)])
    est = np.column_stack([xy * 1.1, np.zeros_like(t)])  # 10% scale error
    s = score_trajectory(t, est, t, ref)
    assert 5.0 < s.rpe_trans_pct < 15.0
