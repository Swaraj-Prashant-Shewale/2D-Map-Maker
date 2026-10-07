"""Tests for the interval-level gyro-vs-odometry yaw consistency check.

Synthetic fixtures only: a constant-rate circle (base_link frame increments,
map-frame truth) driven through the full FusionEngine, so the reorder buffer,
health machinery and EKF interplay are exercised exactly as in replay.
"""

from __future__ import annotations

import numpy as np

from kart_map2d.core.health import Status
from kart_map2d.core.measurements import ImuSample, RelativeMotion
from kart_map2d.estimation.ekf import wrap
from kart_map2d.estimation.fusion import (
    GYRO_INCONSISTENT,
    FusionEngine,
    YawConsistencyConfig,
)

RADIUS = 20.0  # m
SPEED = 5.0  # m/s
OMEGA = SPEED / RADIUS  # rad/s, constant left turn


def _drive_circle(
    engine: FusionEngine,
    duration: float = 40.0,
    gyro_bias: float = 0.0,
    bias_start: float = np.inf,
    bias_stop: float = np.inf,
    odo_channels: tuple[str, ...] = ("vo_zed_right", "vo_zed_left"),
    odo_yaw_rate_bias: tuple[float, ...] | None = None,
    seed: int = 3,
) -> tuple[np.ndarray, float]:
    """Circle drive: 100 Hz IMU + staggered 0.3 s odometric increments.

    Returns the true final (x, y) and yaw. Sensor noise is realistic enough
    not to trip the frozen-value detector.
    """
    rng = np.random.default_rng(seed)
    dt, odo_dt = 0.01, 0.3
    h = OMEGA * odo_dt  # yaw per odometric interval [rad]
    for k in range(int(duration / dt)):
        t = k * dt
        gz = OMEGA + (gyro_bias if bias_start <= t < bias_stop else 0.0)
        gyro = np.array([0.0, 0.0, gz]) + rng.normal(0.0, 0.005, 3)
        accel = np.array([0.0, SPEED * OMEGA, 9.81]) + rng.normal(0.0, 0.05, 3)
        engine.push(ImuSample(t, "imu", gyro, accel), t)
        for i, channel in enumerate(odo_channels):
            if k > int(odo_dt / dt) and (k + 10 * i) % int(odo_dt / dt) == 0:
                dyaw = h + rng.normal(0.0, 0.002)
                if odo_yaw_rate_bias is not None:
                    dyaw += odo_yaw_rate_bias[i] * odo_dt
                engine.push(
                    RelativeMotion(
                        t,
                        channel,
                        t - odo_dt,
                        RADIUS * np.sin(h) + rng.normal(0.0, 0.005),
                        RADIUS * (1.0 - np.cos(h)),
                        dyaw,
                        sigma_translation=0.05,
                        sigma_yaw=0.01,
                        sigma_position=0.005,
                    ),
                    t,
                )
    engine.flush()
    theta = OMEGA * duration
    return RADIUS * np.array([np.sin(theta), 1.0 - np.cos(theta)]), theta


def _final_errors(engine: FusionEngine, true_xy: np.ndarray, true_yaw: float) -> tuple[float, float]:
    _, states, _, _ = engine.trajectory.arrays()
    return float(np.hypot(*(states[-1, :2] - true_xy))), abs(wrap(states[-1, 2] - true_yaw))


def _imu_fault_transitions(engine: FusionEngine, reason: str) -> list[float]:
    return [
        stamp
        for stamp, name, status, why in engine.health.transitions
        if name == "imu" and status == "faulty" and why == reason
    ]


# ------------------------------------------------------------------ clean runs
def test_clean_run_never_blames_the_gyro():
    engine = FusionEngine()
    true_xy, true_yaw = _drive_circle(engine, duration=30.0)
    assert not _imu_fault_transitions(engine, GYRO_INCONSISTENT)
    assert not engine.yaw_monitor.faults
    pos_err, yaw_err = _final_errors(engine, true_xy, true_yaw)
    assert pos_err < 3.0


# ------------------------------------------------------- bias detection + blame
def test_bias_blamed_within_15s_and_ekf_survives_the_switch():
    """0.15 rad/s in-band bias mid-turn: blamed fast, filter recovers."""
    onset, duration = 15.0, 40.0
    engine = FusionEngine()
    true_xy, true_yaw = _drive_circle(engine, duration, gyro_bias=0.15, bias_start=onset)
    faults = _imu_fault_transitions(engine, GYRO_INCONSISTENT)
    assert faults, "the biased gyro was never blamed"
    assert faults[0] - onset < 15.0
    # The blame transfers the measured excess rate into the gyro-bias state,
    # so the re-admitted channel comes back calibrated: one blame suffices
    # and the channel legitimately ends healthy.
    assert len(faults) == 1
    assert engine.health.channels["imu"].status in (Status.OK, Status.DEGRADED)
    assert abs(engine.ekf.x[6] - 0.15) < 0.05  # BG learned the injected bias
    # The heading correction must roughly match the yaw absorbed before blame
    # (bias times detection latency); sign: positive bias -> positive excess.
    _, _, correction = engine.yaw_monitor.faults[0]
    assert 0.05 < correction < 1.0
    # No divergence at the switch: the trajectory finishes near truth even
    # though the (only) gyro went bad mid-turn. The ~2 s of pre-detection
    # travel stays displaced (only heading is corrected retroactively), so the
    # bound is looser than the clean run's ~1.5 m.
    pos_err, yaw_err = _final_errors(engine, true_xy, true_yaw)
    assert pos_err < 6.0
    assert yaw_err < 0.3

    # Reference point: the same drive with the check disabled must be much
    # worse - this is the failure the monitor exists to fix. (The binding
    # "ATE at most half the undetected baseline" gate runs on the real
    # recordings in scripts/validate_gyro_check.py.)
    blind = FusionEngine(yaw_check=YawConsistencyConfig(threshold=1e9, solo_threshold=1e9))
    _drive_circle(blind, duration, gyro_bias=0.15, bias_start=onset)
    blind_err, _ = _final_errors(blind, true_xy, true_yaw)
    assert pos_err < 0.55 * blind_err


def test_bias_recovery_reinstates_the_gyro():
    """Once the bias heals, cooldown + probation re-admit the channel."""
    engine = FusionEngine()
    _drive_circle(engine, duration=45.0, gyro_bias=0.15, bias_start=10.0, bias_stop=20.0)
    faults = _imu_fault_transitions(engine, GYRO_INCONSISTENT)
    assert faults and 10.0 < faults[0] < 20.0
    assert engine.health.channels["imu"].status in (Status.OK, Status.DEGRADED)
    recovered = [
        stamp
        for stamp, name, status, _ in engine.health.transitions
        if name == "imu" and status == "ok" and stamp > faults[0]
    ]
    assert recovered, "the healed gyro was never re-admitted"


# ------------------------------------------------------------ symmetric safety
def test_no_blame_when_odometric_sources_disagree():
    """Odometry split (one +0.1, one -0.1 rad/s): the gyro sits between the
    witnesses, so nothing is blamed and the split is only logged."""
    engine = FusionEngine()
    _drive_circle(engine, duration=25.0, odo_yaw_rate_bias=(0.10, -0.10))
    assert not _imu_fault_transitions(engine, GYRO_INCONSISTENT)
    assert not engine.yaw_monitor.faults
    assert any(kind == "odometry-split" for _, kind, _, _ in engine.yaw_monitor.events)


def test_single_witness_requires_stronger_evidence():
    """One odometric channel vs one gyro is ambiguous: a disagreement that
    clears the corroborated bar (~1.7 sigma) must NOT be blamed solo, while a
    gross one (~4 sigma) must."""
    mild = FusionEngine()
    _drive_circle(mild, duration=30.0, gyro_bias=0.08, bias_start=10.0, odo_channels=("vo_zed_right",))
    assert not _imu_fault_transitions(mild, GYRO_INCONSISTENT)

    gross = FusionEngine()
    _drive_circle(gross, duration=30.0, gyro_bias=0.30, bias_start=10.0, odo_channels=("vo_zed_right",))
    faults = _imu_fault_transitions(gross, GYRO_INCONSISTENT)
    assert faults and faults[0] - 10.0 < 15.0


# ------------------------------------------------------------------- mechanics
def test_interval_overlapping_buffer_edge_is_skipped():
    engine = FusionEngine()
    monitor = engine.yaw_monitor
    engine.health.channel("odom").on_receive(0.0)  # make the witness usable
    rng = np.random.default_rng(0)
    for k in range(300):  # gyro buffered over t = 10..13 s
        t = 10.0 + k * 0.01
        gyro = np.array([0.0, 0.0, 0.1 + rng.normal(0.0, 0.001)])
        monitor.add_imu(ImuSample(t, "imu", gyro, np.zeros(3)))
    # Starts before the retention buffer -> skipped, no statistic.
    monitor.add_relative(RelativeMotion(9.0, "odom", 8.5, 1.0, 0.0, 0.05))
    assert ("imu", "odom") not in monitor._pairs
    # Ends after the newest gyro sample -> skipped.
    monitor.add_relative(RelativeMotion(13.5, "odom", 12.9, 1.0, 0.0, 0.05))
    assert ("imu", "odom") not in monitor._pairs
    # Fully covered -> the pair statistic appears.
    monitor.add_relative(RelativeMotion(12.0, "odom", 11.5, 1.0, 0.0, 0.05))
    assert ("imu", "odom") in monitor._pairs


def test_frozen_gyro_still_caught_by_the_stuck_detector():
    """Regression guard: the frozen-value path is untouched by the monitor."""
    engine = FusionEngine()
    gyro, accel = np.array([0.0, 0.0, 0.3]), np.array([0.1, 0.0, 9.81])
    for k in range(40):
        engine.push(ImuSample(k * 0.01, "imu", gyro, accel), k * 0.01)
    engine.flush()
    imu = engine.health.channels["imu"]
    assert imu.status == Status.FAULTY
    assert imu.reason == "value frozen"
