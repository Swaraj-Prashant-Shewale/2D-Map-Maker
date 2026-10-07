"""Requirement-13 adapters: magnetometer, speedometer, wheel ticks.

All validation here is SYNTHETIC by necessity (the September recordings
carry none of these sensors); the miscalibrated-magnetometer case exercises
the same health-monitor path the live node runs.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from kart_map2d.core.measurements import HeadingMeasurement, ImuSample, RelativeMotion, SpeedMeasurement
from kart_map2d.estimation.fusion import FusionEngine
from kart_map2d.sensors.adapters import MagnetometerAdapter, SpeedometerAdapter, WheelTickAdapter
from kart_map2d.sensors.mag_calibration import (
    CALIBRATION_PROCEDURE,
    MagCalibration,
    MagCalibrationError,
    fit_ellipsoid,
    load_mag_calibration,
    save_mag_calibration,
)


def _time(stamp: float) -> SimpleNamespace:
    sec = int(np.floor(stamp))
    return SimpleNamespace(sec=sec, nanosec=int(round((stamp - sec) * 1e9)))


def _header(stamp: float) -> SimpleNamespace:
    return SimpleNamespace(stamp=_time(stamp), frame_id="")


def _mag_msg(stamp: float, field) -> SimpleNamespace:
    return SimpleNamespace(header=_header(stamp), magnetic_field=SimpleNamespace(x=field[0], y=field[1], z=field[2]))


# ------------------------------------------------------------ ellipsoid fit
def synthetic_mag_samples(n=400, hard=(2e-5, -1e-5, 5e-6), soft_inv=None, norm=5e-5, seed=3):
    """Raw samples of a full slow turn under a known hard/soft-iron pair.

    A +-12 deg tilt wobble rides on the turn (as on a real kart), which is
    what makes the z axis of the ellipsoid observable at all - a perfectly
    planar circle leaves the z hard iron mathematically unconstrained.
    """
    rng = np.random.default_rng(seed)
    yaw = np.linspace(0.0, 2 * np.pi, n)
    dip = 0.38 + 0.21 * np.sin(3.0 * yaw)  # tilt wobble
    truth = np.column_stack([np.cos(yaw), -np.sin(yaw), dip])
    truth *= norm / np.linalg.norm(truth, axis=1, keepdims=True)
    if soft_inv is None:
        soft_inv = np.array([[1.15, 0.05, 0.0], [0.05, 0.9, 0.02], [0.0, 0.02, 1.05]])
    raw = truth @ soft_inv.T + np.asarray(hard) + rng.normal(0, norm * 0.01, (n, 3))
    return raw, yaw


def test_ellipsoid_fit_recovers_hard_iron_sphere_and_heading():
    hard = np.array([2e-5, -1e-5, 5e-6])
    raw, yaw = synthetic_mag_samples(hard=tuple(hard))
    cal = fit_ellipsoid(raw)
    # The planar components are what heading uses; they must come back tight.
    np.testing.assert_allclose(cal.hard_iron[:2], hard[:2], atol=2e-6)
    corrected = cal.correct(raw)
    radii = np.linalg.norm(corrected, axis=1)
    assert radii.std() / radii.mean() < 0.05  # sphere within 5 percent
    assert cal.rms_residual < 0.08 and cal.samples == len(raw)
    # End to end: the corrected planar heading tracks the true yaw.
    headings = np.array([cal.planar_heading(r) for r in raw])
    err = np.degrees(np.abs((headings - yaw + np.pi) % (2 * np.pi) - np.pi))
    assert np.median(err) < 3.0


def test_fit_refuses_too_few_or_stuck_or_partial_data():
    with pytest.raises(MagCalibrationError, match="need at least"):
        fit_ellipsoid(np.zeros((10, 3)))
    with pytest.raises(MagCalibrationError):
        fit_ellipsoid(np.tile([1e-5, 2e-5, 3e-5], (200, 1)))  # stuck sensor
    # No planar spread: the kart never turned.
    line = np.column_stack([np.full(200, 1e-5), np.full(200, 2e-5), np.linspace(0, 1e-5, 200)])
    with pytest.raises(MagCalibrationError, match="360"):
        fit_ellipsoid(line + np.random.default_rng(0).normal(0, 1e-9, (200, 3)))


def test_calibration_yaml_roundtrip(tmp_path):
    cal = fit_ellipsoid(synthetic_mag_samples()[0])
    path = save_mag_calibration(tmp_path / "mag.yaml", cal)
    loaded = load_mag_calibration(path)
    np.testing.assert_allclose(loaded.hard_iron, cal.hard_iron)
    np.testing.assert_allclose(loaded.soft_iron, cal.soft_iron)
    assert loaded.field_norm == pytest.approx(cal.field_norm)


def test_missing_calibration_file_prints_procedure(tmp_path):
    with pytest.raises(MagCalibrationError) as err:
        load_mag_calibration(tmp_path / "none.yaml")
    assert "Drive one SLOW full circle" in CALIBRATION_PROCEDURE
    assert "calibrate_magnetometer.py" in str(err.value)


# --------------------------------------------------------- heading adapter
def _identity_cal(norm=5e-5):
    return MagCalibration(hard_iron=np.zeros(3), soft_iron=np.eye(3), field_norm=norm)


def test_mag_adapter_heading_and_corrections():
    cal = _identity_cal()
    adapter = MagnetometerAdapter(cal, declination_deg=10.0, mount_yaw_deg=-5.0, sigma_deg=8.0, rate_hz=1000.0)
    # Field pointing along +x = facing magnetic north -> heading 0 + corrections.
    m = adapter.push(_mag_msg(1.0, (5e-5, 0.0, 0.0)))
    assert isinstance(m, HeadingMeasurement) and m.channel == "mag"
    assert m.yaw == pytest.approx(np.radians(10.0 - 5.0))
    assert m.sigma == pytest.approx(np.radians(8.0))
    # 90 deg right of north: field appears along +y in body axes -> yaw -90.
    m2 = adapter.push(_mag_msg(2.0, (0.0, 5e-5, 0.0)))
    assert m2.yaw == pytest.approx(np.radians(-90.0 + 10.0 - 5.0))


def test_mag_adapter_decimates_and_rejects_wild_fields():
    adapter = MagnetometerAdapter(_identity_cal(), rate_hz=10.0)
    assert adapter.push(_mag_msg(0.0, (5e-5, 0, 0))) is not None
    assert adapter.push(_mag_msg(0.01, (5e-5, 0, 0))) is None  # decimated
    assert adapter.push(_mag_msg(0.2, (1e-3, 0, 0))) is None  # 20x the calibrated norm
    assert adapter.rejected_field == 1
    assert adapter.push(_mag_msg(0.3, (np.nan, 0, 0))) is None


def test_miscalibrated_mag_is_rejected_by_the_health_monitor():
    """The requirement's negative case: a magnetometer whose calibration is
    wrong (heading ~90 deg off) must be excluded by the health machinery,
    not fused. Straight drive: IMU + odometry hold yaw near 0 while the mag
    insists on +90 deg -> innovation gate rejects every fix and the channel
    goes FAULTY after its 30 consecutive rejects."""
    engine = FusionEngine()
    t0 = 1000.0
    dt = 0.01
    for k in range(1200):
        t = t0 + k * dt
        engine.push(ImuSample(t, "imu", np.zeros(3), np.array([0.0, 0.0, 9.81])), arrival=t)
        if k and k % 10 == 0:
            engine.push(
                RelativeMotion(stamp=t, channel="odom", start_stamp=t - 10 * dt,
                               delta_forward=0.2, delta_left=0.0, delta_yaw=0.0),
                arrival=t,
            )
        if k and k % 25 == 0:  # 4 Hz of confidently wrong heading
            engine.push(
                HeadingMeasurement(t, "mag", yaw=np.pi / 2, sigma=np.radians(3.0)),
                arrival=t,
            )
    engine.flush()
    mag = engine.health.channel("mag")
    assert mag.status.value == "faulty"
    # And the estimate did not get dragged towards the bogus heading.
    assert abs(float(engine.ekf.x[2])) < np.radians(15.0)


def test_well_calibrated_mag_is_accepted():
    engine = FusionEngine()
    t0, dt = 1000.0, 0.01
    for k in range(600):
        t = t0 + k * dt
        engine.push(ImuSample(t, "imu", np.zeros(3), np.array([0.0, 0.0, 9.81])), arrival=t)
        if k and k % 25 == 0:
            engine.push(HeadingMeasurement(t, "mag", yaw=0.0, sigma=np.radians(8.0)), arrival=t)
    engine.flush()
    assert engine.health.healthy("mag")
    assert "heading" in engine.mode()


def _drive_mag_and_gyro(adapter: MagnetometerAdapter, extra_hard=(0.0, 0.0, 0.0), seed=11):
    """A straight -> 90 deg turn -> straight profile: gyro at 100 Hz into
    observe_gyro, mag messages at 50 Hz built from the TRUE heading plus an
    optional UNCOMPENSATED extra hard iron. Returns emitted measurements."""
    rng = np.random.default_rng(seed)
    B, Bz = 2e-5, 4e-5
    t0, dt = 1000.0, 0.01
    yaw = 0.0
    bias = 0.004  # rad/s gyro bias, as the sim uses
    out = []
    for k in range(1500):  # 15 s
        t = t0 + k * dt
        wz = 0.5236 if 5.0 <= t - t0 < 8.0 else 0.0  # 90 deg over 3 s
        yaw += wz * dt
        adapter.observe_gyro(t, wz + bias + rng.normal(0.0, 0.003))
        if k % 2 == 0:  # 50 Hz mag topic (the adapter decimates to its rate)
            field = np.array([B * np.cos(yaw), -B * np.sin(yaw), Bz])
            field = field + np.asarray(extra_hard) + rng.normal(0.0, 3e-7, 3)
            m = adapter.push(_mag_msg(t, field))
            if m is not None:
                out.append(m)
    return out


def test_mag_gyro_consistency_trips_on_uncompensated_hard_iron():
    """The live gap this check closes: a smoothly-wrong (miscalibrated) mag
    is ACCEPTED by the innovation gate once the EKF converges onto it
    (measured live: 573 of 603 accepted). During a turn its heading change
    disagrees with the gyro's integrated turn, which is unfakeable."""
    adapter = MagnetometerAdapter(_identity_cal(norm=4.5e-5), rate_hz=20.0)
    _drive_mag_and_gyro(adapter, extra_hard=(3e-5, -1.5e-5, 0.0))
    assert adapter.tripped, "the mag-gyro consistency check must fire during the turn"
    assert adapter.inconsistency_deg > 20.0
    assert not adapter.usable, "a tripped mag must never be fused again this run"


def test_mag_gyro_consistency_quiet_for_a_calibrated_mag():
    adapter = MagnetometerAdapter(_identity_cal(norm=4.5e-5), rate_hz=20.0)
    assert not adapter.usable or adapter._gyro_last is None  # nothing seen yet
    out = _drive_mag_and_gyro(adapter, extra_hard=(0.0, 0.0, 0.0))
    assert not adapter.tripped, f"false positive: {adapter.inconsistency_deg:.1f} deg"
    assert adapter.validated and adapter.usable, "one agreeing turn must arm the channel"
    assert len(out) > 200  # measurements kept flowing


def test_mag_validation_arms_only_after_a_turn():
    """Validation-armed trust: with gyro flowing, headings are not usable
    before the first turn has cross-validated the calibration (a straight
    cannot distinguish a wrong calibration from a mount offset)."""
    adapter = MagnetometerAdapter(_identity_cal(norm=4.5e-5), rate_hz=20.0)
    B, Bz = 2e-5, 4e-5
    for k in range(300):  # 3 s of STRAIGHT driving
        t = 1000.0 + 0.01 * k
        adapter.observe_gyro(t, 0.0)
        if k % 2 == 0:
            adapter.push(_mag_msg(t, (B, 0.0, Bz)))
    assert not adapter.usable and not adapter.tripped


def test_mag_gyro_consistency_inert_without_gyro_data():
    adapter = MagnetometerAdapter(_identity_cal(norm=4.5e-5), rate_hz=20.0)
    for k in range(200):
        adapter.push(_mag_msg(1000.0 + 0.05 * k, (4.5e-5 * np.cos(k * 0.05), -4.5e-5 * np.sin(k * 0.05), 0.0)))
    assert not adapter.tripped  # no gyro reference -> the check never judges


# ------------------------------------------------------------- speedometer
def test_speed_adapter_float32_uses_receive_stamp():
    adapter = SpeedometerAdapter(kind="float32", sigma_mps=0.2)
    m = adapter.push(SimpleNamespace(data=1.5), receive_stamp=42.0)
    assert isinstance(m, SpeedMeasurement) and m.channel == "speedometer"
    assert m.stamp == 42.0 and m.speed == 1.5 and m.sigma == 0.2
    assert adapter.push(SimpleNamespace(data=float("nan")), 43.0) is None


def test_speed_adapter_twist_stamped_uses_header():
    adapter = SpeedometerAdapter(kind="twist_stamped")
    msg = SimpleNamespace(header=_header(7.25), twist=SimpleNamespace(linear=SimpleNamespace(x=-0.4, y=0.0, z=0.0)))
    m = adapter.push(msg, receive_stamp=99.0)
    assert m.stamp == pytest.approx(7.25) and m.speed == pytest.approx(-0.4)  # reversing passes through


def test_speed_adapter_unknown_kind():
    with pytest.raises(ValueError, match="float32 | twist_stamped"):
        SpeedometerAdapter(kind="canbus").push(SimpleNamespace(data=1.0), 0.0)


# -------------------------------------------------------------- wheel ticks
def test_wheel_ticks_speed_and_quantization_sigma():
    adapter = WheelTickAdapter(ticks_per_metre=100.0, kind="int64", sigma_floor_mps=0.05)
    assert adapter.push(SimpleNamespace(data=1000), 10.0) is None  # anchor
    m = adapter.push(SimpleNamespace(data=1020), 10.1)
    assert m is not None and m.channel == "wheel"
    assert m.speed == pytest.approx(20 / 100.0 / 0.1)  # 2 m/s
    expected_sigma = float(np.hypot(0.05, 2.0 / 100.0 / 0.1))
    assert m.sigma == pytest.approx(expected_sigma)
    # Slower publish: the quantization term shrinks with 1/dt.
    m2 = adapter.push(SimpleNamespace(data=1220), 11.0)
    assert m2.sigma < m.sigma


def test_wheel_ticks_reset_and_gap_reanchor():
    adapter = WheelTickAdapter(ticks_per_metre=100.0, reset_jump_ticks=1e5, max_gap_s=0.5)
    adapter.push(SimpleNamespace(data=500000), 1.0)
    assert adapter.push(SimpleNamespace(data=0), 1.1) is None  # counter reset: silent re-anchor
    m = adapter.push(SimpleNamespace(data=10), 1.2)
    assert m is not None and m.speed == pytest.approx(1.0)
    assert adapter.push(SimpleNamespace(data=20), 3.0) is None  # 1.8 s gap: re-anchor
    # Small negative delta = genuine reversing, NOT a reset.
    m3 = adapter.push(SimpleNamespace(data=10), 3.1)
    assert m3 is not None and m3.speed == pytest.approx(-1.0)


def test_wheel_ticks_joint_state():
    adapter = WheelTickAdapter(ticks_per_metre=1.0, kind="joint_state", joint="rear", sigma_floor_mps=0.01)
    msg1 = SimpleNamespace(header=_header(5.0), name=["front", "rear"], position=[0.0, 100.0])
    msg2 = SimpleNamespace(header=_header(5.5), name=["front", "rear"], position=[0.0, 101.0])
    adapter.push(msg1, 0.0)
    m = adapter.push(msg2, 0.0)
    assert m.stamp == pytest.approx(5.5) and m.speed == pytest.approx(2.0)
    bad = SimpleNamespace(header=_header(6.0), name=["front"], position=[0.0])
    with pytest.raises(ValueError, match="rear"):
        adapter.push(bad, 0.0)


def test_wheel_ticks_validation():
    with pytest.raises(ValueError, match="ticks_per_metre"):
        WheelTickAdapter(ticks_per_metre=0.0)
    with pytest.raises(ValueError, match="int64 | joint_state"):
        WheelTickAdapter(ticks_per_metre=10.0, kind="hall").push(SimpleNamespace(data=1), 0.0)
