"""ROS-message -> measurement adapters for magnetometer, speedometer, wheel ticks.

The fusion core already understands these measurement types
(:class:`HeadingMeasurement`, :class:`SpeedMeasurement` on the channels
``mag`` / ``speedometer`` / ``wheel`` with health configs and EKF hooks in
place); what was missing were the ROS adapters. Like
:mod:`kart_map2d.ros2.converters` these are pure functions/objects over
duck-typed message objects - no rclpy import - so the offline runner, the
live node and the tests share them.

HONESTY (requirement 13): the September recordings contain none of these
topics, so everything here is validated synthetically (unit tests + the
synthetic-topic live test); the first real-data validation happens when the
user records the sensors. README.md lists the exact ``ros2 bag record``
topics to add.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from kart_map2d.core.measurements import HeadingMeasurement, SpeedMeasurement
from kart_map2d.sensors.mag_calibration import MagCalibration

__all__ = [
    "MagnetometerAdapter",
    "SpeedometerAdapter",
    "WheelTickAdapter",
    "wrap_pi",
]


def wrap_pi(angle: float) -> float:
    """Wrap to (-pi, pi]."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _stamp_of(msg) -> float:
    stamp = msg.header.stamp
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


# ---------------------------------------------------------------- magnetometer
@dataclass
class MagnetometerAdapter:
    """``sensor_msgs/MagneticField`` -> planar :class:`HeadingMeasurement`.

    Correction chain (all classical): hard/soft iron from the calibration
    YAML, then the fixed sensor mount yaw (``mount_yaw_deg``: how the sensor
    x axis is rotated CCW from base_link x), then the local magnetic
    declination (``declination_deg``, east positive) to go from magnetic to
    the map's reference direction, then the map-frame yaw offset
    (``map_yaw0``: the true heading of the map frame's +x axis, learned by
    the node from the first fused poses OR configured; default 0 = the map
    x axis points at true declination-corrected north-referenced heading 0).

    Rate limiting: measurements are decimated to ``rate_hz`` (the EKF gains
    nothing from 100 Hz absolute headings). ``sigma_deg`` is the 1-sigma
    heading noise handed to the EKF gate; the health monitor's ``mag``
    channel (faulty after 30 consecutive innovation rejects) is what catches
    a miscalibrated or disturbed magnetometer, exactly as configured in
    :mod:`kart_map2d.estimation.fusion`.
    """

    calibration: MagCalibration
    declination_deg: float = 0.0
    mount_yaw_deg: float = 0.0
    sigma_deg: float = 8.0
    rate_hz: float = 20.0
    map_yaw0: float = 0.0
    min_field_fraction: float = 0.3  # reject fields far off the calibrated norm
    max_field_fraction: float = 3.0
    channel: str = "mag"
    #: Magnetometer-gyro consistency check (classical): over a rolling window
    #: the TOTAL heading change the magnetometer reports must agree with the
    #: integrated gyro turn. A hard/soft-iron miscalibration warps heading as
    #: a function of true heading, so during a turn the two totals disagree
    #: by tens of degrees while the innovation gate alone stays quiet (the
    #: EKF converges onto the smoothly-wrong heading between turns; measured
    #: live: 573 of 603 miscalibrated measurements accepted before this
    #: check existed). Feed the gyro with :meth:`observe_gyro`; without gyro
    #: data the check is inert (never trips).
    gyro_gate_deg: float = 20.0  # min absolute disagreement to call a fault
    gyro_gate_turn_fraction: float = 0.35  # ... and at least this share of the turn
    consistency_window_s: float = 10.0
    min_turn_deg: float = 20.0  # judge only once the window holds a real turn
    min_increments: int = 5

    def __post_init__(self) -> None:
        self._next_emit = -np.inf
        self.rejected_field = 0  # field magnitude outside the plausible band
        self._gyro: deque = deque()  # (stamp, cumulative yaw [rad])
        self._gyro_cum = 0.0
        self._gyro_last: float | None = None
        self._pairs: deque = deque()  # (stamp, err_rad, |turn| rad) per emitted increment
        self._last_emitted: tuple[float, float] | None = None  # (stamp, yaw)
        self.inconsistent = False  # current rolling-window verdict
        self.tripped = False  # latched: the check has fired at least once
        self.validated = False  # latched: one full turn agreed with the gyro
        self.inconsistency_deg = 0.0  # last measured |disagreement| for logging

    @property
    def usable(self) -> bool:
        """Whether headings should be fused right now.

        Validation-armed trust: with gyro data flowing, the magnetometer
        contributes ONLY after its first turn has agreed with the gyro
        (``validated``) and never again once a turn has contradicted it
        (``tripped``). On a straight, a wrong calibration is
        indistinguishable from a mount offset, so pre-validation
        measurements are withheld instead of quietly bending the start of
        the map. Without any gyro feed the check cannot judge and headings
        pass through (an IMU-less rig keeps its only heading source).
        """
        if self.tripped:
            return False
        return self.validated or self._gyro_last is None

    def observe_gyro(self, stamp: float, yaw_rate_radps: float) -> None:
        """Feed the dedicated IMU's yaw rate (base_link z, same time base as
        the mag stamps) so the consistency check can integrate the true turn."""
        if not np.isfinite(stamp) or not np.isfinite(yaw_rate_radps):
            return
        if self._gyro_last is not None:
            dt = stamp - self._gyro_last
            if dt <= 0.0:
                return  # out-of-order arrival: ignore
            if dt < 0.5:  # a gap longer than this carries no usable integral
                self._gyro_cum += yaw_rate_radps * dt
        self._gyro_last = stamp
        self._gyro.append((stamp, self._gyro_cum))
        horizon = stamp - self.consistency_window_s - 1.0
        while len(self._gyro) > 2 and self._gyro[0][0] < horizon:
            self._gyro.popleft()

    def _gyro_cum_at(self, stamp: float) -> float | None:
        g = self._gyro
        if not g or stamp < g[0][0] - 0.05 or stamp > g[-1][0] + 0.05:
            return None
        stamps = [p[0] for p in g]
        values = [p[1] for p in g]
        return float(np.interp(stamp, stamps, values))

    def _check_consistency(self, stamp: float, yaw: float) -> None:
        prev, self._last_emitted = self._last_emitted, (stamp, yaw)
        if prev is None:
            return
        g0 = self._gyro_cum_at(prev[0])
        g1 = self._gyro_cum_at(stamp)
        if g0 is None or g1 is None or not 0.0 < stamp - prev[0] <= 1.0:
            return
        d_gyro = g1 - g0
        err = wrap_pi((yaw - prev[1]) - d_gyro)
        self._pairs.append((stamp, err, abs(d_gyro)))
        while self._pairs and self._pairs[0][0] < stamp - self.consistency_window_s:
            self._pairs.popleft()
        if len(self._pairs) < self.min_increments:
            return
        turn = sum(p[2] for p in self._pairs)
        disagreement = abs(sum(p[1] for p in self._pairs))
        self.inconsistency_deg = float(np.degrees(disagreement))
        if turn >= np.radians(self.min_turn_deg):
            if disagreement >= max(np.radians(self.gyro_gate_deg), self.gyro_gate_turn_fraction * turn):
                self.inconsistent = True
                self.tripped = True
            else:
                self.inconsistent = False
                self.validated = True  # one full turn agreed: arm the channel
        else:
            self.inconsistent = False

    def push(self, msg) -> HeadingMeasurement | None:
        """One MagneticField message -> a HeadingMeasurement or None (decimated/rejected)."""
        stamp = _stamp_of(msg)
        if stamp < self._next_emit:
            return None
        f = msg.magnetic_field
        raw = np.array([f.x, f.y, f.z], dtype=np.float64)
        if not np.all(np.isfinite(raw)):
            return None
        corrected = self.calibration.correct(raw)
        norm = float(np.linalg.norm(corrected))
        ref = float(self.calibration.field_norm) or 1.0
        if not (self.min_field_fraction * ref <= norm <= self.max_field_fraction * ref):
            # A field far from the calibrated magnitude is a local disturbance
            # (steel plate, power cable): drop it here; persistent bad
            # HEADINGS with plausible magnitude are the health monitor's job.
            self.rejected_field += 1
            return None
        self._next_emit = stamp + 1.0 / self.rate_hz if self.rate_hz > 0 else stamp
        yaw = self.calibration.planar_heading(raw)
        yaw = wrap_pi(
            yaw
            + np.radians(self.mount_yaw_deg)
            + np.radians(self.declination_deg)
            - self.map_yaw0
        )
        self._check_consistency(stamp, yaw)
        return HeadingMeasurement(stamp, self.channel, yaw=yaw, sigma=float(np.radians(self.sigma_deg)))


# ---------------------------------------------------------------- speedometer
@dataclass
class SpeedometerAdapter:
    """Speed topic -> :class:`SpeedMeasurement` on channel ``speedometer``.

    ``kind``:
    - ``float32``: ``std_msgs/Float32`` (or Float64: anything with ``.data``)
      in m/s. Float32 has NO header, so the measurement is stamped with the
      receive time the caller passes (the node clock) - document the
      publisher's latency if it matters.
    - ``twist_stamped``: ``geometry_msgs/TwistStamped``; forward speed is
      ``twist.linear.x`` [m/s] in base_link, stamped from the header.

    ``sigma_mps`` is the 1-sigma speed noise. Negative speeds pass through
    (reversing); NaN is dropped.
    """

    kind: str = "float32"  # float32 | twist_stamped
    sigma_mps: float = 0.15
    channel: str = "speedometer"

    def push(self, msg, receive_stamp: float) -> SpeedMeasurement | None:
        if self.kind == "twist_stamped":
            stamp = _stamp_of(msg)
            speed = float(msg.twist.linear.x)
        elif self.kind == "float32":
            stamp = float(receive_stamp)
            speed = float(msg.data)
        else:
            raise ValueError(f"speedometer type {self.kind!r} is not float32 | twist_stamped")
        if not np.isfinite(speed):
            return None
        return SpeedMeasurement(stamp, self.channel, speed=speed, sigma=float(self.sigma_mps))


# ----------------------------------------------------------------- wheel ticks
@dataclass
class WheelTickAdapter:
    """Encoder tick counts -> :class:`SpeedMeasurement` on channel ``wheel``.

    ``kind``:
    - ``int64``: ``std_msgs/Int64`` (or Int32/UInt: anything with ``.data``)
      holding the CUMULATIVE tick count. No header -> stamped with the
      receive time the caller passes.
    - ``joint_state``: ``sensor_msgs/JointState``; the position [rad or
      ticks - set ``ticks_per_metre`` accordingly] of joint ``joint`` (first
      joint when empty), stamped from the header.

    Speed = delta_ticks / ticks_per_metre / delta_t over consecutive
    messages. The sigma is QUANTIZATION-AWARE: a tick counter read over
    ``dt`` is exact to +-1 tick at each end, so

        sigma = sqrt(sigma_floor^2 + (2 / ticks_per_metre / dt)^2)

    - at low speed / high rate the +-2-tick band dominates and the EKF
    weights the measurement accordingly instead of trusting a heavily
    quantized 0.0 m/s. Counter wrap/reset (a negative jump of more than
    ``reset_jump_ticks``) re-anchors instead of emitting a huge negative
    speed. Increments spanning more than ``max_gap_s`` re-anchor too (the
    health monitor sees the gap as a timeout, like the odometry converter).
    """

    ticks_per_metre: float
    kind: str = "int64"  # int64 | joint_state
    joint: str = ""
    sigma_floor_mps: float = 0.05
    max_gap_s: float = 1.0
    reset_jump_ticks: float = 1e6
    channel: str = "wheel"

    def __post_init__(self) -> None:
        if self.ticks_per_metre <= 0:
            raise ValueError("ticks_per_metre must be > 0")
        self._last: tuple[float, float] | None = None  # (stamp, ticks)

    def _extract(self, msg, receive_stamp: float) -> tuple[float, float]:
        if self.kind == "joint_state":
            stamp = _stamp_of(msg)
            names = list(getattr(msg, "name", []) or [])
            positions = list(getattr(msg, "position", []) or [])
            if not positions:
                raise ValueError("JointState message has no position array")
            index = 0
            if self.joint:
                if self.joint not in names:
                    raise ValueError(f"joint {self.joint!r} not in JointState names {names}")
                index = names.index(self.joint)
            return stamp, float(positions[index])
        if self.kind == "int64":
            return float(receive_stamp), float(msg.data)
        raise ValueError(f"wheel_ticks type {self.kind!r} is not int64 | joint_state")

    def push(self, msg, receive_stamp: float) -> SpeedMeasurement | None:
        stamp, ticks = self._extract(msg, receive_stamp)
        if not (np.isfinite(stamp) and np.isfinite(ticks)):
            return None
        last, self._last = self._last, (stamp, ticks)
        if last is None:
            return None
        t0, n0 = last
        dt = stamp - t0
        if not 0.0 < dt <= self.max_gap_s:
            return None  # re-anchored (gap or non-monotonic time)
        dn = ticks - n0
        if dn < 0 and abs(dn) > self.reset_jump_ticks:
            return None  # counter reset/wrap: re-anchor silently (reversing is a small negative dn)
        speed = dn / self.ticks_per_metre / dt
        sigma = float(np.hypot(self.sigma_floor_mps, 2.0 / self.ticks_per_metre / dt))
        return SpeedMeasurement(stamp, self.channel, speed=float(speed), sigma=sigma)
