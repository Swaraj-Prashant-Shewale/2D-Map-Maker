"""Planar pose EKF in which every sensor, including the IMU, is optional.

Design choice that makes the filter degrade gracefully: the process model is a
kinematic constant-turn-rate-and-acceleration model, and *all* sensors enter as
measurements. Conventional filters feed the IMU in as the process input, which
makes the IMU a single point of failure. Here, losing the IMU only means the
yaw rate and acceleration states are observed by the odometry sources instead.

State ``x`` (8):
    0 px, 1 py   position of base_link in the map frame [m]
    2 yaw        heading in the map frame [rad]
    3 v          forward speed [m/s]
    4 r          yaw rate [rad/s]
    5 a          forward acceleration [m/s^2]
    6 bg         gyro z bias [rad/s]
    7 ba         accelerometer x bias, absorbs slope-induced gravity too [m/s^2]

Every update goes through a chi-square innovation gate whose outcome is
reported to the channel's ``ChannelHealth``; the health monitor decides when a
sensor is dropped and when it is allowed back.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.stats import chi2

from kart_map2d.core.health import HealthMonitor
from kart_map2d.core.measurements import (
    HeadingMeasurement,
    ImuSample,
    Measurement,
    PositionFix,
    RelativeMotion,
    SpeedMeasurement,
    VelocityFix,
    ZeroVelocity,
)

PX, PY, YAW, V, R, A, BG, BA = range(8)
N = 8


def wrap(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


@dataclass
class EkfConfig:
    # Continuous-time process noise densities (per sqrt(s)).
    sigma_position: float = 0.05  # unmodelled lateral slip etc. [m/sqrt(s)]
    sigma_yaw: float = 0.002
    sigma_speed: float = 0.05
    sigma_yaw_accel: float = 1.5  # [rad/s^2/sqrt(s)] how fast yaw rate may change
    sigma_jerk: float = 3.0  # [m/s^3/sqrt(s)]
    sigma_gyro_bias: float = 2e-4
    sigma_accel_bias: float = 2e-3

    sigma_gyro: float = 0.02  # gyro z measurement noise after decimation [rad/s]
    # Chassis vibration dominates: 0.5-0.8 m/s^2 std at 100 Hz while driving.
    sigma_accel: float = 0.8

    gate_probability: float = 0.999
    reanchor_after_s: float = 1.5  # speed unobserved this long -> trust the next speed source
    max_lateness_s: float = 0.25  # older measurements are dropped
    max_speed: float = 40.0

    initial_sigma_yaw: float = 1e-3  # map frame is defined by the start pose
    initial_sigma_speed: float = 0.5
    initial_sigma_gyro_bias: float = 0.02
    initial_sigma_accel_bias: float = 0.5


@dataclass
class EkfSnapshot:
    stamp: float
    x: NDArray[np.float64]
    p_diag: NDArray[np.float64]


@dataclass
class UpdateLog:
    stamp: float
    channel: str
    nis: float
    accepted: bool


class PlanarEkf:
    def __init__(self, health: HealthMonitor, config: EkfConfig | None = None) -> None:
        self.config = config or EkfConfig()
        self.health = health
        self.x = np.zeros(N)
        self.P = np.diag(
            [
                1e-4,
                1e-4,
                self.config.initial_sigma_yaw**2,
                self.config.initial_sigma_speed**2,
                0.2**2,
                1.0,
                self.config.initial_sigma_gyro_bias**2,
                self.config.initial_sigma_accel_bias**2,
            ]
        )
        self.stamp: float | None = None
        self.updates: list[UpdateLog] = []
        self.late_drops = 0
        self.last_motion_stamp = -np.inf  # last speed-type evidence
        self.last_turn_stamp = -np.inf  # last yaw-rate-type evidence
        self.reanchors: list[tuple[float, str]] = []
        self._gates = {dim: chi2.ppf(self.config.gate_probability, dim) for dim in (1, 2, 3)}

    # ------------------------------------------------------------------ predict
    def predict(self, stamp: float) -> None:
        if self.stamp is None:
            self.stamp = stamp
            return
        dt = stamp - self.stamp
        if dt <= 0.0:
            return
        # Long gaps are integrated in bounded steps so the linearisation holds.
        steps = int(np.ceil(dt / 0.05))
        h = dt / steps
        for _ in range(steps):
            self._predict_step(h)
        self.stamp = stamp

    def _predict_step(self, dt: float) -> None:
        c = self.config
        x = self.x
        yaw_mid = x[YAW] + 0.5 * x[R] * dt
        cos_m, sin_m = np.cos(yaw_mid), np.sin(yaw_mid)
        travel = x[V] * dt + 0.5 * x[A] * dt * dt
        F = np.eye(N)
        F[PX, YAW] = -travel * sin_m
        F[PX, V] = cos_m * dt
        F[PX, R] = -travel * sin_m * 0.5 * dt
        F[PX, A] = cos_m * 0.5 * dt * dt
        F[PY, YAW] = travel * cos_m
        F[PY, V] = sin_m * dt
        F[PY, R] = travel * cos_m * 0.5 * dt
        F[PY, A] = sin_m * 0.5 * dt * dt
        F[YAW, R] = dt
        F[V, A] = dt

        x[PX] += travel * cos_m
        x[PY] += travel * sin_m
        x[YAW] = wrap(x[YAW] + x[R] * dt)
        x[V] = float(np.clip(x[V] + x[A] * dt, -c.max_speed, c.max_speed))

        q = np.array(
            [
                c.sigma_position,
                c.sigma_position,
                c.sigma_yaw,
                c.sigma_speed,
                c.sigma_yaw_accel,
                c.sigma_jerk,
                c.sigma_gyro_bias,
                c.sigma_accel_bias,
            ]
        )
        self.P = F @ self.P @ F.T + np.diag(q**2 * dt)

    # ------------------------------------------------------------------ update
    def process(self, m: Measurement) -> bool:
        """Predict to ``m.stamp`` and apply ``m``. Returns True if accepted."""
        health = self.health.channel(m.channel)
        if self.stamp is not None and m.stamp < self.stamp - self.config.max_lateness_s:
            self.late_drops += 1
            return False
        if not health.usable(m.stamp):
            return False
        if self.stamp is None or m.stamp > self.stamp:
            self.predict(m.stamp)

        if isinstance(m, ImuSample):
            return self._imu(m)
        if isinstance(m, SpeedMeasurement):
            H = self._row(V)
            ok = self._update(np.array([m.speed]), self.x[[V]], H, np.array([[m.sigma**2]]), m)
            if ok:
                self.last_motion_stamp = m.stamp
            return ok
        if isinstance(m, RelativeMotion):
            return self._relative(m)
        if isinstance(m, PositionFix):
            H = np.zeros((2, N))
            H[0, PX] = H[1, PY] = 1.0
            return self._update(m.position, self.x[[PX, PY]], H, np.eye(2) * m.sigma**2, m)
        if isinstance(m, VelocityFix):
            yaw, v = self.x[YAW], self.x[V]
            predicted = np.array([v * np.cos(yaw), v * np.sin(yaw)])
            H = np.zeros((2, N))
            H[0, YAW], H[0, V] = -v * np.sin(yaw), np.cos(yaw)
            H[1, YAW], H[1, V] = v * np.cos(yaw), np.sin(yaw)
            ok = self._update(m.velocity, predicted, H, np.eye(2) * m.sigma**2, m)
            if ok:
                self.last_motion_stamp = m.stamp
            return ok
        if isinstance(m, HeadingMeasurement):
            return self._update(
                np.array([m.yaw]), self.x[[YAW]], self._row(YAW), np.array([[m.sigma**2]]), m, angular=(0,)
            )
        if isinstance(m, ZeroVelocity):
            H = np.zeros((3, N))
            H[0, V] = H[1, R] = H[2, A] = 1.0
            Rm = np.diag([m.sigma_speed**2, m.sigma_yaw_rate**2, 0.05**2])
            ok = self._update(np.zeros(3), self.x[[V, R, A]], H, Rm, m)
            if ok:
                self.last_motion_stamp = m.stamp
                self.last_turn_stamp = m.stamp
            return ok
        raise TypeError(f"Unsupported measurement {type(m).__name__}")

    def _imu(self, m: ImuSample) -> bool:
        c = self.config
        # Two scalar updates so that a saturated accelerometer does not take the
        # gyro down with it (and vice versa) inside the innovation gate.
        H = np.zeros((1, N))
        H[0, R] = H[0, BG] = 1.0
        ok_gyro = self._update(
            np.array([m.gyro[2]]), np.array([self.x[R] + self.x[BG]]), H, np.array([[c.sigma_gyro**2]]), m, report=False
        )
        H = np.zeros((1, N))
        H[0, A] = H[0, BA] = 1.0
        ok_accel = self._update(
            np.array([m.accel[0]]), np.array([self.x[A] + self.x[BA]]), H, np.array([[c.sigma_accel**2]]), m, report=False
        )
        ok = ok_gyro or ok_accel
        self.health.channel(m.channel).report(ok)
        if ok_gyro:
            self.last_turn_stamp = m.stamp
        return ok

    def _relative(self, m: RelativeMotion) -> bool:
        dt = m.stamp - m.start_stamp
        if dt <= 1e-3:
            return False
        distance = float(np.hypot(m.delta_forward, m.delta_left))
        speed = np.copysign(distance, m.delta_forward) / dt
        yaw_rate = m.delta_yaw / dt
        quality = max(m.quality, 0.05)
        sigma_v = float(np.hypot(m.sigma_translation * distance, np.sqrt(2.0) * m.sigma_position)) / dt / quality
        sigma_r = (m.sigma_yaw / dt) / quality
        # The increment measures the *average* speed over [start, stamp]; with
        # camera frames up to ~1 s apart that differs from the current speed by
        # a*dt/2, so model it rather than ignore it.
        H = np.zeros((2, N))
        H[0, V] = 1.0
        H[0, A] = -0.5 * dt
        H[1, R] = 1.0
        ok = self._update(
            np.array([speed, yaw_rate]),
            np.array([self.x[V] - 0.5 * dt * self.x[A], self.x[R]]),
            H,
            np.diag([sigma_v**2, sigma_r**2]),
            m,
        )
        if ok:
            self.last_motion_stamp = m.stamp
            self.last_turn_stamp = m.stamp
        return ok

    @staticmethod
    def _row(index: int) -> NDArray[np.float64]:
        H = np.zeros((1, N))
        H[0, index] = 1.0
        return H

    def _update(
        self,
        z: NDArray[np.float64],
        predicted: NDArray[np.float64],
        H: NDArray[np.float64],
        Rm: NDArray[np.float64],
        m: Measurement,
        angular: tuple[int, ...] = (),
        report: bool = True,
    ) -> bool:
        y = np.asarray(z, dtype=np.float64) - predicted
        for index in angular:
            y[index] = wrap(y[index])
        reanchor = self._needs_reanchor(m)
        if reanchor:
            # Nothing has observed speed for a while, so the filter's own speed
            # is extrapolation; a gate would make it reject the very sensor
            # that could correct it. Widen the speed states and accept.
            self.P[V, V] += 4.0
            self.P[A, A] += 4.0
            self.reanchors.append((m.stamp, m.channel))
        S = H @ self.P @ H.T + Rm
        S_inv = np.linalg.inv(S)
        nis = float(y @ S_inv @ y)
        accepted = reanchor or nis <= self._gates[len(y)]
        self.updates.append(UpdateLog(m.stamp, m.channel, nis, accepted))
        if report:
            self.health.channel(m.channel).report(accepted)
        if not accepted:
            return False
        K = self.P @ H.T @ S_inv
        self.x = self.x + K @ y
        self.x[YAW] = wrap(self.x[YAW])
        # Joseph form keeps P symmetric positive-definite over long runs.
        I_KH = np.eye(N) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ Rm @ K.T
        return True

    # ------------------------------------------------- cross-check interventions
    def deweight_yaw_rate(self, sigma_rate: float = 0.3, sigma_bias: float = 0.01) -> None:
        """Discard confidence in the yaw-rate and gyro-bias states.

        Called when the sensor that was holding those states (the gyro) is
        blamed by the cross-sensor consistency check: their tight covariance
        is then false confidence, and keeping it would make the innovation
        gate reject the surviving odometric yaw sources at the switch.
        ``sigma_rate`` [rad/s] and ``sigma_bias`` [rad/s] are added in
        quadrature to the current state uncertainty.
        """
        self.P[R, R] += sigma_rate**2
        self.P[BG, BG] += sigma_bias**2

    def learn_gyro_bias(self, delta: float, sigma: float) -> None:
        """Move the gyro-bias state by ``delta`` [rad/s] on external evidence.

        The consistency monitor measures the gyro's excess rate against the
        odometric channels directly, which observes the bias far faster than
        the filter's own slow bias random walk can. Teaching the filter that
        rate at blame time means a re-admitted gyro comes back calibrated
        instead of re-poisoning the yaw state. ``sigma`` [rad/s] is added to
        the bias variance.
        """
        self.x[BG] += delta
        self.P[BG, BG] += sigma**2

    def correct_yaw(self, delta: float, sigma: float) -> None:
        """Remove ``delta`` [rad] of heading absorbed from a faulty gyro.

        A slow gyro fault corrupts the map-frame yaw for as long as it goes
        undetected; the consistency monitor measures that accumulated excess
        (its own yaw increments vs the odometric ones) and takes it back out
        here. ``sigma`` [rad] is the 1-sigma uncertainty of the correction,
        added to the yaw variance so the correction stays honest.
        """
        self.x[YAW] = wrap(self.x[YAW] - delta)
        self.P[YAW, YAW] += sigma**2

    def _needs_reanchor(self, m: Measurement) -> bool:
        motion = isinstance(m, (SpeedMeasurement, RelativeMotion, VelocityFix, ZeroVelocity))
        return motion and m.stamp - self.last_motion_stamp > self.config.reanchor_after_s

    # ------------------------------------------------------------------ output
    def snapshot(self) -> EkfSnapshot:
        return EkfSnapshot(self.stamp or 0.0, self.x.copy(), np.diag(self.P).copy())

    def motion_supported(self, now: float, max_age_s: float = 1.0) -> bool:
        """True while speed and turn rate are observed, not just extrapolated."""
        return now - self.last_motion_stamp < max_age_s and now - self.last_turn_stamp < max_age_s


@dataclass
class Trajectory:
    stamp: list[float] = field(default_factory=list)
    x: list[NDArray[np.float64]] = field(default_factory=list)
    p_diag: list[NDArray[np.float64]] = field(default_factory=list)
    supported: list[bool] = field(default_factory=list)

    def append(self, snap: EkfSnapshot, supported: bool) -> None:
        self.stamp.append(snap.stamp)
        self.x.append(snap.x)
        self.p_diag.append(snap.p_diag)
        self.supported.append(supported)

    def arrays(self) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        return (
            np.asarray(self.stamp),
            np.asarray(self.x).reshape(-1, N),
            np.asarray(self.p_diag).reshape(-1, N),
            np.asarray(self.supported, dtype=bool),
        )

    def pose_at(self, t: NDArray[np.float64]) -> NDArray[np.float64]:
        """Interpolated [px, py, yaw] at times ``t``."""
        stamps, states, _, _ = self.arrays()
        yaw = np.unwrap(states[:, YAW])
        return np.column_stack(
            [np.interp(t, stamps, states[:, PX]), np.interp(t, stamps, states[:, PY]), np.interp(t, stamps, yaw)]
        )
