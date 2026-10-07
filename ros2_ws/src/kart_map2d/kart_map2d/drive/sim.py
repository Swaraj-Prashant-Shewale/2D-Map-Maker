"""Kinematic-bicycle kart simulator for closed-loop tests of the drive stack.

NOT part of the kart deployment (nothing in ``drive_node`` imports it): the
validation script, the unit tests and the real-ROS closed-loop test drive the
controller and the safety layer against this model.

Model (rear-axle kinematic bicycle, the standard low-speed kart model):
``x_r' = v cos(yaw)``, ``y_r' = v sin(yaw)``, ``yaw' = v tan(delta) / L`` for
the rear-axle point; the reported pose is base_link, ``rear_axle_x_m`` behind
which the rear axle sits (URDF: axles at +-0.489 m). Actuators are
first-order lags with limits: steering ``tau_steer`` (0.1 s), a slew limit
and the +-0.52 rad stop; speed ``tau_speed`` with accel / brake limits. No
tyre slip: at <= 3 m/s on a kart the kinematic model is the usual stand-in.

Also here: synthetic IMU / wheel-odometry / planar-LiDAR measurements of the
simulated motion (the real-ROS test publishes these), correlated pose-noise
models, and :func:`run_closed_loop`, which scores a controller on a path
(cross-track RMS / max, heading error, speed tracking, steering saturation,
compute time per step).

Frames and units: map frame, metres, radians (yaw CCW from +x), seconds.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from kart_map2d.drive.controller import (
    KART_MAX_STEER_RAD,
    KART_REAR_AXLE_X_M,
    KART_WHEELBASE_M,
    ControllerConfig,
    PurePursuitController,
    RacingPath,
    wrap_angle,
)

__all__ = [
    "GaussMarkovNoise",
    "KartSim",
    "KartSimConfig",
    "RasterWorld",
    "planar_scan",
    "run_closed_loop",
]


@dataclass
class KartSimConfig:
    wheelbase_m: float = KART_WHEELBASE_M
    rear_axle_x_m: float = KART_REAR_AXLE_X_M
    max_steer_rad: float = KART_MAX_STEER_RAD
    tau_steer_s: float = 0.1  # steering actuator time constant
    max_steer_rate_radps: float = 3.0  # steering servo slew limit
    tau_speed_s: float = 0.25  # drive / brake response
    max_accel_mps2: float = 2.0
    max_decel_mps2: float = 5.0
    substep_s: float = 0.002  # integration step


class KartSim:
    """State: base_link pose (x, y, yaw), speed v, road-wheel angle, time."""

    def __init__(self, x: float = 0.0, y: float = 0.0, yaw: float = 0.0, v: float = 0.0, config: KartSimConfig | None = None) -> None:
        self.config = config or KartSimConfig()
        self.x, self.y, self.yaw, self.v = float(x), float(y), float(yaw), float(v)
        self.steer = 0.0
        self.t = 0.0
        self.yaw_rate = 0.0
        self.accel = 0.0
        self.distance = 0.0

    @property
    def pose(self) -> NDArray[np.float64]:
        return np.array([self.x, self.y, self.yaw])

    def step(self, steer_cmd: float, speed_cmd: float, dt: float) -> None:
        """Advance ``dt`` seconds holding the command (actuator lags inside)."""
        c = self.config
        n = max(int(math.ceil(dt / c.substep_s)), 1)
        h = dt / n
        steer_cmd = float(np.clip(steer_cmd, -c.max_steer_rad, c.max_steer_rad))
        for _ in range(n):
            # Steering: first-order lag, slew-limited, hard stop.
            rate = (steer_cmd - self.steer) / max(c.tau_steer_s, 1e-6)
            rate = float(np.clip(rate, -c.max_steer_rate_radps, c.max_steer_rate_radps))
            self.steer = float(np.clip(self.steer + rate * h, -c.max_steer_rad, c.max_steer_rad))
            # Speed: first-order lag with accel / brake limits, no reversing.
            acc = (speed_cmd - self.v) / max(c.tau_speed_s, 1e-6)
            acc = float(np.clip(acc, -c.max_decel_mps2, c.max_accel_mps2))
            v_new = max(self.v + acc * h, 0.0) if speed_cmd >= 0.0 else self.v + acc * h
            self.accel = (v_new - self.v) / h
            # Rear-axle kinematic bicycle, midpoint integration.
            v_mid = 0.5 * (self.v + v_new)
            r = v_mid * math.tan(self.steer) / c.wheelbase_m
            yaw_mid = self.yaw + 0.5 * r * h
            rx = self.x + c.rear_axle_x_m * math.cos(self.yaw)
            ry = self.y + c.rear_axle_x_m * math.sin(self.yaw)
            rx += v_mid * h * math.cos(yaw_mid)
            ry += v_mid * h * math.sin(yaw_mid)
            self.yaw = wrap_angle(self.yaw + r * h)
            self.x = rx - c.rear_axle_x_m * math.cos(self.yaw)
            self.y = ry - c.rear_axle_x_m * math.sin(self.yaw)
            self.v = v_new
            self.yaw_rate = r
            self.distance += abs(v_mid) * h
            self.t += h

    def imu(self, rng: np.random.Generator | None = None, gyro_sigma: float = 0.003, accel_sigma: float = 0.05, gyro_bias: float = 0.0) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """(gyro, accel) in base_link as an IMU at base_link would measure them.

        With ``rng``: sensor noise plus chassis vibration that grows with speed
        (see :meth:`vibration`), as the real kart's IMU shows.
        """
        g = np.array([0.0, 0.0, self.yaw_rate + gyro_bias])
        a = np.array([self.accel, self.v * self.yaw_rate, 9.80665])
        if rng is not None:
            vib_a, vib_g = self.vibration()
            g = g + rng.normal(0.0, math.hypot(gyro_sigma, vib_g), 3)
            a = a + rng.normal(0.0, math.hypot(accel_sigma, vib_a), 3)
        return g, a

    def vibration(self) -> tuple[float, float]:
        """(accel std [m/s^2], gyro std [rad/s]) of chassis vibration at the current speed.

        The September recordings show 0.5-0.8 m/s^2 of accelerometer vibration
        while driving and ~0.03 parked; the fusion's stillness detector (ZUPT)
        depends on that difference. The ramp saturates by ~0.3 m/s: a real
        kart's chassis buzzes from the first roll of the tyres, and a slower
        ramp makes the SIM's launch look "parked" to the detector - ZUPT then
        clamps speed and acceleration through the first half second, and with
        visual odometry fused as SPEED (never position) that launch gap stays
        in the pose belief for the whole route (measured closed loop: belief
        1.0-1.2 m behind truth, obstacle stops from the phantom offset).
        """
        k = min(abs(self.v) / 0.3, 1.0)
        return 0.6 * k, 0.02 * k


class GaussMarkovNoise:
    """First-order Gauss-Markov pose noise: stationary std ``sigma``, correlation time ``tau``.

    Localization error is not white: an EKF output drifts and is corrected
    at the correction rate. ``sigma = (sx, sy, syaw)`` in [m, m, rad].
    """

    def __init__(self, sigma: tuple[float, float, float], tau_s: float = 0.5, seed: int = 0) -> None:
        self.sigma = np.asarray(sigma, dtype=np.float64)
        self.tau = float(tau_s)
        self.rng = np.random.default_rng(seed)
        self.state = self.rng.normal(0.0, 1.0, 3) * self.sigma

    def sample(self, dt: float) -> NDArray[np.float64]:
        if self.tau <= 0.0:
            return self.rng.normal(0.0, 1.0, 3) * self.sigma
        phi = math.exp(-dt / self.tau)
        self.state = phi * self.state + math.sqrt(1.0 - phi * phi) * self.rng.normal(0.0, 1.0, 3) * self.sigma
        return self.state.copy()


def planar_scan(
    pose: NDArray[np.float64],
    obstacle_xy: NDArray[np.float64],
    cell: float,
    max_range: float = 30.0,
    rays: int = 720,
    extra_boxes: list[tuple[float, float, float, float]] | None = None,
    rng: np.random.Generator | None = None,
    range_sigma: float = 0.02,
) -> NDArray[np.float64]:
    """Ray-cast a 2D LiDAR at ``pose`` against obstacle cells; (M, 2) base_link hits.

    ``obstacle_xy`` are map-frame cell centres of a ``cell``-sized grid
    (e.g. a map package's obstacle cells); ``extra_boxes`` adds axis-aligned
    boxes ``(xmin, ymin, xmax, ymax)`` (injected obstacles). Each ray returns
    its first hit within ``max_range`` (occlusion is real: a box hides the
    wall behind it).
    """
    x, y, yaw = (float(v) for v in pose)
    near = obstacle_xy[np.hypot(obstacle_xy[:, 0] - x, obstacle_xy[:, 1] - y) <= max_range + cell]
    angles = yaw + np.linspace(-math.pi, math.pi, rays, endpoint=False)
    dirs = np.column_stack([np.cos(angles), np.sin(angles)])
    best = np.full(rays, np.inf)
    if len(near):
        # Each obstacle cell as a disc of radius cell/2: ray-circle intersection.
        rel = near - np.array([x, y])
        proj = dirs @ rel.T  # (rays, cells)
        perp2 = np.sum(rel * rel, axis=1)[None, :] - proj * proj
        radius2 = (0.5 * cell) ** 2
        hit = (proj > 0.0) & (perp2 <= radius2)
        dist = np.where(hit, proj - np.sqrt(np.maximum(radius2 - perp2, 0.0)), np.inf)
        best = dist.min(axis=1)
    for box in extra_boxes or []:
        best = np.minimum(best, _ray_box(np.array([x, y]), dirs, box))
    ok = best <= max_range
    rng_m = best[ok]
    if rng is not None:
        rng_m = rng_m + rng.normal(0.0, range_sigma, len(rng_m))
    local_angles = angles[ok] - yaw
    return np.column_stack([rng_m * np.cos(local_angles), rng_m * np.sin(local_angles)])


class RasterWorld:
    """Obstacle cells rasterized once, for fast 2D ray casting (simulated LiDAR).

    ``obstacle_xy`` map-frame cell centres of a ``cell`` grid. :meth:`scan`
    marches every ray in ``step`` increments through the boolean raster
    (vectorized over rays x steps): a 720-ray, 30 m scan is a few ms, cheap
    enough to publish at 10 Hz next to a 200 Hz physics loop.
    """

    def __init__(self, obstacle_xy: NDArray[np.float64], cell: float, pad_m: float = 35.0) -> None:
        pts = np.asarray(obstacle_xy, dtype=np.float64).reshape(-1, 2)
        self.cell = float(cell)
        lo = (pts.min(axis=0) if len(pts) else np.zeros(2)) - pad_m
        hi = (pts.max(axis=0) if len(pts) else np.zeros(2)) + pad_m
        self.origin = lo
        shape = np.ceil((hi - lo) / cell).astype(int) + 1
        self.occ = np.zeros(tuple(shape), dtype=bool)
        if len(pts):
            ij = np.floor((pts - lo) / cell).astype(int)
            self.occ[ij[:, 0], ij[:, 1]] = True

    def scan(
        self,
        pose: NDArray[np.float64],
        max_range: float = 30.0,
        rays: int = 720,
        step: float = 0.05,
        extra_boxes: list[tuple[float, float, float, float]] | None = None,
        rng: np.random.Generator | None = None,
        range_sigma: float = 0.02,
    ) -> NDArray[np.float64]:
        """(M, 2) base_link hits of a 2D LiDAR at ``pose``; boxes occlude like cells."""
        x, y, yaw = (float(v) for v in pose)
        ang = yaw + np.linspace(-math.pi, math.pi, rays, endpoint=False)
        d = np.arange(step, max_range + step, step)
        px = x + np.cos(ang)[:, None] * d[None, :]
        py = y + np.sin(ang)[:, None] * d[None, :]
        ix = np.floor((px - self.origin[0]) / self.cell).astype(int)
        iy = np.floor((py - self.origin[1]) / self.cell).astype(int)
        ok = (ix >= 0) & (ix < self.occ.shape[0]) & (iy >= 0) & (iy < self.occ.shape[1])
        hit = np.zeros(px.shape, dtype=bool)
        hit[ok] = self.occ[ix[ok], iy[ok]]
        for bx0, by0, bx1, by1 in extra_boxes or []:
            hit |= (px >= bx0) & (px <= bx1) & (py >= by0) & (py <= by1)
        any_hit = hit.any(axis=1)
        first = np.argmax(hit, axis=1)
        r = d[first][any_hit]
        if rng is not None:
            r = r + rng.normal(0.0, range_sigma, len(r))
        a = ang[any_hit] - yaw
        return np.column_stack([r * np.cos(a), r * np.sin(a)])


def _ray_box(origin: NDArray[np.float64], dirs: NDArray[np.float64], box: tuple[float, float, float, float]) -> NDArray[np.float64]:
    """Distance along each unit ray to an axis-aligned box (inf when missed)."""
    lo = np.array(box[:2], dtype=np.float64)
    hi = np.array(box[2:], dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / dirs
        t1 = (lo - origin) * inv
        t2 = (hi - origin) * inv
    tmin = np.nanmax(np.minimum(t1, t2), axis=1)
    tmax = np.nanmin(np.maximum(t1, t2), axis=1)
    hit = (tmax >= np.maximum(tmin, 0.0))
    return np.where(hit, np.maximum(tmin, 0.0), np.inf)


@dataclass
class ClosedLoopResult:
    """Scores of one closed-loop run (true kart state vs the path)."""

    t: NDArray[np.float64]
    xyyaw: NDArray[np.float64]
    v: NDArray[np.float64]
    steer_cmd: NDArray[np.float64]
    speed_cmd: NDArray[np.float64]
    v_ref: NDArray[np.float64]
    cross_track: NDArray[np.float64]
    heading_error: NDArray[np.float64]
    step_us: NDArray[np.float64]
    finished: bool
    metrics: dict = field(default_factory=dict)


def run_closed_loop(
    path: RacingPath,
    config: ControllerConfig | None = None,
    sim_config: KartSimConfig | None = None,
    rate_hz: float = 50.0,
    duration_s: float | None = None,
    pose_noise: GaussMarkovNoise | None = None,
    pose_error_fn=None,
    start_pose: NDArray[np.float64] | None = None,
    settle_s: float = 0.0,
) -> ClosedLoopResult:
    """Drive ``path`` with pure pursuit on the kinematic kart; score the TRUE motion.

    The controller sees the true pose corrupted by ``pose_noise`` (Gauss-Markov)
    and/or ``pose_error_fn(t, true_pose) -> [dx, dy, dyaw]`` (map frame; e.g. a
    measured localization-error time series). Cross-track and heading errors
    are measured on the TRUE pose with an independent progress tracker, so a
    controller that cheats with its own projection cannot hide errors.
    ``settle_s`` excludes the first seconds (start transient) from the scores.
    Stops when the controller reports the route finished and the kart is at
    rest, or after ``duration_s`` (default: long enough for the route).
    """
    config = config or ControllerConfig()
    ctrl = PurePursuitController(path, config)
    x0 = np.asarray(start_pose if start_pose is not None else [*path.point_at(0.0), path.heading_at(0.0)], dtype=np.float64)
    sim = KartSim(*x0, config=sim_config)
    dt = 1.0 / rate_hz
    if duration_s is None:
        laps = config.laps if (path.closed and config.laps > 0) else 1
        duration_s = laps * path.length / max(min(config.max_speed_mps, 1e9) * 0.5, 0.2) + 20.0
    rows = []
    s_true = path.project(x0[:2]).s
    s_unwrapped = s_true
    finished = False
    n_steps = int(duration_s * rate_hz)
    for _ in range(n_steps):
        true_pose = sim.pose
        est = true_pose.copy()
        if pose_noise is not None:
            est = est + pose_noise.sample(dt)
        if pose_error_fn is not None:
            est = est + np.asarray(pose_error_fn(sim.t, true_pose), dtype=np.float64)
        est[2] = wrap_angle(est[2])
        t0 = time.perf_counter()
        out = ctrl.step(est, sim.v, dt, yaw_rate=sim.yaw_rate)
        step_us = (time.perf_counter() - t0) * 1e6
        ctrl.commanded(out.speed)
        # Independent truth scoring (own progress window).
        proj = path.project(true_pose[:2], s_hint=path.wrap_s(s_unwrapped), back=2.0, ahead=6.0)
        ds = proj.s - path.wrap_s(s_unwrapped)
        if path.closed:
            ds = (ds + 0.5 * path.length) % path.length - 0.5 * path.length
        s_unwrapped += ds
        v_ref = ctrl.profile_speed_at(s_unwrapped) if out.ok else 0.0
        rows.append(
            (sim.t, *true_pose, sim.v, out.steer, out.speed, out.v_ref, proj.lateral, wrap_angle(true_pose[2] - proj.heading), step_us, v_ref)
        )
        sim.step(out.steer, out.speed, dt)
        if out.finished and sim.v < 0.02:
            finished = True
            break
    a = np.array(rows, dtype=np.float64)
    res = ClosedLoopResult(
        t=a[:, 0],
        xyyaw=a[:, 1:4],
        v=a[:, 4],
        steer_cmd=a[:, 5],
        speed_cmd=a[:, 6],
        v_ref=a[:, 7],
        cross_track=a[:, 8],
        heading_error=a[:, 9],
        step_us=a[:, 10],
        finished=finished,
    )
    moving = (a[:, 0] >= settle_s) & (a[:, 4] > 0.2)
    ct = np.abs(a[moving, 8]) if moving.any() else np.zeros(1)
    he = np.degrees(np.abs(a[moving, 9])) if moving.any() else np.zeros(1)
    # Speed tracking: true speed vs the controller's commanded speed (the
    # rate-limited profile) while cruising, i.e. how well the command is met.
    sp = a[moving, 4] - a[moving, 6] if moving.any() else np.zeros(1)
    # ... and against the un-ramped reference (profile x scale, caps, end braking).
    sp_ref = a[moving, 4] - a[moving, 7] if moving.any() else np.zeros(1)
    sat = np.abs(a[moving, 5]) >= (config.max_steer_rad - 1e-6) if moving.any() else np.zeros(1, bool)
    res.metrics = {
        "duration_s": round(float(a[-1, 0]), 2),
        "distance_m": round(float(sim.distance), 2),
        "finished": bool(finished),
        "cross_track_rms_m": round(float(np.sqrt(np.mean(ct**2))), 4),
        "cross_track_max_m": round(float(ct.max()), 4),
        "cross_track_p95_m": round(float(np.percentile(ct, 95)), 4),
        "heading_error_rms_deg": round(float(np.sqrt(np.mean(he**2))), 3),
        "heading_error_max_deg": round(float(he.max()), 3),
        "speed_error_rms_mps": round(float(np.sqrt(np.mean(sp**2))), 4),
        "speed_ref_error_rms_mps": round(float(np.sqrt(np.mean(sp_ref**2))), 4),
        "mean_speed_mps": round(float(a[moving, 4].mean()) if moving.any() else 0.0, 3),
        "steer_saturation_fraction": round(float(np.mean(sat)), 4),
        "step_us_p50": round(float(np.percentile(a[:, 10], 50)), 1),
        "step_us_p95": round(float(np.percentile(a[:, 10], 95)), 1),
        "step_us_max": round(float(a[:, 10].max()), 1),
        "steps": int(len(a)),
    }
    return res
