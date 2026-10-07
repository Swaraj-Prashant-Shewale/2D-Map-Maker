"""Pure-pursuit path following on a map package's racing line.

The drive stack LOADS the racing line (``racing_line.csv`` of a map package:
arclength, x, y, curvature, speed profile); it never re-plans it. This module
turns that line into steering and speed commands for a kinematic-bicycle kart:

* **Progress tracking, not nearest point.** The kart's position is projected
  onto the path only inside a window around the previous progress ``s``
  (``search_back_m`` / ``search_ahead_m``). An out-and-back route (sept3)
  passes the same spot twice in opposite directions ~2 m apart, and a
  global nearest-point search would jump between the legs; the window makes
  progress monotonic. The first projection is global but must agree with
  the kart's heading (``max_start_heading_deg``) and lie within
  ``max_start_offset_m`` of the path.
* **Pure pursuit from the rear axle.** The target is the path point
  ``lookahead`` metres of arclength ahead (``lookahead = base + gain * v``,
  clamped); with alpha the bearing of that point seen from the rear axle and
  ``ld`` its distance, the bicycle-model steering is
  ``delta = atan(2 * L * sin(alpha) / ld)`` (L = wheelbase), clamped to
  +-``max_steer_rad``. Positive steering turns LEFT (REP-103: y left, yaw
  counter-clockwise), the ``ackermann_msgs`` convention.
* **Speed.** The racing line's own profile (min-curvature planner, built for
  a race car: 8 m/s^2 lateral) is scaled by ``speed_scale``, capped by
  ``max_speed_mps`` and by the kart's lateral-acceleration budget over the
  local curvature, then made dynamically feasible with forward / backward
  passes at the kart's ``max_accel_mps2`` / ``max_decel_mps2`` - so the kart
  brakes BEFORE a tight bend and to a standstill at the end of an open
  route (``stop_margin_m`` short of it), or after ``laps`` laps of a closed
  one. The command is then rate-limited against the previous *commanded*
  speed (:meth:`PurePursuitController.commanded` feeds back what the safety
  layer actually let through, so a safety stop is never "undone" by the
  controller's own ramp).
* **Ends and wrap-around.** On a closed path arclength wraps; the target
  point wraps across the start. On an open path the target beyond the end is
  extrapolated along the final tangent (pure pursuit stays well defined up to
  the stop) and progress saturates at the end.

Frames and units: map frame (planar, metres, yaw CCW from +x, radians);
base_link x forward / y left; speeds m/s, curvature 1/m (CCW positive).
Kart geometry (URDF): wheelbase 0.978 m, axles at x = +-0.489 m from
base_link, steering limit 0.52 rad, wheel radius 0.15 m.

Cost: one :meth:`PurePursuitController.step` is a windowed projection plus a
few interpolations, tens of microseconds in CPython (measured by
``scripts/validate_drive.py``); no allocation grows with the path length.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "KART_MAX_STEER_RAD",
    "KART_REAR_AXLE_X_M",
    "KART_WHEELBASE_M",
    "KART_WHEEL_RADIUS_M",
    "ControlOutput",
    "ControllerConfig",
    "PathProjection",
    "PurePursuitController",
    "RacingPath",
    "wrap_angle",
]

#: Kart geometry from the URDF.
KART_WHEELBASE_M = 0.978
KART_REAR_AXLE_X_M = -0.489  # rear axle position along base_link x
KART_MAX_STEER_RAD = 0.52
KART_WHEEL_RADIUS_M = 0.15


def wrap_angle(angle: float) -> float:
    """Wrap to [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


# ------------------------------------------------------------------ the path
@dataclass(frozen=True)
class PathProjection:
    """Where a point sits relative to the path.

    ``s`` arclength of the closest path point [m] (wrapped for closed paths),
    ``lateral`` signed offset [m] (positive = the point is LEFT of the path
    direction), ``distance`` = |lateral| to the closest segment [m],
    ``heading`` path direction at ``s`` [rad].
    """

    s: float
    lateral: float
    distance: float
    heading: float
    segment: int


class RacingPath:
    """A polyline to follow: positions, arclength, curvature and speed profile.

    ``xy`` (N, 2) map-frame points; a closed path wraps from the last point
    back to the first (no repeated end point). ``s`` is recomputed from the
    geometry (the CSV's ``s_m`` column is only a cross-check).
    """

    def __init__(
        self,
        xy: NDArray[np.float64],
        speed: NDArray[np.float64] | None = None,
        curvature: NDArray[np.float64] | None = None,
        closed: bool = False,
    ) -> None:
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        if len(xy) < 2:
            raise ValueError("a path needs at least 2 points")
        if closed and np.linalg.norm(xy[0] - xy[-1]) < 1e-6:
            xy = xy[:-1]  # tolerate a repeated closing point
        keep = np.r_[True, np.linalg.norm(np.diff(xy, axis=0), axis=1) > 1e-6]
        xy = xy[keep]
        if len(xy) < 2:
            raise ValueError("a path needs at least 2 distinct points")
        self.closed = bool(closed)
        self.xy = xy
        n = len(xy)
        ends = np.roll(xy, -1, axis=0) if self.closed else xy[1:]
        starts = xy if self.closed else xy[:-1]
        self._seg_a = starts
        self._seg_d = ends - starts
        self._seg_len = np.linalg.norm(self._seg_d, axis=1)
        self._seg_s = np.r_[0.0, np.cumsum(self._seg_len)[:-1]]
        self.s = self._seg_s[:n] if self.closed else np.r_[self._seg_s, self._seg_s[-1] + self._seg_len[-1]]
        self.length = float(self._seg_s[-1] + self._seg_len[-1])
        self._heading = np.arctan2(self._seg_d[:, 1], self._seg_d[:, 0])
        # Per-segment columns for the windowed projection (contiguous 1-D
        # arrays) and Python lists for the scalar lookups: the control loop
        # calls these at 50 Hz on ~20-segment windows, where numpy call
        # overhead, not arithmetic, is the cost.
        self._nseg = len(self._seg_s)
        self._ax = np.ascontiguousarray(starts[:, 0])
        self._ay = np.ascontiguousarray(starts[:, 1])
        self._dx = np.ascontiguousarray(self._seg_d[:, 0])
        self._dy = np.ascontiguousarray(self._seg_d[:, 1])
        self._len2 = self._seg_len * self._seg_len
        self._seg_s_list = self._seg_s.tolist()
        self._seg_end_list = (self._seg_s + self._seg_len).tolist()
        self._len_list = self._seg_len.tolist()
        self._ax_list, self._ay_list = self._ax.tolist(), self._ay.tolist()
        self._dx_list, self._dy_list = self._dx.tolist(), self._dy.tolist()
        self._heading_list = self._heading.tolist()
        self.speed = (
            np.full(n, np.inf) if speed is None else np.asarray(speed, dtype=np.float64).reshape(-1)[keep][:n]
        )
        if curvature is None:
            curvature = _menger_curvature(xy, self.closed)
        else:
            curvature = np.asarray(curvature, dtype=np.float64).reshape(-1)[keep][:n]
        self.curvature = curvature

    # ------------------------------------------------------------ builders
    @classmethod
    def from_racing_line(cls, line) -> "RacingPath":
        """From :class:`kart_map2d.mapping.map_package.RacingLineData` (or any
        object with ``path``, ``speed``, ``curvature``, ``closed``)."""
        return cls(line.path, line.speed, line.curvature, bool(line.closed))

    # ------------------------------------------------------------ queries
    def wrap_s(self, s: float) -> float:
        return float(s % self.length) if self.closed else float(s)

    def _interp(self, values: NDArray[np.float64], s: float) -> float:
        if self.closed:
            s = s % self.length
            return float(np.interp(s, np.r_[self.s, self.length], np.r_[values, values[0]]))
        return float(np.interp(s, self.s, values))

    def speed_at(self, s: float) -> float:
        return self._interp(self.speed, s)

    def curvature_at(self, s: float) -> float:
        return self._interp(self.curvature, s)

    def _segment_of(self, s: float) -> int:
        """Index of the segment containing arclength ``s`` (clamped to the ends)."""
        i = bisect.bisect_right(self._seg_s_list, s) - 1
        return 0 if i < 0 else (self._nseg - 1 if i >= self._nseg else i)

    def xy_at(self, s: float) -> tuple[float, float]:
        """:meth:`point_at` as a plain ``(x, y)`` tuple (no array allocation)."""
        s = float(s)
        if self.closed:
            s = s % self.length
        # Beyond an open path's ends the first / last segment is extended
        # (t < 0 or t > 1 on it): a lookahead past the finish stays defined.
        i = self._segment_of(s)
        t = (s - self._seg_s_list[i]) / self._len_list[i]
        return self._ax_list[i] + t * self._dx_list[i], self._ay_list[i] + t * self._dy_list[i]

    def point_at(self, s: float) -> NDArray[np.float64]:
        """Map-frame point at arclength ``s``.

        Closed paths wrap. Open paths are extended along the end tangents
        beyond both ends, so a lookahead past the finish stays well defined.
        """
        return np.array(self.xy_at(s))

    def heading_at(self, s: float) -> float:
        if self.closed:
            s = s % self.length
        return self._heading_list[self._segment_of(float(s))]

    def points_at(self, s: NDArray[np.float64]) -> NDArray[np.float64]:
        """Vectorized :meth:`point_at` for an array of arclengths."""
        s = np.asarray(s, dtype=np.float64)
        if self.closed:
            s = np.mod(s, self.length)
        i = np.clip(np.searchsorted(self._seg_s, s, side="right") - 1, 0, len(self._seg_s) - 1)
        if not self.closed:
            # beyond the ends: extend along the end segments (as point_at)
            i = np.where(s <= 0.0, 0, i)
            i = np.where(s >= self.length, len(self._seg_s) - 1, i)
        t = (s - self._seg_s[i]) / self._seg_len[i]
        return self._seg_a[i] + t[:, None] * self._seg_d[i]

    def points_between(self, s0: float, s1: float, step: float = 0.25) -> NDArray[np.float64]:
        """(M, 2) points from ``s0`` to ``s1`` (wrapping / extrapolating as point_at)."""
        if s1 <= s0:
            return self.point_at(s0)[None, :]
        n = max(int(math.ceil((s1 - s0) / step)), 1)
        return self.points_at(np.linspace(s0, s1, n + 1))

    def project(
        self, xy: NDArray[np.float64], s_hint: float | None = None, back: float = 2.0, ahead: float = 6.0
    ) -> PathProjection:
        """Closest point of the path to ``xy``.

        With ``s_hint`` only segments overlapping ``[s_hint - back, s_hint +
        ahead]`` (wrapped on closed paths) are considered - the progress
        window that keeps an out-and-back route from jumping legs. Without
        it the whole path is searched.
        """
        px, py = float(xy[0]), float(xy[1])
        sel = slice(0, self._nseg) if s_hint is None else self._segments_in(s_hint - back, s_hint + ahead)
        dx, dy = self._dx[sel], self._dy[sel]
        rx = px - self._ax[sel]
        ry = py - self._ay[sel]
        t = (rx * dx + ry * dy) / self._len2[sel]
        np.minimum(np.maximum(t, 0.0, out=t), 1.0, out=t)
        ex = rx - t * dx
        ey = ry - t * dy
        dist2 = ex * ex + ey * ey
        k = int(dist2.argmin())
        seg = k + sel.start if isinstance(sel, slice) else int(sel[k])
        length = self._len_list[seg]
        cross = (self._dx_list[seg] * float(ry[k]) - self._dy_list[seg] * float(rx[k])) / length
        s = self._seg_s_list[seg] + float(t[k]) * length
        return PathProjection(self.wrap_s(s), cross, math.sqrt(float(dist2[k])), self._heading_list[seg], seg)

    def _segments_in(self, s0: float, s1: float) -> slice | NDArray[np.int64]:
        """Segments overlapping ``[s0, s1]`` (wrapped on closed paths): a slice,
        or an index array when the window wraps across s = 0."""
        nseg = self._nseg
        if not self.closed:
            lo = bisect.bisect_left(self._seg_end_list, s0)
            hi = bisect.bisect_right(self._seg_s_list, s1)
            lo = min(max(lo, 0), nseg - 1)
            return slice(lo, min(max(hi, lo + 1), nseg))
        if s1 - s0 >= self.length:
            return slice(0, nseg)
        a, b = s0 % self.length, s1 % self.length
        lo = min(bisect.bisect_left(self._seg_end_list, a), nseg - 1)
        hi = max(bisect.bisect_right(self._seg_s_list, b), 1)
        if a <= b:
            return slice(lo, min(max(hi, lo + 1), nseg))
        # the window wraps across s = 0: [lo, nseg) + [0, hi)
        return np.concatenate((np.arange(lo, nseg), np.arange(0, min(hi, nseg))))


def _menger_curvature(xy: NDArray[np.float64], closed: bool) -> NDArray[np.float64]:
    n = len(xy)
    if n < 3:
        return np.zeros(n)
    if closed:
        a, b, c = np.roll(xy, 1, axis=0), xy, np.roll(xy, -1, axis=0)
    else:
        a, b, c = xy[:-2], xy[1:-1], xy[2:]
    ab, bc, ac = b - a, c - b, c - a
    cross = ab[:, 0] * bc[:, 1] - ab[:, 1] * bc[:, 0]
    den = np.linalg.norm(ab, axis=1) * np.linalg.norm(bc, axis=1) * np.linalg.norm(ac, axis=1)
    k = np.where(den > 1e-12, 2.0 * cross / np.maximum(den, 1e-12), 0.0)
    return k if closed else np.r_[k[0], k, k[-1]]


# ------------------------------------------------------------ controller
@dataclass
class ControllerConfig:
    """Pure-pursuit and speed-profile tunables (kart defaults from the URDF)."""

    wheelbase_m: float = KART_WHEELBASE_M
    rear_axle_x_m: float = KART_REAR_AXLE_X_M
    max_steer_rad: float = KART_MAX_STEER_RAD
    #: Steering command slew limit [rad/s]; 0 = none (the actuator has its own).
    max_steer_rate_radps: float = 0.0
    #: Lookahead = base + gain * speed, clamped to [min, max] [m, s, m, m].
    lookahead_base_m: float = 0.6
    lookahead_gain_s: float = 0.5
    lookahead_min_m: float = 1.0
    lookahead_max_m: float = 4.0
    #: Multiplies the racing line's speed profile before any cap.
    speed_scale: float = 1.0
    #: Hard speed cap of the controller [m/s]; the safety layer caps again.
    max_speed_mps: float = 2.0
    #: Lateral acceleration budget [m/s^2]: v <= sqrt(a / |curvature|).
    max_lat_accel_mps2: float = 2.5
    #: Curvature used for the lateral cap is the max |kappa| within +- this [m]
    #: (single-vertex Menger spikes of the planner output are smoothed over).
    curvature_window_m: float = 1.0
    max_accel_mps2: float = 1.0
    max_decel_mps2: float = 2.0
    #: Open path: come to rest this far before its end [m].
    stop_margin_m: float = 0.3
    #: Drive / brake response time [s]: the kart keeps rolling this long at
    #: the commanded speed before the brake takes hold, so the end-of-route
    #: stop starts braking ``speed * lag`` earlier.
    actuation_lag_s: float = 0.25
    #: Closed path: stop after this many laps (0 = keep lapping).
    laps: int = 0
    #: Progress window for the projection [m] (see RacingPath.project).
    search_back_m: float = 2.0
    search_ahead_m: float = 6.0
    #: Refuse to start farther than this from the path, or pointing more than
    #: this away from its direction.
    max_start_offset_m: float = 3.0
    max_start_heading_deg: float = 60.0
    #: Predict the pose forward this long before steering [s] (actuation delay).
    latency_s: float = 0.0


@dataclass
class ControlOutput:
    """One control cycle's command and diagnostics.

    ``steer`` [rad, + = left] and ``speed`` [m/s] are the command BEFORE the
    safety layer; ``accel`` the signed acceleration the rate limit implies.
    ``cross_track`` [m, + = kart left of the path] and ``heading_error``
    [rad, kart yaw - path yaw] are measured at base_link.
    """

    steer: float
    speed: float
    accel: float
    s: float
    lap: int
    cross_track: float
    heading_error: float
    v_ref: float
    lookahead: float
    target: NDArray[np.float64]
    finished: bool
    ok: bool = True
    reason: str = ""


class PurePursuitController:
    """Follow a :class:`RacingPath` with pure pursuit + a feasible speed profile."""

    def __init__(self, path: RacingPath, config: ControllerConfig | None = None) -> None:
        self.path = path
        self.config = config or ControllerConfig()
        self._s: float | None = None  # progress (unwrapped for closed paths)
        self._s_start = 0.0  # progress when tracking began (laps count from here)
        self._speed_cmd = 0.0
        self._steer_cmd = 0.0
        self._set_profile(self._kart_profile())

    # ------------------------------------------------------------ profile
    def _set_profile(self, v: NDArray[np.float64]) -> None:
        self._profile = v
        p = self.path
        # Python lists for the per-cycle scalar interpolation (closed paths
        # repeat the first vertex at s = length).
        self._prof_s = p.s.tolist() + ([p.length] if p.closed else [])
        self._prof_v = v.tolist() + ([float(v[0])] if p.closed else [])

    def _kart_profile(self) -> NDArray[np.float64]:
        """Per-vertex feasible speed: scaled racing speed, capped, accel/brake-limited."""
        c, p = self.config, self.path
        n = len(p.xy)
        v = np.minimum(np.nan_to_num(p.speed, nan=0.0, posinf=1e9) * c.speed_scale, max(c.max_speed_mps, 0.0))
        # Lateral budget over a curvature window: planner curvature is Menger
        # per 0.5 m vertex and spikes on single kinks.
        ds = p.length / max(n if p.closed else n - 1, 1)
        half = max(int(round(c.curvature_window_m / max(ds, 1e-6))), 0)
        kabs = np.abs(p.curvature)
        if half:
            padded = np.r_[kabs[-half:], kabs, kabs[:half]] if p.closed else np.r_[np.full(half, kabs[0]), kabs, np.full(half, kabs[-1])]
            windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * half + 1)
            kabs = windows.max(axis=1)
        v = np.minimum(v, np.sqrt(c.max_lat_accel_mps2 / np.maximum(kabs, 1e-9)))
        # Forward (accel) and backward (brake) passes; closed paths sweep the wrap.
        seg = np.linalg.norm(np.diff(np.r_[p.xy, p.xy[:1]] if p.closed else p.xy, axis=0), axis=1)
        edges = n if p.closed else n - 1
        for _ in range(8 if p.closed else 1):
            for i in range(edges):
                j = (i + 1) % n
                v[j] = min(v[j], math.sqrt(v[i] ** 2 + 2.0 * c.max_accel_mps2 * seg[i]))
            for i in range(edges - 1, -1, -1):
                j = (i + 1) % n
                v[i] = min(v[i], math.sqrt(v[j] ** 2 + 2.0 * c.max_decel_mps2 * seg[i]))
        return v

    def set_speed_limits(self, max_speed_mps: float | None = None, speed_scale: float | None = None) -> None:
        """Change the speed cap / scale at runtime (e.g. a ROS parameter)."""
        if max_speed_mps is not None:
            self.config.max_speed_mps = float(max_speed_mps)
        if speed_scale is not None:
            self.config.speed_scale = float(speed_scale)
        self._set_profile(self._kart_profile())

    def profile_speed_at(self, s: float) -> float:
        """Kart-feasible profile speed at arclength ``s`` (linear between
        vertices, held beyond an open path's ends; closed paths wrap)."""
        xs, vs = self._prof_s, self._prof_v
        s = float(s)
        if self.path.closed:
            s = s % self.path.length
        if s <= xs[0]:
            return vs[0]
        if s >= xs[-1]:
            return vs[-1]
        i = bisect.bisect_right(xs, s) - 1
        return vs[i] + (s - xs[i]) / (xs[i + 1] - xs[i]) * (vs[i + 1] - vs[i])

    # ------------------------------------------------------------ state
    @property
    def progress(self) -> float | None:
        """Unwrapped arclength progress [m] (None before the first step)."""
        return self._s

    def reset(self) -> None:
        self._s = None
        self._s_start = 0.0
        self._speed_cmd = 0.0
        self._steer_cmd = 0.0

    def commanded(self, speed: float, steer: float | None = None) -> None:
        """Tell the controller what was ACTUALLY sent (after the safety layer).

        The next cycle's accel / brake ramp starts from this speed, so a
        safety stop (or a lower cap) is never ramped back up in one step.
        """
        self._speed_cmd = float(speed)
        if steer is not None:
            self._steer_cmd = float(steer)

    def _end_s(self) -> float:
        c, p = self.config, self.path
        if not p.closed:
            return p.length
        # Laps count from where tracking began (a start pose a hair behind
        # s = 0 projects onto the closing segment at s ~ length).
        return self._s_start + c.laps * p.length if c.laps > 0 else math.inf

    def localize_on_path(self, pose: NDArray[np.float64]) -> tuple[bool, str]:
        """Initialise progress from a pose (global projection + checks)."""
        c = self.config
        proj = self.path.project(pose[:2])
        herr = wrap_angle(float(pose[2]) - proj.heading)
        if proj.distance > c.max_start_offset_m:
            return False, f"kart is {proj.distance:.2f} m from the racing line (max {c.max_start_offset_m} m)"
        if abs(herr) > math.radians(c.max_start_heading_deg):
            # The nearest point may be on the opposite leg of an out-and-back:
            # try the closest point whose direction agrees with the kart.
            best = None
            for s0 in np.arange(0.0, self.path.length, 1.0):
                q = self.path.project(pose[:2], s_hint=s0, back=0.5, ahead=0.5)
                if abs(wrap_angle(float(pose[2]) - q.heading)) <= math.radians(c.max_start_heading_deg):
                    if best is None or q.distance < best.distance:
                        best = q
            if best is None or best.distance > c.max_start_offset_m:
                return False, (
                    f"kart heading differs from the racing line by {math.degrees(herr):.0f} deg "
                    f"(max {c.max_start_heading_deg:.0f})"
                )
            proj = best
        self._s = proj.s
        self._s_start = proj.s
        return True, ""

    # ------------------------------------------------------------ step
    def step(self, pose: NDArray[np.float64], speed_meas: float, dt: float, yaw_rate: float = 0.0) -> ControlOutput:
        """One control cycle for the base_link pose ``[x, y, yaw]`` (map frame).

        ``speed_meas`` [m/s] sets the lookahead; ``dt`` [s] is the time since
        the previous command (rate limits); ``yaw_rate`` [rad/s] is only used
        for the optional latency prediction.
        """
        c, path = self.config, self.path
        pose = np.asarray(pose, dtype=np.float64)
        if c.latency_s > 0.0 and speed_meas > 0.0:
            pose = _predict(pose, speed_meas, yaw_rate, c.latency_s)
        if self._s is None:
            ok, why = self.localize_on_path(pose)
            if not ok:
                return self._hold(pose, why)
        s_prev = self._s
        proj = path.project(pose, s_hint=path.wrap_s(s_prev), back=c.search_back_m, ahead=c.search_ahead_m)
        # Unwrapped progress: take the step from the previous wrapped s.
        ds = proj.s - path.wrap_s(s_prev)
        if path.closed:
            ds = (ds + 0.5 * path.length) % path.length - 0.5 * path.length
        s_now = s_prev + ds
        end_s = self._end_s()
        s_now = min(s_now, end_s)
        self._s = s_now
        lap = int((s_now - self._s_start) // path.length) if path.closed else 0

        # --- steering: pure pursuit from the rear axle
        v_abs = abs(float(speed_meas))
        lookahead = min(max(c.lookahead_base_m + c.lookahead_gain_s * v_abs, c.lookahead_min_m), c.lookahead_max_m)
        # Aim ``lookahead`` along the path from the lateral foot point; with a
        # large cross-track error aim at the circle intersection instead.
        along = math.sqrt(max(lookahead * lookahead - proj.distance * proj.distance, 0.25 * lookahead * lookahead))
        s_target = s_now + along
        if not path.closed:
            s_target = min(s_target, path.length + lookahead)  # extrapolated past the end
        tx, ty = path.xy_at(s_target)
        yaw = float(pose[2])
        cy, sy = math.cos(yaw), math.sin(yaw)
        dx = tx - (float(pose[0]) + c.rear_axle_x_m * cy)
        dy = ty - (float(pose[1]) + c.rear_axle_x_m * sy)
        bx, by = cy * dx + sy * dy, -sy * dx + cy * dy  # target in the rear-axle frame
        ld = max(math.hypot(bx, by), 1e-3)
        alpha = math.atan2(by, bx)
        steer = math.atan(2.0 * c.wheelbase_m * math.sin(alpha) / ld)
        steer = min(max(steer, -c.max_steer_rad), c.max_steer_rad)
        if c.max_steer_rate_radps > 0.0 and dt > 0.0:
            step_max = c.max_steer_rate_radps * dt
            steer = min(max(steer, self._steer_cmd - step_max), self._steer_cmd + step_max)
        self._steer_cmd = steer

        # --- speed: feasible profile, end-of-route braking, rate limits
        v_ref = self.profile_speed_at(s_now)
        remaining = end_s - s_now - (c.stop_margin_m + self._speed_cmd * c.actuation_lag_s if math.isfinite(end_s) else 0.0)
        if math.isfinite(end_s):
            v_ref = min(v_ref, math.sqrt(2.0 * c.max_decel_mps2 * max(remaining, 0.0)))
        finished = math.isfinite(end_s) and remaining <= 0.02
        if finished:
            v_ref = 0.0
        v_ref = max(min(v_ref, c.max_speed_mps), 0.0)
        prev = self._speed_cmd
        if dt > 0.0:
            speed = min(max(v_ref, prev - c.max_decel_mps2 * dt), prev + c.max_accel_mps2 * dt)
        else:
            speed = v_ref
        speed = max(speed, 0.0)
        accel = (speed - prev) / dt if dt > 0.0 else 0.0
        self._speed_cmd = speed
        return ControlOutput(
            steer=steer,
            speed=speed,
            accel=float(accel),
            s=float(s_now),
            lap=lap,
            cross_track=float(proj.lateral),
            heading_error=wrap_angle(yaw - proj.heading),
            v_ref=float(v_ref),
            lookahead=lookahead,
            target=np.array((tx, ty)),
            finished=bool(finished),
        )

    def _hold(self, pose: NDArray[np.float64], reason: str) -> ControlOutput:
        """No valid progress: command a standstill with straight wheels."""
        self._speed_cmd = 0.0
        return ControlOutput(0.0, 0.0, 0.0, float("nan"), 0, float("nan"), float("nan"), 0.0, 0.0, pose[:2].copy(), False, False, reason)

    def path_ahead(self, pose: NDArray[np.float64], distance: float, step: float = 0.25) -> NDArray[np.float64]:
        """Polyline the kart is about to drive: its position, then the path from
        the current progress to ``distance`` metres ahead (safety checks)."""
        if self._s is None:
            return np.asarray(pose[:2], dtype=np.float64)[None, :]
        s0 = self._s
        s1 = min(s0 + max(distance, 0.0), self._end_s() + 1.0) if math.isfinite(self._end_s()) else s0 + max(distance, 0.0)
        pts = self.path.points_between(s0, max(s1, s0 + step), step)
        return np.vstack([np.asarray(pose[:2], dtype=np.float64)[None, :], pts])


def _predict(pose: NDArray[np.float64], v: float, r: float, dt: float) -> NDArray[np.float64]:
    """Constant speed / turn-rate prediction of a base_link pose by ``dt``."""
    yaw_mid = pose[2] + 0.5 * r * dt
    return np.array([pose[0] + v * dt * math.cos(yaw_mid), pose[1] + v * dt * math.sin(yaw_mid), wrap_angle(pose[2] + r * dt)])
