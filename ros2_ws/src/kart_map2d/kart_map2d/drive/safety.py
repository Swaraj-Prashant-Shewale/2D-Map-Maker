"""Safety supervisor: the last word on every drive command.

Every control cycle the node asks :meth:`SafetySupervisor.update` what the
kart may do, and applies :meth:`SafetyDecision.limit` to the controller's
command. The safety layer runs AFTER the controller in the same cycle and
only ever lowers the speed, so no controller output can override a stop;
the node feeds the limited speed back into the controller
(``PurePursuitController.commanded``), so the controller's ramp restarts
from what was actually sent, never from its own wish.

Triggers (each is a named reason in the decision and the diagnostics):

==========================  =================================================  ===========
trigger                     condition                                          clears
==========================  =================================================  ===========
``estop``                   e-stop input True (std_msgs/Bool, latched)         input False + re-engage
``not_engaged``             nobody engaged the kart (``~/engage`` true)        engage
``deadman``                 heartbeat older than ``deadman_timeout_s`` (0=off) heartbeat resumes
``pose_stale``              newest localization sensor data older than         data resumes
                            ``pose_timeout_s`` (0.2 s)
``fusion_degraded``         fusion mode blind / inertial-only for longer than  mode recovers
                            ``degraded_mode_timeout_s`` (0.5 s)
``not_localized``           the map localizer has not converged                converges
``localization_rejected``   ``max_consecutive_rejects`` map fixes in a row     a fix is accepted
                            refused by the matcher, EKF gate or health monitor
``localization_uncertain``  EKF position 1-sigma above ``max_position_sigma_m``  sigma shrinks
``off_path``                the controller cannot place the kart on the racing  back on the line
                            line (too far from it / pointing away)
``obstacle``                live obstacle points inside the swept corridor     path clear for
                            (kart half-width + margin) within the stopping     ``obstacle_clear_s``
                            distance along the planned path
``max_speed``               cap only (``max_speed_mps``, 2.0 m/s default)      -
==========================  =================================================  ===========

Every trigger but ``max_speed`` makes the kart stop: the speed cap ramps
down at ``stop_decel_mps2`` (``estop_decel_mps2`` for the e-stop, and as
hard as needed - up to the e-stop deceleration - to halt short of an
obstacle); steering keeps following the path while braking. A trigger that
clears is held for ``resume_hold_s`` before motion may resume (no stop /
go chatter at a threshold); the e-stop never clears by itself: the input
must read False AND the kart must be engaged again.

Why max_speed defaults to 2.0 m/s: the September recordings (the only data
the maps and the localization were validated on) were driven at ~1.3 m/s;
2.0 m/s is ~50 % above that, keeps the stopping distance at 2 m/s^2 to
1.0 m (+0.6 m reaction at 0.3 s), and is the speed the closed-loop tests
passed at. First on-kart runs should start lower (0.5 m/s) and raise it.

Clock: every method takes ``now`` explicitly - ONE clock (the node's ROS
clock, which is sim time in a bag / simulator test) for every age and hold,
never a mix of wall and ROS time.

Frames and units: map frame metres, seconds, m/s, m/s^2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "DEGRADED_MODES",
    "ObstacleMemory",
    "SafetyConfig",
    "SafetyDecision",
    "SafetyInputs",
    "SafetySupervisor",
    "corridor_arclength",
    "obstacles_on_path",
]

#: FusionEngine.mode() strings in which the kart must not keep driving.
DEGRADED_MODES = ("blind", "inertial-only (dead reckoning)")


@dataclass
class SafetyConfig:
    max_speed_mps: float = 2.0
    pose_timeout_s: float = 0.2
    degraded_mode_timeout_s: float = 0.5
    max_consecutive_rejects: int = 3
    max_position_sigma_m: float = 1.0
    #: Heartbeat (deadman) timeout [s]; 0 disables the deadman.
    deadman_timeout_s: float = 0.0
    stop_decel_mps2: float = 2.0
    estop_decel_mps2: float = 4.0
    resume_hold_s: float = 1.0
    # Obstacle stop.
    obstacle_stop: bool = True
    kart_half_width_m: float = 0.53  # the kart is ~1.06 m wide
    lateral_margin_m: float = 0.25
    front_overhang_m: float = 0.8  # base_link to the front bumper
    reaction_time_s: float = 0.3
    stop_margin_m: float = 0.5  # stop this far short of an obstacle
    #: Also check this far beyond the stopping distance (early detection).
    lookahead_extra_m: float = 1.0
    obstacle_min_points: int = 3
    obstacle_clear_s: float = 1.0


@dataclass
class SafetyInputs:
    """What the supervisor looks at in one cycle (all times on ONE clock).

    ``sensor_stamp`` newest localization measurement stamp (IMU / odometry /
    VO) [s] or None if none yet; ``fusion_mode`` = FusionEngine.mode();
    ``localized`` / ``consecutive_rejects`` / ``position_sigma_m`` from the
    map localizer; ``speed`` the speed commanded LAST cycle after this
    supervisor's limit [m/s] (the stop ramp continues from it);
    ``path_ahead`` (M, 2) map-frame polyline starting AT the kart;
    ``obstacles`` (K, 2) live obstacle points in the map frame (already
    age-filtered, e.g. :meth:`ObstacleMemory.points`). Alternatively
    ``obstacle_ahead`` = ``(distance or None, points)`` already measured
    along the path within :meth:`SafetySupervisor.obstacle_reach` (e.g.
    :meth:`ObstacleMemory.ahead`); when given, ``path_ahead`` /
    ``obstacles`` are not used - the drive node projects each observation
    onto the path ONCE when it arrives instead of every control cycle.
    """

    now: float
    sensor_stamp: float | None
    fusion_mode: str
    localized: bool
    on_path: bool = True
    consecutive_rejects: int = 0
    position_sigma_m: float = 0.0
    speed: float = 0.0
    path_ahead: NDArray[np.float64] | None = None
    obstacles: NDArray[np.float64] | None = None
    obstacle_ahead: tuple[float | None, int] | None = None


@dataclass
class SafetyDecision:
    """``speed_cap`` [m/s] for this cycle; ``stopping`` when any stop trigger is active."""

    speed_cap: float
    stopping: bool
    state: str  # "driving" | "stopping" | "stopped" | "estop" | "holding"
    reasons: list[str] = field(default_factory=list)
    obstacle_distance_m: float | None = None
    decel_mps2: float = 0.0

    def limit(self, speed: float) -> float:
        """The speed actually allowed: ``min(controller speed, cap)``, never negative."""
        return float(max(min(speed, self.speed_cap), 0.0))


def corridor_arclength(
    path_pts: NDArray[np.float64],
    points: NDArray[np.float64],
    half_width: float,
    extra_left: float = 0.0,
    extra_right: float = 0.0,
) -> NDArray[np.float64]:
    """Along-path position [m, from the polyline's first point] of every point
    inside the swept corridor.

    ``path_pts`` (M, 2) polyline; a point is inside when it lies within
    ``half_width`` of its CLOSEST segment and is not behind the first point
    (its projection before the start of the first segment). Returns the
    (K,) positions of the K points inside (the along-path position of a
    nearer-but-not-closest segment end would understate the distance).

    ``extra_left`` / ``extra_right`` [m, >= 0] widen the corridor on ONE side
    of the path (left = +y of the path direction): a kart that is off the
    line sweeps the band between its own position and the line.
    """
    path = np.asarray(path_pts, dtype=np.float64).reshape(-1, 2)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) == 0 or len(path) == 0:
        return np.zeros(0)
    if len(path) == 1:
        d = np.hypot(pts[:, 0] - path[0, 0], pts[:, 1] - path[0, 1])
        return d[d <= half_width]
    extra_left, extra_right = max(float(extra_left), 0.0), max(float(extra_right), 0.0)
    reach = half_width + max(extra_left, extra_right)
    # Cheap prefilter: only points inside the path's bounding box + width.
    lo = path.min(axis=0) - reach
    hi = path.max(axis=0) + reach
    pts = pts[(pts[:, 0] >= lo[0]) & (pts[:, 0] <= hi[0]) & (pts[:, 1] >= lo[1]) & (pts[:, 1] <= hi[1])]
    if len(pts) == 0:
        return np.zeros(0)
    a = path[:-1]
    sx, sy = path[1:, 0] - a[:, 0], path[1:, 1] - a[:, 1]
    seg_len = np.maximum(np.hypot(sx, sy), 1e-9)
    s0 = np.concatenate(([0.0], np.cumsum(seg_len)[:-1]))
    rx = pts[:, 0:1] - a[:, 0]  # (K, M-1)
    ry = pts[:, 1:2] - a[:, 1]
    t_raw = (rx * sx + ry * sy) / (seg_len * seg_len)
    t = np.clip(t_raw, 0.0, 1.0)
    ex = rx - t * sx
    ey = ry - t * sy
    d2 = ex * ex + ey * ey
    k = np.argmin(d2, axis=1)
    rows = np.arange(len(pts))
    along = s0[k] + t[rows, k] * seg_len[k]
    behind = (k == 0) & (t_raw[:, 0] < 0.0)  # behind the kart: not on its way
    inside = d2[rows, k] <= half_width * half_width
    if extra_left > 0.0 or extra_right > 0.0:
        # signed perpendicular offset from the closest segment (+ = left)
        lat = (sx[k] * ry[rows, k] - sy[k] * rx[rows, k]) / seg_len[k]
        tk = t_raw[rows, k]
        on_seg = (tk >= 0.0) & (tk <= 1.0)
        inside |= on_seg & (lat >= -half_width - extra_right) & (lat <= half_width + extra_left)
    inside &= ~behind
    return along[inside]


def obstacles_on_path(
    path_ahead: NDArray[np.float64],
    obstacles: NDArray[np.float64],
    half_width: float,
    max_distance: float,
) -> tuple[float | None, int]:
    """Along-path distance [m] of the nearest obstacle inside the swept corridor.

    ``path_ahead`` (M, 2) starts at the kart; a point is "on the path" when
    it lies within ``half_width`` of its closest segment and its along-path
    position is within ``max_distance`` (:func:`corridor_arclength`).
    Returns ``(distance or None, points inside)``.
    """
    along = corridor_arclength(path_ahead, obstacles, half_width)
    hit = along[along <= max_distance]
    if not len(hit):
        return None, 0
    return float(hit.min()), int(len(hit))


class ObstacleMemory:
    """Recent live obstacle points (map frame) with per-source maximum ages.

    LiDAR points are fresh at 10-20 Hz, so they expire fast
    (``max_age_s['lidar']``); camera ground contacts arrive at a few Hz and
    the camera cannot see the last ~1.7 m in front of the bumper, so they are
    remembered longer. Stored as one batch per observation (a bounded deque),
    so :meth:`points` only touches the batches that are still young - the
    control loop calls it at 50 Hz.

    A batch may also carry ``along``: the (unwrapped) path arclength of its
    points inside the planned path's corridor, projected ONCE when the
    observation arrives (:func:`corridor_arclength`). :meth:`ahead` then
    answers "nearest obstacle ahead of the kart" with a comparison per young
    batch - independent of how many points the sensors produce.
    """

    def __init__(self, max_age_s: dict[str, float] | None = None, max_batches: int = 64) -> None:
        import threading
        from collections import deque

        self.max_age_s = {"lidar": 0.3, "camera": 1.5, **(max_age_s or {})}
        # (stamp, source, (N, 2) points, (K,) corridor arclengths or None)
        self._batches: deque = deque(maxlen=int(max_batches))
        # Sensor worker threads add while the control loop reads: iterating a
        # deque another thread appends to raises "deque mutated during
        # iteration" (measured: 185 in 4.9 M reads under load) - in the drive
        # node that exception would kill the control timer. Readers iterate a
        # snapshot taken under this lock.
        self._lock = threading.Lock()

    def add(
        self,
        points_map: NDArray[np.float64],
        stamp: float,
        source: str = "lidar",
        along: NDArray[np.float64] | None = None,
    ) -> None:
        """Remember one observation's obstacle points (map frame). ``along``:
        optional unwrapped path arclength [m] of those inside the corridor."""
        pts = np.asarray(points_map, dtype=np.float64).reshape(-1, 2)
        if len(pts):
            a = None if along is None else np.asarray(along, dtype=np.float64).reshape(-1)
            with self._lock:
                self._batches.append((float(stamp), source, pts, a))

    def _young(self, stamp: float, source: str, now: float) -> bool:
        return now - stamp <= self.max_age_s.get(source, 0.5) and stamp <= now + 0.5

    def points(self, now: float) -> NDArray[np.float64]:
        with self._lock:
            batches = tuple(self._batches)
        young = [pts for stamp, source, pts, _ in batches if self._young(stamp, source, now)]
        return np.concatenate(young) if young else np.zeros((0, 2))

    def ahead(self, now: float, s_now: float, reach: float) -> tuple[float | None, int]:
        """Nearest corridor obstacle AHEAD of path progress ``s_now``.

        Over the young batches that carry ``along``: the smallest
        ``along - s_now`` in ``[0, reach]`` [m] (None if none) and how many
        points lie in that interval.
        """
        best = math.inf
        count = 0
        with self._lock:
            batches = tuple(self._batches)
        for stamp, source, _, along in batches:
            if along is None or not len(along) or not self._young(stamp, source, now):
                continue
            rel = along - s_now
            m = (rel >= 0.0) & (rel <= reach)
            n = int(np.count_nonzero(m))
            if n:
                count += n
                best = min(best, float(rel[m].min()))
        return (best if count else None), count

    def clear(self) -> None:
        with self._lock:
            self._batches.clear()


class SafetySupervisor:
    """Stateful supervisor; see the module docstring for the trigger table."""

    def __init__(self, config: SafetyConfig | None = None) -> None:
        self.config = config or SafetyConfig()
        self.engaged = False
        self.estop_input = False
        self.estop_latched = False
        self.last_heartbeat: float | None = None
        self._degraded_since: float | None = None
        self._active: dict[str, float] = {}  # trigger -> first time seen (this episode)
        self._clear_since: dict[str, float] = {}  # trigger -> time its condition went false
        self._cap = 0.0
        self._last_now: float | None = None
        #: (time, trigger, "set" | "clear") history for diagnostics and tests.
        self.events: list[tuple[float, str, str]] = []

    # ------------------------------------------------------------ inputs
    def on_estop(self, pressed: bool, now: float) -> None:
        """E-stop input. True latches immediately; False only releases the input."""
        self.estop_input = bool(pressed)
        if pressed and not self.estop_latched:
            self.estop_latched = True
            self.engaged = False
            self.events.append((now, "estop", "set"))

    def on_heartbeat(self, now: float) -> None:
        self.last_heartbeat = float(now)

    def engage(self, on: bool, now: float) -> tuple[bool, str]:
        """Operator request to allow (True) or forbid (False) motion.

        Engaging clears a latched e-stop only once its input reads False.
        """
        if not on:
            self.engaged = False
            self.events.append((now, "not_engaged", "set"))
            return True, "disengaged: the kart stops"
        if self.estop_input:
            return False, "e-stop is still pressed: release it first"
        if self.estop_latched:
            self.estop_latched = False
            self.events.append((now, "estop", "clear"))
        self.engaged = True
        return True, "engaged: the kart may drive while every check passes"

    # ------------------------------------------------------------ update
    def obstacle_reach(self, speed: float) -> float:
        """How far along the path [m] the obstacle check looks at ``speed``:
        bumper overhang + stop margin + reaction distance + braking distance
        at ``stop_decel_mps2`` + ``lookahead_extra_m`` (early detection)."""
        c = self.config
        v = max(float(speed), 0.0)
        return c.front_overhang_m + c.stop_margin_m + v * c.reaction_time_s + v * v / (2.0 * c.stop_decel_mps2) + c.lookahead_extra_m

    def _conditions(self, x: SafetyInputs) -> tuple[dict[str, bool], float | None]:
        c = self.config
        cond: dict[str, bool] = {}
        cond["estop"] = self.estop_latched or self.estop_input
        cond["not_engaged"] = not self.engaged
        if c.deadman_timeout_s > 0.0:
            cond["deadman"] = self.last_heartbeat is None or x.now - self.last_heartbeat > c.deadman_timeout_s
        cond["pose_stale"] = x.sensor_stamp is None or x.now - x.sensor_stamp > c.pose_timeout_s
        if x.fusion_mode in DEGRADED_MODES:
            if self._degraded_since is None:
                self._degraded_since = x.now
        else:
            self._degraded_since = None
        cond["fusion_degraded"] = self._degraded_since is not None and x.now - self._degraded_since > c.degraded_mode_timeout_s
        cond["not_localized"] = not x.localized
        cond["off_path"] = x.localized and not x.on_path
        cond["localization_rejected"] = x.consecutive_rejects >= c.max_consecutive_rejects
        cond["localization_uncertain"] = (not math.isfinite(x.position_sigma_m)) or x.position_sigma_m > c.max_position_sigma_m
        obstacle_distance = None
        distance, count = None, 0
        if c.obstacle_stop:
            if x.obstacle_ahead is not None:
                distance, count = x.obstacle_ahead
            elif x.path_ahead is not None and x.obstacles is not None and len(x.obstacles):
                distance, count = obstacles_on_path(
                    x.path_ahead, x.obstacles, c.kart_half_width_m + c.lateral_margin_m, self.obstacle_reach(x.speed)
                )
        cond["obstacle"] = distance is not None and count >= c.obstacle_min_points
        if cond["obstacle"]:
            obstacle_distance = float(distance)
        return cond, obstacle_distance

    def update(self, x: SafetyInputs) -> SafetyDecision:
        c = self.config
        dt = 0.0 if self._last_now is None else max(x.now - self._last_now, 0.0)
        self._last_now = x.now
        cond, obstacle_distance = self._conditions(x)
        for name, on in cond.items():
            if on:
                self._clear_since.pop(name, None)
                if name not in self._active:
                    self._active[name] = x.now
                    self.events.append((x.now, name, "set"))
            elif name in self._active:
                hold = c.obstacle_clear_s if name == "obstacle" else c.resume_hold_s
                if name in ("estop", "not_engaged"):
                    hold = 0.0  # cleared by the explicit engage() above
                since = self._clear_since.setdefault(name, x.now)
                if x.now - since >= hold:
                    del self._active[name]
                    self._clear_since.pop(name, None)
                    self.events.append((x.now, name, "clear"))
        reasons = sorted(self._active)
        cap_limit = max(c.max_speed_mps, 0.0)
        if reasons:
            decel = c.estop_decel_mps2 if "estop" in self._active else c.stop_decel_mps2
            if obstacle_distance is not None:
                room = obstacle_distance - c.front_overhang_m - c.stop_margin_m
                v = max(x.speed, 0.0)
                need = v * v / (2.0 * max(room, 1e-3)) if room > 0 else math.inf
                decel = min(max(decel, need), c.estop_decel_mps2) if math.isfinite(need) else c.estop_decel_mps2
            # Ramp down from the speed commanded last cycle (``x.speed`` is the
            # safety-limited command, so consecutive cycles form one ramp).
            v = max(x.speed, 0.0)
            self._cap = max(v - decel * dt, 0.0) if dt > 0.0 else v
            state = "estop" if "estop" in self._active else ("stopped" if self._cap <= 1e-6 else "stopping")
            if set(reasons) <= {"not_engaged", "not_localized", "off_path"} and self._cap <= 1e-6:
                state = "holding"
            return SafetyDecision(min(self._cap, cap_limit), True, state, reasons, obstacle_distance, decel)
        self._cap = cap_limit
        return SafetyDecision(cap_limit, False, "driving", [], None, 0.0)

    @property
    def active(self) -> dict[str, float]:
        return dict(self._active)
