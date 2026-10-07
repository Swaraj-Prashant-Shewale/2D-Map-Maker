"""Fusion engine: health monitoring + reorder buffer + EKF + cross-sensor
consistency + mode reporting.

The same object is driven by the offline replay and by the ROS 2 node, so
anything verified offline is what runs on the kart.
"""

from __future__ import annotations

import heapq
import itertools
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from kart_map2d.core.health import ChannelConfig, HealthMonitor, Status
from kart_map2d.core.measurements import ImuSample, Measurement, RelativeMotion, ZeroVelocity
from kart_map2d.estimation.ekf import BG, EkfConfig, PlanarEkf, Trajectory

# NumPy 2 renamed trapz -> trapezoid. The kart's ROS 2 Humble stack must stay on
# NumPy 1.x (its compiled message bindings are built against it), so support both.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz

# What each channel can observe. Used to describe the current operating mode
# and to decide whether poses are good enough to write into the map.
CHANNEL_ROLES = {
    "imu": "inertial",
    "imu_livox": "inertial",
    "lidar_odom": "odometric",
    "visual_odom": "odometric",
    "odom": "odometric",
    "wheel": "odometric",
    "speedometer": "odometric",
    "gnss": "absolute",
    "gnss_vel": "absolute",
    "mag": "heading",
    "zupt": "inertial",
}


def channel_role(name: str) -> str:
    """Role of any channel, INCLUDING dynamically named cameras.

    A ``vo_<camera>`` channel is odometric whatever the camera is called
    (requirement 6: no camera-make names in code); everything unknown is
    "other".
    """
    role = CHANNEL_ROLES.get(name)
    if role is not None:
        return role
    if name.startswith("vo_"):
        return "odometric"
    return "other"


DEFAULT_CHANNEL_CONFIGS = {
    "imu": ChannelConfig(timeout_s=0.1, window=200, faulty_consecutive_rejects=100),
    "imu_livox": ChannelConfig(timeout_s=0.1, window=200, faulty_consecutive_rejects=100),
    "lidar_odom": ChannelConfig(timeout_s=0.3),
    "odom": ChannelConfig(timeout_s=1.0),
    # ZUPTs are only emitted while parked, so silence is normal, not a fault.
    "zupt": ChannelConfig(timeout_s=np.inf, faulty_consecutive_rejects=50),
    "wheel": ChannelConfig(timeout_s=0.3),
    "speedometer": ChannelConfig(timeout_s=0.5),
    "gnss": ChannelConfig(timeout_s=1.5, faulty_consecutive_rejects=8, cooldown_s=5.0),
    "gnss_vel": ChannelConfig(timeout_s=1.5, faulty_consecutive_rejects=8, cooldown_s=5.0),
    "mag": ChannelConfig(timeout_s=0.5, faulty_consecutive_rejects=30, cooldown_s=5.0),
}

#: Any ``vo_<camera>`` channel, whatever the camera's name: frames arrive
#: sparsely (median gap 0.17-0.36 s, dropouts to a few seconds) and VO
#: refuses unsolvable pairs, so silence up to the VO max-gap is normal
#: operation, not staleness.
VO_CHANNEL_CONFIG = ChannelConfig(timeout_s=4.0, faulty_consecutive_rejects=10)


def channel_config(name: str) -> ChannelConfig:
    """Health config for any channel, including dynamically named cameras."""
    config = DEFAULT_CHANNEL_CONFIGS.get(name)
    if config is not None:
        return config
    if name.startswith("vo_"):
        return VO_CHANNEL_CONFIG
    return ChannelConfig()


class StillnessDetector:
    """Declare the kart stationary from IMU vibration alone.

    Parked, the MEMS gyro shows ~0.001 rad/s and the accelerometer ~0.03 m/s^2
    of noise; rolling on any surface raises both by an order of magnitude
    (measured on the September recordings), so a variance threshold over a
    short window separates the two cleanly. Stationary periods then pin speed
    to zero and make the gyro and accelerometer biases observable, which is
    what keeps a camera + IMU only configuration from drifting while parked.
    """

    def __init__(self, window_s: float = 0.4, gyro_std: float = 0.005, accel_std: float = 0.12, rate_hz: float = 10.0):
        self.window_s = window_s
        self.gyro_std = gyro_std
        self.accel_std = accel_std
        self.period = 1.0 / rate_hz
        self._samples: list[ImuSample] = []
        self._next_emit = -np.inf

    def update(self, sample: ImuSample) -> ZeroVelocity | None:
        self._samples.append(sample)
        while self._samples and sample.stamp - self._samples[0].stamp > self.window_s:
            self._samples.pop(0)
        if len(self._samples) < 10 or sample.stamp < self._next_emit:
            return None
        gyro = np.array([s.gyro for s in self._samples])
        accel = np.array([s.accel for s in self._samples])
        # Low variance alone is not stillness: a smooth constant-rate turn has
        # low variance too. Standing still additionally means near-zero mean
        # rotation rate.
        still = (
            gyro.std(axis=0).max() < self.gyro_std
            and float(np.linalg.norm(gyro.mean(axis=0))) < 0.02
            and np.linalg.norm(accel, axis=1).std() < self.accel_std
        )
        if not still:
            return None
        self._next_emit = sample.stamp + self.period
        return ZeroVelocity(sample.stamp, "zupt")


@dataclass
class ModeSample:
    stamp: float
    mode: str
    healthy: tuple[str, ...]


# ------------------------------------------------- gyro-vs-odometry consistency
#: Health-monitor fault reason written by the yaw consistency check.
GYRO_INCONSISTENT = "gyro inconsistent with odometry"


@dataclass
class YawConsistencyConfig:
    """Tuning for the interval-level gyro-vs-odometry yaw cross-check.

    Thresholds are in units of normalized residual (yaw residual over its
    modelled sigma). They were placed against the September recordings:
    the clean signed EWMA never exceeds 0.9 on any (IMU, odometry) pair,
    while a 0.15 rad/s in-band gyro bias drives it past 3 within seconds
    (scripts/validate_gyro_check.py re-measures both).
    """

    buffer_s: float = 8.0  # gyro sample retention [s]; odometric intervals are <3 s
    history_s: float = 30.0  # per-pair residual history for the heading fix [s]
    min_interval_s: float = 0.05  # shorter odometric intervals carry no yaw signal
    edge_tol_s: float = 0.06  # max clamped extrapolation at the interval ends [s]
    max_gyro_gap_s: float = 0.10  # a gyro gap inside the interval -> skip it
    ewma_alpha: float = 0.25  # per-interval EWMA weight of the signed residual
    threshold: float = 1.5  # blame bar with >=2 agreeing odometric witnesses
    solo_threshold: float = 2.5  # stronger bar with a single odometric witness
    trigger_intervals: int = 6  # consecutive intervals above bar before blame
    solo_trigger_intervals: int = 12
    latch_frac: float = 0.6  # stay faulty while |ewma| >= latch_frac * threshold
    rise_frac: float = 0.3  # |ewma| below rise_frac * threshold = "agreeing"
    rise_margin_s: float = 1.0  # heading fix starts this early before the rise [s]
    corroboration_window_s: float = 3.0  # witness validity after its last interval [s]
    rate_frac: float = 0.10  # sigma share of |integrated yaw| (timing/scale slack)
    sigma_floor: float = 0.004  # per-interval model floor [rad]
    correct_heading: bool = True  # undo the yaw absorbed before detection
    learn_bias: bool = True  # teach the EKF the measured excess rate at blame


@dataclass
class _PairState:
    """Disagreement statistics of one (IMU channel, odometric channel) pair."""

    rise_start: float  # last stamp with |ewma| below the rise bar [s]
    ewma: float = 0.0  # signed EWMA of the normalized residual
    streak: int = 0  # consecutive intervals with |ewma| above the blame bar
    last_stamp: float = -np.inf
    # (start_stamp, stamp, odometric delta_yaw [rad], its sigma [rad],
    #  bias-corrected gyro-vs-odometric yaw residual [rad]).
    history: deque = field(default_factory=deque)


class YawConsistencyMonitor:
    """Interval-level cross-check of gyro yaw against odometric yaw.

    Why: a slow in-band gyro bias (e.g. a 0.15 rad/s step) passes every
    per-measurement innovation gate because the CTRA process noise absorbs it
    at 100-400 Hz, and the biased gyro then outvotes the sparse camera yaw.
    Comparing the *integral* of (gyro_z - estimated bias) against each
    odometric channel's delta_yaw over the same interval restores the lost
    signal: the residual grows with the interval length while its sigma does
    not.

    Blame policy (symmetric safety):
    - >= 2 independent odometric witnesses disagreeing with the gyro in the
      same direction -> the gyro is the odd one out: fault it.
    - odometric witnesses disagreeing with the gyro in *opposite* directions
      disagree with each other; the gyro sits between them -> log only.
    - a single witness against a single gyro is genuinely ambiguous. Choice:
      blame the gyro, but only past a much stronger threshold, because a
      wrongly excluded gyro still leaves a working odometric filter while a
      wrongly trusted biased gyro corrupts the whole map.

    All quantities are body-frame yaw increments [rad] on the common clock;
    faults go through ``ChannelHealth.declare_fault`` so the existing
    cooldown/probation machinery governs recovery, and repeated latching
    keeps a persistent fault excluded exactly like the frozen-value detector.
    """

    def __init__(
        self,
        health: HealthMonitor,
        ekf: PlanarEkf,
        trajectory: Trajectory,
        config: YawConsistencyConfig | None = None,
    ) -> None:
        self.health = health
        self.ekf = ekf
        self.trajectory = trajectory
        self.config = config or YawConsistencyConfig()
        self._gyro: dict[str, deque] = {}  # channel -> deque[(stamp, gyro_z)]
        self._pairs: dict[tuple[str, str], _PairState] = {}
        self._last_event: dict[tuple[str, str, str], float] = {}
        self._last_fault_stamp = -np.inf
        #: (stamp, imu channel, heading correction applied [rad]) per blame.
        self.faults: list[tuple[float, str, float]] = []
        #: (stamp, kind, imu channel, odometric channel) log-only events.
        self.events: list[tuple[float, str, str, str]] = []

    # ------------------------------------------------------------------ inputs
    def add_imu(self, m: ImuSample) -> None:
        buf = self._gyro.setdefault(m.channel, deque())
        buf.append((m.stamp, float(m.gyro[2])))
        while buf and m.stamp - buf[0][0] > self.config.buffer_s:
            buf.popleft()

    def add_relative(self, m: RelativeMotion) -> None:
        cfg = self.config
        dt = m.stamp - m.start_stamp
        if dt < cfg.min_interval_s:
            return
        # Only a healthy odometric channel can witness against the gyro.
        if not self.health.healthy(m.channel):
            return
        sigma_odo = m.sigma_yaw / max(m.quality, 0.05)
        for imu_channel, buf in self._gyro.items():
            integrated = self._integrate(buf, m.start_stamp, m.stamp)
            if integrated is None:  # interval overlaps the buffer edge or a gap
                continue
            gyro_yaw, mean_gap = integrated
            residual = gyro_yaw - float(self.ekf.x[BG]) * dt - m.delta_yaw
            sigma = float(
                np.sqrt(
                    sigma_odo**2
                    + (cfg.rate_frac * abs(gyro_yaw)) ** 2
                    + self.ekf.config.sigma_gyro**2 * dt * mean_gap
                    + float(self.ekf.P[BG, BG]) * dt**2
                    + cfg.sigma_floor**2
                )
            )
            pair = self._pairs.setdefault((imu_channel, m.channel), _PairState(rise_start=m.stamp))
            pair.ewma = (1.0 - cfg.ewma_alpha) * pair.ewma + cfg.ewma_alpha * (residual / sigma)
            pair.streak = pair.streak + 1 if abs(pair.ewma) >= cfg.threshold else 0
            if abs(pair.ewma) < cfg.rise_frac * cfg.threshold:
                pair.rise_start = m.stamp
            pair.last_stamp = m.stamp
            pair.history.append((m.start_stamp, m.stamp, m.delta_yaw, sigma_odo, residual))
            while pair.history and m.stamp - pair.history[0][1] > cfg.history_s:
                pair.history.popleft()
            self._evaluate(imu_channel, m.channel, m.stamp)

    def _integrate(self, buf: deque, t0: float, t1: float) -> tuple[float, float] | None:
        """Trapezoid integral of gyro z over [t0, t1] and the mean sample gap.

        Returns None when the interval is not covered: it starts before the
        retention buffer, ends after the newest sample, or spans a gyro gap.
        """
        cfg = self.config
        if len(buf) < 4:
            return None
        samples = np.asarray(buf)
        ts, gz = samples[:, 0], samples[:, 1]
        if t0 < ts[0] - cfg.edge_tol_s or t1 > ts[-1] + cfg.edge_tol_s:
            return None
        i0, i1 = np.searchsorted(ts, (t0, t1))
        stamps = np.concatenate(([t0], ts[i0:i1], [t1]))
        if np.diff(stamps).max(initial=0.0) > cfg.max_gyro_gap_s:
            return None
        values = np.concatenate(([np.interp(t0, ts, gz)], gz[i0:i1], [np.interp(t1, ts, gz)]))
        mean_gap = (t1 - t0) / max(len(stamps) - 1, 1)
        return float(_trapezoid(values, stamps)), mean_gap

    # ---------------------------------------------------------------- decision
    def _evaluate(self, imu_channel: str, odo_channel: str, stamp: float) -> None:
        cfg = self.config
        trigger = self._pairs[(imu_channel, odo_channel)]
        imu_health = self.health.channel(imu_channel)
        if imu_health.status == Status.FAULTY:
            # Hysteresis latch: while the disagreement persists, keep
            # refreshing the cooldown so a persistently biased gyro stays out;
            # once it heals, the EWMA decays and normal recovery takes over.
            if imu_health.reason == GYRO_INCONSISTENT and abs(trigger.ewma) >= cfg.latch_frac * cfg.threshold:
                imu_health.declare_fault(stamp, GYRO_INCONSISTENT)
            return
        if trigger.streak < cfg.trigger_intervals:
            return
        witnesses = {
            odo: pair
            for (imu, odo), pair in self._pairs.items()
            if imu == imu_channel
            and stamp - pair.last_stamp <= cfg.corroboration_window_s
            and self.health.healthy(odo)
        }
        sign = np.sign(trigger.ewma)
        agreeing = [p for p in witnesses.values() if abs(p.ewma) >= cfg.threshold and np.sign(p.ewma) == sign]
        opposing = [p for p in witnesses.values() if abs(p.ewma) >= cfg.threshold and np.sign(p.ewma) != sign]
        if opposing:
            # The odometric sources contradict each other; the gyro sits
            # between them, so there is no case against it.
            self._log(stamp, "odometry-split", imu_channel, odo_channel)
        elif len(agreeing) >= 2:
            self._blame(imu_channel, agreeing, stamp)
        elif len(witnesses) == 1:
            if abs(trigger.ewma) >= cfg.solo_threshold and trigger.streak >= cfg.solo_trigger_intervals:
                self._blame(imu_channel, [trigger], stamp)
        else:
            # A second witness exists but has not (yet) confirmed: wait.
            self._log(stamp, "uncorroborated", imu_channel, odo_channel)

    def _blame(self, imu_channel: str, agreeing: list[_PairState], stamp: float) -> None:
        heading, heading_sigma, rate, rate_sigma = self._window_stats(agreeing)
        self.health.channel(imu_channel).declare_fault(stamp, GYRO_INCONSISTENT)
        # The yaw-rate/bias states were held by the sensor just blamed; drop
        # that confidence so the surviving odometric yaw is not gated out.
        self.ekf.deweight_yaw_rate()
        if self.config.correct_heading and heading is not None:
            self.ekf.correct_yaw(heading, heading_sigma)
        if self.config.learn_bias and rate is not None:
            # Model the fault, not just the exclusion: with the excess rate in
            # the bias state, a later re-admission comes back calibrated
            # instead of re-poisoning the yaw.
            self.ekf.learn_gyro_bias(rate, rate_sigma)
        self._last_fault_stamp = stamp
        self.faults.append((stamp, imu_channel, 0.0 if heading is None else heading))
        self._log(stamp, "blamed", imu_channel, "")

    def _window_stats(self, agreeing: list[_PairState]) -> tuple[float | None, float, float | None, float]:
        """Corrections implied by the disagreement window behind a blame.

        Over each witness's recent intervals - starting where its EWMA left
        the agreement band - this measures (a) the heading the filter absorbed
        in excess of the odometric increments (its own yaw increments vs
        theirs; interval-wise differencing keeps odometric gaps out of the
        sum) and (b) the gyro's excess rate over its modelled bias. Witnesses
        are combined inverse-variance.
        """
        stamps, _, _, _ = self.trajectory.arrays()
        if len(stamps) < 2:
            return None, 0.0, None, 0.0
        start = min(pair.rise_start for pair in agreeing) - self.config.rise_margin_s
        # Never reach back across an earlier correction: its yaw step would
        # be double-counted as freshly absorbed error.
        start = max(start, self._last_fault_stamp, float(stamps[0]))
        yaw_num = yaw_den = rate_num = rate_den = 0.0
        for pair in agreeing:
            intervals = np.asarray([iv for iv in pair.history if iv[0] >= start])
            if len(intervals) == 0:
                continue
            n = len(intervals)
            var = float(np.sum(intervals[:, 3] ** 2))
            span = float(np.sum(intervals[:, 1] - intervals[:, 0]))
            if var <= 0.0 or span <= 0.0:
                continue
            yaw = self.trajectory.pose_at(np.concatenate((intervals[:, 0], intervals[:, 1])))[:, 2]
            absorbed = float(np.sum(yaw[n:] - yaw[:n] - intervals[:, 2]))
            yaw_num += absorbed / var
            yaw_den += 1.0 / var
            # The residuals already discount the modelled bias, so their sum
            # over the covered time is the rate the bias state is missing.
            # var(sum/span) = var / span^2 -> inverse-variance weight below.
            rate_num += float(np.sum(intervals[:, 4])) * span / var
            rate_den += span**2 / var
        heading = heading_sigma = rate = rate_sigma = None
        if yaw_den > 0.0:
            heading = yaw_num / yaw_den
            heading_sigma = max(float(np.sqrt(1.0 / yaw_den)), 0.25 * abs(heading), 0.05)
        if rate_den > 0.0:
            rate = rate_num / rate_den
            rate_sigma = max(float(np.sqrt(1.0 / rate_den)), 0.3 * abs(rate), 0.01)
        return heading, heading_sigma or 0.0, rate, rate_sigma or 0.0

    def _log(self, stamp: float, kind: str, imu_channel: str, odo_channel: str) -> None:
        key = (kind, imu_channel, odo_channel)
        if stamp - self._last_event.get(key, -np.inf) < 2.0:
            return
        self._last_event[key] = stamp
        self.events.append((stamp, kind, imu_channel, odo_channel))


@dataclass
class FusionEngine:
    ekf_config: EkfConfig = field(default_factory=EkfConfig)
    reorder_window_s: float = 0.12
    output_rate_hz: float = 50.0
    yaw_check: YawConsistencyConfig = field(default_factory=YawConsistencyConfig)

    def __post_init__(self) -> None:
        self.health = HealthMonitor(dict(DEFAULT_CHANNEL_CONFIGS), default_factory=channel_config)
        self.ekf = PlanarEkf(self.health, self.ekf_config)
        self.trajectory = Trajectory()
        self.yaw_monitor = YawConsistencyMonitor(self.health, self.ekf, self.trajectory, self.yaw_check)
        self.modes: list[ModeSample] = []
        self._queue: list[tuple[float, int, Measurement]] = []
        self._counter = itertools.count()
        self._latest_arrival = -np.inf
        self._next_output = -np.inf
        self._last_mode: str | None = None
        self._stillness: dict[str, StillnessDetector] = {}

    # ----------------------------------------------------------------- input
    def push(self, m: Measurement, arrival: float | None = None) -> None:
        """Queue a measurement. ``arrival`` is its receive time (defaults to stamp)."""
        health = self.health.channel(m.channel)
        if not health.on_receive(m.stamp, _signature(m)):
            return
        heapq.heappush(self._queue, (m.stamp, next(self._counter), m))
        if isinstance(m, ImuSample):
            detector = self._stillness.setdefault(m.channel, StillnessDetector())
            zupt = detector.update(m)
            if zupt is not None and self.health.channel("zupt").on_receive(zupt.stamp):
                heapq.heappush(self._queue, (zupt.stamp, next(self._counter), zupt))
        self._latest_arrival = max(self._latest_arrival, arrival if arrival is not None else m.stamp)
        self._drain(self._latest_arrival - self.reorder_window_s)

    def flush(self) -> None:
        self._drain(np.inf)

    def _drain(self, until: float) -> None:
        while self._queue and self._queue[0][0] <= until:
            stamp, _, m = heapq.heappop(self._queue)
            self.health.tick(stamp)
            self.ekf.process(m)
            # The cross-check runs after the filter update so a blame decision
            # (fault + covariance surgery) lands before the next measurement.
            if isinstance(m, ImuSample):
                self.yaw_monitor.add_imu(m)
            elif isinstance(m, RelativeMotion) and channel_role(m.channel) == "odometric":
                self.yaw_monitor.add_relative(m)
            self._record(stamp)

    # ----------------------------------------------------------------- output
    def _record(self, stamp: float) -> None:
        if stamp < self._next_output or self.ekf.stamp is None:
            return
        self._next_output = stamp + 1.0 / self.output_rate_hz
        supported = self.ekf.motion_supported(stamp)
        self.trajectory.append(self.ekf.snapshot(), supported)
        mode = self.mode()
        if mode != self._last_mode:
            healthy = tuple(sorted(n for n, h in self.health.channels.items() if h.status in (Status.OK, Status.DEGRADED)))
            self.modes.append(ModeSample(stamp, mode, healthy))
            self._last_mode = mode

    def mode(self) -> str:
        roles = {channel_role(n) for n in self.health.channels if self.health.healthy(n)}
        parts = [role for role in ("absolute", "heading", "odometric", "inertial") if role in roles]
        if not parts:
            return "blind"
        if parts == ["inertial"]:
            return "inertial-only (dead reckoning)"
        return "+".join(parts)

    def pose_trusted(self, stamp: float) -> bool:
        """Whether poses at ``stamp`` are fit to write into the map."""
        return self.ekf.motion_supported(stamp)


def _signature(m: Measurement):
    """The numeric payload used for stuck-sensor detection."""
    for name in ("gyro", "position", "velocity"):
        if hasattr(m, name):
            value = getattr(m, name)
            return np.r_[value, getattr(m, "accel", [])] if name == "gyro" else value
    for name in ("speed", "yaw"):
        if hasattr(m, name):
            return getattr(m, name)
    if hasattr(m, "delta_forward"):
        return np.array([m.delta_forward, m.delta_left, m.delta_yaw, m.stamp])
    return None
