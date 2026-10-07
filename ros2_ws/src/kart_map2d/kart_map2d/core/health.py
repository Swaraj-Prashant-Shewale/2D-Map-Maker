"""Per-sensor health monitoring.

A sensor "fails" on a real kart in a handful of recognisable ways, and each has
a cheap detector:

========================  =============================================
failure                   detector
========================  =============================================
stops publishing          timeout since the last message (STALE)
publishes garbage         plausibility limits, NaN/inf checks
freezes (stuck driver)    identical values repeated too many times
disagrees with the rest   chi-square innovation gate in the estimator
intermittent              rejection ratio over a sliding window
========================  =============================================

A channel marked FAULTY is excluded for a cool-down period and then put on
probation: it must pass the innovation gate several times in a row before it
is trusted again. The estimator asks ``usable()`` before every update, so a
failed sensor simply stops contributing and everything else carries on.
"""

from __future__ import annotations

import enum
from collections import deque
from dataclasses import dataclass, field

import numpy as np


class Status(str, enum.Enum):
    UNSEEN = "unseen"
    OK = "ok"
    DEGRADED = "degraded"  # usable, but a noticeable share is being rejected
    PROBATION = "probation"  # recovering from FAULTY, not yet trusted
    STALE = "stale"  # nothing received within the timeout
    FAULTY = "faulty"  # excluded: stuck, implausible or persistently inconsistent


@dataclass
class ChannelConfig:
    timeout_s: float = 0.5
    window: int = 50  # measurements in the rejection-ratio window
    degraded_reject_ratio: float = 0.2
    faulty_consecutive_rejects: int = 15
    stuck_repeats: int = 25  # bit-identical repeats before calling a sensor frozen
    cooldown_s: float = 2.0
    probation_accepts: int = 10


@dataclass
class ChannelHealth:
    name: str
    config: ChannelConfig
    status: Status = Status.UNSEEN
    last_stamp: float = -np.inf
    received: int = 0
    accepted: int = 0
    rejected: int = 0
    consecutive_rejects: int = 0
    probation_streak: int = 0
    faulty_until: float = -np.inf
    reason: str = ""
    _recent: deque = field(default_factory=deque)
    _last_value: bytes | None = None
    _repeat_count: int = 0

    def on_receive(self, stamp: float, value: np.ndarray | float | None = None) -> bool:
        """Register arrival; return False if the raw value is unusable."""
        self.received += 1
        self.last_stamp = max(self.last_stamp, stamp)
        if self.status in (Status.UNSEEN, Status.STALE):
            # A pause is not evidence of bad data; only FAULTY needs probation.
            self.status = Status.OK
            self.reason = ""
        if value is None:
            return True
        array = np.atleast_1d(np.asarray(value, dtype=np.float64))
        if not np.all(np.isfinite(array)):
            self._fault(stamp, "non-finite value")
            return False
        signature = array.tobytes()
        if signature == self._last_value:
            self._repeat_count += 1
            if self._repeat_count >= self.config.stuck_repeats:
                # Stay latched (and keep refreshing the cooldown) for as long
                # as the value remains frozen; recovery starts only once the
                # sensor produces a different value again.
                self._fault(stamp, "value frozen")
                self._repeat_count = self.config.stuck_repeats
                return False
        else:
            self._repeat_count = 0
            self._last_value = signature
        return True

    def usable(self, stamp: float) -> bool:
        if self.status == Status.FAULTY:
            if stamp < self.faulty_until:
                return False
            self.status = Status.PROBATION
            self.probation_streak = 0
        return self.status not in (Status.UNSEEN, Status.STALE)

    def report(self, accepted: bool) -> None:
        """Called by the estimator after its innovation gate."""
        self._recent.append(accepted)
        if len(self._recent) > self.config.window:
            self._recent.popleft()
        if accepted:
            self.accepted += 1
            self.consecutive_rejects = 0
            if self.status == Status.PROBATION:
                self.probation_streak += 1
                if self.probation_streak >= self.config.probation_accepts:
                    self.status = Status.OK
                    self.reason = ""
        else:
            self.rejected += 1
            self.consecutive_rejects += 1
            self.probation_streak = 0
            if self.consecutive_rejects >= self.config.faulty_consecutive_rejects:
                self._fault(self.last_stamp, "persistent innovation-gate rejections")
                return
        if self.status in (Status.OK, Status.DEGRADED) and len(self._recent) >= 10:
            ratio = 1.0 - sum(self._recent) / len(self._recent)
            self.status = Status.DEGRADED if ratio > self.config.degraded_reject_ratio else Status.OK

    def declare_fault(self, stamp: float, reason: str) -> None:
        """Latch FAULTY on evidence found outside the innovation gate.

        Used by cross-sensor consistency checks (e.g. gyro vs odometric yaw)
        that see a fault the per-measurement gate cannot. Recovery follows the
        normal path: cooldown, then probation. Calling it again while already
        FAULTY refreshes the cooldown, which keeps a persistent fault latched
        the same way the frozen-value detector does.
        """
        self._fault(stamp, reason)

    def tick(self, now: float) -> None:
        if self.status in (Status.UNSEEN, Status.FAULTY, Status.STALE):
            return
        if now - self.last_stamp > self.config.timeout_s:
            self.status = Status.STALE
            self.reason = f"no data for {now - self.last_stamp:.1f} s"

    def _fault(self, stamp: float, reason: str) -> None:
        self.status = Status.FAULTY
        self.reason = reason
        self.faulty_until = stamp + self.config.cooldown_s
        self.consecutive_rejects = 0
        self._repeat_count = 0
        self._recent.clear()


class HealthMonitor:
    """Holds one ``ChannelHealth`` per sensor channel and logs transitions."""

    def __init__(self, configs: dict[str, ChannelConfig] | None = None, default_factory=None) -> None:
        self._configs = configs or {}
        #: Optional ``name -> ChannelConfig`` fallback for channels that are
        #: not in ``configs`` - e.g. ``vo_<camera>`` for ANY camera name
        #: (kart_map2d.estimation.fusion.channel_config). None keeps the
        #: plain ChannelConfig() default.
        self._default_factory = default_factory
        self.channels: dict[str, ChannelHealth] = {}
        self.transitions: list[tuple[float, str, str, str]] = []
        self._last_status: dict[str, Status] = {}

    def channel(self, name: str) -> ChannelHealth:
        if name not in self.channels:
            config = self._configs.get(name)
            if config is None and self._default_factory is not None:
                config = self._default_factory(name)
            self.channels[name] = ChannelHealth(name, config or ChannelConfig())
        return self.channels[name]

    def tick(self, now: float) -> None:
        for name, health in self.channels.items():
            health.tick(now)
            if self._last_status.get(name) != health.status:
                self.transitions.append((now, name, health.status.value, health.reason))
                self._last_status[name] = health.status

    def healthy(self, name: str) -> bool:
        health = self.channels.get(name)
        return health is not None and health.status in (Status.OK, Status.DEGRADED)

    def snapshot(self) -> dict[str, str]:
        return {name: health.status.value for name, health in self.channels.items()}
