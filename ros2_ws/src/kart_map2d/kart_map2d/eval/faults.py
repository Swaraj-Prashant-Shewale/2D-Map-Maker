"""Failure injection for replaying recordings as if sensors had failed.

Faults are declared as plain dicts so that a scenario list reads like a test
plan and can live in a YAML file::

    {"channel": "lidar_odom", "kind": "dropout", "start": 60, "stop": 120}
    {"channel": "imu", "kind": "stuck", "start": 30}
    {"channel": "odom", "kind": "bias", "start": 0, "value": 0.3}
    {"channel": "gnss", "kind": "outliers", "probability": 0.1, "value": 15}

``start``/``stop`` are seconds relative to the run start; omitted means the
whole run.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from kart_map2d.core.measurements import (
    ImuSample,
    Measurement,
    PositionFix,
    RelativeMotion,
    SpeedMeasurement,
)


@dataclass
class Fault:
    channel: str
    kind: str  # dropout | stuck | bias | noise | outliers | scale
    start: float = -np.inf
    stop: float = np.inf
    value: float = 0.0
    probability: float = 1.0

    @classmethod
    def from_dict(cls, raw: dict) -> "Fault":
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})


class FaultInjector:
    def __init__(self, faults: list[Fault], t0: float, seed: int = 7) -> None:
        self.faults = faults
        self.t0 = t0
        self.rng = np.random.default_rng(seed)
        self._frozen: dict[str, Measurement] = {}

    def active(self, channel: str, stamp: float) -> list[Fault]:
        t = stamp - self.t0
        return [f for f in self.faults if f.channel == channel and f.start <= t < f.stop]

    def dropped(self, channel: str, stamp: float) -> bool:
        return any(f.kind == "dropout" for f in self.active(channel, stamp))

    def apply(self, m: Measurement) -> Measurement | None:
        for fault in self.active(m.channel, m.stamp):
            if fault.kind == "dropout":
                return None
            if self.rng.random() > fault.probability:
                continue
            m = self._corrupt(m, fault)
        return m

    def _corrupt(self, m: Measurement, fault: Fault) -> Measurement:
        kind, value = fault.kind, fault.value
        if kind == "stuck":
            frozen = self._frozen.setdefault(m.channel, m)
            return dataclasses.replace(frozen, stamp=m.stamp) if not isinstance(m, RelativeMotion) else dataclasses.replace(
                frozen, stamp=m.stamp, start_stamp=m.start_stamp
            )
        if isinstance(m, ImuSample):
            if kind == "bias":
                return dataclasses.replace(m, gyro=m.gyro + np.array([0.0, 0.0, value]))
            if kind in ("noise", "outliers"):
                return dataclasses.replace(m, gyro=m.gyro + self.rng.normal(0.0, value, 3))
            if kind == "scale":
                return dataclasses.replace(m, gyro=m.gyro * value)
        if isinstance(m, SpeedMeasurement):
            if kind == "bias":
                return dataclasses.replace(m, speed=m.speed + value)
            if kind == "scale":
                return dataclasses.replace(m, speed=m.speed * value)
            if kind in ("noise", "outliers"):
                return dataclasses.replace(m, speed=m.speed + self.rng.normal(0.0, value))
        if isinstance(m, RelativeMotion):
            dt = m.stamp - m.start_stamp
            if kind == "bias":  # value: yaw-rate bias [rad/s]
                return dataclasses.replace(m, delta_yaw=m.delta_yaw + value * dt)
            if kind == "scale":
                return dataclasses.replace(m, delta_forward=m.delta_forward * value, delta_left=m.delta_left * value)
            if kind in ("noise", "outliers"):
                return dataclasses.replace(
                    m,
                    delta_forward=m.delta_forward + self.rng.normal(0.0, value),
                    delta_yaw=m.delta_yaw + self.rng.normal(0.0, value * 0.2),
                )
        if isinstance(m, PositionFix):
            if kind == "bias":
                return dataclasses.replace(m, position=m.position + np.array([value, 0.0]))
            if kind in ("noise", "outliers"):
                return dataclasses.replace(m, position=m.position + self.rng.normal(0.0, value, 2))
        return m
