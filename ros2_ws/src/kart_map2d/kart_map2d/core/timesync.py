"""Map a device clock onto the common clock without hardware sync.

Some devices stamp messages with their own counter (the kart's ``/imu/data``
starts at 0 s, not at ROS time). The receive time is ``device_stamp + offset +
latency`` with latency >= 0, so the lower envelope of ``receive - stamp`` over
a sliding window tracks ``offset`` including slow crystal drift.
"""

from __future__ import annotations

from collections import deque

import numpy as np


class ClockSync:
    def __init__(self, window_s: float = 10.0, epoch_threshold_s: float = 1e8) -> None:
        self.window_s = window_s
        self.epoch_threshold_s = epoch_threshold_s
        self._samples: deque[tuple[float, float]] = deque()
        self._offset: float | None = None

    def correct(self, device_stamp: float, receive_stamp: float) -> float:
        """Return ``device_stamp`` expressed on the common clock."""
        if device_stamp > self.epoch_threshold_s:
            return device_stamp  # already on an epoch clock
        self._samples.append((device_stamp, receive_stamp - device_stamp))
        while self._samples and device_stamp - self._samples[0][0] > self.window_s:
            self._samples.popleft()
        floor = min(delta for _, delta in self._samples)
        # Allow the envelope to rise slowly (clock drift) but drop instantly.
        if self._offset is None or floor < self._offset:
            self._offset = floor
        else:
            self._offset += min(floor - self._offset, 1e-4)
        return device_stamp + self._offset


def batch_correct(device_stamps: np.ndarray, receive_stamps: np.ndarray, window_s: float = 10.0) -> np.ndarray:
    sync = ClockSync(window_s)
    return np.asarray([sync.correct(float(s), float(r)) for s, r in zip(device_stamps, receive_stamps)])


def cross_correlation_lag(
    t_a: np.ndarray, x_a: np.ndarray, t_b: np.ndarray, x_b: np.ndarray, max_lag_s: float = 0.5, step_s: float = 0.0025
) -> tuple[float, float]:
    """Lag to add to ``t_a`` so that signal a best matches signal b."""
    start = max(t_a[0], t_b[0]) + max_lag_s
    stop = min(t_a[-1], t_b[-1]) - max_lag_s
    grid = np.arange(start, stop, step_s)
    reference = np.interp(grid, t_b, x_b)
    reference = reference - reference.mean()
    best_lag, best_corr = 0.0, -np.inf
    for lag in np.arange(-max_lag_s, max_lag_s + step_s, step_s):
        shifted = np.interp(grid, t_a + lag, x_a)
        shifted = shifted - shifted.mean()
        denom = np.linalg.norm(shifted) * np.linalg.norm(reference)
        corr = float(shifted @ reference / denom) if denom > 0 else 0.0
        if corr > best_corr:
            best_lag, best_corr = float(lag), corr
    return best_lag, best_corr
