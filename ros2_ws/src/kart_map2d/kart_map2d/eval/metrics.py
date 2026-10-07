"""Trajectory accuracy metrics against a reference trajectory.

The reference on the September recordings is the LiDAR + IMU solution; any
camera + IMU configuration is scored against it on the overlapping time span.

* ATE  - absolute trajectory error after the best 2D rigid alignment
         (a map built from a local frame has an arbitrary origin/heading).
* RPE  - relative error over ``segment_m`` of travel: translation error as a
         percentage of distance, heading error in degrees. This is what
         decides map quality - local shape, not global placement.
* coverage - share of the reference span for which the estimate is
         "motion-supported" (not pure extrapolation).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class TrajectoryScore:
    ate_rmse_m: float
    ate_max_m: float
    rpe_trans_pct: float
    rpe_yaw_deg: float
    final_error_m: float
    yaw_drift_deg: float
    coverage: float
    distance_m: float
    segments: int

    def as_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def align_se2(source: NDArray[np.float64], target: NDArray[np.float64]) -> tuple[NDArray, NDArray]:
    """Rotation R (2x2) and translation t minimising |R s + t - target|."""
    mu_s, mu_t = source.mean(axis=0), target.mean(axis=0)
    H = (source - mu_s).T @ (target - mu_t)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, mu_t - R @ mu_s


def score_trajectory(
    est_t: NDArray[np.float64],
    est_xyyaw: NDArray[np.float64],
    ref_t: NDArray[np.float64],
    ref_xyyaw: NDArray[np.float64],
    supported: NDArray[np.bool_] | None = None,
    segment_m: float = 10.0,
    sample_dt: float = 0.1,
) -> TrajectoryScore:
    start, stop = max(est_t[0], ref_t[0]), min(est_t[-1], ref_t[-1])
    t = np.arange(start, stop, sample_dt)

    def sample(times, values):
        return np.column_stack(
            [np.interp(t, times, values[:, 0]), np.interp(t, times, values[:, 1]), np.interp(t, times, np.unwrap(values[:, 2]))]
        )

    est, ref = sample(est_t, est_xyyaw), sample(ref_t, ref_xyyaw)
    R, trans = align_se2(est[:, :2], ref[:, :2])
    aligned = est[:, :2] @ R.T + trans
    yaw_offset = np.arctan2(R[1, 0], R[0, 0])
    est_yaw = est[:, 2] + yaw_offset
    err = np.linalg.norm(aligned - ref[:, :2], axis=1)
    yaw_err = np.degrees((est_yaw - ref[:, 2] + np.pi) % (2 * np.pi) - np.pi)

    distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(ref[:, :2], axis=0), axis=1))]
    trans_errors, yaw_errors = [], []
    i = 0
    while True:
        j = int(np.searchsorted(distance, distance[i] + segment_m))
        if j >= len(t):
            break
        d_ref = _relative(ref[i], ref[j])
        d_est = _relative(np.r_[aligned[i], est_yaw[i]], np.r_[aligned[j], est_yaw[j]])
        trans_errors.append(np.linalg.norm(d_est[:2] - d_ref[:2]) / segment_m * 100.0)
        yaw_errors.append(abs(np.degrees((d_est[2] - d_ref[2] + np.pi) % (2 * np.pi) - np.pi)))
        i = j
    coverage = float(np.mean(np.interp(t, est_t, supported.astype(float)) > 0.5)) if supported is not None else 1.0
    return TrajectoryScore(
        ate_rmse_m=float(np.sqrt(np.mean(err**2))),
        ate_max_m=float(err.max()),
        rpe_trans_pct=float(np.mean(trans_errors)) if trans_errors else float("nan"),
        rpe_yaw_deg=float(np.mean(yaw_errors)) if yaw_errors else float("nan"),
        final_error_m=float(err[-1]),
        yaw_drift_deg=float(abs(yaw_err[-1])),
        coverage=coverage,
        distance_m=float(distance[-1]),
        segments=len(trans_errors),
    )


def _relative(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    c, s = np.cos(a[2]), np.sin(a[2])
    d = b[:2] - a[:2]
    return np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1], b[2] - a[2]])
