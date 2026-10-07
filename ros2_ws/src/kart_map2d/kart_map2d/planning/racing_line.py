"""Minimum-curvature racing line and speed profile inside a Corridor.

The line is parameterised by lateral offsets ``alpha_i`` [m] along the *right*
unit normal at each corridor station, bounded by the corridor width minus a
safety margin: ``alpha_i in [-(w_left_i - margin), +(w_right_i - margin)]``.

The objective is the integrated squared curvature approximated by station
second differences with the (uniform) centerline spacing as the parameter -
the linearisation used by the TUMFTM minimum-curvature planner. That choice
matters: because path length per station shrinks toward the inside of a bend,
the linearised objective correctly tightens onto the inside of sustained
corners (on a circular ring it hugs the inner bound) while the exact
arclength-parameterised objective would drift outside. The offsets enter the
residuals linearly, so the problem is a box-bounded linear least squares and
is solved exactly with ``scipy.optimize.lsq_linear`` (an iterative smoother
such as L-BFGS-B stalls on this ill-conditioned quadratic).

Reported curvature is the true geometric (Menger) curvature of the resulting
path; the speed profile is curvature-limited (``v = sqrt(a_lat_max/|kappa|)``)
then forward/backward limited by longitudinal accel/brake, with wrap-around
sweeps so closed-loop profiles are periodic.

Frames and units: map frame, metres, seconds; yaw/curvature CCW-positive.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import lsq_linear

from kart_map2d.mapping.corridor import Corridor, path_normals


@dataclass
class RacingLine:
    """path : (N,2) [m]; closed lines wrap from N-1 back to 0 (no repeat).

    alpha : (N,) signed lateral offset from the corridor centerline [m],
        positive toward the right of the direction of travel.
    curvature : (N,) signed geometric curvature [1/m], CCW positive.
    speed : (N,) profile speed [m/s].
    time_s : lap time (closed) or segment time (open).
    s : (N,) cumulative arclength along the racing line [m].
    """

    path: NDArray[np.float64]
    alpha: NDArray[np.float64]
    curvature: NDArray[np.float64]
    speed: NDArray[np.float64]
    time_s: float
    closed: bool
    s: NDArray[np.float64]


# --------------------------------------------------------------- primitives
def signed_curvature(points: NDArray[np.float64], closed: bool) -> NDArray[np.float64]:
    """Menger curvature per vertex, signed CCW-positive.

    Open paths copy the adjacent interior value onto the end vertices.
    """
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    if n < 3:
        return np.zeros(n)
    if closed:
        a, b, c = np.roll(pts, 1, axis=0), pts, np.roll(pts, -1, axis=0)
    else:
        a, b, c = pts[:-2], pts[1:-1], pts[2:]
    ab, bc, ac = b - a, c - b, c - a
    cross = ab[:, 0] * bc[:, 1] - ab[:, 1] * bc[:, 0]
    denom = np.linalg.norm(ab, axis=1) * np.linalg.norm(bc, axis=1) * np.linalg.norm(ac, axis=1)
    kappa = np.where(denom > 1e-12, 2.0 * cross / np.maximum(denom, 1e-12), 0.0)
    if closed:
        return kappa
    return np.concatenate([[kappa[0]], kappa, [kappa[-1]]])


def _second_diff_matrix(n: int, closed: bool) -> NDArray[np.float64]:
    """Station second-difference operator; (n,n) circulant or (n-2,n) open."""
    if closed:
        d = np.zeros((n, n))
        idx = np.arange(n)
        d[idx, idx] = -2.0
        d[idx, (idx + 1) % n] = 1.0
        d[idx, (idx - 1) % n] = 1.0
        return d
    d = np.zeros((n - 2, n))
    idx = np.arange(n - 2)
    d[idx, idx] = 1.0
    d[idx, idx + 1] = -2.0
    d[idx, idx + 2] = 1.0
    return d


def _optimise_offsets(
    center: NDArray[np.float64],
    n_right: NDArray[np.float64],
    lo: NDArray[np.float64],
    hi: NDArray[np.float64],
    closed: bool,
    max_iter: int,
) -> NDArray[np.float64]:
    """argmin_alpha sum |d2(C + alpha*n_r)|^2 subject to lo <= alpha <= hi.

    P = C + alpha*n_r is affine in alpha, so the station second differences
    stack into residuals ``A @ alpha - b`` and lsq_linear solves the QP to
    optimality. Dense matrices: fine up to a few thousand stations (a 550 m
    run at 0.5 m spacing is ~1100).
    """
    d = _second_diff_matrix(len(center), closed)
    a = np.vstack([d * n_right[:, 0][None, :], d * n_right[:, 1][None, :]])
    b = -np.concatenate([d @ center[:, 0], d @ center[:, 1]])
    hi = np.where(hi - lo < 1e-9, lo + 1e-9, hi)  # lsq_linear wants lb < ub
    res = lsq_linear(a, b, bounds=(lo, hi), method="trf", tol=1e-12, max_iter=max_iter)
    return np.clip(res.x, lo, hi)


def _speed_profile(
    path: NDArray[np.float64],
    kappa: NDArray[np.float64],
    closed: bool,
    a_lat_max: float,
    a_accel: float,
    a_brake: float,
    v_max: float,
) -> tuple[NDArray[np.float64], float, NDArray[np.float64]]:
    """Curvature-limited speed with accel/brake sweeps; returns (v, time, s).

    Closed loops repeat the forward/backward sweeps across the wrap until the
    profile stops changing (speeds only ever decrease, so this converges).
    """
    n = len(path)
    v = np.minimum(v_max, np.sqrt(a_lat_max / np.maximum(np.abs(kappa), 1e-9)))

    if closed:
        seg = np.linalg.norm(np.roll(path, -1, axis=0) - path, axis=1)  # edge i -> i+1 (mod n)
        edges = n
    else:
        seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
        edges = n - 1
    seg = np.maximum(seg, 1e-9)

    for _ in range(64 if closed else 1):
        changed = False
        for i in range(edges):  # forward: accelerating out of slow points
            j = (i + 1) % n
            cap = np.sqrt(v[i] ** 2 + 2.0 * a_accel * seg[i])
            if v[j] > cap + 1e-12:
                v[j] = cap
                changed = True
        for i in range(edges - 1, -1, -1):  # backward: braking into slow points
            j = (i + 1) % n
            cap = np.sqrt(v[j] ** 2 + 2.0 * a_brake * seg[i])
            if v[i] > cap + 1e-12:
                v[i] = cap
                changed = True
        if not changed:
            break

    v_edge = 0.5 * (v[np.arange(edges)] + v[(np.arange(edges) + 1) % n])
    time_s = float(np.sum(seg[:edges] / np.maximum(v_edge, 1e-6)))
    s = np.concatenate([[0.0], np.cumsum(seg[: n - 1])])
    return v, time_s, s


# ------------------------------------------------------------------ planner
def compute_racing_line(
    corridor: Corridor,
    *,
    margin_m: float = 0.3,
    a_lat_max: float = 8.0,
    a_accel: float = 4.0,
    a_brake: float = 6.0,
    v_max: float = 15.0,
    max_iter: int = 500,
) -> RacingLine:
    """Minimum-curvature line inside ``corridor`` plus its speed profile.

    margin_m : lateral safety margin kept off each corridor edge [m] (about
        half the vehicle width plus mapping slack). Stations narrower than
        2*margin collapse to the mid-line of what remains.
    a_lat_max / a_accel / a_brake : friction and longitudinal limits [m/s^2].
    v_max : straight-line speed cap [m/s].
    """
    center = np.asarray(corridor.centerline, dtype=float)
    if len(center) < 4:
        raise ValueError("corridor must have at least 4 stations")
    n_right = -path_normals(center, corridor.closed)

    lo = -np.maximum(corridor.w_left - margin_m, 0.0)
    hi = np.maximum(corridor.w_right - margin_m, 0.0)
    narrow = lo > hi
    if np.any(narrow):  # corridor narrower than the margins: take its middle
        mid = 0.5 * (lo[narrow] + hi[narrow])
        lo = lo.copy()
        hi = hi.copy()
        lo[narrow] = mid
        hi[narrow] = mid

    alpha = _optimise_offsets(center, n_right, lo, hi, corridor.closed, max_iter)
    path = center + alpha[:, None] * n_right

    kappa = signed_curvature(path, corridor.closed)
    speed, time_s, s = _speed_profile(path, kappa, corridor.closed, a_lat_max, a_accel, a_brake, v_max)
    return RacingLine(
        path=path,
        alpha=alpha,
        curvature=kappa,
        speed=speed,
        time_s=time_s,
        closed=corridor.closed,
        s=s,
    )


# ------------------------------------------------------------------- output
def write_racing_line_csv(line: RacingLine, path: Path | str) -> Path:
    """CSV: meta comment, then ``# s_m,x_m,y_m,kappa_radpm,v_mps`` rows."""
    path = Path(path)
    with path.open("w", encoding="ascii", newline="\n") as f:
        f.write(f"# closed={int(line.closed)} time_s={line.time_s:.3f}\n")
        f.write("# s_m,x_m,y_m,kappa_radpm,v_mps\n")
        for s, (x, y), k, v in zip(line.s, line.path, line.curvature, line.speed):
            f.write(f"{s:.3f},{x:.3f},{y:.3f},{k:.5f},{v:.3f}\n")
    return path


def render_racing_line_png(
    line: RacingLine,
    corridor: Corridor,
    path: Path | str,
    *,
    dpi: int = 140,
) -> Path:
    """Racing line coloured by speed inside the corridor outline."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import ListedColormap, Normalize

    path = Path(path)
    left, right = corridor.left_edge(), corridor.right_edge()

    def _wrap(pts: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.vstack([pts, pts[:1]]) if corridor.closed else pts

    fig, ax = plt.subplots(figsize=(8.0, 8.0), dpi=dpi)
    outline = np.vstack([_wrap(left), _wrap(right)[::-1]])
    ax.fill(outline[:, 0], outline[:, 1], color="#f0f0ec", zorder=0)
    ax.plot(_wrap(left)[:, 0], _wrap(left)[:, 1], color="#5b5b5b", lw=1.4, label="corridor edge")
    ax.plot(_wrap(right)[:, 0], _wrap(right)[:, 1], color="#5b5b5b", lw=1.4)
    cl = _wrap(corridor.centerline)
    ax.plot(cl[:, 0], cl[:, 1], color="#9a9a9a", lw=1.0, ls="--", label="centerline")

    pts = _wrap(line.path).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    v = line.speed
    v_seg = 0.5 * (v + np.roll(v, -1)) if line.closed else 0.5 * (v[:-1] + v[1:])
    # Single-hue sequential ramp, truncated so the slow end stays visible.
    cmap = ListedColormap(plt.cm.Blues(np.linspace(0.35, 1.0, 256)))
    lc = LineCollection(segs, cmap=cmap, norm=Normalize(0.0, max(float(v.max()), 1e-6)))
    lc.set_array(v_seg)
    lc.set_linewidth(2.5)
    ax.add_collection(lc)
    cbar = fig.colorbar(lc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("speed [m/s]", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    kind = "lap" if line.closed else "segment"
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"racing line - {kind} time {line.time_s:.2f} s")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
