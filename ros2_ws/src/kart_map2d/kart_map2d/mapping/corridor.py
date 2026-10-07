"""Drivable-corridor extraction from a classified grid and a driven trajectory.

The grid is any planar cell classification (camera-only evidence grid, LiDAR
grid, fused grid): an integer array where one code means "drivable", one means
"obstacle", and every other value - as well as everything outside the array -
is unknown and treated as non-drivable. Better sensors simply produce a better
grid; this module stays sensor-agnostic.

Frames and units:
- Map frame, planar, metres; yaw measured CCW from the x-axis.
- Grid indexing: ``classes[row, col]`` where ``col`` indexes x and ``row``
  indexes y. Cell ``[row, col]`` spans ``x in [origin_x + col*res,
  origin_x + (col+1)*res)`` and likewise for y. ``origin_xy`` is the world
  position of the outer corner of cell ``[0, 0]``.

The extractor resamples the driven path into uniformly spaced stations and
measures, per station, how far the *contiguous* drivable region extends to the
left and right of the path. Anchoring the corridor to where the kart actually
drove makes this work on any open area, not only marked tracks; holes in the
grid should be closed upstream (e.g. morphological closing) before extraction
because any non-drivable cell terminates the lateral sweep.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

# Cell classification codes. These mirror kart_map2d.mapping.evidence_grid
# (kept local so this module stays importable without cv2); override per call
# via drivable_class/obstacle_class if a foreign grid uses other codes.
CLASS_UNKNOWN = 0
CLASS_DRIVABLE = 1
CLASS_OBSTACLE = 2

_OUT_OF_BOUNDS = -1


# ------------------------------------------------------------------ geometry
def path_tangents(points: NDArray[np.float64], closed: bool) -> NDArray[np.float64]:
    """Unit tangents (N,2) of a polyline via central differences.

    Closed polylines wrap around (no duplicated end point expected); open ones
    use one-sided differences at the ends.
    """
    pts = np.asarray(points, dtype=float)
    if closed:
        d = np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0)
    else:
        d = np.empty_like(pts)
        d[1:-1] = pts[2:] - pts[:-2]
        d[0] = pts[1] - pts[0]
        d[-1] = pts[-1] - pts[-2]
    norm = np.linalg.norm(d, axis=1)
    norm[norm < 1e-12] = 1.0
    return d / norm[:, None]


def path_normals(points: NDArray[np.float64], closed: bool) -> NDArray[np.float64]:
    """Left unit normals (N,2): tangent rotated +90 deg (CCW)."""
    t = path_tangents(points, closed)
    return np.column_stack([-t[:, 1], t[:, 0]])


# ------------------------------------------------------------------ corridor
@dataclass
class Corridor:
    """A width-annotated centerline in the map frame.

    centerline : (N,2) station positions [m], uniformly spaced along arclength.
        For ``closed`` corridors the last point does NOT repeat the first;
        the loop wraps from index N-1 back to 0.
    w_left / w_right : (N,) lateral drivable extent [m] to the left/right of
        the direction of travel (left = +90 deg CCW from the tangent).
    closed : the driven path returned to its start with matching direction.
    s : (N,) arclength station coordinate [m] along the centerline.
    """

    centerline: NDArray[np.float64]
    w_left: NDArray[np.float64]
    w_right: NDArray[np.float64]
    closed: bool
    s: NDArray[np.float64]

    def normals_left(self) -> NDArray[np.float64]:
        return path_normals(self.centerline, self.closed)

    def left_edge(self) -> NDArray[np.float64]:
        return self.centerline + self.w_left[:, None] * self.normals_left()

    def right_edge(self) -> NDArray[np.float64]:
        return self.centerline - self.w_right[:, None] * self.normals_left()


def _resolve_grid(
    grid: Any,
    origin_xy: tuple[float, float],
    resolution_m: float,
) -> tuple[NDArray[np.integer], NDArray[np.float64], float]:
    """Accept a bare class array or any object exposing one.

    Three shapes are understood, normalised to ``classes[row=y, col=x]``:
    - a bare 2D int array (``origin_xy``/``resolution_m`` kwargs apply);
    - an ``EvidenceGrid``-like object with ``classify()`` (arrays laid out
      ``(nx, ny)``, hence transposed here), ``origin`` and ``cell``;
    - any object with ``classes`` (2D array, row=y layout) or
      ``classify_array()``, plus ``origin_xy`` and ``resolution_m`` (or
      ``resolution``) attributes.
    """
    if isinstance(grid, np.ndarray):
        classes = grid
    elif hasattr(grid, "classify") and hasattr(grid, "cell"):
        classes = np.asarray(grid.classify()).T  # (nx, ny) -> (ny, nx)
        origin_xy = tuple(np.asarray(grid.origin, dtype=float))
        resolution_m = float(grid.cell)
    else:
        classes = getattr(grid, "classes", None)
        if classes is None and hasattr(grid, "classify_array"):
            classes = grid.classify_array()
        if classes is None:
            raise TypeError("grid must be an ndarray or expose .classify() / .classes")
        origin_xy = tuple(getattr(grid, "origin_xy", origin_xy))
        resolution_m = float(getattr(grid, "resolution_m", getattr(grid, "resolution", resolution_m)))
    classes = np.asarray(classes)
    if classes.ndim != 2:
        raise ValueError(f"class grid must be 2D, got shape {classes.shape}")
    return classes, np.asarray(origin_xy, dtype=float), float(resolution_m)


def _sample_classes(
    classes: NDArray[np.integer],
    origin_xy: NDArray[np.float64],
    resolution_m: float,
    points: NDArray[np.float64],
) -> NDArray[np.int64]:
    """Nearest-cell class per query point; out-of-bounds -> _OUT_OF_BOUNDS."""
    col = np.floor((points[:, 0] - origin_xy[0]) / resolution_m).astype(np.int64)
    row = np.floor((points[:, 1] - origin_xy[1]) / resolution_m).astype(np.int64)
    h, w = classes.shape
    valid = (col >= 0) & (col < w) & (row >= 0) & (row < h)
    out = np.full(len(points), _OUT_OF_BOUNDS, dtype=np.int64)
    out[valid] = classes[row[valid], col[valid]].astype(np.int64)
    return out


def _dedupe(points: NDArray[np.float64], min_step: float = 1e-6) -> NDArray[np.float64]:
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return pts
    keep = np.ones(len(pts), dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(pts, axis=0), axis=1) > min_step
    return pts[keep]


def _detect_closed(traj: NDArray[np.float64], closure_dist_m: float) -> bool:
    """Closed iff the path returns near its start with a matching direction."""
    seg = np.linalg.norm(np.diff(traj, axis=0), axis=1)
    total = float(seg.sum())
    if len(traj) < 8 or total < 4.0 * closure_dist_m:
        return False
    if np.linalg.norm(traj[-1] - traj[0]) > closure_dist_m:
        return False

    # Direction over roughly the first / last metre of travel.
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    probe = min(1.0, 0.25 * total)
    i0 = int(np.searchsorted(cum, probe))
    i1 = int(np.searchsorted(cum, total - probe))
    d_start = traj[max(i0, 1)] - traj[0]
    d_end = traj[-1] - traj[min(i1, len(traj) - 2)]
    n0, n1 = np.linalg.norm(d_start), np.linalg.norm(d_end)
    if n0 < 1e-9 or n1 < 1e-9:
        return False
    return bool(float(d_start @ d_end) / (n0 * n1) > 0.5)


def _resample_stations(
    traj: NDArray[np.float64], spacing_m: float, closed: bool
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Uniform arclength resampling; returns (centerline (N,2), s (N,)).

    Spacing is adjusted so it divides the total length exactly - the
    minimum-curvature objective downstream assumes uniform station spacing.
    """
    pts = traj
    if closed:
        pts = np.vstack([pts, pts[:1]])  # include the closing segment
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    length = float(cum[-1])
    if length < spacing_m:
        raise ValueError(f"trajectory too short ({length:.2f} m) for spacing {spacing_m} m")
    if closed:
        n = max(8, int(round(length / spacing_m)))
        s = np.arange(n) * (length / n)
    else:
        n = max(3, int(round(length / spacing_m)) + 1)
        s = np.linspace(0.0, length, n)
    center = np.column_stack([np.interp(s, cum, pts[:, 0]), np.interp(s, cum, pts[:, 1])])
    return center, s


def _smooth_widths(w: NDArray[np.float64], window_n: int, closed: bool) -> NDArray[np.float64]:
    """Moving average; wraps for closed loops, repeats edges for open paths."""
    k = max(1, int(window_n))
    if k % 2 == 0:
        k += 1
    if k == 1 or len(w) < 3:
        return w.copy()
    k = min(k, len(w) if closed else 2 * len(w) - 1)
    if k % 2 == 0:  # the clamp can land on an even width; keep the kernel odd
        k -= 1
    pad = k // 2
    mode = "wrap" if closed else "edge"
    padded = np.pad(w, pad, mode=mode)
    return np.convolve(padded, np.ones(k) / k, mode="valid")


def extract_corridor(
    grid: Any,
    trajectory_xy: NDArray[np.float64],
    spacing_m: float = 0.5,
    *,
    origin_xy: tuple[float, float] = (0.0, 0.0),
    resolution_m: float = 0.1,
    max_half_width_m: float = 12.0,
    smooth_window_m: float = 2.0,
    closure_dist_m: float | None = None,
    drivable_class: int = CLASS_DRIVABLE,
    obstacle_class: int = CLASS_OBSTACLE,
) -> Corridor:
    """Walk the driven path and measure the drivable corridor around it.

    grid : 2D int class array, or an object exposing one (see _resolve_grid).
        ``origin_xy`` / ``resolution_m`` kwargs apply to bare arrays only.
    trajectory_xy : (M,2) driven positions in the map frame [m], in driving
        order. A loop that returns to its start (within ``closure_dist_m``,
        heading agreeing) is closed and the centerline wraps around.
    spacing_m : nominal station spacing along the path [m].
    max_half_width_m : lateral sweep cap per side [m].
    smooth_window_m : moving-average window applied to the raw widths [m].

    The lateral sweep starts at the station and stops at the first cell that
    is not ``drivable_class`` (obstacle, unknown, or outside the grid) - the
    corridor is the *contiguous* drivable extent. ``obstacle_class`` is
    accepted for symmetry and forward compatibility (e.g. per-cause widths);
    the strict contiguity rule does not distinguish blockers.
    """
    del obstacle_class  # blockers are treated uniformly; kept in the signature
    classes, origin, res = _resolve_grid(grid, origin_xy, resolution_m)
    traj = _dedupe(np.asarray(trajectory_xy, dtype=float))
    if traj.ndim != 2 or traj.shape[1] != 2 or len(traj) < 2:
        raise ValueError("trajectory_xy must be (M,2) with M >= 2")

    if closure_dist_m is None:
        closure_dist_m = max(2.0, 3.0 * spacing_m)
    closed = _detect_closed(traj, closure_dist_m)
    center, s = _resample_stations(traj, spacing_m, closed)
    n_left = path_normals(center, closed)

    # Lateral sweep: sample both sides at half-cell steps out to the cap.
    step = 0.5 * res
    n_samples = int(np.ceil(max_half_width_m / step))
    lateral = (np.arange(1, n_samples + 1) * step)[None, :, None]

    widths = []
    for sign in (+1.0, -1.0):  # +1 -> left, -1 -> right
        probe = center[:, None, :] + sign * lateral * n_left[:, None, :]
        cls = _sample_classes(classes, origin, res, probe.reshape(-1, 2))
        blocked = (cls != drivable_class).reshape(len(center), n_samples)
        first_block = np.where(blocked.any(axis=1), blocked.argmax(axis=1), n_samples)
        widths.append(np.minimum(first_block * step, max_half_width_m))
    w_left, w_right = widths

    # A station whose own cell is not drivable has no corridor there; the
    # smoothing pass then blends it with its neighbours instead of leaving a
    # spurious full-width reading from a sweep that started inside a blocker.
    center_cls = _sample_classes(classes, origin, res, center)
    dead = center_cls != drivable_class
    w_left[dead] = 0.0
    w_right[dead] = 0.0

    ds = s[1] - s[0] if len(s) > 1 else spacing_m
    window_n = int(round(smooth_window_m / max(ds, 1e-9)))
    w_left = _smooth_widths(w_left, window_n, closed)
    w_right = _smooth_widths(w_right, window_n, closed)

    return Corridor(centerline=center, w_left=w_left, w_right=w_right, closed=closed, s=s)


# ------------------------------------------------------------------- writers
def write_track_csv(corridor: Corridor, path: Path | str) -> Path:
    """TUMFTM race-track format: ``# x_m,y_m,w_tr_right_m,w_tr_left_m``."""
    path = Path(path)
    rows = np.column_stack(
        [corridor.centerline[:, 0], corridor.centerline[:, 1], corridor.w_right, corridor.w_left]
    )
    with path.open("w", encoding="ascii", newline="\n") as f:
        f.write("# x_m,y_m,w_tr_right_m,w_tr_left_m\n")
        for x, y, wr, wl in rows:
            f.write(f"{x:.3f},{y:.3f},{wr:.3f},{wl:.3f}\n")
    return path


def _ring(points: NDArray[np.float64], close: bool) -> list[list[float]]:
    coords = [[round(float(x), 3), round(float(y), 3)] for x, y in points]
    if close and coords and coords[0] != coords[-1]:
        coords.append(coords[0])
    return coords


def write_geojson(corridor: Corridor, path: Path | str) -> Path:
    """GeoJSON FeatureCollection in local map-frame metres (not lon/lat).

    Features: centerline, left/right edges (LineStrings, closed loops repeat
    their first vertex) and the corridor outline (Polygon; a closed loop is
    an annulus written as exterior ring plus hole).
    """
    path = Path(path)
    left = corridor.left_edge()
    right = corridor.right_edge()
    if corridor.closed:
        # An annulus must be exterior ring + hole (RFC 7946); stitching the
        # two edges into one ring would make a self-touching slit polygon.
        def _signed_area(pts: NDArray[np.float64]) -> float:
            x, y = pts[:, 0], pts[:, 1]
            return 0.5 * float(x @ np.roll(y, -1) - y @ np.roll(x, -1))

        rings = [left, right]
        areas = [_signed_area(r) for r in rings]
        outer_i = int(np.argmax(np.abs(areas)))
        outer, inner = rings[outer_i], rings[1 - outer_i]
        if areas[outer_i] < 0:  # exterior CCW, hole CW
            outer = outer[::-1]
        if areas[1 - outer_i] > 0:
            inner = inner[::-1]
        corridor_rings = [_ring(outer, True), _ring(inner, True)]
    else:
        corridor_rings = [_ring(np.vstack([left, right[::-1]]), True)]
    features = [
        {
            "type": "Feature",
            "properties": {"kind": "centerline", "closed": corridor.closed},
            "geometry": {"type": "LineString", "coordinates": _ring(corridor.centerline, corridor.closed)},
        },
        {
            "type": "Feature",
            "properties": {"kind": "edge_left"},
            "geometry": {"type": "LineString", "coordinates": _ring(left, corridor.closed)},
        },
        {
            "type": "Feature",
            "properties": {"kind": "edge_right"},
            "geometry": {"type": "LineString", "coordinates": _ring(right, corridor.closed)},
        },
        {
            "type": "Feature",
            "properties": {"kind": "corridor"},
            "geometry": {"type": "Polygon", "coordinates": corridor_rings},
        },
    ]
    doc = {
        "type": "FeatureCollection",
        "features": features,
        "crs_note": "local planar map frame, metres, origin at run start (not WGS84)",
    }
    path.write_text(json.dumps(doc), encoding="ascii")
    return path


def _agg_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _class_rgb(classes: NDArray[np.integer]) -> NDArray[np.float64]:
    """Muted background: unknown grey, drivable near-white, obstacle charcoal."""
    palette = {
        CLASS_DRIVABLE: (0.972, 0.972, 0.960),
        CLASS_OBSTACLE: (0.240, 0.240, 0.240),
    }
    rgb = np.full((*classes.shape, 3), 0.847)  # unknown / anything else
    for code, colour in palette.items():
        rgb[classes == code] = colour
    return rgb


def render_corridor_png(
    corridor: Corridor,
    grid: Any,
    path: Path | str,
    *,
    origin_xy: tuple[float, float] = (0.0, 0.0),
    resolution_m: float = 0.1,
    trajectory_xy: NDArray[np.float64] | None = None,
    dpi: int = 140,
) -> Path:
    """Debug overlay: class grid, corridor edges, centerline, driven path."""
    plt = _agg_pyplot()
    path = Path(path)
    classes, origin, res = _resolve_grid(grid, origin_xy, resolution_m)
    h, w = classes.shape
    extent = (origin[0], origin[0] + w * res, origin[1], origin[1] + h * res)

    fig, ax = plt.subplots(figsize=(8.0, 8.0 * h / max(w, 1)), dpi=dpi)
    ax.imshow(_class_rgb(classes), origin="lower", extent=extent, interpolation="nearest")

    def _closed_xy(pts: NDArray[np.float64]) -> tuple[NDArray, NDArray]:
        if corridor.closed:
            pts = np.vstack([pts, pts[:1]])
        return pts[:, 0], pts[:, 1]

    if trajectory_xy is not None:
        traj = np.asarray(trajectory_xy, dtype=float)
        ax.plot(traj[:, 0], traj[:, 1], color="#7a7a7a", lw=1.0, ls="--", label="driven path")
    ax.plot(*_closed_xy(corridor.centerline), color="#3b6fb5", lw=2.0, label="centerline")
    ax.plot(*_closed_xy(corridor.left_edge()), color="#c66d2a", lw=1.4, label="corridor edge")
    ax.plot(*_closed_xy(corridor.right_edge()), color="#c66d2a", lw=1.4)

    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("corridor" + (" (closed loop)" if corridor.closed else ""))
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
