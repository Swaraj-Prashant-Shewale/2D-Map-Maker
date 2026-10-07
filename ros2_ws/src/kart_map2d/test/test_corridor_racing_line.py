"""Synthetic tests for corridor extraction and the minimum-curvature racing line."""

from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.spatial import cKDTree

from kart_map2d.mapping.corridor import (
    CLASS_DRIVABLE,
    CLASS_OBSTACLE,
    extract_corridor,
    path_normals,
    render_corridor_png,
    write_geojson,
    write_track_csv,
)
from kart_map2d.planning.racing_line import (
    compute_racing_line,
    render_racing_line_png,
    signed_curvature,
    write_racing_line_csv,
)

RES = 0.1


# ------------------------------------------------------------------ fixtures
def _grid_from_path(path_xy, half_width, band=0.6, pad=1.0):
    """Class grid: drivable within half_width of the path, obstacle band around."""
    lo = path_xy.min(axis=0) - (half_width + band + pad)
    hi = path_xy.max(axis=0) + (half_width + band + pad)
    w = int(np.ceil((hi[0] - lo[0]) / RES))
    h = int(np.ceil((hi[1] - lo[1]) / RES))
    xs = lo[0] + (np.arange(w) + 0.5) * RES
    ys = lo[1] + (np.arange(h) + 0.5) * RES
    gx, gy = np.meshgrid(xs, ys)
    d, _ = cKDTree(path_xy).query(np.column_stack([gx.ravel(), gy.ravel()]))
    d = d.reshape(h, w)
    classes = np.zeros((h, w), dtype=np.int8)
    classes[d <= half_width] = CLASS_DRIVABLE
    classes[(d > half_width) & (d <= half_width + band)] = CLASS_OBSTACLE
    return classes, (float(lo[0]), float(lo[1]))


def _u_path():
    """Up the left leg, over a 5 m semicircle, down the right leg (open)."""
    leg1 = np.column_stack([np.zeros(240), np.arange(240) * 0.05])  # (0,0)->(0,12)
    th = np.linspace(np.pi, 0.0, 315)
    arc = np.column_stack([5.0 + 5.0 * np.cos(th), 12.0 + 5.0 * np.sin(th)])
    leg2 = np.column_stack([np.full(240, 10.0), 12.0 - np.arange(240) * 0.05])
    return np.vstack([leg1, arc, leg2])


def _ring_grid(radius=15.0, half_width=2.5, band=0.6):
    """Annulus grid computed radially (no KD-tree needed)."""
    ext = radius + half_width + band + 1.0
    n = int(np.ceil(2 * ext / RES))
    xs = -ext + (np.arange(n) + 0.5) * RES
    gx, gy = np.meshgrid(xs, xs)
    dr = np.abs(np.hypot(gx, gy) - radius)
    classes = np.zeros((n, n), dtype=np.int8)
    classes[dr <= half_width] = CLASS_DRIVABLE
    classes[(dr > half_width) & (dr <= half_width + band)] = CLASS_OBSTACLE
    th = np.arange(0.0, 2 * np.pi - 0.005, 0.01)
    traj = radius * np.column_stack([np.cos(th), np.sin(th)])
    return classes, (-ext, -ext), traj


def _corner_path(radius=6.0):
    """Straight +x, 90 deg left turn of the given radius, straight +y (open)."""
    a = np.column_stack([np.arange(-14.0, 0.0, 0.05), np.zeros(280)])
    th = np.linspace(-np.pi / 2, 0.0, 190)
    arc = np.column_stack([radius * np.cos(th), 6.0 + radius * np.sin(th)])
    b = np.column_stack([np.full(280, 6.0), np.arange(6.0, 20.0, 0.05)])
    return np.vstack([a, arc, b])


# ------------------------------------------------------------------ corridor
def test_u_shape_widths():
    path = _u_path()
    classes, origin = _grid_from_path(path, half_width=2.0)
    cor = extract_corridor(classes, path, spacing_m=0.5, origin_xy=origin, resolution_m=RES)
    assert not cor.closed
    assert len(cor.centerline) == len(cor.w_left) == len(cor.w_right) == len(cor.s)
    interior = slice(3, -3)
    assert np.all(np.abs(cor.w_left[interior] - 2.0) < 0.25)
    assert np.all(np.abs(cor.w_right[interior] - 2.0) < 0.25)
    # Stations follow the driven path.
    d, _ = cKDTree(path).query(cor.centerline)
    assert d.max() < 0.05


def test_obstacle_narrows_corridor():
    path = _u_path()
    classes, origin = _grid_from_path(path, half_width=2.0)
    # Obstacle disc 1.0 m to the LEFT of the up-leg (heading +y -> left = -x).
    h, w = classes.shape
    xs = origin[0] + (np.arange(w) + 0.5) * RES
    ys = origin[1] + (np.arange(h) + 0.5) * RES
    gx, gy = np.meshgrid(xs, ys)
    classes[np.hypot(gx + 1.0, gy - 6.0) <= 0.3] = CLASS_OBSTACLE

    cor = extract_corridor(
        classes, path, spacing_m=0.5, origin_xy=origin, resolution_m=RES, smooth_window_m=0.5
    )
    i = int(np.argmin(np.linalg.norm(cor.centerline - [0.0, 6.0], axis=1)))
    assert cor.w_left[i] < 1.0  # sweep stops at the obstacle
    assert cor.w_right[i] > 1.7  # other side untouched


def test_ring_closed_with_correct_widths():
    classes, origin, traj = _ring_grid()
    cor = extract_corridor(classes, traj, spacing_m=0.5, origin_xy=origin, resolution_m=RES)
    assert cor.closed
    assert np.all(np.abs(cor.w_left - 2.5) < 0.3)
    assert np.all(np.abs(cor.w_right - 2.5) < 0.3)
    # Closed centerline: wrap gap is one station, not a torn seam.
    wrap_gap = np.linalg.norm(cor.centerline[0] - cor.centerline[-1])
    assert wrap_gap < 0.75


# --------------------------------------------------------------- racing line
def test_ring_racing_line_hugs_inside_and_stays_in_bounds():
    classes, origin, traj = _ring_grid()
    cor = extract_corridor(classes, traj, spacing_m=0.5, origin_xy=origin, resolution_m=RES)
    margin = 0.3
    line = compute_racing_line(cor, margin_m=margin)

    assert np.all(line.alpha >= -(cor.w_left - margin) - 1e-6)
    assert np.all(line.alpha <= (cor.w_right - margin) + 1e-6)

    # Minimum-curvature on a ring tightens onto the INSIDE bound: the offset
    # path covers less distance per station there, so its station-parameter
    # second differences shrink with the radius.
    radii = np.linalg.norm(line.path, axis=1)
    r_center = np.linalg.norm(cor.centerline, axis=1).mean()
    assert radii.mean() < r_center - 1.0
    assert radii.min() > r_center - 2.5 - 0.3  # never through the inner edge


def test_ring_speed_profile_periodic_and_within_a_lat():
    classes, origin, traj = _ring_grid()
    cor = extract_corridor(classes, traj, spacing_m=0.5, origin_xy=origin, resolution_m=RES)
    a_lat, a_acc, a_brk, v_max = 8.0, 4.0, 6.0, 15.0
    line = compute_racing_line(cor, a_lat_max=a_lat, a_accel=a_acc, a_brake=a_brk, v_max=v_max)

    assert np.all(line.speed <= v_max + 1e-9)
    assert np.all(line.speed**2 * np.abs(line.curvature) <= a_lat * 1.05)
    assert line.time_s > 0.0

    # Periodicity: the wrap-around edge obeys the same accel/brake limits.
    ds_wrap = np.linalg.norm(line.path[0] - line.path[-1])
    v0, v1 = line.speed[-1], line.speed[0]
    assert v1**2 - v0**2 <= 2.0 * a_acc * ds_wrap + 1e-6
    assert v0**2 - v1**2 <= 2.0 * a_brk * ds_wrap + 1e-6


def test_corner_racing_line_flattens_curvature():
    path = _corner_path(radius=6.0)
    classes, origin = _grid_from_path(path, half_width=3.0)
    cor = extract_corridor(classes, path, spacing_m=0.5, origin_xy=origin, resolution_m=RES)
    assert not cor.closed
    line = compute_racing_line(cor, margin_m=0.3)

    k_center = np.max(np.abs(signed_curvature(cor.centerline, closed=False)))
    k_line = np.max(np.abs(line.curvature))
    assert 0.12 < k_center < 0.22  # sanity: the corner really is ~1/6 1/m
    ratio = k_line / k_center
    assert ratio < 0.85, f"racing line should flatten the corner, got ratio {ratio:.3f}"

    # Longitudinal limits hold along the whole open segment.
    seg = np.linalg.norm(np.diff(line.path, axis=0), axis=1)
    a_long = np.diff(line.speed**2) / (2.0 * seg)
    assert np.all(a_long <= 4.0 + 1e-6)
    assert np.all(a_long >= -(6.0 + 1e-6))


# ------------------------------------------------------------------- writers
def test_writers(tmp_path):
    classes, origin, traj = _ring_grid()
    cor = extract_corridor(classes, traj, spacing_m=0.5, origin_xy=origin, resolution_m=RES)
    line = compute_racing_line(cor)

    csv_path = write_track_csv(cor, tmp_path / "track.csv")
    lines = csv_path.read_text(encoding="ascii").splitlines()
    assert lines[0] == "# x_m,y_m,w_tr_right_m,w_tr_left_m"
    assert len(lines) == len(cor.centerline) + 1
    x, y, wr, wl = (float(v) for v in lines[1].split(","))
    assert abs(np.hypot(x, y) - 15.0) < 0.5 and 2.0 < wr < 3.0 and 2.0 < wl < 3.0

    gj = json.loads((write_geojson(cor, tmp_path / "track.geojson")).read_text(encoding="ascii"))
    kinds = {f["properties"]["kind"] for f in gj["features"]}
    assert kinds == {"centerline", "edge_left", "edge_right", "corridor"}
    poly = next(f for f in gj["features"] if f["properties"]["kind"] == "corridor")
    ring = poly["geometry"]["coordinates"][0]
    assert ring[0] == ring[-1]  # valid closed polygon ring

    rl_csv = write_racing_line_csv(line, tmp_path / "line.csv")
    rl_lines = rl_csv.read_text(encoding="ascii").splitlines()
    assert rl_lines[1] == "# s_m,x_m,y_m,kappa_radpm,v_mps"
    assert len(rl_lines) == len(line.path) + 2

    png1 = render_corridor_png(
        cor, classes, tmp_path / "corridor.png", origin_xy=origin, resolution_m=RES, trajectory_xy=traj
    )
    png2 = render_racing_line_png(line, cor, tmp_path / "line.png")
    assert png1.stat().st_size > 5000
    assert png2.stat().st_size > 5000


def test_geojson_closed_loop_is_annulus_with_hole(tmp_path):
    """A closed corridor polygon must be exterior + hole, not a slit ring."""
    classes, origin, traj = _ring_grid()
    cor = extract_corridor(classes, traj, spacing_m=0.5, origin_xy=origin, resolution_m=RES)
    assert cor.closed
    gj = json.loads(write_geojson(cor, tmp_path / "gj_test.geojson").read_text())
    poly = next(f for f in gj["features"] if f["properties"]["kind"] == "corridor")
    rings = poly["geometry"]["coordinates"]
    assert len(rings) == 2
    for ring in rings:
        r = np.hypot(*np.asarray(ring).T)
        assert ring[0] == ring[-1]
        assert r.max() - r.min() < 1.0  # each ring stays on one edge

    def area(ring):
        a = np.asarray(ring)
        return 0.5 * float(a[:, 0] @ np.roll(a[:, 1], -1) - a[:, 1] @ np.roll(a[:, 0], -1))

    assert area(rings[0]) > 0  # exterior CCW
    assert area(rings[1]) < 0  # hole CW


def test_smooth_widths_length_preserved_small_closed_loop():
    """Window wider than an even-length closed loop must not grow the array."""
    from kart_map2d.mapping.corridor import _smooth_widths

    for n in (8, 9, 10):
        w = np.arange(n, dtype=float)
        out = _smooth_widths(w, window_n=25, closed=True)
        assert len(out) == n
        assert np.all(np.isfinite(out))


def test_evidence_grid_like_object_transposed_layout():
    """Objects with classify()/origin/cell use the (nx, ny) layout of EvidenceGrid."""
    path = np.column_stack([np.linspace(0.0, 20.0, 201), np.zeros(201)])
    classes_row, origin = _grid_from_path(path, half_width=1.5)

    class FakeGrid:
        cell = RES

        def __init__(self, classes_xy, origin_xy):
            self._classes = classes_xy
            self.origin = np.asarray(origin_xy)

        def classify(self):
            return self._classes

    cor = extract_corridor(FakeGrid(classes_row.T, origin), path, spacing_m=0.5)
    interior = slice(2, -2)
    assert np.all(np.abs(cor.w_left[interior] - 1.5) < 0.25)
    assert np.all(np.abs(cor.w_right[interior] - 1.5) < 0.25)


def test_normals_are_left_unit():
    pts = np.column_stack([np.arange(10.0), np.zeros(10)])  # heading +x
    n = path_normals(pts, closed=False)
    assert np.allclose(n, [0.0, 1.0])  # left of +x is +y
    assert np.allclose(np.linalg.norm(n, axis=1), 1.0)


def test_rejects_bad_inputs():
    with pytest.raises(ValueError):
        extract_corridor(np.zeros((4, 4, 2), dtype=np.int8), np.zeros((5, 2)))
    with pytest.raises(ValueError):
        extract_corridor(np.zeros((10, 10), dtype=np.int8), np.zeros((1, 2)))
