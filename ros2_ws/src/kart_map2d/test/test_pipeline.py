"""Tests for the end-to-end pipeline plumbing and the VO replay source.

Everything runs on synthetic fixtures: no recording under inputs/, no GPU and
no model download is needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from kart_map2d.core.measurements import RelativeMotion
from kart_map2d.mapping.evidence_grid import EvidenceGrid
from kart_map2d.pipeline import (
    cameras_for_sources,
    closed_classes,
    corridor_and_line,
    supported_mask,
    write_trajectory_csv,
)
from kart_map2d.replay import (
    VO_ARRIVAL_DELAY_S,
    load_vo_cache,
    run_name,
    save_vo_cache,
    vo_motions,
)


def _motion(t0: float, t1: float, channel: str = "vo_zed_right") -> RelativeMotion:
    return RelativeMotion(
        stamp=t1,
        channel=channel,
        start_stamp=t0,
        delta_forward=0.5 * (t1 - t0),
        delta_left=0.01,
        delta_yaw=0.02,
        sigma_translation=0.08,
        sigma_yaw=0.009,
        sigma_position=0.05,
        quality=0.8,
    )


# ------------------------------------------------------------------ VO cache
def test_vo_cache_roundtrip(tmp_path: Path):
    motions = [_motion(0.0, 0.3), _motion(0.3, 0.7), _motion(0.7, 1.4)]
    cache = tmp_path / "run_vo_zed_right.npz"
    save_vo_cache(cache, motions)
    loaded = load_vo_cache(cache, "vo_zed_right")
    assert loaded == motions  # frozen dataclasses compare by field values


def test_vo_cache_roundtrip_empty(tmp_path: Path):
    cache = tmp_path / "empty.npz"
    save_vo_cache(cache, [])
    assert load_vo_cache(cache, "vo_zed_left") == []


def test_vo_motions_uses_cache_without_touching_frames(tmp_path: Path):
    """A cached run/camera must not open images or even query the run."""
    motions = [_motion(10.0, 10.4)]
    save_vo_cache(tmp_path / "myrun_vo_zed_right.npz", motions)

    class ExplodingRun:
        root = Path("inputs") / "myrun"

        def __getattr__(self, name):  # any data access would fail the test
            raise AssertionError(f"unexpected access to run.{name}")

    out = vo_motions(ExplodingRun(), "zed_right", cache_dir=tmp_path)
    assert out == motions
    assert out[0].channel == "vo_zed_right"


def test_vo_cache_from_other_vo_code_is_not_reused(tmp_path: Path):
    """A cache without the current VO fingerprint (older VO code) is recomputed."""
    rows = np.zeros((1, 9))
    np.savez_compressed(tmp_path / "myrun_vo_zed_right.npz", rows=rows)  # pre-fingerprint cache

    class ExplodingRun:
        root = Path("inputs") / "myrun"

        def __getattr__(self, name):
            raise AssertionError(f"recompute touched run.{name}")

    with pytest.raises(AssertionError, match="recompute touched"):
        vo_motions(ExplodingRun(), "zed_right", cache_dir=tmp_path)


def test_run_name_handles_export_subdirectory():
    assert run_name(SimpleNamespace(root=Path("inputs/foo"))) == "foo"
    assert run_name(SimpleNamespace(root=Path("inputs/foo/export"))) == "foo"


def test_vo_arrival_delay_is_realistic():
    assert 0.05 <= VO_ARRIVAL_DELAY_S <= 0.5


# ------------------------------------------------------------- support gating
def test_supported_mask_interior_and_extrapolation():
    stamps = np.array([0.0, 1.0, 2.0, 3.0])
    supported = np.array([True, True, False, True])
    t = np.array([-0.5, 0.2, 1.9, 2.5, 3.5])
    out = supported_mask(t, stamps, supported)
    # Outside the span -> False; near the unsupported sample -> False;
    # bracketed by supported samples -> True.
    assert out.tolist() == [False, True, False, False, False]


def test_supported_mask_empty_trajectory():
    out = supported_mask(np.array([1.0, 2.0]), np.empty(0), np.empty(0, dtype=bool))
    assert not out.any()


def test_cameras_for_sources():
    sources = ("imu", "vo_zed_right", "lidar_odom", "vo_zed_left")
    assert cameras_for_sources(sources) == ["zed_right", "zed_left"]
    assert cameras_for_sources(("imu", "odom")) == []


def test_closed_classes_bridges_pinholes_but_keeps_obstacles():
    grid = EvidenceGrid(cell=0.5)
    xy = np.array([(x * 0.5, 0.0) for x in range(10)])
    pts = np.column_stack([xy, np.full(len(xy), -0.10)])
    for _ in range(4):
        grid.insert_lidar(pts, np.zeros(3), carve=False)
    raw = np.asarray(grid.classify()).T
    # Punch a pinhole into the drivable row and add an obstacle next to it.
    row, cols = np.nonzero(raw == 1)
    r = row[0]
    grid_cls = raw.copy()
    hole_col = cols[len(cols) // 2]
    grid._lo_drv[hole_col, r] = 0.0  # (nx, ny) layout: clear one cell
    grid._lo_obs[cols[-1], r] = 10.0  # strong obstacle at the row's end
    closed = closed_classes(grid, close_cells=2)
    assert closed[r, hole_col] == 1  # pinhole bridged
    assert closed[r, cols[-1]] == 2  # obstacle never overwritten
    assert closed_classes(EvidenceGrid(), close_cells=2).size == 0  # empty-safe


# ------------------------------------------------------- graceful degradation
def test_corridor_and_line_degrades_on_empty_grid(tmp_path: Path):
    grid = EvidenceGrid()
    traj = np.column_stack([np.linspace(0.0, 5.0, 20), np.zeros(20)])
    corridor_report, line_report = corridor_and_line(grid, traj, tmp_path)
    assert "error" in corridor_report
    assert "error" in line_report
    assert not (tmp_path / "racing_line.csv").exists()


def test_corridor_and_line_on_synthetic_straight(tmp_path: Path):
    """A straight drivable band yields a corridor, a line and all artifacts."""
    grid = EvidenceGrid(cell=0.25)
    xs = np.linspace(0.0, 20.0, 120)
    band = np.array([(x, y) for x in xs for y in np.linspace(-2.0, 2.0, 24)])
    for _ in range(4):  # enough repeated evidence to clear the class threshold
        grid.insert_lidar(
            np.column_stack([band, np.full(len(band), -0.10)]),
            np.zeros(3),
            carve=False,
        )
    traj = np.column_stack([xs, np.zeros_like(xs)])
    corridor_report, line_report = corridor_and_line(grid, traj, tmp_path)
    assert "error" not in corridor_report, corridor_report
    assert corridor_report["length_m"] == pytest.approx(20.0, abs=1.0)
    assert not corridor_report["closed"]
    assert line_report["status"] == "ok"
    assert line_report["time_s"] > 0.0
    for stem in ("corridor.csv", "corridor.geojson", "corridor.png", "racing_line.csv", "racing_line.png"):
        assert (tmp_path / stem).exists(), stem
    # GeoJSON stays parseable ASCII.
    doc = json.loads((tmp_path / "corridor.geojson").read_text(encoding="ascii"))
    assert doc["type"] == "FeatureCollection"


# ------------------------------------------------------------------ artifacts
def test_write_trajectory_csv(tmp_path: Path):
    stamps = np.array([0.0, 0.5])
    states = np.zeros((2, 8))
    states[1, :3] = [1.0, -2.0, 0.5]
    path = write_trajectory_csv(tmp_path / "traj.csv", stamps, states, np.array([True, False]))
    lines = path.read_text(encoding="ascii").strip().splitlines()
    assert lines[0].startswith("#")
    assert lines[1] == "0.0000,0.0000,0.0000,0.00000,1"
    assert lines[2] == "0.5000,1.0000,-2.0000,0.50000,0"
