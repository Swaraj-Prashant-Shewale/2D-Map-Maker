"""Map package: the artifact the drive stack loads (kart_map2d.mapping.map_package).

Synthetic fixtures only: a straight drivable band (as tests/test_pipeline.py)
gives a real corridor + racing line through the same pipeline stage the
offline build and the live node use. kart_map2d is 2D-only: the lidar3d
cloud tests of the research workspace are gone with the feature; a research
lidar3d package must still LOAD here (grid + line), which is tested below.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from kart_map2d.mapping import map_package as mp
from kart_map2d.mapping.evidence_grid import EvidenceGrid
from kart_map2d.pipeline import corridor_and_line, write_trajectory_csv


def _band_grid() -> EvidenceGrid:
    grid = EvidenceGrid(cell=0.25)
    xs = np.linspace(0.0, 20.0, 120)
    band = np.array([(x, y) for x in xs for y in np.linspace(-2.0, 2.0, 24)])
    for _ in range(4):
        grid.insert_lidar(np.column_stack([band, np.full(len(band), -0.10)]), np.zeros(3), carve=False)
    return grid


def _trajectory(n: int = 200):
    stamps = 1788593585.0 + np.arange(n) * 0.1
    xyyaw = np.column_stack([np.linspace(0.0, 20.0, n), np.zeros(n), np.zeros(n)])
    supported = np.ones(n, dtype=bool)
    supported[:3] = False
    return stamps, xyyaw, supported


def _save(out: Path, method: str = "flat", **kw) -> dict:
    stamps, xyyaw, supported = _trajectory()
    return mp.save_map_package(
        out,
        method=method,
        grid=_band_grid(),
        stamps=stamps,
        xyyaw=xyyaw,
        supported=supported,
        source={"kind": "test"},
        sensors=["imu", "vo_zed_right"],
        **kw,
    )


# ------------------------------------------------------------------ roundtrip
def test_save_and_load_roundtrip_flat(tmp_path: Path):
    meta = _save(tmp_path / "pkg")
    assert meta["kind"] == mp.META_KIND and meta["format"] == mp.MAP_PACKAGE_FORMAT
    assert meta["method"] == "flat" and meta["drivable"] is True and meta["problems"] == []
    assert set(meta["files"]) == {"grid.npz", "trajectory.csv", "corridor.csv", "racing_line.csv"}
    assert meta["resolution_m"] == 0.25 and meta["map_frame"]["frame_id"] == "map"
    assert meta["map_frame"]["anchor_xyyaw"] == [0.0, 0.0, 0.0]
    assert meta["software"]["package"] == "kart_map2d" and len(meta["software"]["source_sha256"]) == 64
    assert meta["created"]["utc"].endswith("Z") and meta["sensors"] == ["imu", "vo_zed_right"]

    pkg = mp.load_map_package(tmp_path / "pkg")
    assert pkg.method == "flat" and pkg.drivable
    stamps, xyyaw, supported = _trajectory()
    np.testing.assert_allclose(pkg.stamps, stamps, atol=5e-5)
    np.testing.assert_allclose(pkg.xyyaw, xyyaw, atol=5e-5)
    np.testing.assert_array_equal(pkg.supported, supported)
    np.testing.assert_array_equal(pkg.grid.classify(), _band_grid().classify())
    assert pkg.corridor.centerline.shape[1] == 2 and (pkg.corridor.w_left > 0).any()
    assert len(pkg.racing_line.s) == len(pkg.corridor.centerline)
    assert pkg.racing_line.closed is False and pkg.racing_line.time_s > 0
    assert (tmp_path / "pkg" / "grid.png").exists()  # informative preview, not validated


def test_save_map_package_has_no_cloud_parameter():
    """2D-only surface: save_map_package no longer accepts a cloud (lidar3d
    lives in the research workspace, not on the kart)."""
    import inspect

    assert "cloud" not in inspect.signature(mp.save_map_package).parameters


def test_research_lidar3d_package_loads_as_2d(tmp_path: Path):
    """A lidar3d package written by the research workspace still loads: the
    drive stack needs its planar grid + racing line; the cloud files are
    hash-verified but never parsed here."""
    _save(tmp_path / "pkg")
    meta_path = tmp_path / "pkg" / mp.META_FILENAME
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["method"] = "lidar3d"
    meta_path.write_text(json.dumps(meta), encoding="ascii")
    pkg = mp.load_map_package(tmp_path / "pkg", verify=False, require_drivable=False)
    assert pkg.method == "lidar3d"
    assert pkg.grid.classify().any()


def test_trajectory_csv_matches_the_pipeline_format(tmp_path: Path):
    stamps, xyyaw, supported = _trajectory(7)
    states = np.zeros((7, 8))
    states[:, :3] = xyyaw
    write_trajectory_csv(tmp_path / "a.csv", stamps, states, supported)
    mp.write_trajectory_csv(tmp_path / "b.csv", stamps, xyyaw, supported)
    assert (tmp_path / "a.csv").read_bytes() == (tmp_path / "b.csv").read_bytes()


# -------------------------------------------------------------- validation
def test_tampered_or_partial_files_are_refused(tmp_path: Path):
    _save(tmp_path / "pkg")
    corridor = tmp_path / "pkg" / "corridor.csv"
    text = corridor.read_text(encoding="ascii")
    corridor.write_text(text.replace("0.", "1.", 1), encoding="ascii", newline="\n")  # same size
    with pytest.raises(mp.MapPackageError, match="sha256"):
        mp.load_map_package(tmp_path / "pkg")
    corridor.write_text(text[:-10], encoding="ascii", newline="\n")  # truncated copy
    with pytest.raises(mp.MapPackageError, match="partial copy"):
        mp.load_map_package(tmp_path / "pkg")
    corridor.unlink()
    with pytest.raises(mp.MapPackageError, match="missing"):
        mp.load_map_package(tmp_path / "pkg")


def test_not_a_package_or_newer_format(tmp_path: Path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(mp.MapPackageError, match="not a map package"):
        mp.load_map_package(tmp_path / "empty")
    _save(tmp_path / "pkg")
    meta_path = tmp_path / "pkg" / mp.META_FILENAME
    meta = json.loads(meta_path.read_text(encoding="ascii"))
    meta["format"] = mp.MAP_PACKAGE_FORMAT + 1
    meta_path.write_text(json.dumps(meta), encoding="ascii")
    with pytest.raises(mp.MapPackageError, match="newer than supported"):
        mp.load_map_package(tmp_path / "pkg")


def test_grid_metadata_must_agree_with_the_meta(tmp_path: Path):
    _save(tmp_path / "pkg")
    meta_path = tmp_path / "pkg" / mp.META_FILENAME
    meta = json.loads(meta_path.read_text(encoding="ascii"))
    meta["resolution_m"] = 0.15
    meta_path.write_text(json.dumps(meta), encoding="ascii")
    with pytest.raises(mp.MapPackageError, match="cell 0.25"):
        mp.load_map_package(tmp_path / "pkg")


def test_meta_lists_only_what_this_build_produced(tmp_path: Path):
    """A stale corridor.csv from an earlier run in a reused folder is never claimed."""
    out = tmp_path / "pkg"
    out.mkdir()
    stamps, xyyaw, supported = _trajectory()
    grid = _band_grid()
    grid.save(out / "grid.npz")
    mp.write_trajectory_csv(out / "trajectory.csv", stamps, xyyaw, supported)
    (out / "corridor.csv").write_text("# stale\n1,2,3,4\n", encoding="ascii")
    mp.write_map_meta(
        out,
        method="flat",
        source={"kind": "offline"},
        sensors=["imu"],
        files=["trajectory.csv", "grid.npz", "report.json"],
        corridor_info={"error": "ValueError: too little drivable evidence"},
    )
    meta = json.loads((out / mp.META_FILENAME).read_text(encoding="ascii"))
    assert set(meta["files"]) == {"grid.npz", "trajectory.csv"}
    assert meta["drivable"] is False
    assert any("too little drivable evidence" in p for p in meta["problems"])
    with pytest.raises(mp.MapPackageError, match="not drivable"):
        mp.load_map_package(out)
    pkg = mp.load_map_package(out, require_drivable=False)  # still a valid (partial) map
    assert pkg.corridor is None and not pkg.drivable


def test_required_files_and_method(tmp_path: Path):
    out = tmp_path / "pkg"
    out.mkdir()
    _band_grid().save(out / "grid.npz")
    with pytest.raises(mp.MapPackageError, match="trajectory.csv"):
        mp.write_map_meta(out, method="flat", source={}, sensors=[])
    stamps, xyyaw, supported = _trajectory()
    mp.write_trajectory_csv(out / "trajectory.csv", stamps, xyyaw, supported)
    with pytest.raises(mp.MapPackageError, match="unknown map method"):
        mp.write_map_meta(out, method="lidar", source={}, sensors=[])
    # lidar3d stays a KNOWN method (research packages must load) even though
    # this package never writes one; validation still flags the missing twin.
    mp.write_map_meta(out, method="lidar3d", source={}, sensors=[])
    meta = json.loads((out / mp.META_FILENAME).read_text(encoding="utf-8"))
    # The write-time validation records the missing 3D twin; the 2D loader
    # itself no longer parses clouds, so the record lives in the meta.
    assert any("cloud_map.npz" in p for p in meta.get("problems", []))


def test_save_refuses_to_clobber_and_overwrite_removes_stale_files(tmp_path: Path):
    _save(tmp_path / "pkg")
    with pytest.raises(mp.MapPackageError, match="already holds"):
        _save(tmp_path / "pkg")
    # A stale file with a PACKAGE name must not survive an overwrite.
    (tmp_path / "pkg" / "cloud_map.npz").write_bytes(b"stale")
    meta = _save(tmp_path / "pkg", overwrite=True)
    assert meta["cloud"] is None
    assert not (tmp_path / "pkg" / "cloud_map.npz").exists()
    mp.load_map_package(tmp_path / "pkg")


def test_save_needs_a_trajectory(tmp_path: Path):
    with pytest.raises(mp.MapPackageError, match="trajectory"):
        mp.save_map_package(
            tmp_path / "pkg", method="flat", grid=_band_grid(), stamps=np.array([1.0]),
            xyyaw=np.zeros((1, 3)), supported=np.ones(1, bool), source={}, sensors=[],
        )


def test_copy_map_package_copies_exactly_the_listed_files(tmp_path: Path):
    _save(tmp_path / "pkg")
    (tmp_path / "pkg" / "scratch.tmp").write_text("x", encoding="ascii")
    mp.copy_map_package(tmp_path / "pkg", tmp_path / "kart")
    assert not (tmp_path / "kart" / "scratch.tmp").exists()
    mp.load_map_package(tmp_path / "kart")


def test_a_pipeline_style_folder_is_a_package(tmp_path: Path):
    """The offline pipeline writes its artifacts, then write_map_meta over report['artifacts']."""
    out = tmp_path / "sept3__camera_imu"
    out.mkdir()
    stamps, xyyaw, supported = _trajectory()
    states = np.zeros((len(stamps), 8))
    states[:, :3] = xyyaw
    grid = _band_grid()
    artifacts = {"trajectory_csv": str(write_trajectory_csv(out / "trajectory.csv", stamps, states, supported))}
    grid.save(out / "grid.npz")
    artifacts["grid_npz"] = str(out / "grid.npz")
    corridor, line = corridor_and_line(grid, xyyaw[supported, :2], out)
    for stem in ("corridor.csv", "corridor.geojson", "racing_line.csv"):
        artifacts[stem.replace(".", "_")] = str(out / stem)
    mp.write_map_meta(
        out, method="flat", source={"kind": "offline", "run": "sept3"}, sensors=["imu", "vo_zed_right"],
        files=[Path(p).name for p in artifacts.values()], corridor_info=corridor, racing_line_info=line,
        loop_closed=False,
    )
    pkg = mp.load_map_package(out)
    assert pkg.meta["corridor"]["length_m"] == corridor["length_m"]
    assert pkg.meta["racing_line"]["time_s"] == pytest.approx(line["time_s"], abs=0.01)
    assert pkg.meta["trajectory"]["loop_closed"] is False
