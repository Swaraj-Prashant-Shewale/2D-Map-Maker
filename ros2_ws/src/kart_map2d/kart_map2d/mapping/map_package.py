"""Map package: the on-disk artifact the drive stack loads.

A map package is a plain directory. Every map the project builds - an
offline ``scripts/run_pipeline.py`` output folder or a live node's
``~/save_map`` - is one, so the drive stack has exactly one thing to load:

==================  =========  ==================================================
file                needed     content
==================  =========  ==================================================
``map_meta.json``   always     what this map is: method, map-frame definition,
                               grid origin + resolution, created stamp, source
                               (offline run / live node), sensors used,
                               software version, and the size + sha256 of every
                               package file (so a partial copy is detected)
``grid.npz``        always     :class:`~kart_map2d.mapping.evidence_grid.EvidenceGrid`
                               (``classify()``: 0 unknown / 1 drivable / 2 obstacle)
``trajectory.csv``  always     ``# stamp_s,x_m,y_m,yaw_rad,supported`` - the poses
                               the map was built at (loop-closed when closures
                               were accepted)
``corridor.csv``    to drive   TUMFTM ``# x_m,y_m,w_tr_right_m,w_tr_left_m``
``racing_line.csv`` to drive   ``# s_m,x_m,y_m,kappa_radpm,v_mps`` (+ a
                               ``# closed=.. time_s=..`` header line)
``cloud_map.npz``   lidar3d    :class:`~kart_map2d.mapping.cloud_map.CloudMapBuilder`
``cloud_map.ply``   lidar3d    the same cloud as binary PLY (viewers)
==================  =========  ==================================================

Anything else in the folder (PNG previews, ``report.json``, ``modes.json``,
``corridor.geojson``, ``live_stats.json``) is informative and not validated.
``map_meta.json`` is written LAST and atomically, so a folder without it is
an unfinished build, never a half-valid map.

Map frame (all files): planar, metres, anchored at the kart's pose when the
fusion started - origin = start position, +x = start heading, +y = left,
z up, yaw counter-clockwise from +x in radians (trackmapper conventions).
The grid's ``origin`` is the map-frame position of the outer corner of cell
(0, 0); cell (ix, iy) covers ``origin + [ix, ix + 1) * resolution`` in x and
the same in y.

This module imports only numpy (+ the grid / cloud classes); the corridor and
racing-line stages of :func:`save_map_package` import
:mod:`kart_map2d.pipeline` lazily, which is what the pipeline itself runs.
"""

from __future__ import annotations

import datetime
import functools
import hashlib
import json
import os
import platform
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from kart_map2d.mapping.corridor import Corridor
from kart_map2d.mapping.evidence_grid import EvidenceGrid

__all__ = [
    "MAP_PACKAGE_FORMAT",
    "META_FILENAME",
    "PACKAGE_FILES",
    "MAP_FRAME",
    "MapPackage",
    "MapPackageError",
    "RacingLineData",
    "copy_map_package",
    "load_map_package",
    "save_map_package",
    "software_info",
    "validate_map_package",
    "write_map_meta",
    "write_trajectory_csv",
]

#: map_meta.json schema version; a newer file is refused by older readers.
MAP_PACKAGE_FORMAT = 1
META_KIND = "trackmapper_map_package"
META_FILENAME = "map_meta.json"

#: Role -> file name of every VALIDATED package file.
PACKAGE_FILES: dict[str, str] = {
    "grid": "grid.npz",
    "trajectory": "trajectory.csv",
    "corridor": "corridor.csv",
    "racing_line": "racing_line.csv",
    "cloud_npz": "cloud_map.npz",
    "cloud_ply": "cloud_map.ply",
}
_REQUIRED = ("grid", "trajectory")
_METHODS = ("flat", "lidar3d", "camera3d")  # == output_toggles.OUTPUT_MODES (kept import-free)

#: The map-frame definition every package carries verbatim.
MAP_FRAME: dict[str, Any] = {
    "frame_id": "map",
    "type": "planar",
    "units": "m, rad, s (epoch)",
    "definition": (
        "anchored at the kart's pose when the fusion started: origin = start position, "
        "+x = start heading, +y = left, z up; yaw counter-clockwise from +x"
    ),
    "base_link": "x forward, y left, z up",
    "grid_cell_rule": "cell (ix, iy) covers origin + [i, i+1) * resolution_m on each axis",
    "grid_classes": {"0": "unknown", "1": "drivable", "2": "obstacle"},
}

_TRAJ_HEADER = "# stamp_s,x_m,y_m,yaw_rad,supported"
_CORRIDOR_HEADER = "# x_m,y_m,w_tr_right_m,w_tr_left_m"
_LINE_HEADER = "# s_m,x_m,y_m,kappa_radpm,v_mps"


class MapPackageError(ValueError):
    """A map package that must not be used as written (the message lists why)."""


# ------------------------------------------------------------------ writers
def write_trajectory_csv(
    path: Path | str,
    stamps: NDArray[np.float64],
    xyyaw: NDArray[np.float64],
    supported: NDArray[np.bool_],
) -> Path:
    """``trajectory.csv`` in exactly the offline pipeline's format.

    ``xyyaw`` is (N, >=3) map-frame x, y [m], yaw [rad] (extra columns, e.g.
    full EKF states, are ignored), ``supported`` the motion-support flag.
    """
    path = Path(path)
    states = np.asarray(xyyaw, dtype=np.float64)
    with path.open("w", encoding="ascii", newline="\n") as f:
        f.write(_TRAJ_HEADER + "\n")
        for t, s, ok in zip(stamps, states, supported):
            f.write(f"{t:.4f},{s[0]:.4f},{s[1]:.4f},{s[2]:.5f},{int(ok)}\n")
    return path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


@functools.lru_cache(maxsize=1)
def software_info() -> dict:
    """Which code built the map: a sha256 over the kart_map2d sources + versions.

    The fingerprint is over the package's own ``.py`` files (relative path +
    bytes, sorted), so it identifies the exact code on the dev laptop and on
    the Orin alike, git or not. Cached: the sources do not change while a
    process runs.
    """
    import kart_map2d

    root = Path(kart_map2d.__file__).resolve().parent
    digest = hashlib.sha256()
    files = sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("ascii"))
        digest.update(path.read_bytes())
    info = {
        "package": "kart_map2d",
        "version": getattr(kart_map2d, "__version__", None),
        "source_sha256": digest.hexdigest(),
        "source_files": len(files),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    try:  # cv2 is always there on the kart; never fail a save over a version string
        import cv2

        info["opencv"] = cv2.__version__
    except Exception:  # noqa: BLE001
        pass
    return info


def _read_npz_fields(path: Path, names: Sequence[str]) -> dict:
    with np.load(Path(path)) as data:
        return {n: data[n] for n in names if n in data.files}


def _racing_line_header(path: Path) -> dict:
    """``closed`` / ``time_s`` from racing_line.csv's first comment line."""
    with Path(path).open("r", encoding="ascii") as f:
        first = f.readline()
    out: dict = {}
    for token in first.lstrip("#").split():
        key, _, value = token.partition("=")
        if key == "closed":
            out["closed"] = value == "1"
        elif key == "time_s":
            out["time_s"] = float(value)
    return out


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="ascii", newline="\n") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.generic):
        return _json_safe(obj.item())
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, Path):
        return obj.as_posix()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def write_map_meta(
    out_dir: Path | str,
    *,
    method: str,
    source: Mapping[str, Any],
    sensors: Sequence[str],
    files: Iterable[str] | None = None,
    corridor_info: Mapping[str, Any] | None = None,
    racing_line_info: Mapping[str, Any] | None = None,
    loop_closed: bool | None = None,
    build: Mapping[str, Any] | None = None,
    problems: Sequence[str] = (),
) -> Path:
    """Write ``map_meta.json`` for the package files already in ``out_dir``.

    ``files`` names the artifacts THIS build produced (e.g. the pipeline's
    ``report["artifacts"]``); only package files among them are listed and
    hashed, so a stale ``corridor.csv`` left over from an earlier run in a
    reused folder is never claimed. ``None`` takes every package file present.
    ``corridor_info`` / ``racing_line_info`` are the builder's report
    sections (their ``error`` becomes a listed problem when the file is
    missing). Raises :class:`MapPackageError` when grid.npz or
    trajectory.csv is missing, or ``method`` is unknown. Written atomically.
    """
    out_dir = Path(out_dir)
    if method not in _METHODS:
        raise MapPackageError(f"unknown map method {method!r} (use {' | '.join(_METHODS)})")
    names = set(PACKAGE_FILES.values())
    if files is not None:
        produced = {Path(f).name for f in files} & names
    else:
        produced = {n for n in names if (out_dir / n).is_file()}
    present = {role: name for role, name in PACKAGE_FILES.items() if name in produced and (out_dir / name).is_file()}
    missing = [PACKAGE_FILES[r] for r in _REQUIRED if r not in present]
    if missing:
        raise MapPackageError(f"{out_dir}: cannot write a map package without {', '.join(missing)}")

    problems = list(problems)
    grid = _read_npz_fields(out_dir / PACKAGE_FILES["grid"], ("cell", "origin", "lo_drivable"))
    resolution = float(grid["cell"])
    origin = [float(v) for v in np.asarray(grid["origin"], dtype=np.float64)]
    shape = [int(v) for v in grid["lo_drivable"].shape]

    traj = _load_trajectory(out_dir / PACKAGE_FILES["trajectory"])
    t, xy = traj[:, 0], traj[:, 1:3]
    trajectory = {
        "file": PACKAGE_FILES["trajectory"],
        "poses": int(len(t)),
        "stamp_first_s": float(t[0]) if len(t) else None,
        "stamp_last_s": float(t[-1]) if len(t) else None,
        "distance_m": round(float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()), 2) if len(t) > 1 else 0.0,
        "supported_fraction": round(float(traj[:, 4].mean()), 3) if len(t) else 0.0,
        "loop_closed": loop_closed,
    }
    map_frame = dict(MAP_FRAME)
    map_frame["anchor_stamp_s"] = trajectory["stamp_first_s"]
    map_frame["anchor_xyyaw"] = [round(float(v), 6) for v in traj[0, 1:4]] if len(t) else None

    def section(role: str, info: Mapping[str, Any] | None, fmt: str) -> dict | None:
        name = PACKAGE_FILES[role]
        if role not in present:
            reason = (info or {}).get("error") if info else None
            problems.append(f"{name} missing" + (f": {reason}" if reason else ""))
            return None
        out = {"file": name, "format": fmt}
        out.update({k: v for k, v in (info or {}).items() if k != "error"})
        return out

    corridor = section("corridor", corridor_info, "TUMFTM " + _CORRIDOR_HEADER[2:])
    racing_line = section("racing_line", racing_line_info, _LINE_HEADER[2:])
    if racing_line is not None:
        racing_line.update(_racing_line_header(out_dir / PACKAGE_FILES["racing_line"]))
    if corridor is not None and "closed" not in corridor and racing_line is not None:
        corridor["closed"] = racing_line.get("closed")  # the line inherits the corridor's topology

    cloud = None
    if "cloud_npz" in present:
        fields_ = _read_npz_fields(out_dir / PACKAGE_FILES["cloud_npz"], ("resolution", "keys"))
        cloud = {
            "file": PACKAGE_FILES["cloud_npz"],
            "ply": PACKAGE_FILES["cloud_ply"] if "cloud_ply" in present else None,
            "resolution_m": float(fields_["resolution"]),
            "voxels": int(len(fields_["keys"])),
        }
        if method == "lidar3d" and cloud["voxels"] == 0:
            # e.g. a live lidar3d session whose SE3 odometry never arrived:
            # the flat layers are fine, the twin is not there.
            problems.append(f"{PACKAGE_FILES['cloud_npz']} holds 0 voxels: the lidar3d 3D twin is empty")
    elif method == "lidar3d":
        problems.append(f"{PACKAGE_FILES['cloud_npz']} missing: a lidar3d map carries the 3D twin")

    now = time.time()
    meta = {
        "kind": META_KIND,
        "format": MAP_PACKAGE_FORMAT,
        "method": method,
        "created": {
            "utc": datetime.datetime.fromtimestamp(now, tz=datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "unix_s": round(now, 3),
        },
        "source": dict(source),
        "sensors": list(sensors),
        "map_frame": map_frame,
        "origin": origin,
        "resolution_m": resolution,
        "grid": {"file": PACKAGE_FILES["grid"], "origin_xy_m": origin, "resolution_m": resolution, "shape": shape},
        "trajectory": trajectory,
        "corridor": corridor,
        "racing_line": racing_line,
        "cloud": cloud,
        "drivable": corridor is not None and racing_line is not None,
        "problems": problems,
        "software": software_info(),
        "build": dict(build or {}),
        "files": {
            name: {"bytes": (out_dir / name).stat().st_size, "sha256": _sha256(out_dir / name)}
            for name in sorted(present.values())
        },
    }
    path = out_dir / META_FILENAME
    _atomic_write_text(path, json.dumps(_json_safe(meta), indent=1) + "\n")
    return path


# -------------------------------------------------------------- save (live)
def save_map_package(
    out_dir: Path | str,
    *,
    method: str,
    grid: EvidenceGrid,
    stamps: NDArray[np.float64],
    xyyaw: NDArray[np.float64],
    supported: NDArray[np.bool_],
    source: Mapping[str, Any],
    sensors: Sequence[str],
    loop_closed: bool | None = None,
    corridor_spacing_m: float = 0.5,
    corridor_close_cells: int = 2,
    build: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> dict:
    """Write a complete map package from in-memory map state; return its meta.

    Runs the corridor and racing-line stages exactly as the offline pipeline
    does (:func:`kart_map2d.pipeline.corridor_and_line`, over the
    motion-supported part of the trajectory), so a live map and an offline
    map of the same data are the same kind of artifact. A failing late stage
    is recorded in ``problems`` (the package is then valid but not
    ``drivable``), like the pipeline's ``report.json``.

    ``out_dir`` must not already hold a package unless ``overwrite``; then the
    old package files are removed first so nothing stale survives.
    """
    from kart_map2d.pipeline import corridor_and_line  # lazy: pipeline imports this module

    import cv2

    out_dir = Path(out_dir)
    if (out_dir / META_FILENAME).exists():
        if not overwrite:
            raise MapPackageError(f"{out_dir} already holds a map package (pass overwrite=True)")
        for name in [META_FILENAME, *PACKAGE_FILES.values()]:
            (out_dir / name).unlink(missing_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamps = np.asarray(stamps, dtype=np.float64)
    xyyaw = np.asarray(xyyaw, dtype=np.float64).reshape(-1, 3)
    supported = np.asarray(supported, dtype=bool)
    if len(stamps) < 2:
        raise MapPackageError("cannot save a map without a trajectory (fewer than 2 poses)")

    timing: dict[str, float] = {}
    t0 = time.perf_counter()
    produced = [write_trajectory_csv(out_dir / PACKAGE_FILES["trajectory"], stamps, xyyaw, supported).name]
    grid.save(out_dir / PACKAGE_FILES["grid"])
    cv2.imwrite(str(out_dir / "grid.png"), grid.render(scale=2))
    produced.append(PACKAGE_FILES["grid"])
    timing["grid_trajectory_s"] = round(time.perf_counter() - t0, 3)

    t0 = time.perf_counter()
    skipped: list[str] = []
    corridor_info, line_info = corridor_and_line(
        grid,
        xyyaw[supported, :2],
        out_dir,
        spacing_m=corridor_spacing_m,
        close_cells=corridor_close_cells,
        skipped_previews=skipped,
    )
    produced += [n for n in (PACKAGE_FILES["corridor"], PACKAGE_FILES["racing_line"]) if (out_dir / n).is_file()]
    timing["corridor_line_s"] = round(time.perf_counter() - t0, 3)

    build = {**dict(build or {}), "save_timing_s": timing}
    if skipped:
        build["previews_skipped"] = skipped
    write_map_meta(
        out_dir,
        method=method,
        source=source,
        sensors=sensors,
        files=produced,
        corridor_info=corridor_info,
        racing_line_info=line_info,
        loop_closed=loop_closed,
        build=build,
    )
    return json.loads((out_dir / META_FILENAME).read_text(encoding="ascii"))


# ------------------------------------------------------------------- load
@dataclass
class RacingLineData:
    """racing_line.csv as loaded: arclength s [m], path (N, 2) [m], curvature
    [1/m, CCW +], speed [m/s]; ``closed`` / ``time_s`` from its header."""

    s: NDArray[np.float64]
    path: NDArray[np.float64]
    curvature: NDArray[np.float64]
    speed: NDArray[np.float64]
    closed: bool
    time_s: float | None


@dataclass
class MapPackage:
    """A loaded, validated map package (see the module docstring)."""

    path: Path
    meta: dict
    grid: EvidenceGrid
    stamps: NDArray[np.float64]
    xyyaw: NDArray[np.float64]
    supported: NDArray[np.bool_]
    corridor: Corridor | None = None
    racing_line: RacingLineData | None = None
    problems: list[str] = field(default_factory=list)

    @property
    def method(self) -> str:
        return str(self.meta["method"])

    @property
    def drivable(self) -> bool:
        """Corridor and racing line are present: the drive stack can follow it."""
        return self.corridor is not None and self.racing_line is not None


def _load_rows(path: Path, columns: int) -> NDArray[np.float64]:
    rows = np.loadtxt(path, delimiter=",", comments="#", ndmin=2, dtype=np.float64)
    if rows.size == 0:
        return np.zeros((0, columns))
    if rows.shape[1] != columns:
        raise MapPackageError(f"{path.name}: {rows.shape[1]} columns, expected {columns}")
    return rows


def _load_trajectory(path: Path) -> NDArray[np.float64]:
    return _load_rows(path, 5)


def _check_file(path: Path, entry: Mapping[str, Any], verify: bool) -> str | None:
    if not path.is_file():
        return f"{path.name} is listed in {META_FILENAME} but missing"
    if not verify:
        return None
    size = path.stat().st_size
    if size != entry.get("bytes"):
        return f"{path.name}: {size} bytes, {META_FILENAME} says {entry.get('bytes')} (partial copy or edited)"
    if _sha256(path) != entry.get("sha256"):
        return f"{path.name}: sha256 differs from {META_FILENAME} (corrupt or edited)"
    return None


def validate_map_package(path: Path | str, *, require_drivable: bool = True, verify: bool = True) -> tuple[list[str], MapPackage | None]:
    """Every problem with the package at ``path`` + the loaded package when usable.

    Returns ``(problems, package)``; ``package`` is None when something
    required could not be loaded. ``verify`` checks each file's size and
    sha256 against ``map_meta.json``; ``require_drivable`` makes a missing
    corridor or racing line a problem (the drive stack needs both).
    """
    root = Path(path)
    problems: list[str] = []
    meta_path = root / META_FILENAME
    if not meta_path.is_file():
        return [f"{root}: no {META_FILENAME} - not a map package (or an unfinished build)"], None
    try:
        meta = json.loads(meta_path.read_text(encoding="ascii"))
    except (ValueError, UnicodeDecodeError) as exc:
        return [f"{META_FILENAME}: unreadable ({exc})"], None
    if meta.get("kind") != META_KIND:
        return [f"{META_FILENAME}: kind {meta.get('kind')!r}, expected {META_KIND!r}"], None
    version = meta.get("format")
    if not isinstance(version, int) or version > MAP_PACKAGE_FORMAT:
        return [f"{META_FILENAME}: format {version!r} is newer than supported {MAP_PACKAGE_FORMAT}"], None
    for key in ("method", "map_frame", "origin", "resolution_m", "created", "source", "sensors", "software", "files"):
        if key not in meta:
            problems.append(f"{META_FILENAME}: missing key {key!r}")
    if problems:
        return problems, None
    method = meta["method"]
    if method not in _METHODS:
        problems.append(f"{META_FILENAME}: unknown method {method!r}")

    files: dict = meta["files"]
    for name, entry in sorted(files.items()):
        problem = _check_file(root / name, entry, verify)
        if problem:
            problems.append(problem)
    for role in _REQUIRED:
        if PACKAGE_FILES[role] not in files:
            problems.append(f"{PACKAGE_FILES[role]} is required but not listed in {META_FILENAME}")
    if any(PACKAGE_FILES[r] not in files or not (root / PACKAGE_FILES[r]).is_file() for r in _REQUIRED):
        return problems, None

    try:
        grid = EvidenceGrid.load(root / PACKAGE_FILES["grid"])
    except Exception as exc:  # noqa: BLE001 - reported as a problem
        return problems + [f"grid.npz: cannot load ({type(exc).__name__}: {exc})"], None
    if not np.isclose(grid.cell, float(meta["resolution_m"]), rtol=0, atol=1e-12):
        problems.append(f"grid.npz: cell {grid.cell} m, {META_FILENAME} says {meta['resolution_m']}")
    if not np.allclose(np.asarray(grid.origin, dtype=float), np.asarray(meta["origin"], dtype=float), rtol=0, atol=1e-9):
        problems.append(f"grid.npz: origin {list(grid.origin)}, {META_FILENAME} says {meta['origin']}")

    try:
        traj = _load_trajectory(root / PACKAGE_FILES["trajectory"])
    except (ValueError, MapPackageError) as exc:
        return problems + [f"trajectory.csv: unreadable ({exc})"], None
    if len(traj) < 2:
        problems.append("trajectory.csv: fewer than 2 poses")
    if not np.isfinite(traj).all():
        problems.append("trajectory.csv: non-finite values")
    if len(traj) > 1 and np.any(np.diff(traj[:, 0]) < 0):
        problems.append("trajectory.csv: stamps go backwards")
    package = MapPackage(root, meta, grid, traj[:, 0], traj[:, 1:4], traj[:, 4] > 0.5)

    name = PACKAGE_FILES["corridor"]
    if name in files and (root / name).is_file():
        try:
            rows = _load_rows(root / name, 4)
            if len(rows) < 2 or not np.isfinite(rows).all() or (rows[:, 2:] < 0).any():
                problems.append("corridor.csv: needs >= 2 finite stations with non-negative widths")
            else:
                center = rows[:, :2]
                s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(center, axis=0), axis=1))]
                closed = bool((meta.get("corridor") or {}).get("closed", False))
                package.corridor = Corridor(center, rows[:, 3], rows[:, 2], closed, s)
        except (ValueError, MapPackageError) as exc:
            problems.append(f"corridor.csv: unreadable ({exc})")
    name = PACKAGE_FILES["racing_line"]
    if name in files and (root / name).is_file():
        try:
            rows = _load_rows(root / name, 5)
            header = _racing_line_header(root / name)
            if len(rows) < 4 or not np.isfinite(rows).all() or np.any(np.diff(rows[:, 0]) < 0):
                problems.append("racing_line.csv: needs >= 4 finite rows with increasing s")
            else:
                package.racing_line = RacingLineData(
                    rows[:, 0], rows[:, 1:3], rows[:, 3], rows[:, 4], bool(header.get("closed", False)), header.get("time_s")
                )
        except (ValueError, MapPackageError, IndexError) as exc:
            problems.append(f"racing_line.csv: unreadable ({exc})")
    if require_drivable and not package.drivable:
        why = "; ".join(meta.get("problems") or []) or "corridor.csv / racing_line.csv missing"
        problems.append(f"not drivable: {why}")

    # kart_map2d is the 2D-only package: a lidar3d package written by the
    # research workspace still loads (its planar grid + racing line are what
    # the drive stack needs), but its 3D cloud files are verified by hash
    # only, never parsed - there is no CloudMapBuilder here.
    package.problems = problems
    return problems, package


def load_map_package(path: Path | str, *, require_drivable: bool = True, verify: bool = True) -> MapPackage:
    """Load and validate the map package at ``path`` (see :func:`validate_map_package`).

    Raises :class:`MapPackageError` listing every problem; a returned package
    is complete, internally consistent and byte-identical to what was saved.
    """
    problems, package = validate_map_package(path, require_drivable=require_drivable, verify=verify)
    if problems or package is None:
        raise MapPackageError(f"{path}: not a usable map package:\n  - " + "\n  - ".join(problems))
    return package


def copy_map_package(src: Path | str, dst: Path | str) -> Path:
    """Copy exactly the files a package lists (+ its meta) to ``dst``; verify the copy."""
    src, dst = Path(src), Path(dst)
    meta = json.loads((src / META_FILENAME).read_text(encoding="ascii"))
    dst.mkdir(parents=True, exist_ok=True)
    for name in meta["files"]:
        shutil.copy2(src / name, dst / name)
    shutil.copy2(src / META_FILENAME, dst / META_FILENAME)
    problems, _ = validate_map_package(dst, require_drivable=False)
    if problems:
        raise MapPackageError(f"copy of {src} to {dst} failed validation: {problems}")
    return dst
