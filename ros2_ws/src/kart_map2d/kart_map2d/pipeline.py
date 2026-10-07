"""End-to-end map building: replay -> semantic evidence grid -> corridor -> line.

``build_map`` runs one recording through the fusion replay for localization,
detects and closes trajectory loops (on by default; the corrected trajectory
feeds everything downstream), then fuses camera semantics (and LiDAR, when it
is among the sources) into an
:class:`~kart_map2d.mapping.evidence_grid.EvidenceGrid` at the corrected
poses, extracts the drivable corridor around the driven path and plans a
racing line inside it. Camera + IMU alone is a fully supported configuration;
every other source only sharpens the same artifacts. Segmentation classes
that typically move are routed to the grid's decaying dynamic layer, and the
segmentation backend is selectable (``seg_mode``: the SegFormer network or
the zero-network classical fallback).

This package is 2D-only: ``map_mode`` exists for signature compatibility
with the research workspace and accepts exactly ``"flat"``.

Everything is written to ``outputs/maps/<run>__<config>/`` (flat) or
``outputs/maps/<run>__<config>__<mode>/`` (every other mode, so modes never
overwrite each other) and the pipeline degrades gracefully: a failing late
stage (e.g. too little drivable evidence for a corridor) is recorded in
``report.json`` while the earlier artifacts are still written. Replay,
segmentation and loop closure do not depend on the map mode; :func:`localize`
computes them once so several modes can share them.

Frames and units follow trackmapper conventions: planar map frame anchored at
the run start, metres, yaw CCW from x; base_link x forward / y left / z up.

Deployment footprint: importing this module and running it with
``seg_mode="classical"`` needs only numpy / scipy / opencv (no torch, no
transformers). Two further packages are optional: without ``kiss-icp`` (and
without a cached KISS result) a run whose sources exclude ``lidar_odom``
replays without the LiDAR scoring reference (``score_vs_kiss`` is skipped),
and without ``matplotlib`` the PNG previews are skipped while every map data
artifact (CSV, GeoJSON, npz, grid.png) is still written.
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from kart_map2d import replay as replay_mod
from kart_map2d.core.transforms import invert
from kart_map2d.estimation.fusion import FusionEngine
from kart_map2d.eval.metrics import score_trajectory
from kart_map2d.io.dataset import ExportedRun
from kart_map2d.mapping.corridor import (
    extract_corridor,
    render_corridor_png,
    write_geojson,
    write_track_csv,
)
from kart_map2d.mapping.evidence_grid import (
    CLASS_DRIVABLE,
    CLASS_DYNAMIC,
    CLASS_OBSTACLE,
    DYNAMIC_CLASS_IDS,
    EvidenceGrid,
    _pixel_rays,
)
from kart_map2d.mapping.loop_closure import KeyframeObservation, detect_and_close
from kart_map2d.mapping.map_package import write_map_meta
from kart_map2d.perception.classical_ground import load_or_learn_kart_mask
from kart_map2d.perception.lidar_odometry import LidarOdometryResult
from kart_map2d.perception.segmentation import get_segmenter, load_or_compute
from kart_map2d.perception.segmentation import resolve_seg_mode as _resolve_seg_mode
from kart_map2d.planning.racing_line import (
    compute_racing_line,
    render_racing_line_png,
    write_racing_line_csv,
)


def cameras_for_sources(sources: Sequence[str]) -> list[str]:
    """Cameras named by ``vo_<camera>`` sources, in source order."""
    return [s[len("vo_") :] for s in sources if s.startswith("vo_")]


def supported_mask(
    t: NDArray[np.float64],
    traj_stamps: NDArray[np.float64],
    traj_supported: NDArray[np.bool_],
) -> NDArray[np.bool_]:
    """Whether the EKF pose at each time ``t`` is motion-supported.

    Times outside the trajectory span are unsupported (the pose there would be
    pure extrapolation), and in between the nearest recorded support flags are
    interpolated: only times bracketed by supported samples pass.
    """
    t = np.asarray(t, dtype=np.float64)
    if len(traj_stamps) == 0:
        return np.zeros(len(t), dtype=bool)
    inside = (t >= traj_stamps[0]) & (t <= traj_stamps[-1])
    level = np.interp(t, traj_stamps, np.asarray(traj_supported, dtype=float))
    return inside & (level > 0.5)


def interp_poses(
    t: NDArray[np.float64],
    stamps: NDArray[np.float64],
    xyyaw: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Interpolated [x, y, yaw] at times ``t`` (yaw unwrapped first).

    Mirrors ``Trajectory.pose_at`` but works on a plain trajectory array, so
    the loop-closed trajectory can feed the grid build the same way the raw
    EKF one does.
    """
    t = np.asarray(t, dtype=np.float64)
    yaw = np.unwrap(np.asarray(xyyaw, dtype=np.float64)[:, 2])
    return np.column_stack(
        [np.interp(t, stamps, xyyaw[:, 0]), np.interp(t, stamps, xyyaw[:, 1]), np.interp(t, stamps, yaw)]
    )


# --------------------------------------------------------------- segmentation
#: Cityscapes train ids that typically move (people, riders, vehicles); pixels
#: of these classes go to the grid's decaying dynamic layer, not the final map.
_DYNAMIC_IDS = np.array(sorted(DYNAMIC_CLASS_IDS), dtype=np.uint8)


def resolve_seg_mode(seg_mode: str) -> str:
    """``auto`` picks neural when torch, transformers and the checkpoint are all
    present, else classical (never imports torch; see
    :func:`kart_map2d.perception.segmentation.resolve_seg_mode`)."""
    return _resolve_seg_mode(seg_mode)


def segment_frames(
    frames,
    run_name: str,
    camera: str,
    mode: str,
    K: NDArray[np.float64],
    T_base_cam: NDArray[np.float64],
    cache_root: Path,
    kart_mask: NDArray[np.bool_] | None = None,
) -> list:
    """Segmentation results for ``frames`` with the resolved backend.

    Neural results reuse the ``outputs/seg`` cache. Classical results are
    recomputed every run: the backend keeps a temporal EMA ground model, so a
    per-frame cache would replay stale state into the sequence - and at
    ~8 ms/frame recomputing is cheaper than the neural cache lookup anyway.

    ``kart_mask`` (optional) is the per-camera kart-body mask BOTH backends
    exclude from every scene class - normally the auto-learned one from
    :func:`kart_map2d.perception.classical_ground.load_or_learn_kart_mask`
    (see :func:`localize`); None keeps each backend's geometric default.
    """
    if mode == "neural":
        return load_or_compute(frames, run_name, camera, cache_root=cache_root, kart_mask=kart_mask)
    segmenter = get_segmenter("classical", K=K, T_base_cam=T_base_cam, kart_mask=kart_mask)
    results = []
    for frame in frames:  # in time order: the classical backend is stateful
        image = cv2.imread(frame.image_path)
        if image is None:
            raise FileNotFoundError(f"cannot read camera frame {frame.image_path}")
        results.append(segmenter.segment(image))
    return results


# --------------------------------------------------------------- loop closure
def ground_observations(
    frames,
    segs,
    K: NDArray[np.float64],
    T_base_cam: NDArray[np.float64],
    ground_z: float = -0.10,
    max_range: float = 12.0,
    cell: float = 0.20,
) -> list[KeyframeObservation]:
    """Per-frame drivable / obstacle-contact ground points in base_link.

    Feeds the loop-closure revisit detector with the same ground projection
    the evidence grid uses: drivable pixels intersect the ground plane
    (occluded columns and ranges beyond ``max_range`` rejected - farther
    intersections are too depth-noisy to localise a revisit), obstacles
    contribute only each column's lowest pixel, the ground contact. Points
    are deduplicated to ``cell`` resolution, which is all the patch
    rasterizer keeps anyway.
    """

    def dedupe(points: NDArray[np.float64]) -> NDArray[np.float64]:
        if not len(points):
            return points
        return np.unique(np.floor(points / cell).astype(np.int64), axis=0) * cell + cell / 2

    T_base_cam = np.asarray(T_base_cam, dtype=np.float64)
    shape = segs[0].class_map.shape
    rays = _pixel_rays(K, shape) @ T_base_cam[:3, :3].T
    cam = T_base_cam[:3, 3]
    h, w = shape
    dz = rays[..., 2]
    below = dz < -1e-9
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(below, (ground_z - cam[2]) / dz, np.inf)
        gx = cam[0] + t * rays[..., 0]
        gy = cam[1] + t * rays[..., 1]
        rng = np.hypot(gx - cam[0], gy - cam[1])

    observations = []
    for frame, seg in zip(frames, segs):
        obst = seg.obstacle_mask
        col_limit = np.full(w, np.inf)
        contact = np.empty((0, 2))
        if obst.any():
            cols = np.flatnonzero(obst.any(axis=0))
            rows = (h - 1) - np.argmax(obst[::-1, :][:, cols], axis=0)
            ok = below[rows, cols]
            col_limit[cols[ok]] = rng[rows[ok], cols[ok]]
            keep = ok & (rng[rows, cols] <= max_range)
            contact = np.column_stack([gx[rows[keep], cols[keep]], gy[rows[keep], cols[keep]]])
        sel = seg.drivable_mask & below & (rng <= max_range) & (rng <= col_limit[None, :])
        drv = np.column_stack([gx[sel], gy[sel]])
        observations.append(
            KeyframeObservation(
                stamp=float(frame.stamp), drivable_xy=dedupe(drv), obstacle_xy=dedupe(contact)
            )
        )
    return observations


def _json_safe(obj):
    """Recursively convert numpy scalars so a report section serialises."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


# ------------------------------------------------------------------ artifacts
def write_trajectory_csv(path: Path, stamps, states, supported) -> Path:
    with Path(path).open("w", encoding="ascii", newline="\n") as f:
        f.write("# stamp_s,x_m,y_m,yaw_rad,supported\n")
        for t, s, ok in zip(stamps, states, supported):
            f.write(f"{t:.4f},{s[0]:.4f},{s[1]:.4f},{s[2]:.5f},{int(ok)}\n")
    return Path(path)


def write_modes_json(path: Path, engine: FusionEngine) -> Path:
    """Health / operating-mode timeline: when the fusion changed what it ran on."""
    doc = {
        "modes": [
            {"stamp": m.stamp, "mode": m.mode, "healthy_channels": list(m.healthy)}
            for m in engine.modes
        ],
        "health_transitions": [
            {"stamp": t, "channel": c, "status": s, "reason": r}
            for t, c, s, r in engine.health.transitions
        ],
        "final_health": engine.health.snapshot(),
    }
    Path(path).write_text(json.dumps(doc, indent=1), encoding="ascii")
    return Path(path)


def render_optional(name: str, render, skipped: list | None = None) -> bool:
    """Run one matplotlib PNG renderer (a no-argument callable); False if skipped.

    PNG previews are for people; the kart consumes the data artifacts (CSV,
    GeoJSON, npz). A lean deployment without matplotlib (the mapping
    requirement files leave it out) must therefore still get every data
    artifact and the racing line, so only a missing matplotlib is tolerated
    here - any other error propagates exactly as before. ``name`` (the PNG)
    is appended to ``skipped`` when the preview could not be drawn.
    """
    try:
        render()
    except ImportError as exc:
        if (exc.name or "").split(".")[0] != "matplotlib":
            raise
        if skipped is not None:
            skipped.append(name)
        return False
    return True


def closed_classes(grid: EvidenceGrid, close_cells: int = 2) -> NDArray[np.uint8]:
    """Corridor-ready class array (row=y, col=x) with small drivable holes shut.

    Camera evidence is sparse (per-frame ground splats), so the raw grid has
    pinholes that would terminate the corridor's contiguous lateral sweep after
    centimetres. A morphological closing of the drivable mask bridges gaps up
    to ``close_cells`` cells while obstacle cells always stay blocking.
    """
    classes = np.asarray(grid.classify()).T  # (nx, ny) -> (row=y, col=x)
    if close_cells <= 0 or classes.size == 0:
        return classes.astype(np.uint8)
    k = 2 * int(close_cells) + 1
    drv = (classes == CLASS_DRIVABLE).astype(np.uint8)
    drv = cv2.morphologyEx(drv, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    out = np.where(drv > 0, CLASS_DRIVABLE, 0).astype(np.uint8)
    out[classes == CLASS_OBSTACLE] = CLASS_OBSTACLE
    return out


def corridor_and_line(
    grid: EvidenceGrid,
    trajectory_xy: NDArray[np.float64],
    out_dir: Path,
    spacing_m: float = 0.5,
    close_cells: int = 2,
    skipped_previews: list | None = None,
) -> tuple[dict, dict]:
    """Corridor + racing-line stages with per-stage graceful failure.

    Returns the two report sections; a section is ``{"error": ...}`` when its
    stage failed (e.g. too little drivable evidence), and a racing line is only
    attempted on a successful corridor. Artifacts land in ``out_dir``. PNG
    previews that could not be drawn (no matplotlib) are appended to
    ``skipped_previews`` without failing either stage.
    """
    out_dir = Path(out_dir)
    try:
        classes = closed_classes(grid, close_cells)
        origin = tuple(np.asarray(grid.origin, dtype=float))
        res = float(grid.cell)
        corridor = extract_corridor(
            classes, trajectory_xy, spacing_m=spacing_m, origin_xy=origin, resolution_m=res
        )
        write_track_csv(corridor, out_dir / "corridor.csv")
        write_geojson(corridor, out_dir / "corridor.geojson")
        render_optional(
            "corridor.png",
            lambda: render_corridor_png(
                corridor,
                classes,
                out_dir / "corridor.png",
                origin_xy=origin,
                resolution_m=res,
                trajectory_xy=trajectory_xy,
            ),
            skipped_previews,
        )
        ds = float(corridor.s[1] - corridor.s[0]) if len(corridor.s) > 1 else 0.0
        corridor_report = {
            "stations": int(len(corridor.centerline)),
            "length_m": round(float(corridor.s[-1]) + (ds if corridor.closed else 0.0), 2),
            "closed": bool(corridor.closed),
            "mean_width_left_m": round(float(corridor.w_left.mean()), 3),
            "mean_width_right_m": round(float(corridor.w_right.mean()), 3),
        }
    except Exception as exc:  # noqa: BLE001 - degraded output beats a dead pipeline
        return {"error": f"{type(exc).__name__}: {exc}"}, {"error": "skipped: corridor failed"}

    try:
        line = compute_racing_line(corridor)
        write_racing_line_csv(line, out_dir / "racing_line.csv")
        render_optional(
            "racing_line.png",
            lambda: render_racing_line_png(line, corridor, out_dir / "racing_line.png"),
            skipped_previews,
        )
        line_report = {
            "status": "ok",
            "time_s": round(float(line.time_s), 2),
            "closed": bool(line.closed),
            "peak_abs_curvature_1pm": round(float(np.abs(line.curvature).max()), 4),
            "speed_min_mps": round(float(line.speed.min()), 2),
            "speed_max_mps": round(float(line.speed.max()), 2),
        }
    except Exception as exc:  # noqa: BLE001
        line_report = {"error": f"{type(exc).__name__}: {exc}"}
    return corridor_report, line_report


#: kart_map2d is the 2D package: exactly one map mode.
MAP_MODES = ("flat",)

#: The kart's planar footprint in base_link (x_min, x_max, half_width) [m]:
#: rear bumper to nose with a small margin (wheelbase 0.978 m centred on
#: base_link, ~1.06 m wide). Camera obstacle contacts projecting inside it
#: can only be the kart's own body leaking past the kart mask, so the
#: mapping grid refuses them (EvidenceGrid.kart_footprint).
KART_FOOTPRINT_M = (-1.0, 1.0, 0.6)

#: Minimum camera obstacle-contact range [m] for the kart's forward cameras
#: (EvidenceGrid.min_contact_range; the drive stack's CameraGroundCue uses
#: the same guard at 2.0 m): nearer projections come from pixels a few
#: degrees below the lens - kart silhouette breathing, its contact shadow,
#: unverifiable near-field texture - and move with the camera.
CAMERA_MIN_CONTACT_RANGE_M = 2.5


def map_output_dir(out_root: Path | str, run_name: str, config_name: str, map_mode: str) -> Path:
    """Artifact directory for one run / config / map mode.

    ``flat`` keeps the historical ``<run>__<config>`` so existing references
    stay valid; every other mode appends ``__<mode>``, so building several
    modes of the same config never overwrites one with another.
    """
    suffix = "" if map_mode == "flat" else f"__{map_mode}"
    return Path(out_root) / f"{run_name}__{config_name}{suffix}"


@dataclass
class Localization:
    """Everything ``build_map`` computes before mapping; identical for every map mode.

    Replay, segmentation and loop closure depend only on the run, sources,
    faults and the segmentation / loop-closure settings - never on the map
    mode - so one :func:`localize` result can feed several ``build_map``
    calls (loop closure alone is ~30-40 s on sept3). Consumers must treat it
    as read-only; ``build_map`` deep-copies the report sections it extends.

    stamps / states / p_diag / supported : the raw EKF trajectory arrays
        (``Trajectory.arrays()``; states[:, :3] = map-frame x, y [m], yaw [rad]).
    est_xyyaw : [N, 3] trajectory after loop closure (== states[:, :3] when
        no closure was accepted).
    cam_data : per camera ``(camera, span frames, span segs, used frames,
        used segs, K, T_base_cam)`` - see ``build_map``.
    sections : report sections produced here (``timing_s`` entries,
        ``segmentation``, ``camera_frames``, ``loop_closure``), in report order.
    """

    key: tuple
    inputs: replay_mod.ReplayInputs
    engine: FusionEngine
    stamps: NDArray[np.float64]
    states: NDArray[np.float64]
    p_diag: NDArray[np.float64]
    supported: NDArray[np.bool_]
    est_xyyaw: NDArray[np.float64]
    cam_data: list = field(default_factory=list)
    sections: dict = field(default_factory=dict)


def time_anchor(run: ExportedRun) -> float:
    """Replay clock anchor ``t0`` [epoch s] without running LiDAR odometry.

    The first LiDAR stamp when the recording has LiDAR - exactly the anchor
    ``replay.prepare`` takes from KISS-ICP, so fault windows keep their
    meaning - otherwise the earliest IMU sample or camera frame.
    """
    if any((Path(run.root) / "lidar").glob("chunk_*.npz")):
        return float(run.lidar_stamps()[0])
    firsts = []
    for name in ("imu_data", "livox_imu"):
        imu = run.imu(name)
        if imu is not None and len(imu.stamp):
            firsts.append(float(imu.stamp.min()))
    for camera in run.cameras():
        stamps = [f.stamp for f in run.camera_frames(camera)]
        if stamps:
            firsts.append(float(min(stamps)))
    if not firsts:
        raise ValueError(f"{run.root}: no LiDAR, IMU or camera data to anchor the replay clock")
    return min(firsts)


def prepare_inputs(
    run_dir: Path | str, sources: Sequence[str], cache_dir: Path = Path("outputs/cache")
) -> tuple[replay_mod.ReplayInputs, str | None]:
    """``replay.prepare``, or its LiDAR-free twin when ``lidar_odom`` is not a source.

    ``prepare`` always runs (or loads the cached) KISS-ICP: it anchors the
    replay clock and is the scoring reference. Unless ``lidar_odom`` is a
    source neither role is essential, so a recording without LiDAR, or a
    machine without kiss-icp and without a cached result (the flat deploy
    bundle), replays with :func:`time_anchor` and an empty LiDAR result
    instead - camera + IMU must be sufficient offline too. Returns
    ``(inputs, reason)``; ``reason`` says why there is no KISS-ICP reference
    and is None when there is one (then ``inputs`` is exactly ``prepare``'s).
    """
    run_dir = Path(run_dir)
    needs_lidar = "lidar_odom" in sources
    has_lidar = any((ExportedRun(run_dir).root / "lidar").glob("chunk_*.npz"))
    if needs_lidar or has_lidar:
        try:
            return replay_mod.prepare(run_dir, cache_dir=cache_dir), None
        except ImportError as exc:
            if needs_lidar or (exc.name or "").split(".")[0] != "kiss_icp":
                raise
            reason = "kiss-icp is not installed and there is no cached KISS-ICP result"
    else:
        reason = "the recording has no LiDAR"
    run = ExportedRun(run_dir)
    empty = LidarOdometryResult(np.empty(0), np.empty((0, 4, 4)), np.empty(0))
    return replay_mod.ReplayInputs(run, empty, time_anchor(run)), reason


def _localization_key(run_dir, sources, faults, seg_stride, seg_mode, loop_closure) -> tuple:
    """What a :class:`Localization` depends on (compared, never hashed: faults are dicts)."""
    return (
        str(Path(run_dir).resolve()),
        tuple(sources),
        list(faults or []),
        max(int(seg_stride), 1),
        seg_mode,
        bool(loop_closure),
    )


def localize(
    run_dir: Path | str,
    sources: Sequence[str],
    faults: list | None = None,
    seg_stride: int = 1,
    seg_mode: str = "auto",
    loop_closure: bool = True,
    seg_cache_root: Path = Path("outputs/seg"),
    cache_dir: Path = Path("outputs/cache"),
) -> Localization:
    """Replay, segmentation and loop closure: the map-mode-independent prefix of ``build_map``.

    Arguments mean the same as in :func:`build_map`. Call it once and pass the
    result to several ``build_map(..., localization=...)`` calls to build
    more than one map mode without repeating this work. With an empty
    replay trajectory it returns early (no segmentation); ``build_map`` then
    writes its error report.
    """
    run_dir = Path(run_dir)
    sources = tuple(sources)
    key = _localization_key(run_dir, sources, faults, seg_stride, seg_mode, loop_closure)
    timing: dict = {}
    sections: dict = {"timing_s": timing}

    # ------------------------------------------------------- localization
    t_start = time.perf_counter()
    inputs, no_reference = prepare_inputs(run_dir, sources, cache_dir)
    if no_reference is not None:
        sections["reference"] = {"kiss_icp": "unavailable", "reason": no_reference}
    engine = replay_mod.replay(inputs, sources=sources, faults=faults)
    stamps, states, p_diag, supported = engine.trajectory.arrays()
    timing["replay"] = round(time.perf_counter() - t_start, 2)
    if len(stamps) == 0:
        return Localization(key, inputs, engine, stamps, states, p_diag, supported, states[:, :3].copy(), [], sections)

    # --------------------------------------------------- segmentation
    run = inputs.run
    mode = resolve_seg_mode(seg_mode)
    sections["segmentation"] = {"mode_requested": seg_mode, "mode": mode}
    cameras = cameras_for_sources(sources) or run.cameras()
    sections["camera_frames"] = {}
    # (camera, span frames, span segs, used frames, used segs, K, T_base_cam):
    # the trajectory-span set feeds revisit detection (a patch is only built
    # from locally consistent relative poses, so brief unsupported stretches
    # are fine there), while the grid inserts only motion-supported frames.
    cam_data: list[tuple] = []
    t_seg = 0.0
    t_mask = 0.0
    for camera in cameras:
        frames = run.camera_frames(camera)[:: max(int(seg_stride), 1)]
        span = [f for f in frames if stamps[0] <= f.stamp <= stamps[-1]]
        ok = supported_mask(np.array([f.stamp for f in span]), stamps, supported)
        used = [f for f, good in zip(span, ok) if good]
        sections["camera_frames"][camera] = {
            "total": len(frames),
            "in_span": len(span),
            "inserted": len(used),
        }
        if not span:
            continue
        K, _ = run.camera_intrinsics(camera)
        T_base_cam = run.camera_mount(camera)
        # Auto-learned per-camera kart-body mask (cached; geometric fallback
        # when learning is not possible). The static geometric mask is too
        # low for this kart - the front wheel / suspension rise ~100 px above
        # it, get labelled obstacle and, moving with the camera, paint false
        # obstacle cells right beside the path in every frame (measured:
        # 74 % of the classical near-path obstacle excess traced to those
        # pixels). Both backends receive the same mask.
        t0 = time.perf_counter()
        kart_mask, mask_meta = load_or_learn_kart_mask(span, run_dir.name, camera, K=K, T_base_cam=T_base_cam)
        t_mask += time.perf_counter() - t0
        sections["camera_frames"][camera]["kart_mask"] = {
            k: mask_meta.get(k) for k in ("learned", "mask_fraction", "frames_sampled")
        }
        t0 = time.perf_counter()
        segs = segment_frames(span, run_dir.name, camera, mode, K, T_base_cam, seg_cache_root, kart_mask=kart_mask)
        t_seg += time.perf_counter() - t0
        used_segs = [s for s, good in zip(segs, ok) if good]
        cam_data.append((camera, span, segs, used, used_segs, K, T_base_cam))
    timing["segmentation"] = round(t_seg, 2)
    timing["kart_mask_learn"] = round(t_mask, 2)

    # --------------------------------------------------- loop closure
    # Runs on the finished trajectory (a batch step, not per-frame); the
    # corrected trajectory then feeds every downstream artifact. With no
    # accepted closure detect_and_close returns the input unchanged.
    est_xyyaw = states[:, :3].copy()
    lc_report: dict = {"enabled": bool(loop_closure)}
    if loop_closure and cam_data:
        t0 = time.perf_counter()
        observations: list[KeyframeObservation] = []
        for _, span, segs, _, _, K, T_base_cam in cam_data:
            observations += ground_observations(span, segs, K, T_base_cam)
        observations.sort(key=lambda o: o.stamp)
        closed = detect_and_close(
            stamps,
            est_xyyaw,
            observations,
            position_sigma=np.sqrt(p_diag[:, 0] + p_diag[:, 1]),
        )
        lc_report.update(
            accepted=len(closed.accepted),
            diagnostics=_json_safe(closed.diagnostics),
            closures=[
                {
                    "t_a_s": round(c.stamp_a - float(stamps[0]), 1),
                    "t_b_s": round(c.stamp_b - float(stamps[0]), 1),
                    "score": round(c.score, 3),
                    "chi2": None if c.chi2 is None else round(c.chi2, 1),
                    "accepted": bool(c.accepted),
                }
                for c in closed.closures
            ],
        )
        if closed.accepted:
            est_xyyaw = closed.corrected_xyyaw
        timing["loop_closure"] = round(time.perf_counter() - t0, 2)
    elif loop_closure:
        lc_report["skipped"] = "no camera observations to detect revisits from"
    sections["loop_closure"] = lc_report
    return Localization(key, inputs, engine, stamps, states, p_diag, supported, est_xyyaw, cam_data, sections)

# ------------------------------------------------------------------- pipeline
def build_map(
    run_dir: Path | str,
    sources: Sequence[str],
    faults: list | None = None,
    seg_stride: int = 1,
    name: str | None = None,
    seg_mode: str = "auto",
    loop_closure: bool = True,
    map_mode: str = "flat",
    out_root: Path = Path("outputs/maps"),
    seg_cache_root: Path = Path("outputs/seg"),
    cache_dir: Path = Path("outputs/cache"),
    localization: Localization | None = None,
) -> dict:
    """Build the full map stack for one recording and source configuration.

    run_dir : recording directory (``inputs/<run>``).
    sources : replay sources; camera evidence comes from the cameras named by
        the ``vo_*`` sources (all of the run's cameras when none are named),
        LiDAR evidence is inserted only when ``"lidar_odom"`` is a source.
    faults : optional fault declarations forwarded to the replay injector.
    seg_stride : segment every k-th camera frame (mapping-density knob).
    name : config label; defaults to the sources joined with ``+``.
    seg_mode : segmentation backend, ``auto`` | ``neural`` | ``classical``
        (see :func:`kart_map2d.perception.segmentation.get_segmenter`).
    loop_closure : detect trajectory revisits after the replay and snap the
        loop shut (:mod:`kart_map2d.mapping.loop_closure`); the corrected
        trajectory then feeds the grid build, the corridor and the score.
        With no accepted closure the trajectory is used unchanged.
    map_mode : must be ``"flat"`` - the 2D pipeline is all this package
        builds (kept as a parameter for research-workspace API parity).
    localization : a :func:`localize` result for the same run / sources /
        faults / seg / loop-closure settings, to skip recomputing replay,
        segmentation and loop closure when building several modes (a
        mismatch raises ValueError). The report then carries
        ``"localization_shared": true`` and the shared stage timings.

    Returns the report dict, which is also written as ``report.json`` next to
    the other artifacts in ``<out_root>/<run>__<name>/`` for ``flat`` and
    ``<out_root>/<run>__<name>__<map_mode>/`` otherwise
    (:func:`map_output_dir`).
    """
    if map_mode not in MAP_MODES:
        raise ValueError(
            f"unknown map_mode {map_mode!r}: kart_map2d builds the 2D (flat) map only "
            "(lidar3d / camera3d live in the research workspace, not on the kart)"
        )
    sources = tuple(sources)
    run_dir = Path(run_dir)
    config_name = name or "+".join(sources)
    if localization is not None:
        # Checked before any IO: reusing a localization computed for other
        # inputs would silently map the wrong trajectory.
        expected = _localization_key(run_dir, sources, faults, seg_stride, seg_mode, loop_closure)
        if localization.key != expected:
            raise ValueError(
                "localization was computed for a different run / sources / faults / "
                "seg_stride / seg_mode / loop_closure than this build_map call"
            )
    out_dir = map_output_dir(out_root, run_dir.name, config_name, map_mode)
    out_dir.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "run": run_dir.name,
        "config": config_name,
        "sources": list(sources),
        "map_mode": map_mode,
        "faults": len(faults or []),
        "timing_s": {},
        "artifacts": {},
    }

    # ---------------------- localization: replay, segmentation, loop closure
    if localization is None:
        loc = localize(run_dir, sources, faults, seg_stride, seg_mode, loop_closure, seg_cache_root, cache_dir)
    else:
        loc = localization
        report["localization_shared"] = True
    # Deep copy: the loop-closure section is extended below and a shared
    # localization must stay pristine for the next mode.
    sections = copy.deepcopy(loc.sections)
    report["timing_s"].update(sections.pop("timing_s"))
    inputs, engine = loc.inputs, loc.engine
    stamps, states, supported = loc.stamps, loc.states, loc.supported
    if len(stamps) == 0:
        report["error"] = "replay produced no trajectory"
        (out_dir / "report.json").write_text(json.dumps(report, indent=1), encoding="ascii")
        return report
    report.update(sections)  # segmentation, camera_frames, loop_closure
    run = inputs.run
    cam_data = loc.cam_data
    est_xyyaw = loc.est_xyyaw
    lc_report = report["loop_closure"]

    states_out = states.copy()
    states_out[:, :3] = est_xyyaw
    step = np.linalg.norm(np.diff(est_xyyaw[:, :2], axis=0), axis=1)
    report["trajectory"] = {
        "poses": int(len(stamps)),
        "supported_fraction": round(float(supported.mean()), 3),
        "distance_m": round(float(step.sum()), 1),
        "duration_s": round(float(stamps[-1] - stamps[0]), 1),
    }
    report["artifacts"]["trajectory_csv"] = str(
        write_trajectory_csv(out_dir / "trajectory.csv", stamps, states_out, supported)
    )
    report["artifacts"]["modes_json"] = str(write_modes_json(out_dir / "modes.json", engine))

    # Score against the KISS-ICP base_link reference (loaded by prepare anyway).
    if len(inputs.lidar.stamp) == 0:
        # LiDAR-free replay (prepare_inputs): nothing to score against.
        reason = report.get("reference", {}).get("reason", "no LiDAR odometry")
        report["score_vs_kiss"] = {"skipped": f"no KISS-ICP reference: {reason}"}
    else:
        try:
            T = inputs.run.tf.lookup("base_link", inputs.run.lidar_frame())
            poses = np.einsum("ij,njk,kl->nil", T, inputs.lidar.pose, invert(T))
            ref = np.column_stack(
                [poses[:, 0, 3], poses[:, 1, 3], np.arctan2(poses[:, 1, 0], poses[:, 0, 0])]
            )
            score = score_trajectory(stamps, est_xyyaw, inputs.lidar.stamp, ref, supported)
            report["score_vs_kiss"] = score.as_dict()
            if lc_report.get("accepted"):
                before = score_trajectory(stamps, states[:, :3], inputs.lidar.stamp, ref, supported)
                lc_report["score_before_vs_kiss"] = before.as_dict()
        except Exception as exc:  # noqa: BLE001 - the map is still useful unscored
            report["score_vs_kiss"] = {"error": f"{type(exc).__name__}: {exc}"}

    #: PNG previews that could not be drawn (no matplotlib); data artifacts are unaffected.
    skipped_previews: list[str] = []

    # --------------------------------------------------- camera evidence
    # The kart-rig contact guards are passed here (not grid defaults):
    # they are wrong for e.g. a top-down camera, but for the kart they
    # refuse the self-contacts that pinched the classical corridor.
    grid = EvidenceGrid(
        kart_footprint=KART_FOOTPRINT_M,
        min_contact_range_m=CAMERA_MIN_CONTACT_RANGE_M,
    )
    t_cam = 0.0
    for camera, _, _, used, used_segs, K, T_base_cam in cam_data:
        if not used:
            continue
        poses_xyyaw = interp_poses(np.array([f.stamp for f in used]), stamps, est_xyyaw)
        t0 = time.perf_counter()
        for frame, seg, pose in zip(used, used_segs, poses_xyyaw):
            # Typically-moving classes go to the decaying dynamic layer so
            # crossing pedestrians never bake into the final map. The stamp
            # is always passed so the grid can also HISTORY-route contacts
            # that land on camera-verified-free cells (unknown-class movers
            # and far-contact smear; see EvidenceGrid.insert_camera).
            dyn = np.isin(seg.class_map, _DYNAMIC_IDS)
            grid.insert_camera(
                seg.drivable_mask,
                seg.obstacle_mask,
                K,
                T_base_cam,
                pose,
                dynamic_mask=dyn if dyn.any() else None,
                stamp=float(frame.stamp),
            )
        t_cam += time.perf_counter() - t0
    report["timing_s"]["camera_insert"] = round(t_cam, 2)

    # ---------------------------------------------------- LiDAR evidence
    if "lidar_odom" in sources:
        t0 = time.perf_counter()
        T = run.tf.lookup("base_link", run.lidar_frame())
        R, t_vec = T[:3, :3], T[:3, 3]
        n_used = 0
        for scan in run.lidar_scans():
            if not supported_mask(np.array([scan.stamp]), stamps, supported)[0]:
                continue
            pose = interp_poses(np.array([scan.stamp]), stamps, est_xyyaw)[0]
            pts_base = scan.points.astype(np.float64) @ R.T + t_vec
            # The stamp lets history-routing move hits on verifiably-free
            # cells (someone stepping onto the driven lane) to the dynamic layer.
            grid.insert_lidar(pts_base, pose, stamp=float(scan.stamp))
            n_used += 1
        report["lidar"] = {"scans_inserted": n_used}
        report["timing_s"]["lidar_insert"] = round(time.perf_counter() - t0, 2)

    cls = grid.classify()
    report["grid"] = {
        "shape": list(grid.shape),
        "cell_m": grid.cell,
        "drivable_cells": int((cls == CLASS_DRIVABLE).sum()),
        "obstacle_cells": int((cls == CLASS_OBSTACLE).sum()),
        "unknown_cells": int(cls.size - (cls != 0).sum()),
        # live view only; these decay and are never part of the final map
        "dynamic_cells_live": int((grid.classify(include_dynamic=True) == CLASS_DYNAMIC).sum()),
    }
    grid.save(out_dir / "grid.npz")
    cv2.imwrite(str(out_dir / "grid.png"), grid.render(scale=2))
    report["artifacts"]["grid_npz"] = str(out_dir / "grid.npz")
    report["artifacts"]["grid_png"] = str(out_dir / "grid.png")

    # ------------------------------------------------ corridor + racing line
    t0 = time.perf_counter()
    traj_xy = est_xyyaw[supported, :2]
    report["corridor"], report["racing_line"] = corridor_and_line(
        grid, traj_xy, out_dir, skipped_previews=skipped_previews
    )
    report["timing_s"]["corridor_line"] = round(time.perf_counter() - t0, 2)
    for stem in ("corridor.csv", "corridor.geojson", "corridor.png", "racing_line.csv", "racing_line.png"):
        if (out_dir / stem).exists() and stem not in skipped_previews:
            report["artifacts"][stem.replace(".", "_")] = str(out_dir / stem)
    if skipped_previews:
        # Only present on installs without matplotlib (lean deploy bundles).
        report["previews_skipped"] = {"reason": "matplotlib not installed", "files": skipped_previews}

    # Map package: map_meta.json, written last and only over the artifacts
    # THIS run produced, makes the folder loadable by the drive stack
    # (kart_map2d.mapping.map_package.load_map_package), exactly like a
    # live node's ~/save_map. Additive: every artifact above is unchanged.
    try:
        meta_path = write_map_meta(
            out_dir,
            method=map_mode,
            source={"kind": "offline", "run": run_dir.name, "config": config_name, "recording": run_dir.as_posix()},
            sensors=list(sources),
            files=[Path(p).name for p in report["artifacts"].values()],
            corridor_info=report.get("corridor"),
            racing_line_info=report.get("racing_line"),
            loop_closed=bool(lc_report.get("accepted")),
            build={
                "seg_mode": report.get("segmentation", {}).get("mode"),
                "faults": report["faults"],
                "report": "report.json",
            },
        )
        report["artifacts"]["map_meta_json"] = str(meta_path)
    except Exception as exc:  # noqa: BLE001 - the artifacts above stay valid either way
        report["map_package"] = {"error": f"{type(exc).__name__}: {exc}"}

    (out_dir / "report.json").write_text(json.dumps(report, indent=1), encoding="ascii")
    report["artifacts"]["report_json"] = str(out_dir / "report.json")
    return report
