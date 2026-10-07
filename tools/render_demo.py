"""Demo video of the KART TARGET configuration: ONE camera + the dedicated IMU.

Self-contained in the 2D project (adapted from the research workspace's
scripts/render_video.py, rebuilt on kart_map2d's pipeline API). It renders
results/demo_2d_single_cam_imu.mp4 from the sept3 recording:

- LEFT  panel: the camera frame with the classical segmentation overlay
  exactly as the mapper sees it (drivable green, obstacle red, dynamic
  orange, kart body dimmed), plus a legend.
- RIGHT panel: the 2D evidence grid GROWING from camera evidence ONLY -
  NO LIDAR ANYWHERE: localization is camera+IMU, and the script asserts
  the [sensors] lidar switch is 0, uses no lidar source, and traps any
  insert_lidar call. What you see is exactly what the kart's target
  sensor set produces - including its noise; nothing is dressed up.
- BOTTOM status bar: time, speed, fusion mode, per-channel health dots.
- ENDING (held): the finished map with corridor edges and the racing line
  coloured by planned speed, plus a title card of honest numbers.

    .venv/Scripts/python.exe tools/render_demo.py
    (options: --run, --camera, --fps, --out, --limit N for a quick test)

The output goes to results/ (the never-upload list; it contains recorded
imagery of identifiable people - keep it local).
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import sys
import time
from bisect import bisect_right
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from kart_map2d.config.master_config import load_master_config
from kart_map2d.mapping.corridor import extract_corridor
from kart_map2d.mapping.evidence_grid import CLASS_DRIVABLE, CLASS_DYNAMIC, CLASS_OBSTACLE, DYNAMIC_CLASS_IDS, EvidenceGrid
from kart_map2d.pipeline import closed_classes, interp_poses, localize, supported_mask
from kart_map2d.planning.racing_line import compute_racing_line

PROJECT_ROOT = _bootstrap.PROJECT_ROOT

# ------------------------------------------------------------------- layout
CAM_W, CAM_H = 960, 600
MAP_W, MAP_H = 640, 600
BAR_H = 100
CANVAS_W, CANVAS_H = CAM_W + MAP_W, CAM_H + BAR_H

FONT = cv2.FONT_HERSHEY_SIMPLEX
_DYNAMIC_IDS = np.array(sorted(DYNAMIC_CLASS_IDS), dtype=np.uint8)

COL_DRIVABLE = (40, 210, 40)
COL_OBSTACLE = (30, 30, 235)
COL_DYNAMIC = (0, 165, 255)
COL_KART = (70, 70, 70)
COL_TRAJ = (255, 140, 0)
COL_TEXT = (235, 235, 235)
HEALTH_COLOURS = {
    "ok": (60, 200, 60),
    "degraded": (40, 210, 230),
    "probation": (40, 210, 230),
    "faulty": (40, 40, 230),
    "stale": (150, 150, 150),
    "unseen": (150, 150, 150),
}


def put_text(img, text, org, scale=0.5, colour=COL_TEXT, thick=1) -> None:
    x, y = org
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        cv2.putText(img, text, (x + dx, y + dy), FONT, scale, (0, 0, 0), thick, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, colour, thick, cv2.LINE_AA)


# --------------------------------------------------------------- left panel
def overlay_segmentation(image: NDArray[np.uint8], seg) -> NDArray[np.uint8]:
    out = image.astype(np.float32)
    dyn = np.isin(seg.class_map, _DYNAMIC_IDS)
    masks = (
        (seg.drivable_mask, COL_DRIVABLE, 0.35),
        (seg.obstacle_mask & ~dyn, COL_OBSTACLE, 0.35),
        (dyn, COL_DYNAMIC, 0.45),
    )
    for mask, colour, alpha in masks:
        if mask.any():
            out[mask] = (1.0 - alpha) * out[mask] + alpha * np.array(colour, dtype=np.float32)
    kart = seg.kart_mask
    if kart.any():
        out[kart] *= 0.35
    return out.astype(np.uint8)


def draw_legend(panel: NDArray[np.uint8]) -> None:
    entries = (
        ("drivable", COL_DRIVABLE),
        ("obstacle", COL_OBSTACLE),
        ("dynamic (person/vehicle)", COL_DYNAMIC),
        ("kart body (masked out)", COL_KART),
    )
    x0, y0, row = 10, 34, 22
    box = panel[y0 - 18 : y0 - 18 + 8 + row * len(entries), x0 - 6 : x0 + 250]
    box[:] = (box * 0.35).astype(np.uint8)
    for i, (label, colour) in enumerate(entries):
        y = y0 + i * row
        cv2.rectangle(panel, (x0, y - 11), (x0 + 16, y + 3), colour, -1)
        put_text(panel, label, (x0 + 24, y + 2), 0.45)


# -------------------------------------------------------------- right panel
def render_grid_bgr(grid: EvidenceGrid, now: float, live: bool = True) -> NDArray[np.uint8]:
    nx, ny = grid.shape
    if nx == 0 or ny == 0:
        return np.full((16, 16, 3), 110, dtype=np.uint8)
    cls = grid.classify(include_dynamic=live, now=now)
    img = np.full((nx, ny, 3), 110, dtype=np.uint8)
    drv = cls == CLASS_DRIVABLE
    conf = np.clip((grid.drivability() - 0.5) * 2.0, 0.0, 1.0)
    img[drv, 0] = 40
    img[drv, 1] = (110 + 120 * conf[drv]).astype(np.uint8)
    img[drv, 2] = 40
    obs = cls == CLASS_OBSTACLE
    conf_o = np.clip((grid.occupancy() - 0.5) * 2.0, 0.0, 1.0)
    img[obs, 0] = 30
    img[obs, 1] = 30
    img[obs, 2] = (140 + 115 * conf_o[obs]).astype(np.uint8)
    if live:
        dyn = cls == CLASS_DYNAMIC
        if dyn.any():
            conf_d = np.clip((grid.dynamic_occupancy(now) - 0.5) * 2.0, 0.0, 1.0)
            img[dyn, 0] = 20
            img[dyn, 1] = (110 + 50 * conf_d[dyn]).astype(np.uint8)
            img[dyn, 2] = (200 + 55 * conf_d[dyn]).astype(np.uint8)
    return img.transpose(1, 0, 2)[::-1].copy()


class MapView:
    """Auto-zoom onto the explored region; the view box only grows."""

    def __init__(self, margin_m: float = 3.0, min_span_m: float = 14.0, max_px_per_cell: float = 6.0):
        self.margin = margin_m
        self.min_span = min_span_m
        self.max_scale = max_px_per_cell
        self.box: list[float] | None = None

    def update_box(self, grid: EvidenceGrid) -> None:
        nx, ny = grid.shape
        if nx == 0:
            return
        known = grid.classify(include_dynamic=True) != 0
        xs = np.flatnonzero(known.any(axis=1))
        ys = np.flatnonzero(known.any(axis=0))
        if not len(xs):
            return
        ox, oy = grid.origin
        c = grid.cell
        box = [
            ox + xs[0] * c - self.margin,
            ox + (xs[-1] + 1) * c + self.margin,
            oy + ys[0] * c - self.margin,
            oy + (ys[-1] + 1) * c + self.margin,
        ]
        for axis in (0, 2):
            span = box[axis + 1] - box[axis]
            if span < self.min_span:
                pad = 0.5 * (self.min_span - span)
                box[axis] -= pad
                box[axis + 1] += pad
        if self.box is None:
            self.box = box
        else:
            self.box = [
                min(self.box[0], box[0]),
                max(self.box[1], box[1]),
                min(self.box[2], box[2]),
                max(self.box[3], box[3]),
            ]

    def render(self, grid: EvidenceGrid, now: float, live: bool = True):
        full = render_grid_bgr(grid, now, live=live)
        panel = np.full((MAP_H, MAP_W, 3), 110, dtype=np.uint8)
        nx, ny = grid.shape
        if nx == 0 or self.box is None:
            return panel, None
        ox, oy = grid.origin
        c = grid.cell
        c0 = int(np.clip(np.floor((self.box[0] - ox) / c), 0, nx - 1))
        c1 = int(np.clip(np.ceil((self.box[1] - ox) / c), c0 + 1, nx))
        iy0 = int(np.clip(np.floor((self.box[2] - oy) / c), 0, ny - 1))
        iy1 = int(np.clip(np.ceil((self.box[3] - oy) / c), iy0 + 1, ny))
        r0 = ny - iy1
        crop = full[r0 : ny - iy0, c0:c1]
        f = min(MAP_W / crop.shape[1], MAP_H / crop.shape[0], self.max_scale)
        w, h = max(int(crop.shape[1] * f), 1), max(int(crop.shape[0] * f), 1)
        interp = cv2.INTER_NEAREST if f >= 1.0 else cv2.INTER_AREA
        resized = cv2.resize(crop, (w, h), interpolation=interp)
        x_off, y_off = (MAP_W - w) // 2, (MAP_H - h) // 2
        panel[y_off : y_off + h, x_off : x_off + w] = resized

        def world_to_px(xy: NDArray[np.float64]) -> NDArray[np.float64]:
            xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
            u = (xy[:, 0] - ox) / c - c0
            v = (ny - (xy[:, 1] - oy) / c) - r0
            return np.column_stack([u * f + x_off, v * f + y_off])

        bar = int(5.0 / c * f)
        y, x1 = MAP_H - 16, MAP_W - 16
        cv2.line(panel, (x1 - bar, y), (x1, y), (255, 255, 255), 2)
        put_text(panel, "5 m", (x1 - bar, y - 8), 0.45)
        return panel, world_to_px


def draw_trajectory(panel, world_to_px, xy, colour=COL_TRAJ, thick=2) -> None:
    if world_to_px is None or len(xy) < 2:
        return
    px = world_to_px(xy[:: max(1, len(xy) // 600)]).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(panel, [px], False, colour, thick, cv2.LINE_AA)


def draw_pose_arrow(panel, world_to_px, pose_xyyaw) -> None:
    if world_to_px is None:
        return
    x, y, yaw = pose_xyyaw
    tip = np.array([x + 1.8 * np.cos(yaw), y + 1.8 * np.sin(yaw)])
    p0, p1 = world_to_px(np.array([[x, y], tip]))
    p0, p1 = tuple(p0.astype(int)), tuple(p1.astype(int))
    cv2.arrowedLine(panel, p0, p1, (0, 0, 0), 5, cv2.LINE_AA, tipLength=0.45)
    cv2.arrowedLine(panel, p0, p1, (255, 255, 255), 2, cv2.LINE_AA, tipLength=0.45)


# --------------------------------------------------------------- status bar
class HealthTimeline:
    def __init__(self, engine, channels):
        self.channels = list(channels)
        self._events: dict[str, tuple[list[float], list[str]]] = {c: ([], []) for c in self.channels}
        for stamp, channel, status, _reason in engine.health.transitions:
            if channel in self._events:
                self._events[channel][0].append(stamp)
                self._events[channel][1].append(status)

    def at(self, t: float) -> dict[str, str]:
        out = {}
        for c in self.channels:
            stamps, statuses = self._events[c]
            i = bisect_right(stamps, t)
            out[c] = statuses[i - 1] if i else "unseen"
        return out


def mode_at(modes, t: float) -> str:
    stamps = [m.stamp for m in modes]
    i = bisect_right(stamps, t)
    return modes[i - 1].mode if i else "starting"


def draw_status_bar(canvas, t_rel, speed, mode, health: dict[str, str], frame_no, n_frames, traj_label):
    bar = canvas[CAM_H:, :]
    bar[:] = (24, 24, 24)
    cv2.line(canvas, (0, CAM_H), (CANVAS_W, CAM_H), (80, 80, 80), 1)
    put_text(bar, f"t = {t_rel:5.1f} s", (14, 34), 0.65)
    put_text(bar, f"speed {speed:4.2f} m/s", (170, 34), 0.65)
    put_text(bar, f"frame {frame_no:3d}/{n_frames}", (390, 34), 0.65)
    put_text(bar, f"fusion mode: {mode}", (14, 74), 0.55)
    put_text(bar, f"trajectory: {traj_label}", (390, 74), 0.55)
    x = 950
    put_text(bar, "sensor health:", (x, 26), 0.5)
    for name, status in health.items():
        colour = HEALTH_COLOURS.get(status, (150, 150, 150))
        cv2.circle(bar, (x + 8, 48), 7, colour, -1)
        cv2.circle(bar, (x + 8, 48), 7, (0, 0, 0), 1)
        put_text(bar, name, (x - 2, 78), 0.42)
        x += 150


# ------------------------------------------------------------------- ending
def speed_colours(speeds: NDArray[np.float64]) -> NDArray[np.uint8]:
    lo, hi = float(speeds.min()), float(speeds.max())
    norm = (speeds - lo) / max(hi - lo, 1e-6)
    return cv2.applyColorMap((norm * 255).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_TURBO)[:, 0]


def draw_final_map(panel, world_to_px, corridor, line, traj_xy):
    draw_trajectory(panel, world_to_px, traj_xy, COL_TRAJ, 1)
    if corridor is not None and world_to_px is not None:
        for edge in (corridor.left_edge(), corridor.right_edge()):
            px = world_to_px(edge).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(panel, [px], corridor.closed, (255, 255, 255), 2, cv2.LINE_AA)
    if line is not None and world_to_px is not None:
        px = world_to_px(line.path).astype(np.int32)
        cols = speed_colours(line.speed)
        for i in range(len(px) - 1):
            cv2.line(panel, tuple(px[i]), tuple(px[i + 1]), tuple(int(v) for v in cols[i]), 3, cv2.LINE_AA)
        x0, y0, w = 12, 52, 160
        grad = cv2.applyColorMap(np.tile(np.linspace(0, 255, w).astype(np.uint8), (12, 1)), cv2.COLORMAP_TURBO)
        panel[y0 : y0 + 12, x0 : x0 + w] = grad
        put_text(panel, "racing line speed [m/s]", (x0, y0 - 8), 0.42)
        put_text(panel, f"{line.speed.min():.1f}", (x0, y0 + 28), 0.42)
        put_text(panel, f"{line.speed.max():.1f}", (x0 + w - 30, y0 + 28), 0.42)


def draw_title_card(canvas, lines):
    x0, y0, row = 40, 120, 34
    w, h = 600, 40 + row * len(lines)
    roi = canvas[y0 : y0 + h, x0 : x0 + w]
    roi[:] = (roi * 0.2).astype(np.uint8)
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (200, 200, 200), 1)
    for i, text in enumerate(lines):
        put_text(canvas, text, (x0 + 20, y0 + 34 + i * row), 0.62, thick=1 if i else 2)


def _forbid_lidar(grid: EvidenceGrid) -> None:
    """NO LIDAR IN THIS VIDEO: any insert_lidar call is a bug, not a feature."""

    def _refuse(*_a, **_k):
        raise AssertionError("render_demo is camera+IMU only: insert_lidar must never run")

    grid.insert_lidar = _refuse  # type: ignore[method-assign]


# --------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default=str(PROJECT_ROOT / "inputs" / "sept3_20260905_130304"))
    ap.add_argument("--camera", default="zed_right", help="the ONE camera (the kart target)")
    ap.add_argument("--cfg", default=str(PROJECT_ROOT / "kart_master.cfg"))
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--hold-s", type=float, default=4.0)
    ap.add_argument("--out", default=str(PROJECT_ROOT / "results"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    # ---------------------------------------------------- honesty guards
    cfg = load_master_config(args.cfg)
    assert not cfg["sensors"]["lidar"], (
        "the demo renders the kart TARGET configuration: [sensors] lidar must be 0 "
        f"(it is 1 in {args.cfg}) - the user explicitly rejected a LiDAR-fed demo map"
    )
    sources = ("imu", f"vo_{args.camera}")
    assert not any("lidar" in s for s in sources)

    t_wall = time.perf_counter()
    run_dir = Path(args.run)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------- localization (camera+IMU only)
    print(f"[localize] sources={sources} (camera+IMU only, classical segmentation)")
    loc = localize(
        run_dir,
        sources,
        seg_mode="classical",
        seg_cache_root=PROJECT_ROOT / "outputs" / "seg",
        cache_dir=PROJECT_ROOT / "outputs" / "cache",
    )
    stamps, supported = loc.stamps, loc.supported
    est_xyyaw = loc.est_xyyaw
    engine = loc.engine
    lc = loc.sections.get("loop_closure", {})
    n_closures = int(lc.get("accepted", 0) or 0)
    traj_label = f"loop-closed EKF ({n_closures} closures)" if n_closures else "raw EKF (no closures accepted)"
    speeds = np.abs(loc.states[:, 3])
    print(f"[localize] {len(stamps)} poses, closures={n_closures}")

    cam_entry = next(c for c in loc.cam_data if c[0] == args.camera)
    _, span, segs, _used, _used_segs, K, T_base_cam = cam_entry
    ok = supported_mask(np.array([f.stamp for f in span]), stamps, supported)
    frames = span
    if args.limit:
        frames, segs, ok = frames[: args.limit], segs[: args.limit], ok[: args.limit]
    n_run = len(frames)
    n_hold = int(round(args.hold_s * args.fps))
    n_total = n_run + n_hold
    preview_at = set(np.linspace(0, n_total - 1, 4).astype(int).tolist())
    print(f"[timeline] {n_run} {args.camera} frames + {n_hold} hold = {n_total}")

    grid = EvidenceGrid(cell=float(cfg["mapping"]["cell_size_m"]))
    _forbid_lidar(grid)
    view = MapView()
    health = HealthTimeline(engine, sources)
    t_start = float(stamps[0])

    def insert_camera_frame(frame, seg):
        pose = interp_poses(np.array([frame.stamp]), stamps, est_xyyaw)[0]
        dyn = np.isin(seg.class_map, _DYNAMIC_IDS)
        if dyn.any():
            grid.insert_camera(seg.drivable_mask, seg.obstacle_mask, K, T_base_cam, pose,
                               dynamic_mask=dyn, stamp=float(frame.stamp))
        else:
            grid.insert_camera(seg.drivable_mask, seg.obstacle_mask, K, T_base_cam, pose)

    # ------------------------------------------------------------- writer
    video_path = out_dir / "demo_2d_single_cam_imu.mp4"
    writer = None
    codec_used = None
    for codec in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*codec), args.fps, (CANVAS_W, CANVAS_H))
        if writer.isOpened():
            codec_used = codec
            break
        writer.release()
    if writer is None or not writer.isOpened():
        raise RuntimeError("no working codec among ('avc1', 'mp4v')")
    print(f"[writer] {video_path} codec={codec_used}")

    # ----------------------------------------------------------- run frames
    written = 0
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    for i, (frame, seg) in enumerate(zip(frames, segs)):
        t = float(frame.stamp)
        if ok[i]:
            insert_camera_frame(frame, seg)
        view.update_box(grid)

        image = cv2.imread(frame.image_path)
        if image is None:
            raise FileNotFoundError(frame.image_path)
        left = overlay_segmentation(image, seg)
        if left.shape[:2] != (CAM_H, CAM_W):
            left = cv2.resize(left, (CAM_W, CAM_H), interpolation=cv2.INTER_AREA)
        put_text(left, f"camera {args.camera} + classical segmentation (mapper's view)", (10, CAM_H - 12), 0.5)
        draw_legend(left)

        right, world_to_px = view.render(grid, now=t, live=True)
        past = est_xyyaw[stamps <= t, :2]
        draw_trajectory(right, world_to_px, past)
        draw_pose_arrow(right, world_to_px, interp_poses(np.array([t]), stamps, est_xyyaw)[0])
        put_text(right, "2D evidence grid - CAMERA EVIDENCE ONLY", (10, 20), 0.5)
        put_text(right, "localization: one camera + dedicated IMU", (10, MAP_H - 44), 0.42)
        put_text(right, "(no LiDAR anywhere in this video)", (10, MAP_H - 28), 0.42)

        canvas[:CAM_H, :CAM_W] = left
        canvas[:CAM_H, CAM_W:] = right
        draw_status_bar(canvas, t - t_start, float(np.interp(t, stamps, speeds)), mode_at(engine.modes, t),
                        health.at(t), i + 1, n_run, traj_label)
        writer.write(canvas)
        written += 1
        if (written - 1) in preview_at:
            cv2.imwrite(str(out_dir / f"demo_preview_{sorted(preview_at).index(written - 1) + 1}.jpg"), canvas)
        if written % 20 == 0:
            print(f"[frames] {written}/{n_run}")

    # ----------------------------------------------------------- finish map
    view.update_box(grid)
    cls = grid.classify()
    n_drv = int((cls == CLASS_DRIVABLE).sum())
    n_obs = int((cls == CLASS_OBSTACLE).sum())
    step = np.linalg.norm(np.diff(est_xyyaw[:, :2], axis=0), axis=1)
    path_len = float(step.sum())

    corridor = line = None
    corridor_err = ""
    try:
        classes = closed_classes(grid, int(cfg["mapping"]["corridor_close_cells"]))
        corridor = extract_corridor(
            classes,
            est_xyyaw[supported, :2],
            spacing_m=float(cfg["mapping"]["corridor_spacing_m"]),
            origin_xy=tuple(np.asarray(grid.origin, dtype=float)),
            resolution_m=float(grid.cell),
        )
        line = compute_racing_line(corridor)
    except Exception as exc:  # noqa: BLE001 - the demo still ends with the map
        corridor_err = f"{type(exc).__name__}: {exc}"
        print(f"[warn] corridor/racing line failed: {corridor_err}")

    right, world_to_px = view.render(grid, now=float(stamps[-1]), live=False)
    draw_final_map(right, world_to_px, corridor, line, est_xyyaw[:, :2])
    put_text(right, "finished camera-only map + corridor + racing line", (10, 20), 0.5)
    canvas[:CAM_H, CAM_W:] = right

    card = [
        "KART TARGET CONFIG - sept3",
        "sensors: ONE camera + dedicated IMU (no LiDAR)",
        f"driven path: {path_len:.1f} m in {stamps[-1] - stamps[0]:.0f} s",
        f"drivable cells: {n_drv} ({grid.cell:.2f} m), obstacle cells: {n_obs}",
        f"loop closures accepted: {n_closures}",
    ]
    if corridor is not None:
        card.append(f"corridor: {len(corridor.centerline)} stations, {corridor.s[-1]:.1f} m")
    if line is not None:
        card.append(f"racing line: {line.time_s:.1f} s at {line.speed.min():.1f}-{line.speed.max():.1f} m/s")
    draw_title_card(canvas, card)

    for _ in range(n_hold):
        writer.write(canvas)
        written += 1
        if (written - 1) in preview_at:
            cv2.imwrite(str(out_dir / f"demo_preview_{sorted(preview_at).index(written - 1) + 1}.jpg"), canvas)
    writer.release()

    # ------------------------------------------------------------ verify
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot reopen {video_path}")
    count = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps_read = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, count // 2)
    ok_mid, _ = cap.read()
    cap.release()
    size_mb = video_path.stat().st_size / 1e6
    assert count == written and count > 0, f"frame count mismatch: wrote {written}, file has {count}"
    assert ok_mid, "middle frame failed to decode"

    report = {
        "video": str(video_path),
        "codec": codec_used,
        "frames_written": written,
        "frames_in_file": count,
        "fps": fps_read,
        "duration_s": round(written / args.fps, 1),
        "size_mb": round(size_mb, 2),
        "resolution": f"{CANVAS_W}x{CANVAS_H}",
        "sources": list(sources),
        "lidar_used": False,
        "loop_closures": n_closures,
        "drivable_cells": n_drv,
        "obstacle_cells": n_obs,
        "path_length_m": round(path_len, 1),
        "corridor_error": corridor_err,
        "wall_s": round(time.perf_counter() - t_wall, 1),
    }
    (out_dir / "demo_render_report.json").write_text(json.dumps(report, indent=1), encoding="ascii")
    print("[done] " + json.dumps(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
