"""Offline replay runner: build a 2D map from a recording, honouring [sensors].

    .venv/Scripts/python.exe tools/replay_run.py inputs/sept3_20260905_130304
    .venv/Scripts/python.exe tools/replay_run.py inputs/sept3_20260905_130304 --cfg kart_master.cfg
    .venv/Scripts/python.exe tools/replay_run.py inputs/... --set sensors.lidar=1 --set sensors.camera_2=1

The [sensors] switches of kart_master.cfg decide which replay sources feed
the fusion engine, exactly like the live node's subscriptions:

    imu -> imu, livox_imu -> imu_livox, odom -> odom,
    lidar -> lidar_odom (KISS-ICP odometry + 2D grid evidence; needs the
             kiss-icp package or a cached result),
    camera / camera_2 -> vo_<recorded camera> (matched to the recording via
             the [camera] image_topic / name, see --camera).

Output folders are named by the sensor combination (requirement 9), e.g.
``outputs/sept3_20260905_130304__imu+cam/`` - different combinations NEVER
overwrite each other. Every output folder is a loadable map package.

HONESTY: magnetometer / speedometer / wheel_ticks switches are accepted but
the September recordings contain none of that data - the runner says so and
continues without those channels. First real-data validation happens when
you record them (README.md lists the topics to add to ros2 bag record).
"""

import _bootstrap  # noqa: F401

import argparse
import json
import sys
import time
from pathlib import Path

from kart_map2d.config.master_config import MasterConfigError, combo_name, load_master_config
from kart_map2d.io.dataset import ExportedRun
from kart_map2d import replay as replay_mod

PROJECT_ROOT = _bootstrap.PROJECT_ROOT


def _match_recorded_camera(cfg_cam: dict, recorded: list[str], taken: set[str]) -> str | None:
    """The recording's camera folder that corresponds to one [camera] section."""
    name = str(cfg_cam.get("name") or "")
    if name in recorded and name not in taken:
        return name
    topic = str(cfg_cam.get("image_topic") or "")
    for cam in recorded:
        if cam not in taken and cam in topic:
            return cam
    for cam in recorded:  # positional fallback: first untaken recorded camera
        if cam not in taken:
            return cam
    return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", help="recording directory (e.g. inputs/sept3_20260905_130304)")
    parser.add_argument("--cfg", default=str(PROJECT_ROOT / "kart_master.cfg"), help="kart_master.cfg (default: the project's)")
    parser.add_argument("--set", action="append", default=[], metavar="sensors.NAME=0|1", help="override a [sensors] switch")
    parser.add_argument("--camera", action="append", default=[], metavar="CFGNAME=RECORDED",
                        help="map a cfg camera to a recorded camera folder (e.g. camera=zed_right)")
    parser.add_argument("--out-root", default=str(PROJECT_ROOT / "outputs"), help="output root (default: outputs/)")
    parser.add_argument("--seg-stride", type=int, default=1, help="segment every k-th frame")
    parser.add_argument("--no-loop-closure", action="store_true")
    args = parser.parse_args(argv)

    overrides = {}
    for token in args.set:
        key, _, value = token.partition("=")
        overrides[key.strip().lower().removeprefix("sensors.")] = int(value or "1")
    try:
        cfg = load_master_config(args.cfg, overrides=overrides)
    except MasterConfigError as exc:
        for problem in exc.problems:
            print(problem)
        return 2
    for line in cfg.warnings:
        print(f"WARNING: {line}")

    run_dir = Path(args.run)
    if not run_dir.exists():
        print(f"recording {run_dir} does not exist (is the inputs/ junction in place?)")
        return 2
    run = ExportedRun(run_dir)
    recorded_cameras = run.cameras()
    s = cfg["sensors"]

    # ------------------------------------------------ sources from [sensors]
    sources: list[str] = []
    if s["imu"]:
        sources.append("imu")
    if s["livox_imu"]:
        sources.append("imu_livox")
    if s["odom"]:
        sources.append("odom")
    if s["lidar"]:
        sources.append("lidar_odom")
    forced = dict(pair.split("=", 1) for pair in args.camera)
    taken: set[str] = set()
    heights: dict[str, float] = {}
    for switch, section in (("camera", "camera"), ("camera_2", "camera_2")):
        if not s[switch]:
            continue
        cam_cfg = cfg[section]
        cfg_name = str(cam_cfg["name"]) or section
        recorded = forced.get(cfg_name) or _match_recorded_camera(
            {"name": cam_cfg["name"], "image_topic": cam_cfg["image_topic"]}, recorded_cameras, taken
        )
        if recorded is None:
            print(f"[{section}]: the recording has no unmatched camera folder (has: {recorded_cameras}); skipped")
            continue
        taken.add(recorded)
        sources.append(f"vo_{recorded}")
        if cam_cfg["height_m"] is not None:
            heights[recorded] = float(cam_cfg["height_m"])
        print(f"[{section}] -> recorded camera {recorded!r}")
    replay_mod.set_camera_heights(heights)

    for missing in ("magnetometer", "speedometer", "wheel_ticks"):
        if s[missing]:
            print(
                f"NOTE: [sensors] {missing} = 1, but this recording contains no {missing} data - "
                "the channel is skipped offline. Validation for it is synthetic until you record "
                "the sensor (see README.md for the ros2 bag record topics)."
            )

    if not sources:
        print("no replayable sources enabled - turn on at least [sensors] imu or a camera")
        return 2

    combo = combo_name(cfg)
    name = combo.replace("+", "+")  # explicit: folder names carry the combo
    seg_mode = str(cfg["segmentation"]["mode"])
    print(f"sources: {sources}; segmentation: {seg_mode}; output: {args.out_root}/{run_dir.name}__{combo}")

    from kart_map2d.pipeline import build_map

    t0 = time.perf_counter()
    report = build_map(
        run_dir,
        sources,
        seg_stride=max(int(args.seg_stride), 1),
        name=name,
        seg_mode=seg_mode,
        loop_closure=not args.no_loop_closure and bool(cfg["mapping"]["loop_closure"]),
        out_root=Path(args.out_root),
        seg_cache_root=Path(args.out_root) / "seg",
        cache_dir=Path(args.out_root) / "cache",
    )
    dt = time.perf_counter() - t0
    out_dir = Path(args.out_root) / f"{run_dir.name}__{name}"
    print(f"done in {dt:.1f} s -> {out_dir}")
    for key in ("trajectory", "grid", "corridor", "racing_line", "score_vs_kiss", "loop_closure"):
        if key in report:
            print(f"  {key}: {json.dumps(report[key], default=str)[:200]}")
    return 0 if "error" not in report else 1


if __name__ == "__main__":
    sys.exit(main())
