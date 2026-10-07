"""Re-encode an exported camera's frames into an mp4 (device-source testing).

    .venv/Scripts/python.exe tools/frames_to_video.py inputs/sept3_20260905_130304 zed_right --out outputs/sept3_zed_right.mp4

Writes the recording's JPEG frames at the recorded timestamps' MEDIAN fps
(cv2.VideoWriter mp4v) plus a sidecar ``<out>.stamps.csv`` with the original
stamps, and a calibration YAML next to it (from the export's camera_info).
This is the input for the requirement-14 proof: run the full mapping
pipeline with [camera] source = device pointing at the mp4 + the YAML, with
ZERO ROS image topics subscribed, and compare against the ros2-source
baseline (gate: ATE within 15 percent on the same frames).
"""

import _bootstrap  # noqa: F401

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

from kart_map2d.camera.model import CameraModel, save_calibration_yaml
from kart_map2d.io.dataset import ExportedRun


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", help="recording directory (inputs/<run>)")
    parser.add_argument("camera", help="recorded camera name (e.g. zed_right)")
    parser.add_argument("--out", required=True, help="output .mp4 path")
    args = parser.parse_args(argv)

    run = ExportedRun(Path(args.run))
    frames = run.camera_frames(args.camera)
    if len(frames) < 2:
        print(f"camera {args.camera!r} has {len(frames)} frames - nothing to encode")
        return 2
    stamps = np.array([f.stamp for f in frames])
    fps = 1.0 / float(np.median(np.diff(stamps)))
    first = cv2.imread(frames[0].image_path)
    if first is None:
        print(f"cannot read {frames[0].image_path}")
        return 2
    h, w = first.shape[:2]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    codec = "MJPG" if out.suffix.lower() == ".avi" else "mp4v"
    # Ask for maximum quality; FFMPEG's mp4v honours VIDEOWRITER_PROP_QUALITY
    # on some builds and ignores it on others - measure the PSNR, do not trust
    # the flag (validation measured 36-38 dB for default mp4v).
    writer = cv2.VideoWriter(
        str(out), cv2.VideoWriter_fourcc(*codec), fps, (w, h),
        params=[int(cv2.VIDEOWRITER_PROP_QUALITY), 100],
    )
    if not writer.isOpened():
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*codec), fps, (w, h))
    if not writer.isOpened() and codec == "mp4v":
        # Some OpenCV builds ship without an mp4 muxer; MJPG-in-AVI always works.
        out = out.with_suffix(".avi")
        print(f"mp4v unavailable in this OpenCV build; writing MJPG {out.name} instead")
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    if not writer.isOpened():
        print("cv2.VideoWriter failed to open (no usable codec)")
        return 2
    written = 0
    for frame in frames:
        img = cv2.imread(frame.image_path)
        if img is None:
            continue
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        writer.write(img)
        written += 1
    writer.release()

    with open(f"{out}.stamps.csv", "w", newline="", encoding="ascii") as f:
        cw = csv.writer(f)
        cw.writerow(["frame_index", "stamp_s"])
        for i, t in enumerate(stamps):
            cw.writerow([i, f"{t:.6f}"])

    K, (iw, ih) = run.camera_intrinsics(args.camera)
    yaml_path = out.with_suffix(".calibration.yaml")
    save_calibration_yaml(yaml_path, CameraModel(K, (iw, ih)))  # exports are undistorted (D=0)

    print(f"wrote {out}: {written} frames at {fps:.2f} fps ({w}x{h})")
    print(f"wrote {out}.stamps.csv and {yaml_path}")
    print("Point [camera] at it:  source = device, device = " + str(out).replace("\\", "/") + f", calibration_file = {yaml_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
