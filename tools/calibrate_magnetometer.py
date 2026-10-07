"""Fit a magnetometer hard/soft-iron calibration from a rosbag or CSV.

    .venv/Scripts/python.exe tools/calibrate_magnetometer.py --bag <bag_dir> --topic /mag --out config/mag_calibration.yaml
    .venv/Scripts/python.exe tools/calibrate_magnetometer.py --csv samples.csv --out config/mag_calibration.yaml

Recording procedure (also printed by the node when the calibration is
missing): drive one SLOW full 360 deg circle (30-60 s) on open ground away
from steel, recording the sensor_msgs/MagneticField topic. The fit is a
classical least-squares ellipsoid (no learned models); degenerate inputs
are refused with the reason.

CSV format: three columns x,y,z (tesla), with or without a header line.
"""

import _bootstrap  # noqa: F401

import argparse
import sys
from pathlib import Path

import numpy as np

from kart_map2d.sensors.mag_calibration import MagCalibrationError, fit_ellipsoid, save_mag_calibration


def samples_from_bag(bag_dir: Path, topic: str) -> np.ndarray:
    from rosbags.rosbag2 import Reader
    from rosbags.typesys import Stores, get_typestore

    typestore = get_typestore(Stores.ROS2_HUMBLE)
    rows = []
    with Reader(bag_dir) as reader:
        connections = [c for c in reader.connections if c.topic == topic]
        if not connections:
            available = sorted({c.topic for c in reader.connections})
            raise SystemExit(f"topic {topic!r} not in the bag; available: {available}")
        for connection, _, raw in reader.messages(connections=connections):
            msg = typestore.deserialize_cdr(raw, connection.msgtype)
            f = msg.magnetic_field
            rows.append((f.x, f.y, f.z))
    return np.asarray(rows, dtype=np.float64)


def samples_from_csv(path: Path) -> np.ndarray:
    rows = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        parts = [p for p in line.replace(",", " ").split() if p]
        if len(parts) < 3:
            continue
        try:
            rows.append((float(parts[0]), float(parts[1]), float(parts[2])))
        except ValueError:
            continue  # header line
    return np.asarray(rows, dtype=np.float64)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--bag", help="rosbag2 directory holding the magnetometer topic")
    source.add_argument("--csv", help="CSV with x,y,z columns in tesla")
    parser.add_argument("--topic", default="/mag", help="MagneticField topic in the bag")
    parser.add_argument("--out", required=True, help="output calibration YAML")
    args = parser.parse_args(argv)

    samples = samples_from_bag(Path(args.bag), args.topic) if args.bag else samples_from_csv(Path(args.csv))
    print(f"{len(samples)} samples")
    try:
        cal = fit_ellipsoid(samples)
    except MagCalibrationError as exc:
        print(str(exc))
        return 2
    path = save_mag_calibration(Path(args.out), cal)
    print(f"wrote {path}")
    print(f"  hard iron: {np.array2string(cal.hard_iron, precision=6)}")
    print(f"  field norm: {cal.field_norm:.6g}  rms residual: {cal.rms_residual:.3f} of norm")
    print("Set [magnetometer] calibration_file to this path and declination_deg for your site.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
