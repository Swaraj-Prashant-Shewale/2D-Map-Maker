"""Validate a kart_master.cfg without starting anything.

    ros2 run kart_map2d check_cfg /path/to/kart_master.cfg
    python -m kart_map2d.config.check_cfg kart_master.cfg [--set sensors.lidar=1]

Exit 0 when the file is valid (warnings printed but tolerated), 2 otherwise
with every problem listed - the same validation the nodes run at startup.
"""

from __future__ import annotations

import argparse
import sys

from kart_map2d.config.master_config import (
    MasterConfigError,
    combo_name,
    load_master_config,
    to_drive_params_dict,
    to_mapping_params_dict,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cfg", help="path to kart_master.cfg")
    parser.add_argument("--set", action="append", default=[], metavar="sensors.NAME=0|1")
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
    print(f"OK: {args.cfg}")
    print(f"sensors on: {', '.join(cfg.sensors_on()) or 'none'}  (combo name: {combo_name(cfg)})")
    for line in cfg.warnings:
        print(f"WARNING: {line}")
    mapping = to_mapping_params_dict(cfg)
    print(f"mapping: seg={mapping['seg_mode']}, cameras={list(mapping['cameras'])}, "
          f"imu={mapping['imu_topic'] or 'off'}, lidar={mapping['lidar_topic'] or 'off'}")
    try:
        drive = to_drive_params_dict(cfg)
        print(f"drive: map_package={drive['map_package'] or '(unset)'}, max_speed={drive['max_speed_mps']} m/s")
    except MasterConfigError as exc:
        print(f"drive translation: not driveable as configured ({'; '.join(exc.problems)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
