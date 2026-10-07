"""Write a camera calibration YAML from a camera_info source, ONCE.

From a live topic (needs a sourced ROS 2 environment - Orin or container):
    python3 tools/dump_camera_info.py --topic /camera/camera_info --out config/camera_calibration.yaml

From a recorded bag (works on the laptop, no ROS needed):
    .venv/Scripts/python.exe tools/dump_camera_info.py --bag <bag_dir> --topic /zed_right/.../camera_info --out config/camera_calibration.yaml

The YAML (kart_map2d/camera/model.py format) carries intrinsics AND the
lens-distortion model/coefficients, and is what [camera] calibration_file
points at for source = device / gstreamer cameras.
"""

import _bootstrap  # noqa: F401

import argparse
import sys
from pathlib import Path

import numpy as np

from kart_map2d.camera.model import CameraModel, save_calibration_yaml


def from_bag(bag_dir: Path, topic: str) -> CameraModel:
    from rosbags.rosbag2 import Reader
    from rosbags.typesys import Stores, get_typestore

    typestore = get_typestore(Stores.ROS2_HUMBLE)
    with Reader(bag_dir) as reader:
        connections = [c for c in reader.connections if c.topic == topic]
        if not connections:
            available = sorted(c.topic for c in reader.connections if "camera_info" in c.topic)
            raise SystemExit(f"topic {topic!r} not in the bag; camera_info topics: {available}")
        for connection, _, raw in reader.messages(connections=connections):
            msg = typestore.deserialize_cdr(raw, connection.msgtype)
            return _model_from_msg(msg)
    raise SystemExit(f"no message on {topic!r}")


def _model_from_msg(msg) -> CameraModel:
    K = np.asarray(msg.k, dtype=np.float64).reshape(3, 3)
    D = np.asarray(getattr(msg, "d", []) or [], dtype=np.float64)
    model = str(getattr(msg, "distortion_model", "") or "none")
    return CameraModel(K, (int(msg.width), int(msg.height)), D, model)


def from_live(topic: str, timeout_s: float) -> CameraModel:
    import rclpy
    from sensor_msgs.msg import CameraInfo

    rclpy.init()
    node = rclpy.create_node("dump_camera_info")
    holder: list = []
    node.create_subscription(CameraInfo, topic, lambda msg: holder.append(msg), 10)
    try:
        import time

        deadline = time.monotonic() + timeout_s
        while not holder and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.2)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    if not holder:
        raise SystemExit(f"no CameraInfo on {topic!r} within {timeout_s:.0f} s - is the driver running?")
    return _model_from_msg(holder[0])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--topic", required=True, help="sensor_msgs/CameraInfo topic")
    parser.add_argument("--bag", default=None, help="read from this rosbag2 directory instead of live")
    parser.add_argument("--out", required=True, help="output calibration YAML path")
    parser.add_argument("--timeout", type=float, default=10.0, help="live wait [s]")
    args = parser.parse_args(argv)

    model = from_bag(Path(args.bag), args.topic) if args.bag else from_live(args.topic, args.timeout)
    path = save_calibration_yaml(Path(args.out), model)
    print(f"wrote {path}: {model.size[0]}x{model.size[1]}, distortion {model.model} "
          f"({'nonzero' if model.has_distortion else 'zero/rectified'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
