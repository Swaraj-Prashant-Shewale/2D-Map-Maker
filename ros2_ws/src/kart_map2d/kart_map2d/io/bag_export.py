"""Convert a rosbag2 recording into compact arrays for fast offline replay.

A 34 GB bag is dominated by raw BGRA images and LiDAR clouds. Decoding it once
into NumPy archives and downscaled JPEGs makes every later experiment a few
seconds instead of several minutes, and it removes the ROS dependency from the
mapping code.

Usage:
    python -m kart_map2d.io.bag_export inputs/sept2_20260905_124659

The argument is one recording folder under inputs/ (metadata.yaml + the
.db3); the export lands in inputs/<recording>/export/ (see inputs/README.txt).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

TYPESTORE = get_typestore(Stores.ROS2_HUMBLE)

_POINT_DTYPES = {1: "<i1", 2: "<u1", 3: "<i2", 4: "<u2", 5: "<i4", 6: "<u4", 7: "<f4", 8: "<f8"}


def _stamp(header) -> float:
    return header.stamp.sec + header.stamp.nanosec * 1e-9


def _quat_to_list(q) -> list[float]:
    return [q.x, q.y, q.z, q.w]


def _cloud_dtype(msg) -> np.dtype:
    return np.dtype(
        {
            "names": [f.name for f in msg.fields],
            "formats": [_POINT_DTYPES[f.datatype] for f in msg.fields],
            "offsets": [f.offset for f in msg.fields],
            "itemsize": msg.point_step,
        }
    )


class _CloudChunkWriter:
    """Buffer LiDAR scans and flush them in chunks to bound memory use."""

    def __init__(self, directory: Path, scans_per_chunk: int = 600) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.scans_per_chunk = scans_per_chunk
        self.chunk_index = 0
        self._reset()

    def _reset(self) -> None:
        self.stamps: list[float] = []
        self.receive: list[float] = []
        self.counts: list[int] = []
        self.xyz: list[np.ndarray] = []
        self.intensity: list[np.ndarray] = []
        self.dt: list[np.ndarray] = []

    def add(self, stamp: float, receive: float, cloud: np.ndarray) -> None:
        xyz = np.column_stack([cloud["x"], cloud["y"], cloud["z"]]).astype(np.float32)
        finite = np.all(np.isfinite(xyz), axis=1)
        rng = np.linalg.norm(xyz, axis=1)
        # The MID-360 returns points on the kart itself inside ~0.3 m.
        keep = finite & (rng > 0.3) & (rng < 60.0)
        cloud = cloud[keep]
        xyz = xyz[keep]
        if "timestamp" in cloud.dtype.names and len(cloud):
            # Per-point capture time relative to the scan header, for de-skewing.
            dt = (cloud["timestamp"].astype(np.float64) * 1e-9 - stamp).astype(np.float32)
        else:
            dt = np.zeros(len(cloud), dtype=np.float32)
        self.stamps.append(stamp)
        self.receive.append(receive)
        self.counts.append(len(cloud))
        self.xyz.append(xyz.astype(np.float16))
        self.intensity.append(np.clip(cloud["intensity"], 0, 255).astype(np.uint8))
        self.dt.append(dt.astype(np.float16))
        if len(self.stamps) >= self.scans_per_chunk:
            self.flush()

    def flush(self) -> None:
        if not self.stamps:
            return
        np.savez(
            self.directory / f"chunk_{self.chunk_index:04d}.npz",
            stamp=np.asarray(self.stamps),
            receive=np.asarray(self.receive),
            count=np.asarray(self.counts, dtype=np.int32),
            xyz=np.concatenate(self.xyz),
            intensity=np.concatenate(self.intensity),
            dt=np.concatenate(self.dt),
        )
        self.chunk_index += 1
        self._reset()


def export_bag(bag_dir: Path, out_dir: Path, image_scale: float = 0.5, jpeg_quality: int = 90) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    imus: dict[str, list] = {}
    odom: list = []
    scans: list = []
    scan_meta: dict = {}
    camera_info: dict[str, dict] = {}
    camera_rows: dict[str, list] = {}
    static_tf: list[dict] = []
    summary: dict = {"bag": str(bag_dir), "topics": {}}
    cloud_writer = _CloudChunkWriter(out_dir / "lidar")

    with Reader(bag_dir) as reader:
        for connection in reader.connections:
            summary["topics"][connection.topic] = {
                "type": connection.msgtype,
                "count": connection.msgcount,
            }
        for connection, timestamp, raw in reader.messages():
            topic = connection.topic
            receive = timestamp * 1e-9
            msg = TYPESTORE.deserialize_cdr(raw, connection.msgtype)
            kind = connection.msgtype

            if kind == "sensor_msgs/msg/Imu":
                imus.setdefault(topic, []).append(
                    (
                        _stamp(msg.header),
                        receive,
                        msg.angular_velocity.x,
                        msg.angular_velocity.y,
                        msg.angular_velocity.z,
                        msg.linear_acceleration.x,
                        msg.linear_acceleration.y,
                        msg.linear_acceleration.z,
                    )
                )
                summary["topics"][topic]["frame_id"] = msg.header.frame_id
            elif kind == "nav_msgs/msg/Odometry":
                pose = msg.pose.pose
                twist = msg.twist.twist
                odom.append(
                    (
                        _stamp(msg.header),
                        receive,
                        pose.position.x,
                        pose.position.y,
                        pose.position.z,
                        *_quat_to_list(pose.orientation),
                        twist.linear.x,
                        twist.linear.y,
                        twist.angular.z,
                    )
                )
                summary["topics"][topic]["frame_id"] = msg.header.frame_id
                summary["topics"][topic]["child_frame_id"] = msg.child_frame_id
            elif kind == "sensor_msgs/msg/LaserScan":
                ranges = np.asarray(msg.ranges, dtype=np.float32)
                scans.append((_stamp(msg.header), receive, ranges.astype(np.float16)))
                scan_meta = {
                    "frame_id": msg.header.frame_id,
                    "angle_min": float(msg.angle_min),
                    "angle_increment": float(msg.angle_increment),
                    "range_min": float(msg.range_min),
                    "range_max": float(msg.range_max),
                }
            elif kind == "sensor_msgs/msg/PointCloud2":
                cloud = np.frombuffer(bytes(msg.data), dtype=_cloud_dtype(msg))
                cloud_writer.add(_stamp(msg.header), receive, cloud)
                summary["topics"][topic]["frame_id"] = msg.header.frame_id
            elif kind == "sensor_msgs/msg/Image":
                name = topic.strip("/").split("/")[0]
                directory = out_dir / name
                directory.mkdir(exist_ok=True)
                channels = len(msg.data) // (msg.height * msg.width)
                image = np.frombuffer(bytes(msg.data), np.uint8).reshape(msg.height, msg.width, channels)
                encoding = msg.encoding.lower()
                if encoding.startswith("rgb"):
                    image = image[:, :, 2::-1] if channels >= 3 else image
                elif channels == 4:
                    image = image[:, :, :3]
                if image_scale != 1.0:
                    image = cv2.resize(image, None, fx=image_scale, fy=image_scale, interpolation=cv2.INTER_AREA)
                rows = camera_rows.setdefault(name, [])
                filename = f"{len(rows):06d}.jpg"
                cv2.imwrite(str(directory / filename), image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
                rows.append((_stamp(msg.header), receive, filename))
            elif kind == "sensor_msgs/msg/CameraInfo":
                name = topic.strip("/").split("/")[0]
                if name not in camera_info:
                    camera_info[name] = {
                        "frame_id": msg.header.frame_id,
                        "width": int(msg.width),
                        "height": int(msg.height),
                        "K": np.asarray(msg.k).reshape(3, 3).tolist(),
                        "D": np.asarray(msg.d).tolist(),
                        "distortion_model": msg.distortion_model,
                        "export_scale": image_scale,
                    }
            elif kind == "tf2_msgs/msg/TFMessage" and topic == "/tf_static":
                for transform in msg.transforms:
                    t = transform.transform.translation
                    static_tf.append(
                        {
                            "parent": transform.header.frame_id,
                            "child": transform.child_frame_id,
                            "translation": [t.x, t.y, t.z],
                            "rotation_xyzw": _quat_to_list(transform.transform.rotation),
                        }
                    )
            elif kind == "std_msgs/msg/String" and topic == "/robot_description":
                (out_dir / "robot_description.urdf").write_text(msg.data, encoding="utf-8")

    cloud_writer.flush()
    for topic, rows in imus.items():
        array = np.asarray(rows, dtype=np.float64)
        np.savez(
            out_dir / f"imu_{topic.strip('/').replace('/', '_')}.npz",
            stamp=array[:, 0],
            receive=array[:, 1],
            gyro=array[:, 2:5],
            accel=array[:, 5:8],
        )
    if odom:
        array = np.asarray(odom, dtype=np.float64)
        np.savez(
            out_dir / "odom.npz",
            stamp=array[:, 0],
            receive=array[:, 1],
            position=array[:, 2:5],
            quat_xyzw=array[:, 5:9],
            twist=array[:, 9:12],
        )
    if scans:
        np.savez(
            out_dir / "scan.npz",
            stamp=np.asarray([s[0] for s in scans]),
            receive=np.asarray([s[1] for s in scans]),
            ranges=np.stack([s[2] for s in scans]),
            **{k: np.asarray(v) for k, v in scan_meta.items() if k != "frame_id"},
        )
        summary["scan_frame_id"] = scan_meta.get("frame_id")
    for name, rows in camera_rows.items():
        np.savez(
            out_dir / name / "index.npz",
            stamp=np.asarray([r[0] for r in rows]),
            receive=np.asarray([r[1] for r in rows]),
            filename=np.asarray([r[2] for r in rows]),
        )
    (out_dir / "camera_info.json").write_text(json.dumps(camera_info, indent=2), encoding="utf-8")
    (out_dir / "tf_static.json").write_text(json.dumps(static_tf, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bag", type=Path, help="rosbag2 directory (contains metadata.yaml), e.g. inputs/<recording>")
    parser.add_argument("--out", type=Path, default=None, help="default: <bag>/export")
    parser.add_argument("--image-scale", type=float, default=0.5)
    args = parser.parse_args()
    summary = export_bag(args.bag, args.out or args.bag / "export", image_scale=args.image_scale)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
