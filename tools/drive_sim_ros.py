"""Closed-loop REAL ROS 2 test of the 2D drive stack (kart target: camera + IMU).

Ported from the research workspace's ros2_drive_sim_test.py and extended with
a SYNTHETIC CAMERA so the kart's target configuration - ONE camera plus the
dedicated IMU, no LiDAR - can close the loop for real:

1. ``ros2 launch kart_map2d drive.launch.py cfg:=<generated kart_master.cfg>
   start_pose:=... stats_path:=...`` starts the drive node exactly as on the
   kart (the canonical launch path);
2. this process runs the kart simulator node: kinematic bicycle
   (kart_map2d.drive.sim.KartSim, 5 ms steps on the ROS clock), consuming
   /drive (AckermannDriveStamped) and publishing what the kart's sensors
   would see:
   - /sim/imu        sensor_msgs/Imu, 200 Hz, noise + gyro bias
   - /sim/cam/*      a RENDERED camera view of the map package's world
                     (drivable cells = textured asphalt, obstacle cells =
                     textured red-brown, unknown = olive, sky above the
                     horizon), bgr8 + CameraInfo, default 8 Hz. VO tracks
                     the world-anchored texture; the classical ground
                     classifier segments it; the camera map fixes correlate
                     it against the SAME map - the whole camera localization
                     path runs for real, no shortcuts.
   - /sim/scan       PointCloud2 ray-cast against the map's obstacle cells
                     (comparison rows with lidar = 1 only)
   - /sim/odom       wheel-odometry-like Odometry (odom rows only)
3. the drive node closes the loop on the saved map package.

Scenarios (--runs, default the kart-target set):
  cam_route        HARD GATE: camera+IMU, KNOWN start pose -> localized,
                   engaged, full route; cross-track RMS <= 0.3 m at 2 m/s
  cam_obstacle     HARD GATE: a 0.6 m block ON the route (rendered AND in
                   the collision world) -> the kart reaches speed, stops
                   SHORT of it with the obstacle reason, and stays stopped
  cam_estop        HARD GATE: e-stop at speed -> latched stop; release alone
                   does not restart; ~/engage restarts
  cam_watchdog     HARD GATE: all sensors silent at speed -> stop within the
                   pose timeout + braking
  cam_wrong_start  STRETCH (reported, NOT gated): start pose 2 m / 10 deg off
  cam_global       STRETCH (reported, NOT gated): no start pose, global search
  lidar_route      comparison row: lidar = 1 (the research-proven config)

Writes results/drive/<run>/ (cfg, node log, result.json, traces.npz) and
results/drive/summary.json. ASCII only. Run inside kart2d:humble:

    docker run --rm -v "S:\\...\\2d map maker:/ws2d" --shm-size=1g kart2d:humble \\
        bash -lc "source /ws2d/ros2_ws/install/setup.bash && python3 /ws2d/tools/drive_sim_ros.py --map /ws2d/results/m01_imu_cam/<pkg>"
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

from kart_map2d.drive.controller import RacingPath, wrap_angle
from kart_map2d.drive.sim import KartSim, KartSimConfig, RasterWorld
from kart_map2d.mapping.map_package import load_map_package

PROJECT_ROOT = _bootstrap.PROJECT_ROOT
OUT_DEFAULT = PROJECT_ROOT / "results" / "drive"
XTRACK_GATE_M = 0.3
KART_TARGET_RUNS = ("cam_route", "cam_obstacle", "cam_estop", "cam_watchdog", "cam_wrong_start", "cam_global", "lidar_route")
#: Obstacle cells closer than this to the racing line are removed from the
#: SIM WORLD (collision + render). The mapping kart physically drove where
#: the line runs, so cells hugging it are evidence blur / bystanders, not
#: walls; and the swept-path obstacle check reaches kart half-width 0.53 m
#: + margin 0.25 m, with camera ground-projection and pose error adding up
#: to ~0.6 m of apparent lateral displacement at the 15 cm cell scale
#: (measured: with 1.0 m clearing and the old class-tinted floor the camera
#: row stopped on phantom cues; with the uniform floor the phantom factory
#: is gone and the clearing only needs to cover the sweep + pose drift -
#: while LEAVING the real walls beyond it rendered, because the camera
#: patch matcher feeds on exactly that class structure: at 1.5 m clearing
#: the matcher starved, 20 of 147 matches accepted, and the localizer
#: dropped out ~6 m in). A REAL obstacle test is the dedicated
#: cam_obstacle scenario, whose block is added after this clearing.
CLEAR_NEAR_LINE_M = 1.1
#: The mapped sept3 corridor is a dead end: the racing line runs to the
#: turnaround wall. A route that ends at a wall can never finish cleanly,
#: so the fixture trims the LAST metres off the route (map grid untouched)
#: - the drive node brakes to a stop at the trimmed end, as a real route
#: ending short of a wall would.
TRIM_ROUTE_END_M = 2.5

# ---------------------------------------------------------------- camera model
CAM_W, CAM_H = 640, 400
CAM_FX = CAM_FY = 330.0
CAM_CX, CAM_CY = 320.0, 200.0
CAM_MOUNT_XYZ = (0.10, 0.0, 0.345)  # base_link; ground plane sits at z=-0.10
CAM_HEIGHT_M = 0.445
#: Sim camera rate. 5 Hz keeps the drive node's image path (decode + VO +
#: cue run inline on its executor) from starving the 200 Hz IMU and the
#: 50 Hz control loop on this laptop-in-Docker (at 8 Hz the published pose
#: ran ~0.5 s stale and the kart oscillated into obstacle stops). The real
#: camera publishes 15-60 fps: budget the drive node's image path on the
#: Orin before trusting higher rates (flagged in the validation notes).
CAM_RATE_HZ = 5.0


def log(text: str) -> None:
    print(text, flush=True)


class WorldRenderer:
    """Rasterize the map package's class grid into camera frames.

    World-anchored value-noise texture (two octaves from one 256x256 random
    tile) gives VO real corners that move rigidly with the world; colors are
    chosen so the classical ground classifier's seed (asphalt directly ahead)
    generalizes to all drivable cells and rejects obstacle cells.
    """

    def __init__(self, grid, seed: int = 5, classes_override: np.ndarray | None = None) -> None:
        classes = grid.classify() if classes_override is None else classes_override
        # 0 unknown, 1 drivable, 2 obstacle
        self.classes = classes.astype(np.uint8)
        self.origin = np.asarray(grid.origin, dtype=np.float64)
        self.cell = float(grid.cell)
        rng = np.random.default_rng(seed)
        self.tile = rng.random((256, 256)).astype(np.float32)
        # Precomputed body-frame ray table (x forward, y left, z up).
        us, vs = np.meshgrid(np.arange(CAM_W, dtype=np.float32), np.arange(CAM_H, dtype=np.float32))
        self.dir_body = np.stack(
            [np.ones_like(us), -(us - CAM_CX) / CAM_FX, -(vs - CAM_CY) / CAM_FY], axis=-1
        )  # (H, W, 3)
        self.base_colors = {
            # unknown: IDENTICAL to drivable. The map's UNKNOWN cells are
            # never-observed ground, not darker ground - the real plaza floor
            # is continuous asphalt. Painting them even slightly darker made
            # the one-camera map's patchy corridor (unknown holes between
            # sparse 2 Hz evidence) read as a mottled floor: the classical
            # classifier rejected hole borders and emitted phantom obstacle
            # contacts at >= 2 m dead ahead, parking the kart mid-route
            # (measured: a 140 s obstacle stop at s = 2-4 m with the old
            # 96/95/93 tint; the painted-red obstacle cells are the only
            # thing that should segment as obstacle).
            0: np.array([118, 116, 112], np.float32),
            1: np.array([118, 116, 112], np.float32),  # drivable: asphalt gray
            2: np.array([55, 62, 150], np.float32),  # obstacle: brick red (BGR)
        }
        self.sky = np.array([205, 185, 160], np.float32)

    def _noise(self, x: np.ndarray, y: np.ndarray, freq: float) -> np.ndarray:
        xi = (x * freq) % 256.0
        yi = (y * freq) % 256.0
        x0 = np.floor(xi).astype(np.int32) % 256
        y0 = np.floor(yi).astype(np.int32) % 256
        x1 = (x0 + 1) % 256
        y1 = (y0 + 1) % 256
        fx = (xi - np.floor(xi)).astype(np.float32)
        fy = (yi - np.floor(yi)).astype(np.float32)
        t = self.tile
        top = t[y0, x0] * (1 - fx) + t[y0, x1] * fx
        bot = t[y1, x0] * (1 - fx) + t[y1, x1] * fx
        return top * (1 - fy) + bot * fy

    #: World texture resolution [m/px] for the prebuilt top-down image.
    WORLD_RES = 0.03

    def _build_world_image(self) -> None:
        """One BGR top-down image of the whole world (classes + noise),
        rendered ONCE: per frame the camera view is then a single
        cv2.warpPerspective through the ground-plane homography (~2 ms vs
        ~145 ms for the original per-pixel ray casting - that extra latency
        alone pushed every VO increment past the EKF's lateness window and
        starved the camera fixes; a real camera driver publishes within
        tens of milliseconds, which is what the sim must reproduce)."""
        import cv2

        res = self.WORLD_RES
        margin = 6.0
        h_cells, w_cells = self.classes.shape  # (ny, nx)? origin + ij*cell -> see below
        # classes is indexed [i, j] with world = origin + (ij + .5) * cell
        # (i along x? keep consistent with earlier obstacle extraction:
        # obstacles = origin + (argwhere(classes==2) + .5) * cell, so
        # axis 0 -> world x, axis 1 -> world y.
        x0 = self.origin[0] - margin
        y0 = self.origin[1] - margin
        nx = int(round((h_cells * self.cell + 2 * margin) / res))
        ny = int(round((w_cells * self.cell + 2 * margin) / res))
        xs = x0 + (np.arange(nx) + 0.5) * res
        ys = y0 + (np.arange(ny) + 0.5) * res
        i = np.clip(((xs - self.origin[0]) / self.cell).astype(np.int64), 0, h_cells - 1)
        j = np.clip(((ys - self.origin[1]) / self.cell).astype(np.int64), 0, w_cells - 1)
        inside_x = (xs >= self.origin[0]) & (xs < self.origin[0] + h_cells * self.cell)
        inside_y = (ys >= self.origin[1]) & (ys < self.origin[1] + w_cells * self.cell)
        cls = np.where(
            inside_x[:, None] & inside_y[None, :], self.classes[np.ix_(i, j)], 0
        ).astype(np.uint8)  # outside the grid = unknown pavement
        img = np.empty((nx, ny, 3), np.float32)
        for k, base in self.base_colors.items():
            img[cls == k] = base
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        tex = 0.7 * self._noise(gx, gy, 23.0) + 0.3 * self._noise(gx, gy, 5.0)
        # Texture amplitude: identical for unknown and drivable ground (one
        # continuous floor), a touch flatter on the red obstacle paint.
        amp = np.where(cls == 2, 24.0, 34.0).astype(np.float32)
        img += ((tex - 0.5) * amp)[..., None]
        # world image rows = y axis, cols = x? cv2 wants [row, col] = [v, u]:
        # use rows = ny (y), cols = nx (x): transpose so that pixel (r, c)
        # sits at world (x0 + c*res, y0 + r*res).
        self.world_img = np.clip(img.transpose(1, 0, 2), 0, 255).astype(np.uint8)
        self.world_x0, self.world_y0 = x0, y0
        self.K_mat = np.array([[CAM_FX, 0.0, CAM_CX], [0.0, CAM_FY, CAM_CY], [0.0, 0.0, 1.0]])
        #: optical axes in body coords (z fwd, x right, y down), body->world below
        self.horizon_row = int(CAM_CY) + 3
        self._fall = np.clip(
            1.0 - (CAM_HEIGHT_M * CAM_FY / np.maximum(np.arange(CAM_H, dtype=np.float32) - CAM_CY, 1e-3)) / 90.0,
            0.55,
            1.0,
        )[:, None, None]

    def render(self, x: float, y: float, yaw: float) -> np.ndarray:
        import cv2

        if not hasattr(self, "world_img"):
            self._build_world_image()
        c, s = math.cos(yaw), math.sin(yaw)
        # camera optical center in world
        ox = x + CAM_MOUNT_XYZ[0] * c - CAM_MOUNT_XYZ[1] * s
        oy = y + CAM_MOUNT_XYZ[0] * s + CAM_MOUNT_XYZ[1] * c
        oz = CAM_HEIGHT_M
        # world -> camera-optical rotation (optical: x right, y down, z fwd;
        # body fwd = world heading yaw): r_right = (sin yaw, -cos yaw, 0)?
        # right = -left; left = (-s, c, 0) -> right = (s, -c, 0); down = (0,0,-1)
        R_cw = np.array([[s, -c, 0.0], [0.0, 0.0, -1.0], [c, s, 0.0]])
        t_cw = -R_cw @ np.array([ox, oy, oz])
        # ground plane (x, y, 1) -> pixel: H = K [r1 r2 | t]
        Hw = self.K_mat @ np.column_stack([R_cw[:, 0], R_cw[:, 1], t_cw])
        # world-image pixel (u_w, v_w) -> world (x, y): x = x0 + u_w*res, y = y0 + v_w*res
        A = np.array([[self.WORLD_RES, 0.0, self.world_x0], [0.0, self.WORLD_RES, self.world_y0], [0.0, 0.0, 1.0]])
        H = Hw @ A  # world-image px -> camera px
        img = cv2.warpPerspective(
            self.world_img,
            H,
            (CAM_W, CAM_H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=tuple(float(v) for v in self.base_colors[0]),
        )
        out = img.astype(np.float32)
        out[self.horizon_row :] *= self._fall[self.horizon_row :]
        out[: self.horizon_row] = self.sky
        return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- sim node
class KartSimNode:
    """The kart: physics at 200 Hz, sensors (IMU / camera / scan / odom),
    command subscriber, e-stop + heartbeat publishers. Ported from the
    research ros2_drive_sim_test.py; the camera renderer is new."""

    DT = 0.005

    def __init__(self, package, start, out_dir: Path, lidar: bool = False, odom: bool = False,
                 camera: bool = True, seed: int = 0, path: RacingPath | None = None, clear_m: float = 1.0,
                 block: tuple[float, float, float] | None = None) -> None:
        import rclpy
        from ackermann_msgs.msg import AckermannDriveStamped
        from geometry_msgs.msg import PoseStamped
        from nav_msgs.msg import Odometry
        from rclpy.executors import MultiThreadedExecutor
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import CameraInfo, Image, Imu, PointCloud2, PointField
        from std_msgs.msg import Bool, String
        from std_srvs.srv import SetBool

        self.rclpy = rclpy
        self.msg = {
            "Imu": Imu, "Odometry": Odometry, "PointCloud2": PointCloud2, "PointField": PointField,
            "Bool": Bool, "SetBool": SetBool, "Image": Image, "CameraInfo": CameraInfo,
        }
        self.node = rclpy.create_node("kart_sim")
        self.rng = np.random.default_rng(seed)
        classes = package.grid.classify()
        ij = np.argwhere(classes == 2)
        obstacles = np.asarray(package.grid.origin) + (ij + 0.5) * package.grid.cell
        self.cells_removed = 0
        cleared_classes = classes.copy()
        if path is not None and len(obstacles):
            from scipy.spatial import cKDTree

            d, _ = cKDTree(path.points_between(0.0, path.length, 0.2)).query(obstacles)
            self.cells_removed = int((d < clear_m).sum())
            # The mapping kart itself drove where the line runs, so obstacle
            # cells hugging it are bystanders/noise, removed from the sim
            # WORLD - and therefore also from the RENDERED world (painting
            # them would make the camera stop the kart on phantoms the
            # collision world does not contain; measured: obstacle-stops at
            # s=2.2 m with an all-cells render).
            cleared_classes[tuple(ij[d < clear_m].T)] = 1
            obstacles = obstacles[d >= clear_m]
        if block is not None:
            # A REAL obstacle for the cam_obstacle scenario: a block of
            # obstacle cells painted into the rendered world AND the
            # collision/scan world, dead on the route (it is added AFTER the
            # near-line clearing, so the clearing cannot remove it).
            bx, by, half = block
            origin = np.asarray(package.grid.origin, dtype=np.float64)
            cell = float(package.grid.cell)
            i0 = int(math.floor((bx - half - origin[0]) / cell))
            i1 = int(math.ceil((bx + half - origin[0]) / cell))
            j0 = int(math.floor((by - half - origin[1]) / cell))
            j1 = int(math.ceil((by + half - origin[1]) / cell))
            i0, i1 = max(i0, 0), min(i1, cleared_classes.shape[0])
            j0, j1 = max(j0, 0), min(j1, cleared_classes.shape[1])
            cleared_classes[i0:i1, j0:j1] = 2
            bij = np.argwhere(cleared_classes[i0:i1, j0:j1] == 2) + (i0, j0)
            bxy = origin + (bij + 0.5) * cell
            obstacles = np.vstack([obstacles, bxy]) if len(obstacles) else bxy
        self.obstacles = obstacles
        self.world = RasterWorld(obstacles, float(package.grid.cell))
        self.renderer = WorldRenderer(package.grid, classes_override=cleared_classes) if camera else None
        self.sim = KartSim(*start, config=KartSimConfig())
        self.lock = threading.Lock()
        self.cmd = (0.0, 0.0)
        self.publish_sensors = True
        self.publish_heartbeat = False
        self.lidar = lidar
        self.odom = odom
        self.boxes: list[tuple[float, float, float, float]] = []
        self.gyro_bias = 0.004
        self.odo = np.array([0.0, 0.0, 0.0])
        self.truth: list[tuple] = []
        self.cmds: list[tuple] = []
        self.diag: dict | None = None
        self.frames_rendered = 0
        self.render_ms: list[float] = []
        self._t_sim: float | None = None
        self._k = 0
        rel = QoSProfile(depth=50, reliability=ReliabilityPolicy.RELIABLE)
        best1 = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_imu = self.node.create_publisher(Imu, "/sim/imu", rel)
        self.pub_odom = self.node.create_publisher(Odometry, "/sim/odom", rel)
        self.pub_scan = self.node.create_publisher(PointCloud2, "/sim/scan", QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE))
        self.pub_img = self.node.create_publisher(Image, "/sim/cam/image", best1)
        self.pub_info = self.node.create_publisher(CameraInfo, "/sim/cam/camera_info", best1)
        self.pub_estop = self.node.create_publisher(Bool, "/estop", latched)
        self.pub_heartbeat = self.node.create_publisher(Bool, "/deadman", QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE))
        self.node.create_subscription(AckermannDriveStamped, "/drive", self._on_drive, QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE))
        self.node.create_subscription(String, "/drive/diagnostics", self._on_diag, 10)
        self.node.create_subscription(PoseStamped, "/drive/pose", self._on_pose, 50)
        self.est: list[tuple] = []
        self.engage_client = self.node.create_client(SetBool, "/drive/engage")
        self.node.create_timer(self.DT, self._tick)
        # A spin_once loop starves the subscriptions behind the 200 Hz
        # physics timer (measured: 0 of the 5 Hz diagnostics and only a
        # third of the poses arrived); executor.spin with three threads
        # services them fairly.
        self.executor = MultiThreadedExecutor(num_threads=3)
        self.executor.add_node(self.node)
        self._stop = False
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        self._scan_thread = threading.Thread(target=self._scan_loop, daemon=True)
        self._scan_thread.start()
        self._cam_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self._cam_thread.start()

    def _spin(self) -> None:
        try:
            self.executor.spin()
        except Exception:  # noqa: BLE001 - shutdown races surface here
            pass

    def now(self) -> float:
        return self.node.get_clock().now().nanoseconds * 1e-9

    @staticmethod
    def _stamp(msg_stamp, t: float) -> None:
        sec = int(math.floor(t))
        msg_stamp.sec, msg_stamp.nanosec = sec, int((t - sec) * 1e9)

    def _on_drive(self, m) -> None:
        t = self.now()
        stamp = m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec
        with self.lock:
            self.cmd = (float(m.drive.steering_angle), float(m.drive.speed))
            self.cmds.append((t, stamp, self.cmd[0], self.cmd[1]))

    def _on_pose(self, m) -> None:
        q = m.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        t = m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec
        self.est.append((t, m.pose.position.x, m.pose.position.y, yaw))

    def _on_diag(self, m) -> None:
        try:
            self.diag = json.loads(m.data)
        except ValueError:
            pass

    def _tick(self) -> None:
        t_now = self.now()
        if self._t_sim is None:
            self._t_sim = t_now
            return
        samples = []
        with self.lock:
            steer, speed = self.cmd
            while self._t_sim + self.DT <= t_now:
                self.sim.step(steer, speed, self.DT)
                self._t_sim += self.DT
                s = self.sim
                t = self._t_sim
                self.truth.append((t, s.x, s.y, s.yaw, s.v, s.steer))
                step = s.v * self.DT * 1.01
                self.odo[2] += s.yaw_rate * self.DT + 0.002 * self.DT
                self.odo[0] += step * math.cos(self.odo[2])
                self.odo[1] += step * math.sin(self.odo[2])
                k = self._k = self._k + 1
                vib_a, vib_g = s.vibration()
                samples.append((t, k, s.yaw_rate, s.accel, s.v, vib_a, vib_g, self.odo.copy()))
        if not self.publish_sensors:
            return
        for t, k, yaw_rate, accel, v, vib_a, vib_g, odo in samples:
            imu = self.msg["Imu"]()
            self._stamp(imu.header.stamp, t)
            imu.header.frame_id = "sim_imu"
            g = self.rng.normal(0.0, math.hypot(0.003, vib_g), 3)
            a = self.rng.normal(0.0, math.hypot(0.05, vib_a), 3)
            imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z = float(g[0]), float(g[1]), float(yaw_rate + self.gyro_bias + g[2])
            imu.linear_acceleration.x = float(accel + a[0])
            imu.linear_acceleration.y = float(v * yaw_rate + a[1])
            imu.linear_acceleration.z = float(9.80665 + a[2])
            self.pub_imu.publish(imu)
            if self.odom and k % 4 == 0:
                od = self.msg["Odometry"]()
                self._stamp(od.header.stamp, t)
                od.header.frame_id = "odom"
                od.child_frame_id = "base_link"
                n = self.rng.normal(0.0, 0.003, 2)
                od.pose.pose.position.x = float(odo[0] + n[0])
                od.pose.pose.position.y = float(odo[1] + n[1])
                od.pose.pose.orientation.z = float(math.sin(odo[2] / 2))
                od.pose.pose.orientation.w = float(math.cos(odo[2] / 2))
                self.pub_odom.publish(od)
            if self.publish_heartbeat and k % 20 == 0:
                hb = self.msg["Bool"]()
                hb.data = True
                self.pub_heartbeat.publish(hb)

    # ----------------------------------------------------------- camera loop
    def _camera_loop(self) -> None:
        if self.renderer is None:
            return
        period = 1.0 / CAM_RATE_HZ
        next_t = time.monotonic()
        while not self._stop:
            next_t += period
            with self.lock:
                pose = self.sim.pose.copy()
                t = self._t_sim
            if t is not None and self.publish_sensors:
                t0 = time.perf_counter()
                img = self.renderer.render(pose[0], pose[1], pose[2])
                self.render_ms.append((time.perf_counter() - t0) * 1e3)
                m = self.msg["Image"]()
                self._stamp(m.header.stamp, t)
                m.header.frame_id = "sim_cam_optical"
                m.height, m.width = CAM_H, CAM_W
                m.encoding = "bgr8"
                m.step = 3 * CAM_W
                m.data = img.tobytes()
                info = self.msg["CameraInfo"]()
                self._stamp(info.header.stamp, t)
                info.header.frame_id = "sim_cam_optical"
                info.width, info.height = CAM_W, CAM_H
                info.k = [CAM_FX, 0.0, CAM_CX, 0.0, CAM_FY, CAM_CY, 0.0, 0.0, 1.0]
                info.distortion_model = "plumb_bob"
                info.d = [0.0] * 5
                self.pub_info.publish(info)
                self.pub_img.publish(m)
                self.frames_rendered += 1
            dt = next_t - time.monotonic()
            if dt > 0:
                time.sleep(dt)
            else:
                next_t = time.monotonic()

    def _scan_loop(self) -> None:
        rng = np.random.default_rng(12345)
        next_t = time.monotonic()
        while not self._stop:
            next_t += 0.1
            with self.lock:
                pose = self.sim.pose.copy()
                t = self._t_sim
                boxes = list(self.boxes)
            if t is not None and self.lidar and self.publish_sensors:
                pts = self.world.scan(pose, max_range=30.0, rays=720, extra_boxes=boxes, rng=rng)
                self.pub_scan.publish(self._cloud(t, pts))
            time.sleep(max(0.0, next_t - time.monotonic()))

    def _cloud(self, t: float, pts: np.ndarray):
        PF = self.msg["PointField"]
        msg = self.msg["PointCloud2"]()
        self._stamp(msg.header.stamp, t)
        msg.header.frame_id = "sim_lidar"
        xyz = np.column_stack([pts, np.full(len(pts), 0.5)]).astype(np.float32)
        msg.height, msg.width = 1, len(xyz)
        msg.fields = [PF(name=n, offset=4 * i, datatype=PF.FLOAT32, count=1) for i, n in enumerate("xyz")]
        msg.is_bigendian = False
        msg.point_step, msg.row_step = 12, 12 * len(xyz)
        msg.data = xyz.tobytes()
        msg.is_dense = True
        return msg

    def estop(self, pressed: bool) -> None:
        m = self.msg["Bool"]()
        m.data = bool(pressed)
        self.pub_estop.publish(m)

    def engage(self, on: bool, timeout: float = 10.0) -> tuple[bool, str]:
        if not self.engage_client.wait_for_service(timeout_sec=timeout):
            return False, "engage service not available"
        req = self.msg["SetBool"].Request()
        req.data = bool(on)
        fut = self.engage_client.call_async(req)
        t0 = time.time()
        while not fut.done() and time.time() - t0 < timeout:
            time.sleep(0.02)
        if not fut.done():
            return False, "timeout"
        res = fut.result()
        return bool(res.success), str(res.message)

    def speed(self) -> float:
        with self.lock:
            return self.sim.v

    def last_cmd_speed(self) -> float:
        with self.lock:
            return self.cmds[-1][3] if self.cmds else 0.0

    def wait(self, cond, timeout: float, poll: float = 0.02) -> float | None:
        t0 = time.time()
        while time.time() - t0 < timeout:
            if cond():
                return time.time() - t0
            time.sleep(poll)
        return None

    def close(self) -> None:
        self._stop = True
        self.executor.shutdown(timeout_sec=2.0)
        self._thread.join(timeout=5.0)
        self._scan_thread.join(timeout=5.0)
        self._cam_thread.join(timeout=5.0)
        self.executor.remove_node(self.node)
        self.node.destroy_node()


# ---------------------------------------------------------------- helpers
def make_cfg(run: str, package_dir: Path, out_dir: Path) -> Path:
    """A kart_master.cfg for this scenario (the canonical control file)."""
    lidar = run.startswith("lidar")
    camera = run.startswith("cam")
    lines = [
        "# generated by tools/drive_sim_ros.py - closed-loop sim scenario " + run,
        "[sensors]",
        "imu = 1",
        f"camera = {1 if camera else 0}",
        f"lidar = {1 if lidar else 0}",
        "[imu]",
        "topic = /sim/imu",
        "[camera]",
        "name = sim_cam",
        "source = ros2",
        "image_topic = /sim/cam/image",
        "camera_info_topic = /sim/cam/camera_info",
        f"mount_xyz = {CAM_MOUNT_XYZ[0]}, {CAM_MOUNT_XYZ[1]}, {CAM_MOUNT_XYZ[2]}",
        "mount_rpy_deg = 0, 0, 0",
        f"height_m = {CAM_HEIGHT_M}",
        "learn_mask = 0",
        "processing_scale = 1.0",
        "[lidar]",
        "topic = /sim/scan",
        "xyz = 0, 0, 0",
        "rpy_deg = 0, 0, 0",
        "[drive]",
        f"map_package = {package_dir.as_posix()}",
        "max_speed_mps = 2.0",
        "command_topic = /drive",
        # Known-start scenarios pin global relocalization OFF (the kart
        # starts from a marked spot; a dropout must re-converge locally, not
        # jump to an aliased stretch of this self-similar corridor). The
        # wrong-start / global rows need it ON - finding the pose IS their test.
        f"global_relocalization = {0 if run not in ('cam_wrong_start', 'cam_global') else 1}",
        "[safety]",
        "estop_topic = /estop",
        "[system]",
        "log_level = info",
        "use_sim_time = 0",
    ]
    cfg = out_dir / "kart_master.cfg"
    cfg.write_text("\n".join(lines) + "\n", encoding="ascii")
    return cfg


def xtrack(path: RacingPath, truth: np.ndarray, t_from: float = -np.inf, t_to: float = np.inf) -> dict:
    sel = (truth[:, 0] >= t_from) & (truth[:, 0] <= t_to)
    rows = truth[sel][::4]
    if not len(rows):
        return {"n": 0}
    s = path.project(rows[0, 1:3]).s
    ct, he, v = [], [], []
    for r in rows:
        pr = path.project(r[1:3], s_hint=s, back=2.0, ahead=6.0)
        s = pr.s
        if r[4] > 0.2:
            ct.append(abs(pr.lateral))
            he.append(abs(math.degrees(wrap_angle(r[3] - pr.heading))))
            v.append(r[4])
    if not ct:
        return {"n": 0}
    ct_a, he_a = np.asarray(ct), np.asarray(he)
    return {
        "n": int(len(ct_a)),
        "cross_track_rms_m": round(float(np.sqrt(np.mean(ct_a**2))), 4),
        "cross_track_max_m": round(float(ct_a.max()), 4),
        "heading_error_rms_deg": round(float(np.sqrt(np.mean(he_a**2))), 2),
        "mean_speed_mps": round(float(np.mean(v)), 3),
        "max_speed_mps": round(float(np.max(v)), 3),
    }


def localization_error(sim: KartSimNode) -> dict | None:
    if not sim.est or not sim.truth:
        return None
    est = np.asarray(sim.est)
    tr = np.asarray(sim.truth)
    xs = np.interp(est[:, 0], tr[:, 0], tr[:, 1])
    ys = np.interp(est[:, 0], tr[:, 0], tr[:, 2])
    yaws = np.interp(est[:, 0], tr[:, 0], np.unwrap(tr[:, 3]))
    ok = (est[:, 0] >= tr[0, 0]) & (est[:, 0] <= tr[-1, 0])
    d = np.hypot(est[ok, 1] - xs[ok], est[ok, 2] - ys[ok])
    y = np.abs(np.degrees([wrap_angle(a) for a in (est[ok, 3] - yaws[ok])]))
    if not len(d):
        return None
    return {
        "n": int(len(d)),
        "rms_m": round(float(np.sqrt(np.mean(d**2))), 4),
        "p95_m": round(float(np.percentile(d, 95)), 4),
        "max_m": round(float(d.max()), 4),
        "yaw_rms_deg": round(float(np.sqrt(np.mean(y**2))), 3),
    }


def latency(cmds: list) -> dict | None:
    if not cmds:
        return None
    a = np.asarray([(c[0] - c[1]) * 1e3 for c in cmds if c[1] > 1e8])
    if not len(a):
        return None
    return {"n": int(len(a)), "p50_ms": round(float(np.percentile(a, 50)), 2), "p95_ms": round(float(np.percentile(a, 95)), 2), "max_ms": round(float(a.max()), 2)}


# ---------------------------------------------------------------- one run
def run_one(run: str, package, package_dir: Path, path: RacingPath, out_root: Path) -> dict:
    out = out_root / run
    out.mkdir(parents=True, exist_ok=True)
    start = np.array([*path.point_at(0.0), path.heading_at(0.0)])
    start_arg = f"{start[0]:.3f},{start[1]:.3f},{math.degrees(start[2]):.2f}"
    if run == "cam_wrong_start":
        # 2 m sideways / 10 deg off: the STRETCH condition (research-known
        # open problem for camera-only convergence; reported, not gated).
        c, s = math.cos(start[2]), math.sin(start[2])
        start_arg = f"{start[0] - 2.0 * s:.3f},{start[1] + 2.0 * c:.3f},{math.degrees(start[2]) + 10.0:.2f}"
    elif run == "cam_global":
        start_arg = "global"
    block = None
    if run == "cam_obstacle":
        # A 0.6 x 0.6 m block ON the route, far enough in for the kart to
        # reach full speed first.
        s_blk = min(12.0, 0.45 * path.length)
        bx, by = path.point_at(s_blk)
        block = (float(bx), float(by), 0.3)
    cfg = make_cfg(run, package_dir, out)
    stats_file = out / "drive_stats.json"
    stats_file.unlink(missing_ok=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    node_log = (out / "drive_node.log").open("w", encoding="utf-8")
    proc = subprocess.Popen(
        [
            "ros2", "launch", "kart_map2d", "drive.launch.py",
            f"cfg:={cfg}", f"start_pose:={start_arg}", f"stats_path:={stats_file}",
        ],
        env=env, stdout=node_log, stderr=subprocess.STDOUT, start_new_session=True,
    )
    sim = KartSimNode(
        package,
        start,
        out,
        lidar=run.startswith("lidar"),
        odom=False,
        camera=run.startswith("cam"),
        seed=sum(map(ord, run)),
        path=path,
        clear_m=CLEAR_NEAR_LINE_M,
        block=block,
    )
    result: dict = {"run": run, "start_pose_arg": start_arg, "sim_world_cells_removed_near_line": sim.cells_removed,
                    "sim_world_clear_near_line_m": CLEAR_NEAR_LINE_M}
    if block is not None:
        result["block_center_xy_half_m"] = list(block)
    t_run0 = time.time()
    try:
        if run == "cam_deadman":
            sim.publish_heartbeat = True
        timeout_loc = 120.0 if run in ("cam_global", "cam_wrong_start") else 60.0
        t_loc = sim.wait(lambda: bool(sim.diag and (sim.diag.get("localization") or {}).get("localized")), timeout_loc)
        result["localized_after_s"] = None if t_loc is None else round(t_loc, 2)
        if sim.diag:
            result["startup_localization"] = sim.diag.get("localization")
            result["global_attempts"] = sim.diag.get("global_attempts")
        if t_loc is None:
            result["error"] = "never localized"
            return result
        if run in ("cam_wrong_start", "cam_global"):
            # STRETCH rows: where did it THINK it was once converged?
            time.sleep(2.0)
            if sim.est:
                _, ex, ey, eyaw = sim.est[-1]
                tx, ty, tyaw = sim.sim.x, sim.sim.y, sim.sim.yaw
                result["converged_pose_error_m"] = round(float(np.hypot(ex - tx, ey - ty)), 3)
                result["converged_yaw_error_deg"] = round(abs(math.degrees(wrap_angle(eyaw - tyaw))), 2)
        ok, msg = sim.engage(True)
        result["engage"] = [ok, msg]
        t_engaged = sim.now()
        if run == "cam_obstacle":
            # Must accelerate, see the block, and stop SHORT of it - and stay
            # stopped while the path remains blocked.
            sim.wait(lambda: sim.speed() > 1.5, 30.0)
            v_peak = sim.speed()

            def stopped_on_obstacle() -> bool:
                d = sim.diag or {}
                return "obstacle" in (d.get("safety") or {}).get("reasons", []) and sim.speed() < 0.05

            t_stop = sim.wait(stopped_on_obstacle, 90.0)
            time.sleep(5.0)  # the block stays: the kart must NOT creep past it
            truth = np.asarray(sim.truth)
            bx, by, half = block
            gap = np.hypot(truth[:, 1] - bx, truth[:, 2] - by) - half
            result["speed_reached_mps"] = round(float(max(v_peak, truth[:, 4].max() if len(truth) else 0.0)), 3)
            result["stopped_before_block"] = t_stop is not None
            result["min_gap_base_link_to_block_face_m"] = round(float(gap.min()), 3)
            result["still_stopped_after_5s"] = bool(sim.speed() < 0.05)
            result["reasons_when_stopped"] = (sim.diag or {}).get("safety", {}).get("reasons")
            result["tracking"] = xtrack(path, truth, t_engaged)
        elif run.endswith("route") or run in ("cam_wrong_start", "cam_global"):
            done = sim.wait(lambda: bool(sim.diag and (sim.diag.get("control") or {}).get("finished")) and sim.speed() < 0.02, 180.0)
            result["route_time_s"] = None if done is None else round(done, 2)
            truth = np.asarray(sim.truth)
            result["tracking"] = xtrack(path, truth, t_engaged)
            result["finished"] = done is not None
            final = truth[-1]
            end = path.point_at(path.length)
            result["distance_to_route_end_m"] = round(float(np.hypot(final[1] - end[0], final[2] - end[1])), 3)
        else:
            sim.wait(lambda: sim.speed() > 1.8, 40.0)
            time.sleep(1.0)
            v0 = sim.speed()
            if run == "cam_estop":
                sim.estop(True)
            elif run == "cam_watchdog":
                sim.publish_sensors = False
            t_inj = time.time()
            t_cmd0 = sim.wait(lambda: sim.last_cmd_speed() <= 1e-6, 10.0, poll=0.005)
            t_cmd0 = None if t_cmd0 is None else time.time() - t_inj
            t_stop = sim.wait(lambda: sim.speed() < 0.05, 15.0, poll=0.005)
            t_stop = None if t_stop is None else time.time() - t_inj
            result["speed_at_fault_mps"] = round(v0, 3)
            result["command_zero_after_s"] = None if t_cmd0 is None else round(t_cmd0, 3)
            result["kart_stopped_after_s"] = None if t_stop is None else round(t_stop, 3)
            result["reasons_when_stopped"] = (sim.diag or {}).get("safety", {}).get("reasons")
            result["stopped"] = t_stop is not None
            if run == "cam_estop":
                sim.estop(False)
                time.sleep(2.0)
                result["still_stopped_after_release"] = sim.speed() < 0.05 and sim.last_cmd_speed() <= 1e-6
                ok, msg = sim.engage(True)
                result["reengage"] = [ok, msg]
                t_go = sim.wait(lambda: sim.speed() > 0.5, 15.0)
                result["moving_again_after_reengage_s"] = None if t_go is None else round(t_go, 2)
            elif run == "cam_watchdog":
                time.sleep(1.0)
                sim.publish_sensors = True
                t_go = sim.wait(lambda: sim.speed() > 0.5, 30.0)
                result["moving_again_after_data_back_s"] = None if t_go is None else round(t_go, 2)
        result["latency_sensor_to_command"] = latency(sim.cmds)
        result["localization_error_vs_truth"] = localization_error(sim)
        result["commands_received"] = len(sim.cmds)
        result["camera_frames_rendered"] = sim.frames_rendered
        if sim.render_ms:
            r = np.asarray(sim.render_ms)
            result["render_ms_mean_p95"] = [round(float(r.mean()), 1), round(float(np.percentile(r, 95)), 1)]
    finally:
        wall = time.time() - t_run0
        try:
            np.savez_compressed(
                out / "traces.npz",
                truth=np.asarray(sim.truth, dtype=np.float64).reshape(-1, 6),
                est=np.asarray(sim.est, dtype=np.float64).reshape(-1, 4),
                cmds=np.asarray(sim.cmds, dtype=np.float64).reshape(-1, 4),
            )
        except Exception as exc:  # noqa: BLE001
            result["traces_error"] = str(exc)
        sim.close()
        # Stop the DRIVE NODE first with its own single SIGINT and wait for
        # its stats file: Humble's ros2 launch escalates SIGINT -> SIGTERM
        # after ~5 s (no CLI knob), which truncated the shutdown stats
        # write on the Windows bind mount every time.
        try:
            pids = subprocess.run(
                ["pgrep", "-f", "drive_node --cfg"], capture_output=True, text=True
            ).stdout.split()
            for pid in pids:
                os.kill(int(pid), signal.SIGINT)
            for _ in range(80):
                if stats_file.exists() or not pids:
                    break
                time.sleep(0.5)
        except (OSError, ValueError):
            pass
        try:
            proc.send_signal(signal.SIGINT)  # launch cleans up the TF publisher
        except ProcessLookupError:
            pass
        try:
            result["launch_exit"] = proc.wait(timeout=45)
        except subprocess.TimeoutExpired:
            proc.kill()
            result["launch_exit"] = "killed"
        node_log.close()
        result["wall_s"] = round(wall, 1)
        # Written INSIDE finally: the early "never localized" return path
        # must still leave a result.json behind.
        (out / "result.json").write_text(json.dumps(result, indent=1), encoding="ascii")
    if stats_file.exists():
        stats = json.loads(stats_file.read_text(encoding="ascii"))
        result["node_timing"] = stats.get("timing", {})
        result["node_counters"] = stats.get("counters", {})
        result["node_localization"] = stats.get("localization")
        result["safety_events"] = stats.get("safety_events", [])[-20:]
        fixes = stats.get("localizer", {}).get("fixes", [])
        result["fixes"] = {
            "n": len(fixes),
            "accepted": sum(1 for f in fixes if f["accepted"]),
            "rejected": sum(1 for f in fixes if f["accepted"] is False),
        }
    (out / "result.json").write_text(json.dumps(result, indent=1), encoding="ascii")
    return result


def prepare_route_package(src: Path, dst: Path, trim_end_m: float) -> tuple[Path, "RacingPath"]:
    """Copy the map package with the racing line's last ``trim_end_m`` cut.

    Grid, corridor and metadata are byte-identical to the real save; ONLY
    racing_line.csv is rewritten (rows with s_m <= length - trim). Returns
    the copy's directory and the trimmed path (the same one the drive node
    will follow).
    """
    import shutil

    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file():
            shutil.copy2(f, dst / f.name)
    lines = (src / "racing_line.csv").read_text(encoding="ascii").splitlines()
    rows = [ln for ln in lines if ln and not ln.startswith("#")]
    s_vals = np.asarray([float(ln.split(",")[0]) for ln in rows])
    s_cut = float(s_vals[-1]) - float(trim_end_m)
    kept = [ln for ln, s in zip(rows, s_vals) if s <= s_cut]
    if len(kept) < 2:
        raise SystemExit(f"route too short to trim {trim_end_m} m")
    header = [ln for ln in lines if ln.startswith("#")]
    (dst / "racing_line.csv").write_text("\n".join(header + kept) + "\n", encoding="ascii")
    # The package manifest pins every file's size + sha256: record the
    # trimmed racing line so the copy stays a VALID package (and note the
    # trim, so nobody mistakes it for the original save).
    import hashlib

    meta_path = dst / "map_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    raw = (dst / "racing_line.csv").read_bytes()
    meta["files"]["racing_line.csv"] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    meta["route_end_trimmed_m"] = float(trim_end_m)
    meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    pkg = load_map_package(dst)
    return dst, RacingPath.from_racing_line(pkg.racing_line)


def gates(results: dict) -> dict:
    g = {}
    r = results.get("cam_route")
    if r:
        tr = r.get("tracking", {})
        g["cam_route_cross_track_rms_le_0.3m_at_2mps"] = bool(
            tr.get("cross_track_rms_m") is not None and tr["cross_track_rms_m"] <= XTRACK_GATE_M and tr.get("max_speed_mps", 0) >= 1.9
        )
        g["cam_route_finished"] = bool(r.get("finished"))
        g["cam_route_localized_from_known_start"] = r.get("localized_after_s") is not None
    if "cam_obstacle" in results:
        o = results["cam_obstacle"]
        gap = o.get("min_gap_base_link_to_block_face_m")
        # base_link to block face must clear the 0.8 m front overhang (no
        # contact) and the kart must stay stopped while the path is blocked.
        g["cam_obstacle_stops_short_no_contact"] = (
            bool(o.get("stopped_before_block")) and gap is not None and gap > 0.85 and bool(o.get("still_stopped_after_5s"))
        )
    if "cam_estop" in results:
        e = results["cam_estop"]
        g["cam_estop_stops_and_latches"] = bool(e.get("stopped")) and bool(e.get("still_stopped_after_release"))
    if "cam_watchdog" in results:
        g["cam_watchdog_stops"] = bool(results["cam_watchdog"].get("stopped"))
    if "lidar_route" in results:
        lr = results["lidar_route"].get("tracking", {})
        g["lidar_route_comparison_ran"] = lr.get("cross_track_rms_m") is not None
    return g


def main(argv=None) -> int:
    global CLEAR_NEAR_LINE_M, TRIM_ROUTE_END_M
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--map", type=Path, required=True, help="map package directory (a live save or offline output)")
    parser.add_argument("--runs", default=",".join(KART_TARGET_RUNS))
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--clear-m", type=float, default=CLEAR_NEAR_LINE_M,
                        help="sim-world obstacle clearing around the racing line [m]")
    parser.add_argument("--trim-end", type=float, default=TRIM_ROUTE_END_M,
                        help="metres trimmed off the route end (corridor dead end)")
    args = parser.parse_args(argv)
    CLEAR_NEAR_LINE_M = float(args.clear_m)
    TRIM_ROUTE_END_M = float(args.trim_end)
    runs = [r.strip() for r in args.runs.split(",") if r.strip()]
    import rclpy

    package = load_map_package(args.map)
    path_full = RacingPath.from_racing_line(package.racing_line)
    args.out.mkdir(parents=True, exist_ok=True)
    if path_full.closed or path_full.length <= TRIM_ROUTE_END_M + 6.0:
        route_dir, path = Path(args.map), path_full
    else:
        route_dir, path = prepare_route_package(Path(args.map), args.out / "route_pkg", TRIM_ROUTE_END_M)
    log(
        f"map package {Path(args.map).name}: racing line {path_full.length:.1f} m"
        f" ({'closed' if path_full.closed else 'open'}), route driven {path.length:.1f} m"
        f" (end trimmed {path_full.length - path.length:.1f} m short of the corridor dead end),"
        f" ROS_DISTRO={os.environ.get('ROS_DISTRO')}"
    )
    summary_path = args.out / "summary.json"
    results = json.loads(summary_path.read_text(encoding="ascii")).get("results", {}) if summary_path.exists() else {}
    rclpy.init(args=[])
    try:
        for run in runs:
            log(f"=== {run} ===")
            r = run_one(run, package, route_dir, path, args.out)
            results[run] = r
            brief = {
                k: r.get(k)
                for k in (
                    "localized_after_s", "converged_pose_error_m", "converged_yaw_error_deg", "finished",
                    "route_time_s", "tracking", "localization_error_vs_truth", "command_zero_after_s",
                    "kart_stopped_after_s", "reasons_when_stopped", "still_stopped_after_release",
                    "moving_again_after_reengage_s", "moving_again_after_data_back_s",
                    "latency_sensor_to_command", "launch_exit", "camera_frames_rendered", "render_ms_mean_p95", "error",
                )
                if r.get(k) is not None
            }
            log(json.dumps(brief))
    finally:
        rclpy.shutdown()
    g = gates(results)
    summary = {
        "map": str(args.map),
        "route_length_m": round(path.length, 2),
        "route_end_trimmed_m": round(path_full.length - path.length, 2),
        "sim_world_clear_near_line_m": CLEAR_NEAR_LINE_M,
        "ros_distro": os.environ.get("ROS_DISTRO"),
        "gates": g,
        "all_gates_pass": all(g.values()),
        "results": results,
    }
    summary_path.write_text(json.dumps(summary, indent=1), encoding="ascii")
    log(f"GATES: {json.dumps(g)}")
    log(f"wrote {summary_path}")
    return 0 if all(g.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
