# Maveric 2D map maker (kart_map2d)

The go-kart's STANDALONE 2D mapping + autonomous drive stack. This folder is
what the kart team edits from now on; the research workspace
(`../Virtual map maker`) stays untouched as the reference. Code was ported
from there after its classical-segmentation fix (corridor 0.19 m -> 1.46 m,
7 loop closures on sept3) and refactored to be camera-agnostic, 2D-only and
Orin-only. **This README is laptop documentation - it is on the never-upload
list (`KART_UPLOAD.txt`).**

Scope: 2D drivable/obstacle evidence grid, corridor, racing line, loop
closure, map packages + `~/save_map`, the live mapping node, the drive stack
(map-relative localization, pure pursuit, safety layer), classical
segmentation by default (SegFormer optional), offline laptop tools.
NOT here: lidar3d / camera3d / cloud maps, research evaluation scripts, the
demo renderer, anything Jazzy/Thor. Target: Jetson Orin, JetPack 6 =
Ubuntu 22.04 = ROS 2 Humble, Python 3.10, numpy < 2.

## Layout

```
kart_master.cfg          THE control file (sensors, cameras, mapping, drive, safety)
KART_UPLOAD.txt          what goes on the kart and what never does
results/                 validation evidence (sensor matrix, drive sim, latency, clocks)
ros2_ws/src/kart_map2d/  the ament_python package (code, launch, config, tests)
tools/                   laptop tools (replay runner, bag export, uploader, calibrators)
requirements/            orin-base.txt (numpy<2), orin-neural.txt (torch note), laptop.txt
docker/                  kart2d:humble - the laptop's Orin-userland test container
inputs/                  JUNCTION into the research workspace's recordings (never uploaded)
outputs/                 laptop test artifacts (gitignored)
.venv/                   Windows test environment
```

## Laptop setup (Windows)

```powershell
cd "S:\Coding_Tools\My_Projects\Maveric\2d map maker"
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements\laptop.txt
# the test suite (366 tests: 359 pass + 7 torch-only skips; setup.cfg already
# passes -q - adding another -q would hide the final count line):
.venv\Scripts\python.exe -m pytest ros2_ws\src\kart_map2d\test -p no:cacheprovider
```

The Humble container mirrors the Orin userland (build once):

```powershell
docker build -t kart2d:humble -f docker/Dockerfile docker
# colcon build + test exactly as on the Orin (PowerShell, not Git Bash - it mangles -v):
docker run --rm -v "S:\Coding_Tools\My_Projects\Maveric\2d map maker:/ws2d" --shm-size=1g kart2d:humble `
  bash -lc "cd /ws2d/ros2_ws && colcon build --symlink-install && colcon test --packages-select kart_map2d && colcon test-result --verbose"
```

## kart_master.cfg - the one control file

Every behaviour switch lives there, with a comment block per key (what it
does, what changing it affects, safe range/unit, why the default). Shipped
defaults are the kart target: `[sensors] imu = 1, camera = 1`, everything
else 0 - ONE camera plus the dedicated IMU, camera as the only visual input.
`lidar` is strictly an off-by-default experiment switch. Validate any edit
without starting a node:

```
ros2 run kart_map2d check_cfg kart_master.cfg          # on the kart / container
.venv\Scripts\python.exe -m kart_map2d.config.check_cfg kart_master.cfg   # laptop (from ros2_ws\src\kart_map2d)
```

Rules enforced before anything runs: typo -> did-you-mean with the line
number; out-of-range -> the allowed range; all sensors off -> error; IMU-only
-> dead-reckoning warning; `gnss = 1` -> "not implemented" error;
magnetometer without a calibration file -> refusal that prints the exact
recording procedure. UTF-8/UTF-16(BOM)/CRLF/trailing comments all parse.
Launch arguments flip switches without editing: `lidar:=1`.
The file is read ONCE at startup; while driving only `max_speed_mps` and
`speed_scale` are live (`ros2 param set /drive ...`).

## Running

Mapping (kart or container; the launch file also publishes the static
`base_link -> imu_link` TF and can start a usb_cam driver):

```bash
ros2 launch kart_map2d mapping.launch.py cfg:=/abs/path/kart_master.cfg
ros2 service call /kart_map2d/save_map std_srvs/srv/Trigger    # or save_on_shutdown = 1
```

Save BEFORE Ctrl-C when running under `ros2 launch`: launch escalates
SIGINT -> SIGTERM after ~5 s, which is shorter than a shutdown save's loop
closure. Either call the service first (the normal flow), or add the launch
argument `sigterm_timeout:=120` if you rely on `save_on_shutdown`
(`--sigterm-timeout` is not a ros2 launch flag; the launch ARGUMENT is accepted).
(Run colcon-less - `python3 -m kart_map2d.ros2.node --cfg ...` - and Ctrl-C
waits for the save; clean exit 0 is container-verified.)

Bag replay test (full pipeline against a September recording):

```bash
ros2 launch kart_map2d replay.launch.py cfg:=... bag:=/path/to/sept3_20260905_130304
```

(Replay = sim time + `--clock 200`; live = wall clock. The decided clock
rules and the two measured replay traps - SHM profile coverage and
bind-mount bags - are in `results/clock_handling.md` and the guide.)

Driving (loads ONE map package; starts DISENGAGED; stands first,
`max_speed_mps` low):

```bash
ros2 launch kart_map2d drive.launch.py cfg:=...
ros2 service call /drive/engage std_srvs/srv/SetBool "{data: true}"
```

Offline mapping on the laptop (no ROS; honours `[sensors]`, output folders
named by the combination so experiments never overwrite each other -
requirement 9):

```powershell
.venv\Scripts\python.exe tools\replay_run.py inputs\sept3_20260905_130304
# -> outputs\sept3_20260905_130304__imu+cam\  (a loadable map package)
.venv\Scripts\python.exe tools\replay_run.py inputs\sept3_20260905_130304 --set sensors.lidar=1
# -> ...__imu+cam+lidar\   (needs kiss-icp installed, or a cached KISS result)
```

## Bringing up a new camera (no ZED assumed anywhere)

1. **Source** - `[camera] source = ros2` (topics), `device`
   (cv2.VideoCapture index/path/video file) or `gstreamer` (Orin CSI;
   needs apt `python3-opencv` - pip wheels have no GStreamer).
2. **Intrinsics + distortion** - from the CameraInfo topic, or a calibration
   YAML (`tools/dump_camera_info.py` writes one from a live topic or a bag).
   plumb_bob / rational_polynomial / equidistant are handled by undistorting
   POINTS and one precomputed ray table per camera - full frames are never
   warped (compute budget). Any resolution; `processing_scale` sets the
   working size. Encodings: bgr8 / rgb8 / bgra8 / rgba8 / mono8 /
   CompressedImage (jpeg, png).
3. **Extrinsics** - `frame = auto` (TF via the image header frame_id),
   `frame = <tf frame>`, or the manual `mount_xyz` / `mount_rpy_deg`
   fallback (always available, always used when the lookup fails).
4. **Height calibration** (the monocular scale anchor, replaces the old
   ZED-keyed table): tape-measure a straight 20+ m; drive it slowly; build
   the map offline (`tools/replay_run.py`); compare the mapped distance D_est
   with the true D; set `height_m := height_m * D / D_est`; repeat once.
   September values: zed_right 0.445 m, zed_left 0.545 m.
5. **Kart mask** - leave `learn_mask = 1` (auto-learned from temporally
   static pixels while driving; the classical-fix learner) or paint a PNG
   (white = kart body) and set `mask_file`. An unmasked kart body paints
   false obstacles beside the path every frame - this exact failure pinched
   the research corridor to 0.2 m before the fix.

## All-sensor input - honesty section (requirement 13)

The adapters for **magnetometer** (`sensor_msgs/MagneticField` -> hard/soft
iron correction -> planar heading), **speedometer** (`Float32` m/s or
`TwistStamped`) and **wheel ticks** (`Int64` counter or `JointState`,
quantization-aware sigma) are implemented and unit-tested, and a
miscalibrated magnetometer is proven (synthetically) to be excluded by the
health monitor. BUT: the September recordings contain none of these
sensors, so NO real-data validation exists yet. It happens the first time
you record them - add to your next recording:

```bash
ros2 bag record /imu/data <your camera topics> /mag /speed /wheel_ticks
```

then run `tools/replay_run.py` on it and compare `[sensors]` combinations.
Magnetometer first needs `tools/calibrate_magnetometer.py` (slow 360 deg
circle; the node refuses to start without the YAML and prints the procedure).

## Requirement-14 proof (camera without ROS) - EXECUTED

```powershell
.venv\Scripts\python.exe tools\frames_to_video.py inputs\sept3_20260905_130304 zed_right --out outputs\sept3_zed_right.mp4
```

writes the mp4 (median-fps), a stamps CSV and the calibration YAML. The
container then ran the live pipeline with `[camera] source = device`
pointing at the video + YAML and ZERO ROS image topics subscribed (proved
in each run's `node_info.txt`); the IMU still arrived over DDS from the bag.
Measured (results/sensor_matrix.md, "Camera WITHOUT ROS"):
- device, NEAR-LOSSLESS avi (MJPG): ATE 2.00 m - within 15 percent of the
  ros2-source draws (mean 2.16 m) -> the device-source abstraction is
  PROVEN equivalent, gate MET on a faithful encode;
- device, lossy mp4v: ATE ~3.5 m (36-38 dB PSNR) - the extra error is the
  CODEC, not the source path (the avi row isolates it). Use a lossless /
  high-bitrate encode for device-source validation.
- `source = gstreamer` under pip OpenCV refuses at startup with the apt
  python3-opencv hint (exit 2, no crash); real CSI capture is Orin-only.

## What carried over unchanged (requirement 11)

The proven live-node behaviours: /imu/data QoS depth 2000 + backlog
draining, the IMU arrival-clock cap with the starved-executor fix, clean
Ctrl-C exit 0, `~/save_map` -> map package (same `map_meta.json` format -
research and 2D packages load in either project), save-time loop closure in
a spawned subprocess, never-queue camera/LiDAR workers, the safety layer
(e-stop latch, deadman, watchdogs, footprint-aware obstacle stop), the
drive node's fixed-at-verification behaviours (obstacle corridor around the
KART not the racing line, node-clock obstacle aging, rejected-match
evidence, locked deque access, at-rest global-search restart).

Plus the classical fix, now wired into the LIVE node too (the research node
had it offline-only): learned/loaded per-camera kart masks, the kart
footprint + 2.5 m min-contact grid guards, stamp-always camera inserts
(history routing for unknown-class movers and facade smear).

## Measured (this port, real September data)

`tools/replay_run.py inputs/sept3_20260905_130304` on the Windows laptop
(numpy 2.5 / OpenCV 5 venv), against the research workspace's published
classical-fix numbers - **digit-exact parity**:

| configuration | corridor L/R [m] | loop closures | ATE vs KISS [m] |
|---|---|---|---|
| `imu+cam` (zed_right, the kart target) | 1.133 / 1.179 | 3 | 1.4588 |
| `imu+cam+cam2` (both cameras)          | 1.462 / 1.335 | 7 | 1.2589 |

(Re-measured on this stack - numpy 2.5 / OpenCV 5 - so the one-camera
closure count and ATE differ slightly from the research machine's numpy
1.26 / OpenCV 4.10 figures of 1.013/1.154 m, 5 closures, 1.5182 m; the
container, which matches the Orin, reproduces the research numbers - see
results/sensor_matrix.md's offline table.)
Kart mask auto-learned online path: mask_fraction 0.3105 (research: 0.3105).
Laptop stage timings for the 50 s recording (one camera): replay 4.3 s,
segmentation 1.5 s (14.2 ms/frame at 960x600 - the 15 ms budget holds even
on this slower stack; the research machine measured 7-9 ms), camera grid
insert 4.8 s, save-time loop closure 26.3 s, corridor + racing line 1.4 s.
Latency-relevant cfg knobs and their measured stage: `[segmentation] rate_hz`
and `scale` (segmentation + insert per frame), `[odometry] orb_features` /
`max_corners` (VO solve), `[mapping] cell_size_m` (insert + corridor),
`[mapping] loop_closure` (save time only, never per frame). Orin per-stage
numbers come from the live health topic (`latency_ms`) during the next
engineer's on-device run.

Suites: Windows 359 passed + 7 torch-only skips (366 tests, re-counted
2026-10-01); kart2d:humble container (colcon build --symlink-install +
colcon test) 366 tests, 0 errors, 0 failures, 7 skipped. Container smoke: `ros2 run kart_map2d check_cfg` works, the
mapping node starts under real rclpy from kart_master.cfg and exits 0 on
SIGINT, the drive node refuses a missing map package with exit 2.

LIVE bag replay (`ros2 launch kart_map2d replay.launch.py` in kart2d:humble,
real DDS, sept3 bag copied to container disk, shipped cfg = imu+cam):
14,858 /imu/data messages fused with 4 EKF late drops, 101 camera frames ->
49 VO solves -> 35 segmented grid inserts, `/kart_map2d/save_map` wrote a
hash-verified drivable package (corridor 42.8 m). Live per-stage latencies
(this laptop under Docker, health-topic `latency_ms`, mean/p95/max ms):
IMU callback 2.5/5.0/94, output timer 1.8/4.5/76, image decode 20/63/127,
VO solve 143/335/679, segmentation 31/56/522, grid insert 267/501/682
(workers, off the control path); RSS 440 MB. Loop closures at the live
save: 0 (35 observations from one camera at the 2 Hz mapping rate - the
offline build of the same data, which segments every frame, gets 5;
research live one-camera runs behaved the same).

## Known gaps / next steps

- The full validation MATRIX has now run (results/sensor_matrix.md,
  results/drive/): every [sensors] combination live on the sept3 bag, three
  repeat draws of the default, the req-14 device-source runs (ros2 vs
  source=device with ZERO ROS image topics), the non-ZED republished
  streams (bgr8 / jpeg CompressedImage / injected plumb_bob distortion),
  the synthetic mag/speed/wheel topics alongside the bag (incl. the
  miscalibrated-mag health rejection), and the closed-loop drive sim. What
  remains for the next engineer: on-Orin aarch64 timings, real CSI
  GStreamer capture, neural segmentation (needs the JetPack torch wheel),
  and first REAL mag/speed/wheel data (none in the September bags).
- Camera+IMU DRIVING - what the closed-loop ROS 2 sim measured on sept3
  (results/drive/, kart2d:humble). START CONDITIONS camera-only driving
  supports TODAY:
  - KNOWN start pose (marked grid spot, `start_pose:=x,y,yaw_deg`, and set
    `[drive] global_relocalization = 0` on a self-similar course): the
    camera map localizer converges in ~10 s and keeps the kart ON the line
    - cross-track RMS 0.07 m, max 0.16 m at the 2.0 m/s cap, 135 camera
      fixes accepted / 0 rejected, pose error bounded <= 0.85 m - when the
    map carries near-field visual structure (the dense OFFLINE-built map;
    the sparse live 2 Hz save is too thin for drive-time patch fixes: 20 of
    147 matches accepted before dropout - build the drive map offline).
  - WRONG start (2 m / 10 deg) or UNKNOWN start: NOT supported by camera
    alone. This plaza corridor is longitudinally self-similar, so the
    camera localizer either converges onto the wrong prior (~2 m off) or,
    from fully unknown, correctly REFUSES to engage (global search scores
    0.82-0.89, no distinct winner). The safety layer keeps the kart stopped
    while unlocalized. Multi-hypothesis / direction-selective fixes are the
    queued design.
  - LiDAR comparison row (`lidar:=1`): completes the full route, pose error
    0.05 m, cross-track RMS 0.03 m - the robust option when the start may
    be wrong or the course repeats itself.
  - e-stop (latched; release alone does not restart, re-engage does),
    sensor watchdog (stop within the pose timeout + braking), and the
    swept-path obstacle stop (stops >= 0.85 m short of a real block ON the
    route and stays stopped) all PASS. Mapping is unaffected by any of this.
- mag/speed/wheel: synthetic validation only (see the honesty section).
- gnss: not implemented; `gnss = 1` refuses by design.
- Real-hardware unknowns inherited from the research audit: aarch64
  timings, NVIDIA torch wheel, real DDS/shm sizing, chrony/PTP, motor
  bridge, steering sign, hardware e-stop, multi-hour memory growth.
