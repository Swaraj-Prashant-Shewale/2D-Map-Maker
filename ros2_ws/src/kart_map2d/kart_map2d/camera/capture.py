"""Non-ROS camera sources: cv2.VideoCapture devices/files and GStreamer.

The camera is NOT required to be a ROS topic ([camera] ``source`` in
kart_master.cfg):

- ``ros2``      - image + camera_info topics; handled by the node's
                  subscriptions, not by this module.
- ``device``    - ``cv2.VideoCapture``: a V4L2 index ("0"), a device path
                  ("/dev/video0") or a video file (testing against recorded
                  footage). Intrinsics come from a calibration YAML
                  (:mod:`kart_map2d.camera.model`) because there is no
                  camera_info topic. Frames are stamped AT CAPTURE from the
                  system clock (``time.time()``); the driver's internal
                  buffering (typically one frame, ~1/fps s) is NOT corrected
                  for - documented, measured latency belongs in the rig's
                  calibration notes.
- ``gstreamer`` - a full GStreamer pipeline string ending in ``appsink``
                  (the Orin CSI port). HONEST NOTE: pip's
                  ``opencv-python-headless`` is built WITHOUT GStreamer -
                  ``source = gstreamer`` under it fails at open. On the Orin
                  install the apt build (``sudo apt install python3-opencv``)
                  for this mode; ``device`` and video files work with either
                  build.

All sources feed the SAME per-camera worker path as ROS frames: the capture
thread stamps the frame and hands it to the node callback; VO / segmentation
never know where a frame came from. One frame in flight per camera, never a
queue - a frame captured while the previous callback still runs replaces the
older one exactly like the ROS path's pending-frame slot.
"""

from __future__ import annotations

import threading
import time
from typing import Callable

import numpy as np
from numpy.typing import NDArray

__all__ = ["CaptureConfig", "CameraCapture", "gstreamer_available", "open_capture"]


class CaptureConfig:
    """How to open one non-ROS camera (subset of [camera] in kart_master.cfg)."""

    def __init__(
        self,
        source: str,  # "device" | "gstreamer"
        device: str = "0",  # V4L2 index, device path, or a video file
        gstreamer_pipeline: str = "",
        width: int = 0,  # 0 = leave the driver's default
        height: int = 0,
        fps: float = 0.0,
        max_rate_hz: float = 0.0,  # 0 = deliver every frame
        loop_file: bool = False,  # video files: loop at EOF (soak tests)
    ) -> None:
        self.source = source
        self.device = str(device)
        self.gstreamer_pipeline = gstreamer_pipeline
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.max_rate_hz = float(max_rate_hz)
        self.loop_file = bool(loop_file)


def gstreamer_available() -> bool:
    """True when this OpenCV build was compiled with GStreamer support."""
    import cv2

    try:
        info = cv2.getBuildInformation()
    except Exception:  # noqa: BLE001 - never break startup over a probe
        return False
    for line in info.splitlines():
        if "GStreamer" in line:
            return "YES" in line.upper()
    return False


def open_capture(config: CaptureConfig):
    """An opened ``cv2.VideoCapture`` for ``config``, or raise with the fix."""
    import cv2

    if config.source == "gstreamer":
        if not config.gstreamer_pipeline:
            raise ValueError("source = gstreamer needs gstreamer_pipeline in [camera]")
        if not gstreamer_available():
            raise RuntimeError(
                "this OpenCV build has no GStreamer support (pip's opencv-python-headless "
                "never does). On the Orin: sudo apt install python3-opencv and remove/shadow "
                "the pip wheel for this mode; or use source = device for a V4L2 camera."
            )
        cap = cv2.VideoCapture(config.gstreamer_pipeline, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            raise RuntimeError(f"GStreamer pipeline did not open: {config.gstreamer_pipeline!r}")
        return cap

    if config.source != "device":
        raise ValueError(f"unknown capture source {config.source!r} (use device | gstreamer)")
    dev: int | str = config.device
    if str(dev).lstrip("-").isdigit():
        dev = int(dev)
    cap = cv2.VideoCapture(dev)
    if not cap.isOpened():
        raise RuntimeError(
            f"cv2.VideoCapture({dev!r}) did not open - check the device index/path "
            "(a video file path must exist; on Linux the user needs rw on /dev/video*)"
        )
    if config.width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
    if config.height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
    if config.fps > 0:
        cap.set(cv2.CAP_PROP_FPS, config.fps)
    return cap


class CameraCapture:
    """One capture thread reading frames and handing them to a callback.

    ``on_frame(stamp: float, bgr: np.ndarray)`` runs ON THE CAPTURE THREAD;
    the node's callback only stores the frame and pokes the VO worker (the
    same cheap work its ROS image callback does). Frame stamps are
    ``clock()`` at the moment ``read()`` returned (default ``time.time()``,
    the system clock - the same epoch the dedicated IMU's corrected stamps
    live on when the node runs on the wall clock).

    A VIDEO FILE is paced to its own timeline instead of being decoded flat
    out (decoding is much faster than real time; unpaced, a 50 s file would
    flood the pipeline in seconds with near-identical stamps and the rate
    cap would simply discard most of its content):

    - with a ``<file>.stamps.csv`` sidecar (written by
      ``tools/frames_to_video.py``: ``frame_index,stamp_s`` rows), frame k is
      emitted when ``clock()`` reaches the sidecar time - absolutely, when
      the clock starts BEFORE the recording (a ``use_sim_time`` bag replay:
      frames then reproduce the recorded timeline exactly), otherwise
      relative to the first frame (wall clock: recorded gaps, shifted to
      now);
    - without a sidecar, frames are paced by the container's fps metadata
      (``CAP_PROP_FPS``, falling back to the configured fps, else 30).

    A live device (V4L2 index / path) is untouched: ``read()`` blocks at the
    driver's own rate. Before the first frame the thread waits for a valid
    clock (an epoch time): under ``use_sim_time`` the node clock is 0 until
    the first /clock message, and frames stamped 0 would be unusable.
    Errors stop the thread and are reported through ``on_error`` once.
    """

    def __init__(
        self,
        name: str,
        config: CaptureConfig,
        on_frame: Callable[[float, NDArray[np.uint8]], None],
        on_error: Callable[[str], None] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.name = name
        self.config = config
        self.on_frame = on_frame
        self.on_error = on_error or (lambda text: None)
        self.clock = clock
        self.frames = 0
        self.failed_reads = 0
        self._cap = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Open the device (raises with the fix on failure) and start reading."""
        self._cap = open_capture(self.config)
        self._thread = threading.Thread(target=self._run, name=f"kart2d-capture-{self.name}", daemon=True)
        self._thread.start()

    #: A clock below this is not an epoch time (use_sim_time before /clock).
    _EPOCH_MIN_S = 1e8

    def _file_pacer(self):
        """For a video-file source: a callable(frame_index) that blocks until
        the frame's emission time, or None for a live device."""
        import os

        if self.config.source != "device" or not os.path.isfile(self.config.device):
            return None
        stamps: list[float] = []
        sidecar = self.config.device + ".stamps.csv"
        if os.path.isfile(sidecar):
            try:
                with open(sidecar, "r", encoding="ascii") as f:
                    for line in f:
                        parts = line.strip().split(",")
                        if len(parts) >= 2 and not parts[0].lstrip("-").startswith(("f", "#")):
                            stamps.append(float(parts[1]))
            except (OSError, ValueError):
                stamps = []
        fps = 0.0
        try:
            import cv2

            fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
        except Exception:  # noqa: BLE001 - metadata is best effort
            pass
        if fps <= 0.0:
            fps = self.config.fps if self.config.fps > 0 else 30.0
        state: dict = {}

        def wait_for(index: int) -> None:
            if not state:
                now = self.clock()
                # Absolute sidecar times only when the clock is BEHIND the
                # recording (a use_sim_time replay of the same timeline).
                state["absolute"] = bool(stamps) and now < stamps[0]
                state["t_start"] = now
            if stamps:
                s = stamps[min(index, len(stamps) - 1)]
                target = s if state["absolute"] else state["t_start"] + (s - stamps[0])
            else:
                target = state["t_start"] + index / fps
            while not self._stop.is_set() and self.clock() < target:
                time.sleep(0.005)

        return wait_for

    def _run(self) -> None:
        min_period = 1.0 / self.config.max_rate_hz if self.config.max_rate_hz > 0 else 0.0
        last_emit = -np.inf
        consecutive_failures = 0
        # Wait for a valid (epoch) clock: under use_sim_time the node clock
        # is 0 until the first /clock message and stamps would be unusable.
        while not self._stop.is_set() and self.clock() < self._EPOCH_MIN_S:
            time.sleep(0.01)
        pacer = self._file_pacer()
        frame_index = 0
        while not self._stop.is_set():
            if pacer is not None:
                pacer(frame_index)
                if self._stop.is_set():
                    break
            ok, frame = self._cap.read()
            frame_index += 1
            stamp = self.clock()
            if not ok or frame is None:
                self.failed_reads += 1
                consecutive_failures += 1
                if self.config.loop_file and self.config.source == "device":
                    import cv2

                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                if consecutive_failures >= 25:
                    self.on_error(
                        f"camera {self.name}: {consecutive_failures} consecutive failed reads - "
                        "device unplugged or video file ended; capture stopped"
                    )
                    return
                time.sleep(0.02)
                continue
            consecutive_failures = 0
            if min_period > 0.0 and stamp - last_emit < min_period:
                continue  # rate cap: drop, never queue
            last_emit = stamp
            self.frames += 1
            try:
                if frame.ndim == 2:  # mono capture -> the pipeline expects BGR-compatible input
                    self.on_frame(stamp, frame)
                else:
                    self.on_frame(stamp, np.ascontiguousarray(frame[:, :, :3]))
            except Exception as exc:  # noqa: BLE001 - a bad frame must not kill capture
                self.on_error(f"camera {self.name}: frame callback failed: {type(exc).__name__}: {exc}")

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:  # noqa: BLE001 - best effort at shutdown
                pass
