"""Non-ROS camera sources: cv2.VideoCapture device/file capture (req 14).

A real video file (written here with cv2.VideoWriter) stands in for a V4L2
device: the capture thread must deliver its frames, stamp them at capture,
respect the rate cap, and stop cleanly. GStreamer coverage is limited to
the honest capability probe - pip's OpenCV has no GStreamer, and the test
must state that rather than pretend.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from kart_map2d.camera.capture import CameraCapture, CaptureConfig, gstreamer_available, open_capture


def write_test_video(path: Path, frames: int = 12, w: int = 64, h: int = 48, fps: float = 30.0) -> list:
    """An MJPG AVI (portable across OpenCV builds) with per-frame markers."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    assert writer.isOpened(), "cv2.VideoWriter could not open (MJPG missing?)"
    written = []
    for k in range(frames):
        img = np.full((h, w, 3), 30, np.uint8)
        img[:, : (k + 1) * 4] = (0, 0, 255)  # widening marker per frame
        writer.write(img)
        written.append(img)
    writer.release()
    return written


def test_device_source_reads_a_video_file(tmp_path: Path):
    video = tmp_path / "cam.avi"
    write_test_video(video, frames=10)
    got: list[tuple[float, np.ndarray]] = []
    done = threading.Event()

    def on_frame(stamp, bgr):
        got.append((stamp, bgr.copy()))
        if len(got) >= 10:
            done.set()

    capture = CameraCapture("cam", CaptureConfig("device", device=str(video)), on_frame)
    capture.start()
    assert done.wait(10.0), f"only {len(got)} frames arrived"
    capture.stop()
    stamps = [s for s, _ in got]
    assert all(b >= a for a, b in zip(stamps, stamps[1:]))  # capture-time stamps rise
    assert got[0][1].shape == (48, 64, 3)
    # Frames arrive in order: the marker widens monotonically.
    widths = [int((frame[:, :, 2] > 200).sum(axis=1).max()) for _, frame in got]
    assert widths == sorted(widths)
    assert capture.frames >= 10


def test_device_source_rate_cap_drops_frames(tmp_path: Path):
    video = tmp_path / "cam.avi"
    write_test_video(video, frames=30)
    got = []
    capture = CameraCapture(
        "cam",
        CaptureConfig("device", device=str(video), max_rate_hz=15.0),
        lambda stamp, bgr: got.append(stamp),
    )
    capture.start()
    time.sleep(1.0)
    capture.stop()
    # 30 frames decoded near-instantly; the 15 Hz cap lets only ~a dozen through.
    assert 1 <= len(got) < 30
    gaps = np.diff(got)
    if len(gaps):
        assert gaps.min() >= 1.0 / 15.0 - 1e-3


def test_capture_stops_after_file_end_with_error_report(tmp_path: Path):
    video = tmp_path / "cam.avi"
    write_test_video(video, frames=3)
    errors = []
    capture = CameraCapture("cam", CaptureConfig("device", device=str(video)), lambda s, b: None, on_error=errors.append)
    capture.start()
    deadline = time.monotonic() + 10.0
    while not errors and time.monotonic() < deadline:
        time.sleep(0.05)
    capture.stop()
    assert errors and "capture stopped" in errors[0]
    assert capture.frames == 3


def test_file_replay_paces_by_fps_and_keeps_every_frame(tmp_path: Path):
    """A video FILE must replay at its own pace (12 frames at 30 fps = 0.37 s),
    not decode flat out - otherwise a rate cap throws away content and every
    stamp collapses onto the decode burst (the requirement-14 defect)."""
    video = tmp_path / "cam.avi"
    write_test_video(video, frames=12, fps=30.0)
    got = []
    capture = CameraCapture("cam", CaptureConfig("device", device=str(video)), lambda s, b: got.append(s))
    t0 = time.monotonic()
    capture.start()
    deadline = time.monotonic() + 10.0
    while len(got) < 12 and time.monotonic() < deadline:
        time.sleep(0.02)
    elapsed = time.monotonic() - t0
    capture.stop()
    assert len(got) == 12, f"paced replay must deliver every frame, got {len(got)}"
    assert elapsed >= 11 / 30.0 - 0.05, f"12 frames at 30 fps must take ~0.37 s, took {elapsed:.3f}"
    assert got[-1] - got[0] >= 11 / 30.0 - 0.08  # stamps spread over the file's own timeline


def test_file_replay_follows_stamps_sidecar_absolutely_on_a_replay_clock(tmp_path: Path):
    """With a <file>.stamps.csv sidecar and a clock BEHIND the recording (a
    use_sim_time bag replay), frames must come out AT the recorded stamps."""
    video = tmp_path / "cam.avi"
    write_test_video(video, frames=6, fps=30.0)
    base = 1.8e9  # an epoch in the recording's past-relative future
    stamps = [base + 0.30 + 0.15 * k for k in range(6)]
    with open(str(video) + ".stamps.csv", "w", encoding="ascii") as f:
        f.write("frame_index,stamp_s\n")
        for k, s in enumerate(stamps):
            f.write(f"{k},{s:.6f}\n")
    m0 = time.monotonic()
    clock = lambda: base + (time.monotonic() - m0)  # starts BEFORE stamps[0]  # noqa: E731
    got = []
    capture = CameraCapture("cam", CaptureConfig("device", device=str(video)), lambda s, b: got.append(s), clock=clock)
    capture.start()
    deadline = time.monotonic() + 10.0
    while len(got) < 6 and time.monotonic() < deadline:
        time.sleep(0.01)
    capture.stop()
    assert len(got) == 6
    err = [g - s for g, s in zip(got, stamps)]
    assert all(e >= -1e-3 for e in err), f"a frame ran AHEAD of its sidecar stamp: {err}"
    assert max(err) < 0.12, f"frames must track the sidecar stamps, offsets {err}"


def test_capture_waits_for_a_valid_epoch_clock(tmp_path: Path):
    """use_sim_time before the first /clock message reads 0: no frame may be
    captured (and stamped 0) until the clock becomes an epoch time."""
    video = tmp_path / "cam.avi"
    write_test_video(video, frames=4, fps=30.0)
    state: dict = {}
    clock = lambda: 0.0 if not state else state["base"] + (time.monotonic() - state["m0"])  # noqa: E731
    got = []
    capture = CameraCapture("cam", CaptureConfig("device", device=str(video)), lambda s, b: got.append(s), clock=clock)
    capture.start()
    time.sleep(0.25)
    assert got == [], "frames were captured while the clock was invalid (0)"
    state.update(base=1.9e9, m0=time.monotonic())
    deadline = time.monotonic() + 10.0
    while len(got) < 4 and time.monotonic() < deadline:
        time.sleep(0.02)
    capture.stop()
    assert len(got) == 4 and all(s >= 1.9e9 for s in got)


def test_missing_device_raises_with_fix():
    with pytest.raises(RuntimeError, match="did not open"):
        open_capture(CaptureConfig("device", device="Z:/does/not/exist.avi"))


def test_unknown_source_rejected():
    with pytest.raises(ValueError, match="device | gstreamer"):
        open_capture(CaptureConfig("webcam"))


def test_gstreamer_probe_is_honest():
    """pip opencv has NO GStreamer: the probe must say so, and the open must
    fail with the documented apt-python3-opencv hint (not crash cryptically)."""
    available = gstreamer_available()
    if available:
        pytest.skip("this OpenCV build has GStreamer; the refusal path does not apply")
    with pytest.raises(RuntimeError, match="apt install python3-opencv"):
        open_capture(CaptureConfig("gstreamer", gstreamer_pipeline="videotestsrc ! appsink"))


def test_gstreamer_needs_pipeline():
    with pytest.raises(ValueError, match="gstreamer_pipeline"):
        open_capture(CaptureConfig("gstreamer"))
