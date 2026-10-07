"""CameraModel: distortion via POINTS and ray tables only - never frames.

The rule under test (requirement 6): plumb_bob / rational_polynomial /
equidistant CameraInfo distortion is handled by undistorting matched points
and by one precomputed per-camera ray table; with zero distortion every
path is BIT-IDENTICAL to the proven pinhole math.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from kart_map2d.camera.model import (
    CameraModel,
    load_calibration_yaml,
    save_calibration_yaml,
)
from kart_map2d.mapping.evidence_grid import EvidenceGrid, _pixel_rays

K = np.array([[420.0, 0.0, 320.0], [0.0, 421.5, 200.0], [0.0, 0.0, 1.0]])
SIZE = (640, 400)
D_PLUMB = np.array([-0.28, 0.07, 1e-4, -2e-4, -0.006])


# ------------------------------------------------------------- zero-D paths
def test_zero_distortion_is_pinhole_bit_identical():
    model = CameraModel(K, SIZE)
    assert not model.has_distortion
    pts = np.array([[10.0, 20.0], [630.0, 390.0], [320.0, 200.0]])
    out = model.undistort_pixels(pts)
    assert out is pts  # literally the identity: zero cost on the proven path
    rays = model.pixel_rays((400, 640))
    reference = _pixel_rays(K, (400, 640))
    assert np.array_equal(rays, reference)  # bit-identical to the grid's own table


def test_empty_model_string_means_none():
    model = CameraModel(K, SIZE, np.zeros(5), "")
    assert model.model == "none" and not model.has_distortion
    # all-zero D under a named model is also "already rectified"
    model2 = CameraModel(K, SIZE, np.zeros(5), "plumb_bob")
    assert not model2.has_distortion


def test_unknown_distortion_model_is_refused():
    with pytest.raises(ValueError, match="not supported"):
        CameraModel(K, SIZE, np.array([0.1]), "weird_lens")


# ------------------------------------------------------------ points paths
def test_undistort_points_matches_cv2_reference():
    model = CameraModel(K, SIZE, D_PLUMB, "plumb_bob")
    assert model.has_distortion
    pts = np.array([[100.0, 80.0], [500.0, 350.0], [320.0, 200.0], [40.0, 380.0]])
    got = model.undistort_pixels(pts)
    want = cv2.undistortPoints(pts.reshape(-1, 1, 2), K, D_PLUMB, P=K).reshape(-1, 2)
    np.testing.assert_allclose(got, want, atol=1e-9)
    # The principal point barely moves; corners move by many pixels.
    assert np.linalg.norm(got[2] - pts[2]) < 0.5
    assert np.linalg.norm(got[3] - pts[3]) > 2.0


def test_undistort_roundtrips_through_projection():
    """Project a known ray through the distortion, undistort the pixel, and
    the ideal-pinhole reprojection must land on the original ray."""
    model = CameraModel(K, SIZE, D_PLUMB, "plumb_bob")
    ray = np.array([[0.21, -0.12, 1.0]])
    distorted = cv2.projectPoints(ray, np.zeros(3), np.zeros(3), K, D_PLUMB)[0].reshape(-1, 2)
    ideal = model.undistort_pixels(distorted)
    back = model.normalized_rays(distorted)
    np.testing.assert_allclose(back[0, :2], ray[0, :2], atol=1e-6)
    ideal_expected = (K @ (ray[0] / ray[0, 2]))[:2]
    np.testing.assert_allclose(ideal[0], ideal_expected, atol=1e-6)


def test_equidistant_uses_fisheye_undistort():
    D_fish = np.array([-0.05, 0.01, -0.002, 0.0005])
    model = CameraModel(K, SIZE, D_fish, "equidistant")
    pts = np.array([[80.0, 60.0], [600.0, 380.0]])
    got = model.undistort_pixels(pts)
    want = cv2.fisheye.undistortPoints(pts.reshape(-1, 1, 2), K, D_fish.reshape(-1, 1), P=K).reshape(-1, 2)
    np.testing.assert_allclose(got, want, atol=1e-9)


def test_rational_polynomial_accepted():
    D8 = np.array([-0.3, 0.08, 1e-4, -1e-4, -0.01, 0.05, 0.002, -0.001])
    model = CameraModel(K, SIZE, D8, "rational_polynomial")
    pts = np.array([[50.0, 50.0]])
    want = cv2.undistortPoints(pts.reshape(-1, 1, 2), K, D8, P=K).reshape(-1, 2)
    np.testing.assert_allclose(model.undistort_pixels(pts), want, atol=1e-9)


# --------------------------------------------------------------- ray tables
def test_distorted_ray_table_matches_per_pixel_undistortion():
    model = CameraModel(K, SIZE, D_PLUMB, "plumb_bob").scaled(0.1)  # 64x40: cheap
    h, w = 40, 64
    rays = model.pixel_rays((h, w))
    assert rays.shape == (h, w, 3)
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    ideal = model.undistort_pixels(np.column_stack([uu.ravel(), vv.ravel()]))
    np.testing.assert_allclose(rays[..., 0].ravel(), (ideal[:, 0] - model.K[0, 2]) / model.K[0, 0], atol=1e-12)
    np.testing.assert_allclose(rays[..., 2], 1.0)
    # cached: the same object comes back
    assert model.pixel_rays((h, w)) is rays


def test_grid_insert_with_rays_equals_default_for_zero_distortion():
    """insert_camera(rays=...) with the pinhole table must reproduce the
    default path EXACTLY (the live node always passes rays now)."""
    model = CameraModel(np.array([[60.0, 0, 48.0], [0, 60.0, 30.0], [0, 0, 1.0]]), (96, 60))
    h, w = 60, 96
    drivable = np.zeros((h, w), bool)
    drivable[38:55, 20:70] = True
    obstacle = np.zeros((h, w), bool)
    obstacle[30:38, 40:50] = True
    T = np.eye(4)
    T[:3, :3] = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])  # optical->base
    T[2, 3] = 0.5
    pose = np.zeros(3)

    a = EvidenceGrid(cell=0.2)
    a.insert_camera(drivable, obstacle, model.K, T, pose)
    b = EvidenceGrid(cell=0.2)
    b.insert_camera(drivable, obstacle, model.K, T, pose, rays=model.pixel_rays((h, w)))
    np.testing.assert_array_equal(a.classify(), b.classify())
    np.testing.assert_array_equal(a.drivability(), b.drivability())


def test_grid_insert_rejects_mismatched_ray_table():
    grid = EvidenceGrid(cell=0.2)
    model = CameraModel(np.array([[60.0, 0, 48.0], [0, 60.0, 30.0], [0, 0, 1.0]]), (96, 60))
    with pytest.raises(ValueError, match="rays shape"):
        grid.insert_camera(
            np.zeros((60, 96), bool), np.zeros((60, 96), bool), model.K, np.eye(4), np.zeros(3),
            rays=model.pixel_rays((30, 48)),
        )


def test_no_full_frame_undistortion_api_exists():
    """The budget rule is structural: CameraModel exposes no frame warp."""
    api = {name for name in dir(CameraModel) if not name.startswith("_")}
    assert "undistort_image" not in api and "remap" not in api
    src = Path(CameraModel.__module__.replace(".", "/"))
    module_file = Path(__file__).resolve().parents[1] / "kart_map2d" / "camera" / "model.py"
    text = module_file.read_text(encoding="utf-8")
    assert "cv2.remap" not in text and "initUndistortRectifyMap" not in text and "cv2.undistort(" not in text
    _ = src


# ------------------------------------------------------------------ scaling
def test_scaled_model_halves_K_and_keeps_D():
    model = CameraModel(K, SIZE, D_PLUMB, "plumb_bob")
    half = model.scaled(0.5)
    np.testing.assert_allclose(half.K[:2], K[:2] * 0.5)
    assert half.K[2, 2] == 1.0 and half.size == (320, 200)
    np.testing.assert_array_equal(half.D, D_PLUMB)  # normalized coords: unchanged
    assert model.scaled(1.0) is model


# --------------------------------------------------------------- YAML round
def test_calibration_yaml_roundtrip(tmp_path: Path):
    model = CameraModel(K, SIZE, D_PLUMB, "plumb_bob")
    path = save_calibration_yaml(tmp_path / "cal.yaml", model)
    loaded = load_calibration_yaml(path)
    np.testing.assert_allclose(loaded.K, model.K)
    np.testing.assert_allclose(loaded.D, model.D)
    assert loaded.model == "plumb_bob" and loaded.size == SIZE


def test_calibration_yaml_accepts_ros_calibration_layout(tmp_path: Path):
    (tmp_path / "ros.yaml").write_text(
        "image_width: 640\nimage_height: 400\n"
        "camera_matrix:\n  rows: 3\n  cols: 3\n  data: [420, 0, 320, 0, 421.5, 200, 0, 0, 1]\n"
        "distortion_model: plumb_bob\n"
        "distortion_coefficients:\n  rows: 1\n  cols: 5\n  data: [-0.28, 0.07, 0.0001, -0.0002, -0.006]\n",
        encoding="ascii",
    )
    loaded = load_calibration_yaml(tmp_path / "ros.yaml")
    np.testing.assert_allclose(loaded.K, K)
    assert loaded.has_distortion


def test_missing_calibration_file_names_the_fix(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="dump_camera_info"):
        load_calibration_yaml(tmp_path / "nope.yaml")
