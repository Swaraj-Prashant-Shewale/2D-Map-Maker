"""Synthetic-geometry tests for kart_map2d.perception.visual_odometry.

A textured ground plane is rendered through a pinhole camera at two poses
related by a known SE(2) step; the recovered RelativeMotion must match.
No files from inputs/ are needed.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from kart_map2d.core.measurements import CameraFrame
from kart_map2d.core.transforms import make_transform, rpy_matrix
from kart_map2d.perception.visual_odometry import (
    VisualOdometry,
    VoConfig,
    _default_kart_mask,
    _remap_tables,
    _skew,
    run_visual_odometry,
    yaw_prior_from_imu,
)

W, H = 960, 600
K = np.array([[340.0, 0.0, 480.0], [0.0, 361.0, 300.0], [0.0, 0.0, 1.0]])
GROUND_Z = -0.10
CAM_XYZ = np.array([-0.06, -0.27, 0.462])  # like the kart's right camera

TEX_RES = 0.02  # m per texture pixel
TEX_SIZE = 3000  # 60 m x 60 m of ground
RNG = np.random.default_rng(20260928)

# Rich, non-periodic ground texture (noise blobs at several scales plus
# scattered sharp dots so corner detectors have something to bite on).
# The std is reduced in float64: numpy 1.x and 2.x round a float32 reduction
# differently in the last ulp (0.28232101 vs 0.28232110 here), which flipped
# a few texture pixels by 1 grey level between the Windows venv (numpy 2)
# and the ROS containers (numpy 1) - enough to change which 5-point sample
# the raw RANSAC kept. With it the scene is bit-identical on every build.
_tex = np.zeros((TEX_SIZE, TEX_SIZE), np.float32)
for sigma, amp in ((1, 40.0), (3, 50.0), (9, 40.0), (27, 30.0)):
    layer = cv2.GaussianBlur(RNG.normal(0.0, 1.0, (TEX_SIZE, TEX_SIZE)).astype(np.float32), (0, 0), sigma)
    _tex += amp * layer / max(float(layer.std(dtype=np.float64)), 1e-6)
for _ in range(4000):
    cx, cy = RNG.integers(2, TEX_SIZE - 2, 2)
    _tex[cy - 1 : cy + 2, cx - 1 : cx + 2] += float(RNG.choice([-90.0, 90.0]))
TEXTURE = np.clip(128.0 + _tex, 0, 255).astype(np.uint8)


def nominal_mount() -> np.ndarray:
    """base_link <- camera optical frame, level and forward-looking."""
    optical_to_base = rpy_matrix(-np.pi / 2, 0.0, -np.pi / 2)
    return make_transform(optical_to_base, CAM_XYZ)


def se3_of_se2(x: float, y: float, yaw: float) -> np.ndarray:
    T = np.eye(4)
    c, s = np.cos(yaw), np.sin(yaw)
    T[:2, :2] = [[c, -s], [s, c]]
    T[:2, 3] = [x, y]
    return T


def render(T_w_base: np.ndarray, T_base_cam: np.ndarray) -> np.ndarray:
    """Render the ground plane seen by the camera at the given base pose."""
    T_w_c = T_w_base @ T_base_cam
    R_cw = T_w_c[:3, :3].T
    C = T_w_c[:3, 3]
    # Homography ground (X, Y on z=GROUND_Z) -> image pixel.
    cols = np.column_stack([R_cw[:, 0], R_cw[:, 1], R_cw @ (np.array([0.0, 0.0, GROUND_Z]) - C)])
    H_img_ground = K @ cols
    # Texture px -> ground metres (texture centred on the world origin).
    T_tex = np.array([[TEX_RES, 0.0, -TEX_SIZE / 2 * TEX_RES], [0.0, TEX_RES, -TEX_SIZE / 2 * TEX_RES], [0, 0, 1.0]])
    # Sample the texture at (H_img_ground @ T_tex)^-1 of every image pixel
    # through fixed-point remap tables built in numpy: cv2.warpPerspective's
    # bilinear kernel differs between OpenCV 4.10 and 5.0 (up to 2 grey
    # levels), which made the synthetic frames themselves build-dependent.
    xy, frac, _ = _remap_tables(np.linalg.inv(H_img_ground @ T_tex), H, W, (TEX_SIZE, TEX_SIZE), nearest=False)
    img = cv2.remap(TEXTURE, xy, frac, cv2.INTER_LINEAR)
    # Pixels whose ray does not hit the plane in front of the camera are sky.
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    rays = np.stack([uu, vv, np.ones_like(uu)], axis=-1) @ np.linalg.inv(K).T @ T_w_c[:3, :3].T
    lam = (GROUND_Z - C[2]) / np.where(np.abs(rays[..., 2]) < 1e-9, -1e-9, rays[..., 2])
    img[lam <= 0.5] = 128
    return img


def write_frames(tmp_path: Path, poses: list[np.ndarray], stamps: list[float], T_base_cam=None) -> list[CameraFrame]:
    T_base_cam = nominal_mount() if T_base_cam is None else T_base_cam
    frames = []
    for i, (T, t) in enumerate(zip(poses, stamps)):
        path = tmp_path / f"f{i:03d}.png"
        cv2.imwrite(str(path), render(T, T_base_cam))
        frames.append(CameraFrame(t, str(path), "test"))
    return frames


def make_vo(**kwargs) -> VisualOdometry:
    return VisualOdometry("test", K, (W, H), nominal_mount(), ground_z=GROUND_Z, **kwargs)


# --------------------------------------------------------------------- basics
def test_channel_and_first_frame(tmp_path):
    vo = make_vo()
    assert vo.channel == "vo_test"
    frames = write_frames(tmp_path, [np.eye(4)], [0.0])
    assert vo.process(frames[0]) is None  # nothing to relate the frame to


def test_forward_step_recovered(tmp_path):
    step = 0.4
    frames = write_frames(tmp_path, [np.eye(4), se3_of_se2(step, 0.0, 0.0)], [0.0, 0.25])
    vo = make_vo()
    motions = run_visual_odometry(vo, frames)
    assert len(motions) == 1
    m = motions[0]
    assert m.channel == "vo_test"
    assert m.start_stamp == 0.0 and m.stamp == 0.25
    assert m.delta_forward == pytest.approx(step, abs=0.06)
    assert abs(m.delta_left) < 0.05
    assert abs(m.delta_yaw) < 0.02
    assert 0.0 < m.quality <= 1.0


def test_turn_step_recovered(tmp_path):
    dx, dy, dyaw = 0.3, 0.04, 0.12
    frames = write_frames(tmp_path, [np.eye(4), se3_of_se2(dx, dy, dyaw)], [0.0, 0.3])
    m = run_visual_odometry(make_vo(), frames)[0]
    assert m.delta_yaw == pytest.approx(dyaw, abs=0.02)
    assert m.delta_forward == pytest.approx(dx, abs=0.06)
    assert m.delta_left == pytest.approx(dy, abs=0.06)


def test_stationary_pair_is_zero_with_tight_sigma(tmp_path):
    frames = write_frames(tmp_path, [np.eye(4), np.eye(4)], [0.0, 0.3])
    m = run_visual_odometry(make_vo(), frames)[0]
    assert m.delta_forward == 0.0 and m.delta_left == 0.0 and m.delta_yaw == 0.0
    assert m.sigma_position <= 0.02
    assert m.quality >= 0.5


def test_featureless_pair_yields_nothing(tmp_path):
    for i in range(2):
        img = np.full((H, W), 128, np.uint8)  # no texture at all
        cv2.imwrite(str(tmp_path / f"g{i}.png"), img)
    frames = [CameraFrame(0.3 * i, str(tmp_path / f"g{i}.png"), "test") for i in range(2)]
    vo = make_vo()
    assert run_visual_odometry(vo, frames) == []
    assert vo.diagnostics[-1].status in ("too_few_matches", "pose_failed")


def test_long_gap_is_skipped(tmp_path):
    frames = write_frames(tmp_path, [np.eye(4), se3_of_se2(0.4, 0, 0)], [0.0, 10.0])
    vo = make_vo()
    assert run_visual_odometry(vo, frames) == []
    assert vo.diagnostics[-1].status == "gap"


def test_near_duplicate_frames_are_merged(tmp_path):
    poses = [np.eye(4), se3_of_se2(0.01, 0, 0), se3_of_se2(0.4, 0, 0)]
    frames = write_frames(tmp_path, poses, [0.0, 0.02, 0.3])
    vo = make_vo()
    motions = run_visual_odometry(vo, frames)
    # The 20 ms frame must not create its own increment; the emitted one
    # spans the whole interval from the first frame.
    assert [d.status for d in vo.diagnostics][0] == "merged"
    assert len(motions) == 1
    assert motions[0].start_stamp == 0.0 and motions[0].stamp == 0.3


def test_scale_comes_from_camera_height(tmp_path):
    """Doubling the assumed camera height must double the recovered step."""
    step = 0.4
    frames = write_frames(tmp_path, [np.eye(4), se3_of_se2(step, 0, 0)], [0.0, 0.25])
    tall = VisualOdometry("test", K, (W, H), nominal_mount(), ground_z=GROUND_Z - 0.562, config=VoConfig())
    m = run_visual_odometry(tall, frames)[0]
    assert m.delta_forward == pytest.approx(2 * step, rel=0.15)


# ----------------------------------------------------------------- imu prior
def test_yaw_prior_from_imu_integrates_rate():
    stamps = np.linspace(0.0, 10.0, 1001)
    prior = yaw_prior_from_imu(stamps, np.full(1001, 0.2))
    assert prior(2.0, 3.5) == pytest.approx(0.3, abs=1e-6)
    assert prior(3.5, 2.0) == pytest.approx(-0.3, abs=1e-6)


def test_bad_vision_yaw_is_replaced_by_prior(tmp_path):
    dyaw = 0.12
    frames = write_frames(tmp_path, [np.eye(4), se3_of_se2(0.3, 0.0, dyaw)], [0.0, 0.3])
    # A prior that contradicts the vision yaw wins the gate and is used.
    vo = make_vo(yaw_prior=lambda t0, t1: 0.5)
    m = run_visual_odometry(vo, frames)[0]
    assert m.delta_yaw == pytest.approx(0.5, abs=1e-9)
    assert vo.diagnostics[-1].yaw_from_prior
    # An agreeing prior leaves the (more precise) vision yaw in place.
    vo2 = make_vo(yaw_prior=lambda t0, t1: dyaw)
    m2 = run_visual_odometry(vo2, write_frames(tmp_path, [np.eye(4), se3_of_se2(0.3, 0.0, dyaw)], [10.0, 10.3]))[0]
    assert m2.delta_yaw == pytest.approx(dyaw, abs=0.02)
    assert not vo2.diagnostics[-1].yaw_from_prior


# ------------------------------------------------------- pitch self-calibration
def test_mount_pitch_self_calibrates(tmp_path):
    """A camera secretly pitched must be discovered and the scale corrected."""
    pitch_err = np.radians(2.5)
    # The real camera looks *down* by pitch_err relative to the nominal mount.
    true_mount = make_transform(rpy_matrix(0.0, pitch_err, 0.0), [0, 0, 0]) @ nominal_mount()
    true_mount[:3, 3] = CAM_XYZ
    poses, stamps = [], []
    for i in range(16):
        poses.append(se3_of_se2(0.35 * i, 0.0, 0.0))
        stamps.append(0.25 * i)
    frames = write_frames(tmp_path, poses, stamps, T_base_cam=true_mount)
    vo = make_vo()  # believes the nominal mount
    motions = run_visual_odometry(vo, frames)
    assert vo.calibrated
    assert np.degrees(abs(vo.mount_pitch)) == pytest.approx(2.5, abs=1.2)
    # A 2.5 deg pitch error scales the ground ranges by ~17 %/deg, so the
    # uncalibrated head of the run overestimates badly; once the running
    # median has converged the residual bias must be far smaller (the last
    # ~1 deg of estimator bias is what the per-camera height calibration in
    # scripts/validate_vo.py absorbs on the real rig).
    early = [m for m in motions if m.stamp <= stamps[6]]
    late = [m for m in motions if m.stamp > stamps[10]]
    assert late, "expected increments after calibration warm-up"
    err_early = np.mean([abs(m.delta_forward - 0.35) for m in early])
    err_late = np.mean([abs(m.delta_forward - 0.35) for m in late])
    assert err_late < err_early
    for m in late:
        assert m.delta_forward == pytest.approx(0.35, rel=0.22)


def test_pitch_can_be_pinned(tmp_path):
    cfg = VoConfig(mount_pitch_init=np.radians(-2.0), self_calibrate_pitch=False)
    vo = make_vo(config=cfg)
    assert vo.mount_pitch == pytest.approx(np.radians(-2.0))
    frames = write_frames(tmp_path, [np.eye(4), se3_of_se2(0.4, 0, 0)], [0.0, 0.25])
    run_visual_odometry(vo, frames)
    assert vo.mount_pitch == pytest.approx(np.radians(-2.0))  # unchanged


# ------------------------------------------------------------------ kart mask
def test_default_mask_mirrors_with_mount_side():
    right = _default_kart_mask((W, H), kart_on_left=True)
    left = _default_kart_mask((W, H), kart_on_left=False)
    assert right.shape == (H, W)
    # Kart body region blocked on the correct side.
    assert right[int(0.8 * H), int(0.1 * W)] == 0
    assert right[int(0.8 * H), int(0.9 * W)] == 255
    assert left[int(0.8 * H), int(0.9 * W)] == 0
    assert left[int(0.8 * H), int(0.1 * W)] == 255
    # Bottom strip always blocked, sky never.
    assert right[H - 1].max() == 0
    assert right[0].min() == 255


def test_masked_kart_region_ignored(tmp_path):
    """A big static blob where the kart sits must not drag the estimate."""
    step = 0.4
    frames = []
    for i, T in enumerate([np.eye(4), se3_of_se2(step, 0, 0)]):
        img = render(T, nominal_mount())
        cv2.circle(img, (int(0.2 * W), int(0.85 * H)), 60, 30, -1)
        cv2.circle(img, (int(0.3 * W), int(0.9 * H)), 25, 220, -1)
        path = tmp_path / f"m{i}.png"
        cv2.imwrite(str(path), img)
        frames.append(CameraFrame(0.25 * i, str(path), "test"))
    m = run_visual_odometry(make_vo(), frames)[0]
    assert m.delta_forward == pytest.approx(step, abs=0.06)


# ------------------------------------------- portable pose + IPM (OpenCV 4/5)
def _tracks(vo: VisualOdometry, pose2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """LK tracks between the origin view and ``pose2`` of the synthetic ground."""
    g0 = vo._clahe.apply(render(np.eye(4), nominal_mount()))
    g1 = vo._clahe.apply(render(pose2, nominal_mount()))
    return vo._track_lk(g0, g1, 0.25, None)


def test_pose_independent_of_match_order():
    """ORB returns the same matches in a platform-dependent order; the pose must not care."""
    vo = make_vo()
    p1, p2 = _tracks(vo, se3_of_se2(0.4, 0.0, 0.0))
    R, t, inlier = vo._recover_pose(p1, p2)
    perm = np.random.default_rng(3).permutation(len(p1))
    Rp, tp, inlier_p = vo._recover_pose(p1[perm], p2[perm])
    assert np.array_equal(R, Rp) and np.array_equal(t, tp)
    assert np.array_equal(inlier_p, inlier[perm])  # mask follows the caller's order


def test_refined_pose_does_not_depend_on_the_ransac_sample():
    """Root cause of the OpenCV 4.10 / 5.0 split in the tests above.

    RANSAC keeps one 5-point sample's model, unrefined; which sample wins
    changes with any input perturbation (here: the order it sees the
    matches in), and those models scatter by tens of mrad. The polished
    pose is a fit to all the data and must not depend on the sample.
    """
    dyaw = 0.12
    p1, p2 = _tracks(make_vo(), se3_of_se2(0.3, 0.04, dyaw))
    rng = np.random.default_rng(0)
    perms = [rng.permutation(len(p1)) for _ in range(8)]
    raw = make_vo(config=VoConfig(pose_refine=False))
    vo = make_vo()
    raw_yaw = [raw._base_yaw(raw._solve_pose(p1[q], p2[q])[0]) for q in perms]
    yaw = [vo._base_yaw(vo._solve_pose(p1[q], p2[q])[0]) for q in perms]
    assert np.ptp(raw_yaw) > 5e-3  # the minimal-sample scatter being fixed
    assert np.ptp(yaw) < 1e-4
    assert np.median(yaw) == pytest.approx(dyaw, abs=0.004)


def test_sampson_jacobian_matches_finite_differences():
    vo = make_vo()
    rng = np.random.default_rng(1)
    y1 = np.column_stack([rng.uniform(-0.8, 0.8, (50, 2)), np.ones(50)])
    y2 = np.column_stack([y1[:, :2] + rng.normal(0.0, 0.02, (50, 2)), np.ones(50)])
    R = cv2.Rodrigues(np.array([0.01, -0.02, 0.005]))[0]
    t = np.array([0.1, 0.05, -1.0]) / np.linalg.norm([0.1, 0.05, -1.0])
    B = vo._tangent_basis(t)
    J = vo._sampson_jacobian(R, t, B, y1, y2, vo._sampson(_skew(t) @ R, y1, y2))

    def resid(d):
        Rd, td = vo._perturb(R, t, d, B)
        return vo._sampson(_skew(td) @ Rd, y1, y2)[0]

    numeric = np.column_stack([(resid(e * 1e-7) - resid(-e * 1e-7)) / 2e-7 for e in np.eye(5)])
    assert np.abs(J - numeric).max() < 1e-6 * np.abs(numeric).max()


def test_in_front_is_the_recover_pose_cheirality():
    vo = make_vo()
    p1, p2 = _tracks(vo, se3_of_se2(0.4, 0.0, 0.0))
    E, mask = cv2.findEssentialMat(p1, p2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0, maxIters=1000)
    _, R, t, pose_mask = cv2.recoverPose(E, p1, p2, K, mask=mask.copy())
    y1 = np.column_stack([p1, np.ones(len(p1))]) @ np.linalg.inv(K).T
    y2 = np.column_stack([p2, np.ones(len(p2))]) @ np.linalg.inv(K).T
    ours = vo._in_front(R, t.ravel(), y1, y2) & (mask.ravel() > 0)
    assert (ours == (pose_mask.ravel() > 0)).mean() > 0.99
    assert vo._in_front(R, -t.ravel(), y1, y2).mean() < 0.05  # the wrong twin is behind


def test_remap_tables_reproduce_warp_perspective():
    """The portable IPM warp is OpenCV 4's warpPerspective, on every OpenCV.

    OpenCV 4.x quantises source coordinates to 1/32 px, exactly like the
    tables, so there the two are bit-identical; 5.x samples at float
    coordinates, which moves high-gradient pixels by a few grey levels.
    """
    vo = make_vo()
    image = vo._clahe.apply(render(np.eye(4), nominal_mount()))
    cfg = vo.config
    rows = int(round((cfg.ipm_search_fwd[1] - cfg.ipm_search_fwd[0]) / cfg.ipm_res))
    cols = int(round(2 * cfg.ipm_search_half / cfg.ipm_res))
    Hm = vo._ipm_homography(cfg.ipm_search_fwd[1], cfg.ipm_search_half, 0.03)
    xy, frac, near = _remap_tables(Hm, rows, cols, (W, H))
    ours = cv2.remap(image, xy, frac, cv2.INTER_LINEAR).astype(np.int16)
    ref = cv2.warpPerspective(image, Hm, (cols, rows), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    diff = np.abs(ours - ref.astype(np.int16))
    if cv2.__version__.startswith("4."):
        assert diff.max() == 0
    else:  # measured on 5.0 with this sharp-dotted texture: max 4, mean 0.47
        assert diff.max() <= 6 and np.percentile(diff, 99) <= 3 and diff.mean() < 1.0
    mask_ours = cv2.remap(vo.mask, near, None, cv2.INTER_NEAREST)
    mask_ref = cv2.warpPerspective(vo.mask, Hm, (cols, rows), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_NEAREST)
    if cv2.__version__.startswith("4."):
        assert np.array_equal(mask_ours, mask_ref)
    else:
        assert (mask_ours == mask_ref).mean() > 0.999


def test_ipm_view_is_the_same_on_every_warp_backend():
    """The native warp is used only where it equals the portable tables."""
    image = make_vo()._clahe.apply(render(np.eye(4), nominal_mount()))
    cfg = VoConfig()
    args = (image, cfg.ipm_search_fwd, cfg.ipm_search_half, 0.02)
    auto = make_vo(config=VoConfig(ipm_warp="auto"))._ipm_view(*args)
    tables = make_vo(config=VoConfig(ipm_warp="tables"))._ipm_view(*args)
    assert np.array_equal(auto[0], tables[0]) and np.array_equal(auto[1], tables[1])


# ---------------------------------------------------- lens distortion (req 6)
def _distort_frames(tmp_path: Path, frames, D):
    """Re-render the synthetic frames as a DISTORTED camera would see them:
    inject plumb_bob distortion by sampling each rectified frame at the
    undistorted position of every output pixel, cropped to the valid
    interior box with the principal point shifted accordingly (exactly what
    a real distorted sensor of that FOV would deliver + its CameraInfo)."""
    xs, ys = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    pts = np.stack([xs.ravel(), ys.ravel()], 1).astype(np.float64).reshape(-1, 1, 2)
    und = cv2.undistortPoints(pts, K, D, P=K).reshape(H, W, 2)
    valid = (und[..., 0] >= 0) & (und[..., 0] <= W - 1) & (und[..., 1] >= 0) & (und[..., 1] <= H - 1)
    cx, cy = W // 2, H // 2
    for f in np.arange(0.98, 0.4, -0.02):
        hw, hh = int(f * W / 2), int(f * H / 2)
        if valid[cy - hh : cy + hh, cx - hw : cx + hw].all():
            x0, x1, y0, y1 = cx - hw, cx + hw, cy - hh, cy + hh
            break
    mx = und[y0:y1, x0:x1, 0].astype(np.float32)
    my = und[y0:y1, x0:x1, 1].astype(np.float32)
    Kc = K.copy()
    Kc[0, 2] -= x0
    Kc[1, 2] -= y0
    out = []
    for i, fr in enumerate(frames):
        img = cv2.imread(fr.image_path, cv2.IMREAD_GRAYSCALE)
        dimg = cv2.remap(img, mx, my, cv2.INTER_LINEAR)
        path = tmp_path / f"d{i:03d}.png"
        cv2.imwrite(str(path), dimg)
        out.append(CameraFrame(fr.stamp, str(path), "test"))
    return out, Kc, (x1 - x0, y1 - y0)


def test_distort_pixels_inverts_undistort_pixels():
    from kart_map2d.camera.model import CameraModel

    D = np.array([-0.28, 0.08, 0.0005, -0.0005, 0.0])
    model = CameraModel(K, (W, H), D, "plumb_bob")
    pts = np.array([[100.0, 100.0], [480.0, 300.0], [900.0, 560.0], [60.0, 520.0]])
    round_trip = model.undistort_pixels(model.distort_pixels(pts))
    assert np.abs(round_trip - pts).max() < 0.2


def test_forward_scale_survives_lens_distortion(tmp_path):
    """The requirement-6/req-3 regression: a plumb_bob-distorted stream with
    the matching CameraInfo must recover the SAME metric step as the
    rectified baseline. Before the distortion-aware IPM sampling the scale
    came out ~13% low on real data (ATE +47% live) while NCC stayed
    confident; this pins the fixed behaviour on synthetic ground truth."""
    from kart_map2d.camera.model import CameraModel

    step = 0.5
    poses = [np.eye(4), se3_of_se2(step, 0.0, 0.0)]
    frames = write_frames(tmp_path, poses, [0.0, 0.25])
    D = np.array([-0.28, 0.08, 0.0005, -0.0005, 0.0])
    dframes, Kc, sz = _distort_frames(tmp_path, frames, D)
    vo = VisualOdometry(
        "test", Kc, sz, nominal_mount(), ground_z=GROUND_Z,
        camera_model=CameraModel(Kc, sz, D, "plumb_bob"),
    )
    motions = run_visual_odometry(vo, dframes)
    assert len(motions) == 1, f"pair not solved: {vo.diagnostics[-1] if vo.diagnostics else 'no diag'}"
    m = motions[0]
    assert m.delta_forward == pytest.approx(step, rel=0.10), (
        f"distorted-stream scale off: {m.delta_forward:.3f} vs {step} "
        f"(source {vo.diagnostics[-1].scale_source})"
    )
    assert abs(m.delta_left) < 0.08
    assert abs(m.delta_yaw) < 0.03
