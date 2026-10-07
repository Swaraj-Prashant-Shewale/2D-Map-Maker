"""Tests for kart_map2d.perception.classical_ground (synthetic scenes only)."""

from __future__ import annotations

import importlib.util
import json

import cv2
import numpy as np
import pytest

from kart_map2d.perception import segmentation as seg
from kart_map2d.perception.classical_ground import (
    KART_ID,
    KART_MASK_LEARN_VERSION,
    ClassicalGroundSegmenter,
    ClassicalSegmentationResult,
    kart_body_mask,
    learn_kart_mask,
    load_or_learn_kart_mask,
)

H, W = 300, 480  # test resolution; module defaults are resolution-relative

# Scene palette (BGR): grey pavement, blue sky, saturated hedge green.
GROUND = (150, 155, 160)
SKY = (235, 190, 120)
HEDGE = (40, 130, 30)


def _scene(rng: np.random.Generator, obstacle: bool = True) -> np.ndarray:
    """Synthetic frame: sky above the horizon, noisy pavement below, and an
    optional hedge block standing on the ground on the right."""
    img = np.empty((H, W, 3), dtype=np.float32)
    horizon = int(0.483 * H)
    img[:horizon] = SKY
    img[horizon:] = GROUND
    img += rng.normal(0.0, 2.0, img.shape)
    if obstacle:
        img[int(0.52 * H) : int(0.85 * H), int(0.78 * W) :] = HEDGE
    return np.clip(img, 0, 255).astype(np.uint8)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = int(np.count_nonzero(a | b))
    return np.count_nonzero(a & b) / union if union else 1.0


# ------------------------------------------------------------- kart mask


def test_kart_body_mask_matches_neural_default() -> None:
    # The classical module keeps its own torch-free copy; pin them together.
    # Since the 2026-09-30 classical-corridor fix these geometric masks are
    # the shared FALLBACK only: the production pipeline learns a per-camera
    # kart mask from temporal stillness (load_or_learn_kart_mask) and hands
    # the SAME learned mask to both backends. The pin stays deliberately -
    # whenever learning is unavailable (live node today, synthetic tests,
    # brand-new camera before its first drive) both backends must still
    # agree on the fallback pixel-for-pixel, or their cached artifacts stop
    # being comparable.
    np.testing.assert_array_equal(kart_body_mask(600, 960), seg.default_kart_mask(600, 960))
    np.testing.assert_array_equal(kart_body_mask(H, W), seg.default_kart_mask(H, W))


# ---------------------------------------------------------------- result


def test_result_partition_class_map_and_fraction() -> None:
    drv = np.zeros((4, 4), dtype=bool)
    obs = np.zeros((4, 4), dtype=bool)
    neu = np.zeros((4, 4), dtype=bool)
    kart = np.zeros((4, 4), dtype=bool)
    drv[2:], obs[1], neu[0], kart[3, 3] = True, True, True, True
    drv[3, 3] = False
    result = ClassicalSegmentationResult(drv, obs, neu, kart, quality=0.8)
    cmap = result.class_map
    assert cmap.dtype == np.uint8
    assert cmap[3, 0] == 0 and cmap[1, 0] == 2 and cmap[0, 0] == 10 and cmap[3, 3] == KART_ID
    assert result.drivable_fraction == pytest.approx(7 / 15)
    # The synthesised class map reproduces the masks through the neural
    # result type, so both backends are interchangeable downstream.
    twin = seg.SegmentationResult(cmap)
    np.testing.assert_array_equal(twin.drivable_mask, drv)
    np.testing.assert_array_equal(twin.obstacle_mask, obs)
    np.testing.assert_array_equal(twin.neutral_mask, neu)
    np.testing.assert_array_equal(twin.kart_mask, kart)


# -------------------------------------------------------------- segment


def test_segment_synthetic_scene_categories() -> None:
    rng = np.random.default_rng(7)
    segmenter = ClassicalGroundSegmenter()
    result = segmenter.segment(_scene(rng))

    assert result.quality == 1.0
    # Masks partition the frame.
    total = (
        result.drivable_mask.astype(int)
        + result.obstacle_mask.astype(int)
        + result.neutral_mask.astype(int)
        + result.kart_mask.astype(int)
    )
    assert (total == 1).all()
    np.testing.assert_array_equal(result.kart_mask, kart_body_mask(H, W))

    # Open pavement ahead is drivable; the hedge and the sky are not.
    assert result.drivable_mask[int(0.7 * H), int(0.45 * W)]
    assert result.drivable_mask[int(0.55 * H), int(0.3 * W)]
    assert not result.drivable_mask[int(0.7 * H), int(0.9 * W)]  # hedge
    # The obstacle assert sits at 0.6 H: rows within kart_ring_px above the
    # kart mask are neutral by design (the near band moves with the camera
    # and must never emit contacts), and at 0.7 H / 0.9 W the corner
    # triangle's ring reaches the hedge.
    assert result.obstacle_mask[int(0.6 * H), int(0.9 * W)]
    assert result.neutral_mask[int(0.2 * H), int(0.5 * W)]  # sky
    assert not result.obstacle_mask[: int(0.4 * H)].any()  # above horizon

    # Bulk coverage: most of the visible pavement is recovered.
    ground = np.zeros((H, W), dtype=bool)
    ground[int(0.52 * H) :, : int(0.75 * W)] = True
    ground &= ~result.kart_mask
    assert (result.drivable_mask & ground).sum() / ground.sum() > 0.85

    assert segmenter.frames_processed == 1
    assert segmenter.ms_per_frame > 0.0


def test_exposure_change_between_frames_is_normalised() -> None:
    rng = np.random.default_rng(11)
    base = _scene(rng)
    darker = np.clip(base.astype(np.float32) * 0.55, 0, 255).astype(np.uint8)

    segmenter = ClassicalGroundSegmenter()
    first = segmenter.segment(base)
    second = segmenter.segment(darker)  # auto-exposure dropped the gain
    assert second.quality == 1.0
    assert _iou(first.drivable_mask, second.drivable_mask) > 0.9


def test_soft_shadow_stays_drivable() -> None:
    rng = np.random.default_rng(13)
    img = _scene(rng).astype(np.float32)
    # Cast shadow with a penumbra: darken a band of the pavement to 45%,
    # feathering the edges over ~21 rows (a hard step is indistinguishable
    # from an obstacle contact edge and is deliberately kept out). The band
    # lies below the seed trapezoid: the model stays sunlit and must still
    # accept the shadow (a shadow swallowing the seed itself is the
    # contamination case).
    gain = np.ones((H, 1), dtype=np.float32)
    top, bottom = int(0.70 * H), int(0.84 * H)
    gain[top:bottom] = 0.45
    kernel = np.ones(21, dtype=np.float32) / 21.0
    gain[:, 0] = np.convolve(gain[:, 0], kernel, mode="same")
    shadowed = np.clip(img * gain[:, :, None], 0, 255).astype(np.uint8)

    result = ClassicalGroundSegmenter().segment(shadowed)
    band = np.zeros((H, W), dtype=bool)
    band[top + 4 : bottom - 4, int(0.15 * W) : int(0.7 * W)] = True
    band &= ~result.kart_mask
    assert (result.drivable_mask & band).sum() / band.sum() > 0.7
    # Sunlit ground above the shadow stays drivable too.
    assert result.drivable_mask[int(0.60 * H), int(0.4 * W)]


def test_contaminated_seed_falls_back_to_previous_model() -> None:
    rng = np.random.default_rng(17)
    segmenter = ClassicalGroundSegmenter()
    clean = segmenter.segment(_scene(rng))
    assert clean.quality == 1.0

    # An obstacle parked directly ahead: high-variance clutter over the
    # whole seed trapezoid.
    blocked = _scene(rng)
    r0, r1 = int(0.55 * H), int(0.75 * H)
    c0, c1 = int(0.15 * W), int(0.7 * W)
    blocked[r0:r1, c0:c1] = rng.integers(0, 256, (r1 - r0, c1 - c0, 3), dtype=np.uint8)

    result = segmenter.segment(blocked)
    assert result.quality == pytest.approx(0.35)
    # Judged by the remembered ground model, the clutter is an obstacle.
    assert result.obstacle_mask[int(0.65 * H), int(0.4 * W)]
    assert not result.drivable_mask[int(0.65 * H), int(0.4 * W)]


def test_contaminated_seed_without_history_has_low_quality() -> None:
    rng = np.random.default_rng(19)
    blocked = _scene(rng)
    blocked[int(0.55 * H) : int(0.75 * H), int(0.15 * W) : int(0.7 * W)] = rng.integers(
        0, 256, (int(0.75 * H) - int(0.55 * H), int(0.7 * W) - int(0.15 * W), 3), dtype=np.uint8
    )
    result = ClassicalGroundSegmenter().segment(blocked)
    assert result.quality == pytest.approx(0.2)


def test_geometry_from_intrinsics_sets_horizon() -> None:
    # Camera 0.5 m above ground, optical axis pitched 10 deg down, looking
    # along base x: horizon = cy - fy * tan(10 deg).
    pitch = np.deg2rad(10.0)
    fy, cy = 300.0, 150.0
    K = np.array([[300.0, 0.0, 240.0], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    # camera <- base rotation: base forward -> (0, -sin p, cos p) in the
    # optical frame (x right, y down, z forward), base left -> -x, base up
    # -> (0, -cos p, -sin p); the mount stores base <- camera.
    R_cb = np.array(
        [
            [0.0, -1.0, 0.0],
            [-np.sin(pitch), 0.0, -np.cos(pitch)],
            [np.cos(pitch), 0.0, -np.sin(pitch)],
        ]
    )
    T = np.eye(4)
    T[:3, :3] = R_cb.T
    T[2, 3] = 0.40  # ground_z = -0.10 -> camera 0.5 m above ground
    segmenter = ClassicalGroundSegmenter(K=K, T_base_cam=T)
    expected = cy - fy * np.tan(pitch)
    assert segmenter._horizon_row_px(H, W) == pytest.approx(expected, abs=1e-6)

    result = segmenter.segment(_scene(np.random.default_rng(23)))
    pad = segmenter.horizon_pad_px * H / 600.0
    band_top = int(round(expected - pad))
    assert result.neutral_mask[: band_top - 1].all()
    # Below the band, neutral exists ONLY in the ring above the kart mask
    # (unverifiable pixels demoted from obstacle; see kart_ring_px).
    below_neutral = result.neutral_mask.copy()
    below_neutral[: band_top + 2] = False
    ring = segmenter._geometry(H, W).ring_full
    assert not (below_neutral & ~ring).any()


def test_segmenter_handles_odd_resolutions() -> None:
    rng = np.random.default_rng(29)
    img = np.clip(
        np.full((131, 173, 3), GROUND, dtype=np.float32) + rng.normal(0, 2, (131, 173, 3)),
        0,
        255,
    ).astype(np.uint8)
    result = ClassicalGroundSegmenter().segment(img)
    assert result.drivable_mask.shape == (131, 173)


# --------------------------------------------------------------- factory
#: Constructing the neural backend imports torch; a torch-free install skips.
needs_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None or importlib.util.find_spec("transformers") is None,
    reason="the neural backend needs torch + transformers",
)


@needs_torch
def test_factory_returns_requested_backend() -> None:
    classical = seg.get_segmenter("classical")
    assert isinstance(classical, ClassicalGroundSegmenter)
    neural = seg.get_segmenter("neural")  # lazy: no model load on construction
    assert isinstance(neural, seg.Segmenter)
    with pytest.raises(ValueError):
        seg.get_segmenter("quantum")


@needs_torch
def test_factory_auto_mode_follows_weight_availability(monkeypatch) -> None:
    monkeypatch.setattr(seg, "neural_weights_available", lambda: False)
    assert isinstance(seg.get_segmenter("auto"), ClassicalGroundSegmenter)
    monkeypatch.setattr(seg, "neural_weights_available", lambda: True)
    assert isinstance(seg.get_segmenter("auto"), seg.Segmenter)


def test_factory_forwards_kwargs() -> None:
    segmenter = seg.get_segmenter("classical", work_width=160, color_tau=5.0)
    assert segmenter.work_width == 160 and segmenter.color_tau == 5.0


# ------------------------------------------------------- kart mask learning
LH, LW = 240, 320  # learner test resolution


def _moving_sequence(n: int = 20, kart: bool = True, seed: int = 3):
    """Synthetic drive: textured ground scrolls towards the camera, uniform
    sky stands still, and (optionally) a static dark kart blob is anchored
    at the bottom edge with a protrusion above the geometric mask."""
    rng = np.random.default_rng(seed)
    horizon = int(0.45 * LH)
    texture = (rng.integers(90, 200, (4 * LH, LW))).astype(np.uint8)
    frames, drivables, obstacles = [], [], []
    geo = kart_body_mask(LH, LW)
    blob = np.zeros((LH, LW), dtype=bool)
    blob[int(0.62 * LH) :, int(0.38 * LW) : int(0.62 * LW)] = True  # protrudes above geo
    for i in range(n):
        img = np.empty((LH, LW, 3), dtype=np.uint8)
        img[:horizon] = (235, 190, 120)  # constant sky
        scroll = texture[3 * LH - 14 * i : 3 * LH - 14 * i + (LH - horizon)]
        img[horizon:] = scroll[..., None]
        if kart:
            img[blob] = (35, 30, 40)  # static dark body, bottom-anchored
        frames.append(img)
        drv = np.zeros((LH, LW), dtype=bool)
        drv[horizon:] = True
        drv[blob | geo] = False
        drivables.append(drv)
        obs = np.zeros((LH, LW), dtype=bool)
        obs[blob & ~geo] = True  # a per-frame classifier calls the blob obstacle
        obstacles.append(obs)
    return frames, drivables, obstacles, blob


def test_learn_kart_mask_finds_static_body_not_sky_or_ground() -> None:
    frames, drivables, obstacles, blob = _moving_sequence()
    mask = learn_kart_mask(frames, drivables, obstacles)
    assert mask is not None and mask.shape == (LH, LW)
    core = cv2.erode(blob.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    assert (mask & core).sum() / core.sum() > 0.9  # body found (incl. protrusion)
    assert not mask[: int(0.3 * LH)].any()  # constant sky is NOT body (anchoring)
    ground = np.zeros((LH, LW), dtype=bool)
    ground[int(0.5 * LH) : int(0.9 * LH), : int(0.25 * LW)] = True
    assert (mask & ground).sum() / ground.sum() < 0.05  # moving ground stays scene


def test_learn_kart_mask_needs_scene_motion() -> None:
    frames, drivables, obstacles, _ = _moving_sequence(n=20)
    static = [frames[0]] * 20  # parked kart: nothing moves
    assert learn_kart_mask(static, [drivables[0]] * 20, [obstacles[0]] * 20) is None


def test_learn_kart_mask_refuses_implausibly_large_mask() -> None:
    # Motion confined to the top strip while 2/3 of the image is a still,
    # bottom-anchored dark region: "kart" would cover most of the scene.
    rng = np.random.default_rng(11)
    frames = []
    for _ in range(20):
        img = np.full((LH, LW, 3), 40, dtype=np.uint8)
        img[: int(0.3 * LH)] = rng.integers(0, 255, (int(0.3 * LH), LW, 3), dtype=np.uint8)
        frames.append(img)
    assert learn_kart_mask(frames) is None


def test_load_or_learn_kart_mask_caches_and_invalidates(tmp_path, monkeypatch) -> None:
    from kart_map2d.core.measurements import CameraFrame
    import kart_map2d.perception.classical_ground as cg

    frames_img, _, _, _ = _moving_sequence()
    cam_frames = []
    for i, img in enumerate(frames_img):
        p = tmp_path / f"{i:06d}.jpg"
        cv2.imwrite(str(p), img)
        cam_frames.append(CameraFrame(float(i), str(p), "camX"))

    calls = {"n": 0}
    real = cg.learn_kart_mask

    def counting(*a, **kw):
        calls["n"] += 1
        return real(*a, **kw)

    monkeypatch.setattr(cg, "learn_kart_mask", counting)
    root = tmp_path / "cache"
    mask1, meta1 = load_or_learn_kart_mask(cam_frames, "runX", "camX", cache_root=root)
    assert calls["n"] == 1 and meta1["learned"] is True
    assert mask1 is not None
    # union with the geometric fallback is part of the contract
    assert ((mask1 | kart_body_mask(LH, LW)) == mask1).all()
    assert (root / "runX" / "camX.png").exists()

    mask2, meta2 = load_or_learn_kart_mask(cam_frames, "runX", "camX", cache_root=root)
    assert calls["n"] == 1  # served from cache
    np.testing.assert_array_equal(mask1, mask2)

    # A version bump (algorithm change) invalidates the cached mask.
    meta_path = root / "runX" / "camX.json"
    doc = json.loads(meta_path.read_text(encoding="ascii"))
    doc["version"] = KART_MASK_LEARN_VERSION - 1
    meta_path.write_text(json.dumps(doc), encoding="ascii")
    load_or_learn_kart_mask(cam_frames, "runX", "camX", cache_root=root)
    assert calls["n"] == 2


# ------------------------------------------- protrusion / contact regression
def _pinhole_camera():
    """The synthetic kart camera from test_evidence_grid: 0.56 m above ground."""
    K = np.array([[60.0, 0.0, 32.0], [0.0, 60.0, 24.0], [0.0, 0.0, 1.0]])
    T = np.eye(4)
    T[:3, :3] = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    T[:3, 3] = (0.30, 0.0, 0.46)
    return K, T


def test_kart_protrusion_never_contacts_but_real_obstacle_does() -> None:
    """THE classical-corridor regression (2026-09-30): a kart-body part
    poking above the geometric mask must not create obstacle contacts near
    the kart, while a real obstacle ahead still does.

    Grid-level: the protrusion's pixels label 'obstacle' (a segmenter that
    knows no better), their rays land inside the kart footprint / below the
    minimum contact range, and insert_camera must refuse them; the wall at
    5.9 m must map. The grid is built with the SAME guard values the
    offline pipeline passes (they are off by default: a top-down camera
    may legitimately see closer contacts).
    """
    from kart_map2d.mapping.evidence_grid import CLASS_OBSTACLE, EvidenceGrid
    from kart_map2d.pipeline import CAMERA_MIN_CONTACT_RANGE_M, KART_FOOTPRINT_M

    K, T = _pinhole_camera()
    h, w = 48, 64
    drivable = np.zeros((h, w), dtype=bool)
    drivable[31:, :] = True
    obstacle = np.zeros((h, w), dtype=bool)
    obstacle[26:31, :] = True  # real wall: lowest row 30 -> contact x = 5.9 m
    # Kart protrusion above the (unused) geometric mask: bottom rows around
    # the centre; their rays meet the ground right at the kart.
    obstacle[40:, 24:40] = True  # rows 40.. -> ground x <= 0.3+0.56*60/16 = 2.4 m
    drivable[40:, 24:40] = False

    g = EvidenceGrid(
        kart_footprint=KART_FOOTPRINT_M, min_contact_range_m=CAMERA_MIN_CONTACT_RANGE_M
    )
    g.insert_camera(drivable, obstacle, K, T, np.zeros(3))
    cls = g.classify()
    ij = np.argwhere(cls == CLASS_OBSTACLE)
    assert len(ij), "the real wall must be mapped"
    centers = g.cell_to_world(ij)
    r = np.hypot(centers[:, 0], centers[:, 1])
    assert r.min() > 2.4, f"kart protrusion painted an obstacle at {r.min():.2f} m"
    assert g.classify_points([[5.9, 0.0]])[0] == CLASS_OBSTACLE  # wall present


def test_classical_ring_demotes_near_band_but_keeps_far_obstacle() -> None:
    """Segmenter-level: pixels just above the kart mask are neutral (never
    contacts), while an obstacle outside the ring stays obstacle."""
    rng = np.random.default_rng(31)
    img = _scene(rng)  # hedge on the right
    protrusion = kart_body_mask(H, W).copy()
    protrusion[int(0.80 * H) :, int(0.40 * W) : int(0.55 * W)] = True  # learned bump
    segmenter = ClassicalGroundSegmenter(kart_mask=protrusion)
    result = segmenter.segment(img)
    ring = segmenter._geometry(H, W).ring_full
    # Nothing in the ring is ever obstacle - by construction.
    assert not (result.obstacle_mask & ring).any()
    # The hedge outside the ring is still an obstacle.
    assert result.obstacle_mask[int(0.6 * H), int(0.9 * W)]
    # Partition still holds with a custom mask + ring.
    total = (
        result.drivable_mask.astype(int)
        + result.obstacle_mask.astype(int)
        + result.neutral_mask.astype(int)
        + result.kart_mask.astype(int)
    )
    assert (total == 1).all()


def test_camera_history_routing_absorbs_transient_on_verified_ground() -> None:
    """A contact landing on a cell the camera itself verified drivable goes
    to the decaying dynamic layer (a dog, a far-facade smear), while a
    contact on unverified ground stays a static obstacle."""
    from kart_map2d.mapping.evidence_grid import CLASS_DYNAMIC, CLASS_OBSTACLE, EvidenceGrid

    K, T = _pinhole_camera()
    h, w = 48, 64
    all_drv = np.ones((h, w), dtype=bool)
    none = np.zeros((h, w), dtype=bool)
    g = EvidenceGrid()
    for k in range(4):  # verify the ground ahead as drivable
        g.insert_camera(all_drv, none, K, T, np.zeros(3), stamp=100.0 + 0.2 * k)
    target = np.array([5.9, 0.0])
    assert g.logodds_at([target])[0, 0] >= g.cam_route_drv_min

    obstacle = np.zeros((h, w), dtype=bool)
    obstacle[26:31, :] = True  # contact line at x = 5.9 m
    drv = all_drv & ~obstacle
    g.insert_camera(drv, obstacle, K, T, np.zeros(3), stamp=101.0)
    # Routed: not a static obstacle, visible in the live (dynamic) view.
    assert g.classify_points([target])[0] != CLASS_OBSTACLE
    live = g.classify(include_dynamic=True, now=101.0)
    ij = g.world_to_cell(target.reshape(1, 2))[0]
    assert live[ij[0], ij[1]] == CLASS_DYNAMIC

    # The same frame into a FRESH grid (no drivable history): static obstacle.
    g2 = EvidenceGrid()
    g2.insert_camera(drv, obstacle, K, T, np.zeros(3), stamp=101.0)
    assert g2.classify_points([target])[0] == CLASS_OBSTACLE
    # And without a stamp the routing is off entirely (legacy behaviour).
    g3 = EvidenceGrid()
    for k in range(4):
        g3.insert_camera(all_drv, none, K, T, np.zeros(3))
    g3.insert_camera(drv, obstacle, K, T, np.zeros(3))
    assert g3.classify_points([target])[0] == CLASS_OBSTACLE
