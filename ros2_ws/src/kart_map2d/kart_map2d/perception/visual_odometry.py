"""Monocular visual odometry for sparse, irregularly timed camera frames.

Frames arrive at a median gap of 0.17-0.36 s with dropouts up to a few
seconds, so this is pair-wise wide-baseline odometry rather than video
tracking: every consecutive frame pair is solved from scratch and a pair that
cannot be solved confidently yields *no* increment instead of a guess.

Per pair, three stages:

1. **Correspondences** - GFTT corners tracked by pyramidal LK with a
   motion-predicted seed (forward-backward checked); ORB + ratio-test
   matching takes over for wide baselines or when LK starves.
2. **Rotation** - essential matrix with RANSAC on canonically ordered
   matches, cheirality-checked decomposition, then a least-squares /
   Cauchy polish of (R, t) over all matches (portable across OpenCV
   builds, see :meth:`VisualOdometry._refine_pose`); the yaw of the implied
   base_link rotation is gated against (and can be replaced by) an optional
   IMU yaw prior.
3. **Metric translation** - the ground plane is the scale reference: both
   frames are warped to a metric top-down view (inverse perspective mapping
   at the known camera height, through build-independent fixed-point remap
   tables, see :func:`_remap_tables`) and the yaw-compensated view of the second
   frame is located inside the first with masked, high-passed NCC template
   matching. Dense alignment survives the low-texture concrete that starves
   sparse features; a sparse ray/ground-plane fit is kept as a fallback.

The camera's true attitude against the ground is *self-calibrated*: for
planar motion the epipolar translation direction must be parallel to
the ground, so the running median of its elevation in the nominal base frame
directly measures the mount pitch error (the September mounts are ~2 deg off
their URDF). The estimate feeds the IPM warp and ground classification.

Frames follow trackmapper conventions: base_link x forward / y left / z up
with the ground near ``z = ground_z``; camera OPTICAL frame z forward /
x right / y down; the relative pose is ``R, t`` with
``X_cam2 = R @ X_cam1 + s * t`` and ``|t| = 1``.

An IMU is optional: when a ``yaw_prior`` callback is supplied it seeds the
flow prediction, gates the vision yaw and rescues pairs whose epipolar
geometry fails (blur), but the class runs camera-only without it.
"""

from __future__ import annotations

import functools
import math
from collections import deque
from dataclasses import dataclass
from typing import Callable, Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from kart_map2d.core.measurements import CameraFrame, RelativeMotion
from kart_map2d.core.transforms import invert

YawPrior = Callable[[float, float], float]
"""Integrated base_link yaw change [rad] between two stamps [s]."""


def yaw_prior_from_imu(stamp: NDArray[np.float64], gyro_z: NDArray[np.float64]) -> YawPrior:
    """Build a yaw-prior callback from IMU samples (base_link z gyro, rad/s).

    Returns the trapezoidal integral of the yaw rate between two stamps; the
    gyro bias contributes < 1 mrad over the sub-second intervals VO uses.
    """
    cumulative = np.concatenate([[0.0], np.cumsum(0.5 * (gyro_z[1:] + gyro_z[:-1]) * np.diff(stamp))])

    def prior(t0: float, t1: float) -> float:
        a, b = np.interp([t0, t1], stamp, cumulative)
        return float(b - a)

    return prior


@dataclass
class VoConfig:
    """Tuning knobs; defaults measured on the September kart recordings."""

    # Feature tracking (GFTT corners + pyramidal LK with a warp prediction).
    max_corners: int = 1500
    gftt_quality: float = 0.005
    gftt_min_distance: int = 10
    lk_win: int = 21
    lk_levels: int = 4
    lk_max_gap_s: float = 1.0  # longer pairs go straight to descriptor matching
    fb_max_px: float = 1.5  # forward-backward track consistency
    # Descriptor fallback for wide baselines / failed LK.
    orb_features: int = 3000
    orb_ratio: float = 0.78
    # Pose estimation. Every robust-estimator parameter is pinned explicitly
    # (never an OpenCV default) so OpenCV 4.x and 5.x run the same estimator.
    min_matches: int = 25
    min_inliers: int = 15
    ransac_px: float = 1.0
    ransac_prob: float = 0.999
    ransac_max_iters: int = 1000
    # Polish of the RANSAC pose (see _refine_pose): least squares on the
    # consensus set, then a Cauchy(ransac_px) fit over every match. False
    # keeps the raw minimal-sample RANSAC model (the pre-2026-09-29 VO).
    pose_refine: bool = True
    pose_refine_max_iters: int = 20  # per stage; bounds the worst case
    pose_refine_coarse_tol: float = 1e-3  # consensus-LSQ stage [rad]
    pose_refine_tol: float = 1e-5  # final stage [rad]: 0.05 mm at the IPM template
    min_parallax_px: float = 1.2  # below: translation indistinguishable from 0
    max_gap_s: float = 3.5  # do not relate frames across longer dropouts
    min_pair_dt_s: float = 0.09  # merge near-duplicate frames into one interval
    # IMU yaw gate: reject vision yaw further than gate_abs + gate_rel*|prior|.
    yaw_gate_abs: float = 0.04
    yaw_gate_rel: float = 0.35
    # Inverse-perspective-mapped ground alignment (metric translation).
    ipm_res: float = 0.03  # m per top-down pixel
    ipm_search_fwd: tuple[float, float] = (1.2, 8.7)  # m, frame-1 window
    ipm_template_fwd: tuple[float, float] = (3.0, 6.0)  # m, frame-2 template
    ipm_search_half: float = 3.4  # m lateral half-width, frame-1 window
    ipm_template_half: float = 2.3  # m lateral half-width, frame-2 template
    ipm_highpass_sigma: float = 6.0  # px, removes sun shading before NCC
    # "auto": OpenCV-4 fixed-point warp semantics on every build - the native
    # warpPerspective where it is bit-identical (OpenCV <= 4.10, the kart),
    # numpy remap tables elsewhere (5.x). "tables" forces the tables.
    ipm_warp: str = "auto"
    ncc_min: float = 0.32  # accept threshold with epipolar support
    ncc_blind: float = 0.5  # accept threshold when only the IMU gives yaw
    ncc_peak_margin: float = 0.05  # lead over the best rival peak (aliasing)
    ncc_peak_radius_px: int = 5  # rival peaks closer than this are the same
    # Sparse ray/ground-plane fallback. The range cap is deliberately short:
    # beyond ~9 m a "ground" ray cannot be told from a building base or hedge
    # sitting just above the horizon band, which biases the scale low.
    min_ground_points: int = 10
    pixel_sigma: float = 0.6  # px feature noise for range weighting
    max_ground_range_m: float = 9.0
    min_ground_range_m: float = 0.8
    max_step_rel_sigma: float = 0.25
    max_step_abs_sigma: float = 0.1  # m
    # Mount-pitch self-calibration from the epipolar translation elevation.
    pitch_window: int = 60  # pairs in the running median
    pitch_min_samples: int = 8  # calibrated once this many are collected
    pitch_limit: float = np.radians(6.0)
    pitch_min_parallax_px: float = 4.0  # t direction is noise below this
    mount_pitch_init: float | None = None  # rad, seeds/replaces the estimate
    self_calibrate_pitch: bool = True  # False pins the pitch to the init value
    # Sanity.
    max_speed_mps: float = 12.0


@dataclass
class VoPairDiagnostics:
    """Per-pair internals, kept for validation scripts (not used by fusion)."""

    start_stamp: float
    stamp: float
    status: str  # emitted | stationary | rotation_only | <failure reason>
    n_matches: int = 0
    n_inliers: int = 0
    parallax_px: float = 0.0
    scale_source: str = ""  # ipm_ncc | ground_se2 | ""
    ncc: float = 0.0
    speed_mps: float = 0.0
    yaw_rate: float = 0.0
    yaw_from_prior: bool = False
    method: str = ""  # lk | orb | imu
    pitch_deg: float = 0.0  # self-calibrated mount pitch at solve time


def _skew(v: NDArray[np.float64]) -> NDArray[np.float64]:
    """3 x 3 cross-product matrix: ``_skew(v) @ x == v x x``."""
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


_SKEW_BASIS = np.array([_skew(e) for e in np.eye(3)])  # [e_j]x, j = 0..2
_EYE3 = np.eye(3)
_EYE5 = np.eye(5)


def _remap_tables(
    H: NDArray[np.float64], rows: int, cols: int, src_size: tuple[int, int], nearest: bool = True
) -> tuple[NDArray[np.int16], NDArray[np.uint16], NDArray[np.int16] | None]:
    """Fixed-point cv2.remap tables sampling ``src`` at ``H @ (col, row, 1)``.

    Replaces ``cv2.warpPerspective(src, H, (cols, rows), WARP_INVERSE_MAP)``,
    whose INTER_LINEAR kernel changed after OpenCV 4.10 (2-4 grey levels
    apart on the same inputs; 4.x quantises source coordinates to 1/32 px
    internally, 5.0 does not). Here the map is computed in float64 numpy and
    quantised to 1/32 px (OpenCV's INTER_BITS = 5, ties to even like its
    cvRound) before OpenCV sees it; ``cv2.remap(src, xy, frac,
    INTER_LINEAR)`` is then bit-identical on 4.x and 5.x (verified 4.10 vs
    5.0), and on OpenCV <= 4.10 it equals warpPerspective bit for bit:
    OpenCV 4 semantics on every build (see :func:`_native_warp_is_portable`
    for when the native call is used instead). Returns (integer xy
    [rows, cols, 2] int16, 5+5-bit fraction uint16, and with ``nearest``
    the whole-pixel table for INTER_NEAREST). Coordinates are clamped just
    outside the source so out-of-view samples read the zero border, as
    warpPerspective's BORDER_CONSTANT did.
    """
    w, h = src_size
    u = np.arange(cols, dtype=np.float64)
    v = np.arange(rows, dtype=np.float64)[:, None]
    Wh = (H[2, 0] * u + H[2, 2]) + H[2, 1] * v
    behind = Wh <= 1e-9  # never inside the IPM window; guarded for general use
    any_behind = bool(behind.any())
    if any_behind:
        Wh[behind] = 1.0
    x = ((H[0, 0] * u + H[0, 2]) + H[0, 1] * v) / Wh
    y = ((H[1, 0] * u + H[1, 2]) + H[1, 1] * v) / Wh
    np.clip(x, -2.0, w + 1.0, out=x)
    np.clip(y, -2.0, h + 1.0, out=y)
    if any_behind:
        x[behind] = -2.0
        y[behind] = -2.0
    near = None
    if nearest:
        near = np.empty((rows, cols, 2), np.int16)
        near[..., 0] = np.rint(x)
        near[..., 1] = np.rint(y)
    for c in (x, y):  # nearest 1/32 px, ties to even like OpenCV's cvRound
        c *= 32.0
        np.rint(c, out=c)
    fx = x.astype(np.int32)
    fy = y.astype(np.int32)
    xy = np.empty((rows, cols, 2), np.int16)
    xy[..., 0] = fx >> 5
    xy[..., 1] = fy >> 5
    frac = ((fy & 31) << 5 | (fx & 31)).astype(np.uint16)
    return xy, frac, near


@functools.lru_cache(maxsize=1)
def _native_warp_is_portable() -> bool:
    """Does this OpenCV's warpPerspective reproduce :func:`_remap_tables` exactly?

    True on OpenCV <= 4.10 (its kernel is the same 1/32 px fixed-point
    remap), False on 5.0. Probed once per process on a random image through
    a generic homography, so a build whose kernel differs in any way (a newer
    release, another SIMD path) falls back to the portable tables by itself.
    Where it holds, the IPM uses the native call: identical pixels, no cost
    (the numpy tables cost ~2.5 ms per pair on this CPU).
    """
    src = np.random.default_rng(0).integers(0, 256, (120, 160), dtype=np.uint8)
    H = np.array([[0.93, 0.05, 7.3], [-0.04, 1.08, 5.1], [2e-4, 3e-4, 1.0]])
    xy, frac, near = _remap_tables(H, 96, 128, (160, 120))
    for interp, tables in (
        (cv2.INTER_LINEAR, cv2.remap(src, xy, frac, cv2.INTER_LINEAR)),
        (cv2.INTER_NEAREST, cv2.remap(src, near, None, cv2.INTER_NEAREST)),
    ):
        native = cv2.warpPerspective(
            src, H, (128, 96), flags=cv2.WARP_INVERSE_MAP | interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        if not np.array_equal(tables, native):
            return False
    return True


def _default_kart_mask(size: tuple[int, int], kart_on_left: bool) -> NDArray[np.uint8]:
    """255 = usable, 0 = kart body.

    The chassis and front wheel sit in the lower-inboard region of the image
    (the cameras are mounted at +-0.27 m lateral); the polygon was traced on
    the September frames with margin for wheel steering. The opposite lower
    corner and the bottom strip are masked as a guard band.
    """
    w, h = size
    mask = np.full((h, w), 255, np.uint8)

    def scaled(poly: Sequence[tuple[float, float]]) -> NDArray[np.int32]:
        pts = np.array(poly, np.float64)
        if not kart_on_left:
            pts[:, 0] = 1.0 - pts[:, 0]
        return np.round(pts * [w - 1, h - 1]).astype(np.int32)

    kart = [(0.0, 0.58), (0.45, 0.60), (0.62, 0.70), (0.74, 1.0), (0.0, 1.0)]
    guard = [(1.0, 0.88), (0.80, 1.0), (1.0, 1.0)]
    cv2.fillPoly(mask, [scaled(kart)], 0)
    cv2.fillPoly(mask, [scaled(guard)], 0)
    mask[int(0.95 * h) :, :] = 0
    return mask


class VisualOdometry:
    """Pair-wise monocular VO for one camera, emitting base_link increments.

    Feed frames of a single camera in stamp order through :meth:`process`;
    each call relates the new frame to the previous one and returns a
    :class:`RelativeMotion` on channel ``vo_<camera>`` or ``None`` when the
    pair cannot be solved confidently.

    Parameters
    ----------
    camera:
        Camera name; sets the measurement channel to ``vo_<camera>``.
    K, image_size:
        Pinhole intrinsics of the (already undistorted) images.
    T_base_cam:
        4x4 base_link <- camera optical frame (nominal mount; the pitch error
        against the real ground is self-calibrated online).
    ground_z:
        Ground height in base_link [m]; with the mount height this fixes the
        camera's height above the ground plane, the source of metric scale.
    yaw_prior:
        Optional callback (e.g. :func:`yaw_prior_from_imu`); ``None`` runs
        camera-only.
    kart_mask:
        Optional uint8 mask (255 = usable); ``None`` uses the built-in kart
        polygon, mirrored according to which side of the kart the camera is
        mounted on.
    camera_model:
        Optional :class:`kart_map2d.camera.model.CameraModel` AT the working
        resolution. When it carries lens distortion, matched feature POINTS
        are undistorted to the ideal pinhole ``K`` before the pose solve
        (never the frames - compute budget). The dense IPM ground alignment
        still assumes the pinhole ``K``; under strong distortion its NCC
        self-rejects and the sparse (undistorted-point) ground step takes
        over. ``None`` (rectified stream): behaviour is bit-identical to the
        proven research VO.
    """

    def __init__(
        self,
        camera: str,
        K: NDArray[np.float64],
        image_size: tuple[int, int],
        T_base_cam: NDArray[np.float64],
        ground_z: float = -0.10,
        yaw_prior: YawPrior | None = None,
        kart_mask: NDArray[np.uint8] | None = None,
        config: VoConfig | None = None,
        camera_model=None,
    ) -> None:
        self.channel = f"vo_{camera}"
        self.camera_model = camera_model if (camera_model is not None and getattr(camera_model, "has_distortion", False)) else None
        self.K = np.asarray(K, dtype=np.float64)
        self.K_inv = np.linalg.inv(self.K)
        self._sampson_w = np.array([1.0 / self.K[0, 0] ** 2, 1.0 / self.K[1, 1] ** 2])
        self.image_size = image_size
        self.T_base_cam = np.asarray(T_base_cam, dtype=np.float64)
        self.T_cam_base = invert(self.T_base_cam)
        self.cam_height = float(self.T_base_cam[2, 3] - ground_z)
        self.yaw_prior = yaw_prior
        self.config = config or VoConfig()
        # A camera on the right of the kart (y < 0) sees the chassis image-left.
        if kart_mask is None:
            kart_mask = _default_kart_mask(image_size, kart_on_left=self.T_base_cam[1, 3] < 0)
        self.mask = kart_mask
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # All ORB / matcher parameters explicit (the non-feature-count ones
        # are OpenCV's defaults), so a release that changes a default cannot
        # silently change the fallback matcher.
        self._orb = cv2.ORB_create(
            nfeatures=self.config.orb_features,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=31,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31,
            fastThreshold=12,
        )
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self._prev: tuple[float, NDArray[np.uint8]] | None = None
        self._prev_speed = 0.0  # m/s, seeds the flow prediction
        self._prev_yaw_rate = 0.0  # rad/s, used only when there is no IMU
        self._elevations: deque[float] = deque(maxlen=self.config.pitch_window)
        if self.config.mount_pitch_init is not None:
            # Seed the running median so the first pairs already use the
            # calibrated mount; fresh measurements take over as they arrive.
            self._elevations.extend([float(self.config.mount_pitch_init)] * self.config.pitch_min_samples)
        self.diagnostics: list[VoPairDiagnostics] = []

    # ------------------------------------------------------------------ public
    @property
    def mount_pitch(self) -> float:
        """Self-calibrated mount pitch error [rad].

        Negative means the camera actually looks *up* relative to the URDF
        mount (the September karts measure about -3 deg). Zero until enough
        translation directions have been collected.
        """
        cfg = self.config
        if not cfg.self_calibrate_pitch:
            return float(cfg.mount_pitch_init or 0.0)
        if len(self._elevations) < cfg.pitch_min_samples:
            return 0.0
        return float(np.clip(np.median(self._elevations), -cfg.pitch_limit, cfg.pitch_limit))

    @property
    def calibrated(self) -> bool:
        return len(self._elevations) >= self.config.pitch_min_samples

    def process(self, frame: CameraFrame) -> RelativeMotion | None:
        """Relate ``frame`` to the previous one; returns the increment or None."""
        image = cv2.imread(frame.image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None
        gray = self._clahe.apply(image)
        if self._prev is None:
            self._prev = (frame.stamp, gray)
            return None
        t0, prev = self._prev
        dt = frame.stamp - t0
        if 0.0 <= dt < self.config.min_pair_dt_s:
            # Near-duplicate frame: keep the older anchor so the next pair
            # spans a long enough baseline instead of measuring jitter.
            self._log(t0, frame.stamp, "merged")
            return None
        self._prev = (frame.stamp, gray)
        if not 0.0 < dt <= self.config.max_gap_s:
            self._log(t0, frame.stamp, "gap")
            return None
        return self._solve_pair(prev, gray, t0, frame.stamp, dt)

    def reset(self) -> None:
        """Forget the previous frame (e.g. after a known discontinuity)."""
        self._prev = None
        self._prev_speed = 0.0
        self._prev_yaw_rate = 0.0

    # ------------------------------------------------------------- pair solver
    def _solve_pair(
        self, prev: NDArray[np.uint8], gray: NDArray[np.uint8], t0: float, t1: float, dt: float
    ) -> RelativeMotion | None:
        cfg = self.config
        prior_yaw = self.yaw_prior(t0, t1) if self.yaw_prior is not None else None
        method = "lk"
        p1 = p2 = None
        if dt <= cfg.lk_max_gap_s:
            p1, p2 = self._track_lk(prev, gray, dt, prior_yaw)
        if p1 is None or len(p1) < cfg.min_matches:
            method = "orb"
            p1, p2 = self._match_orb(prev, gray)

        if p1 is not None and self.camera_model is not None:
            # Lens distortion: features were measured on the DISTORTED image;
            # map the matched points onto the ideal pinhole K before any
            # epipolar / ground geometry (points only, never frames).
            p1 = self.camera_model.undistort_pixels(p1)
            p2 = self.camera_model.undistort_pixels(p2)

        if p1 is None or len(p1) < cfg.min_matches:
            n = 0 if p1 is None else len(p1)
            return self._blind_or_fail(prev, gray, t0, t1, dt, prior_yaw, "too_few_matches", n)

        # Stationary short-circuit: when the raw flow is sub-pixel there is no
        # epipolar geometry to solve; the frames agree, so the kart stood still.
        raw_flow = float(np.median(np.linalg.norm(p2 - p1, axis=1)))
        if raw_flow < cfg.min_parallax_px:
            self._prev_speed, self._prev_yaw_rate = 0.0, 0.0
            self._log(t0, t1, "stationary", n_matches=len(p1), parallax_px=raw_flow, method=method)
            return RelativeMotion(
                stamp=t1,
                channel=self.channel,
                start_stamp=t0,
                delta_forward=0.0,
                delta_left=0.0,
                delta_yaw=0.0,
                sigma_translation=0.1,
                sigma_yaw=0.004,
                sigma_position=0.02,
                quality=0.7,
            )

        pose = self._recover_pose(p1, p2)
        if pose is None:
            return self._blind_or_fail(prev, gray, t0, t1, dt, prior_yaw, "pose_failed", len(p1))
        R, t, inlier = pose
        q1, q2 = p1[inlier], p2[inlier]
        inlier_ratio = len(q1) / len(p1)

        # Parallax after removing rotation: how much of the flow is translation.
        warped = self._warp_rotation_only(q1, R)
        parallax = float(np.median(np.linalg.norm(q2 - warped, axis=1)))
        diag = VoPairDiagnostics(
            t0,
            t1,
            "",
            n_matches=len(p1),
            n_inliers=len(q1),
            parallax_px=parallax,
            method=method,
            pitch_deg=float(np.degrees(self.mount_pitch)),
        )

        delta_yaw, yaw_from_prior = self._gate_yaw(self._base_yaw(R), prior_yaw)
        diag.yaw_from_prior = yaw_from_prior

        if parallax < cfg.min_parallax_px:
            # Rotating (or creeping) essentially in place: the translation
            # direction is unobservable but bounded by the missing parallax.
            self._prev_speed = 0.0
            self._prev_yaw_rate = delta_yaw / dt
            diag.status, diag.yaw_rate = "rotation_only", delta_yaw / dt
            self.diagnostics.append(diag)
            return RelativeMotion(
                stamp=t1,
                channel=self.channel,
                start_stamp=t0,
                delta_forward=0.0,
                delta_left=0.0,
                delta_yaw=delta_yaw,
                sigma_translation=0.5,
                sigma_yaw=0.01,
                sigma_position=0.05,
                quality=0.5 * inlier_ratio + 0.2,
            )

        # Mount-pitch self-calibration: planar motion means the translation
        # direction is ground-parallel, so its elevation in the nominal base
        # frame measures the pitch error. Needs clear parallax to be meaningful.
        if parallax >= cfg.pitch_min_parallax_px and len(q1) >= 2 * cfg.min_inliers:
            d_base = self.T_base_cam[:3, :3] @ (-R.T @ t)
            if d_base[0] < 0:
                d_base = -d_base
            self._elevations.append(float(np.arctan2(d_base[2], np.hypot(d_base[0], d_base[1]))))

        # With lens distortion the IPM samples through the model's forward
        # distortion (see _ipm_view), so it stays the primary scale source
        # for every stream.
        step = self._ipm_translation(prev, gray, delta_yaw, cfg.ncc_min)
        source = "ipm_ncc"
        if step is None:
            step = self._sparse_ground_step(q1, q2, R, t, delta_yaw)
            source = "ground_se2"
        if step is None:
            diag.status = "no_scale"
            self.diagnostics.append(diag)
            return None
        (dfwd, dleft), sigma_step, strength = step
        distance = float(np.hypot(dfwd, dleft))
        if distance > cfg.max_speed_mps * dt:
            diag.status = "no_scale"
            self.diagnostics.append(diag)
            return None

        diag.scale_source, diag.ncc = source, strength
        quality = float(np.clip(0.2 + 0.4 * inlier_ratio + 0.4 * strength, 0.0, 1.0))
        if yaw_from_prior:
            quality *= 0.7
        if not self.calibrated:
            quality *= 0.6
            sigma_step *= 2.0
        self._prev_speed = distance / dt
        self._prev_yaw_rate = delta_yaw / dt
        diag.status, diag.speed_mps, diag.yaw_rate = "emitted", distance / dt, delta_yaw / dt
        self.diagnostics.append(diag)
        return RelativeMotion(
            stamp=t1,
            channel=self.channel,
            start_stamp=t0,
            delta_forward=dfwd,
            delta_left=dleft,
            delta_yaw=delta_yaw,
            sigma_translation=0.08,
            sigma_yaw=0.008 + 0.02 * (1.0 - inlier_ratio),
            sigma_position=float(np.clip(sigma_step, 0.02, 0.3)),
            quality=quality,
        )

    def _blind_or_fail(
        self,
        prev: NDArray[np.uint8],
        gray: NDArray[np.uint8],
        t0: float,
        t1: float,
        dt: float,
        prior_yaw: float | None,
        reason: str,
        n_matches: int,
    ) -> RelativeMotion | None:
        """Feature stages failed (blur, texture loss): try IMU yaw + dense IPM.

        Without an IMU the pair is dropped honestly; with one, the ground
        alignment alone can still deliver a metric increment, at a stricter
        NCC threshold because the epipolar cross-check is missing.
        """
        cfg = self.config
        if prior_yaw is None:
            self._log(t0, t1, reason, n_matches=n_matches)
            return None
        step = self._ipm_translation(prev, gray, prior_yaw, cfg.ncc_blind)
        if step is None:
            self._log(t0, t1, reason, n_matches=n_matches)
            return None
        (dfwd, dleft), sigma_step, strength = step
        distance = float(np.hypot(dfwd, dleft))
        if distance > cfg.max_speed_mps * dt:
            self._log(t0, t1, reason, n_matches=n_matches)
            return None
        if not self.calibrated:
            sigma_step *= 2.0
        self._prev_speed = distance / dt
        self._prev_yaw_rate = prior_yaw / dt
        self._log(
            t0,
            t1,
            "emitted",
            n_matches=n_matches,
            method="imu",
            scale_source="ipm_ncc",
            ncc=strength,
            speed_mps=distance / dt,
            yaw_rate=prior_yaw / dt,
            yaw_from_prior=True,
            pitch_deg=float(np.degrees(self.mount_pitch)),
        )
        return RelativeMotion(
            stamp=t1,
            channel=self.channel,
            start_stamp=t0,
            delta_forward=dfwd,
            delta_left=dleft,
            delta_yaw=prior_yaw,
            sigma_translation=0.12,
            sigma_yaw=0.01,
            sigma_position=float(np.clip(sigma_step, 0.03, 0.3)),
            quality=float(np.clip(0.3 * strength + 0.2, 0.0, 1.0)) * (1.0 if self.calibrated else 0.6),
        )

    # ---------------------------------------------------------- correspondence
    def _track_lk(
        self, prev: NDArray[np.uint8], gray: NDArray[np.uint8], dt: float, prior_yaw: float | None
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]] | tuple[None, None]:
        cfg = self.config
        # Every parameter explicit (OpenCV defaults shown), so a release that
        # changes a default cannot silently change the tracker.
        corners = cv2.goodFeaturesToTrack(
            prev,
            maxCorners=cfg.max_corners,
            qualityLevel=cfg.gftt_quality,
            minDistance=cfg.gftt_min_distance,
            mask=self.mask,
            blockSize=3,
            useHarrisDetector=False,
            k=0.04,
        )
        if corners is None or len(corners) < cfg.min_matches:
            return None, None
        p1 = corners.reshape(-1, 2).astype(np.float32)
        guess = self._predict_flow(p1, dt, prior_yaw).astype(np.float32)
        lk = dict(
            winSize=(cfg.lk_win, cfg.lk_win),
            maxLevel=cfg.lk_levels,
            criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, 0.01),
            minEigThreshold=1e-4,
        )
        p2, ok, _ = cv2.calcOpticalFlowPyrLK(
            prev, gray, p1, guess.copy(), flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **lk
        )
        back, ok_b, _ = cv2.calcOpticalFlowPyrLK(gray, prev, p2, p1.copy(), flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **lk)
        fb = np.linalg.norm(back - p1, axis=1)
        w, h = self.image_size
        good = (
            (ok.ravel() == 1)
            & (ok_b.ravel() == 1)
            & (fb < cfg.fb_max_px)
            & (p2[:, 0] >= 0)
            & (p2[:, 0] < w)
            & (p2[:, 1] >= 0)
            & (p2[:, 1] < h)
        )
        return p1[good].astype(np.float64), p2[good].astype(np.float64)

    def _predict_flow(self, p1: NDArray[np.float32], dt: float, prior_yaw: float | None) -> NDArray[np.float32]:
        """Seed LK with the flow of the expected motion.

        Rotation comes from the IMU prior (or the previous yaw rate), the
        translation from the previous speed assuming points below the horizon
        lie on the ground plane; points above it get rotation-only flow
        (infinite depth). A wrong seed only slows LK down, it cannot corrupt
        the result past the forward-backward check.
        """
        dyaw = prior_yaw if prior_yaw is not None else self._prev_yaw_rate * dt
        step = self._prev_speed * dt
        cy, sy = np.cos(dyaw), np.sin(dyaw)
        T_b1_b2 = np.eye(4)
        T_b1_b2[:2, :2] = [[cy, -sy], [sy, cy]]
        T_b1_b2[0, 3] = step
        T_c2_c1 = invert(self.T_cam_base @ T_b1_b2 @ self.T_base_cam)
        n = self._ground_normal_cam()
        rays = np.column_stack([p1, np.ones(len(p1))]) @ self.K_inv.T
        depth_to_ground = rays @ n
        on_ground = depth_to_ground > self.cam_height / self.config.max_ground_range_m
        lam = np.where(on_ground, self.cam_height / np.where(on_ground, depth_to_ground, 1.0), 1e6)
        pts = rays * lam[:, None]
        moved = pts @ T_c2_c1[:3, :3].T + T_c2_c1[:3, 3]
        uv = moved @ self.K.T
        behind = uv[:, 2] < 1e-6
        uv = uv[:, :2] / np.where(behind, 1.0, uv[:, 2])[:, None]
        return np.where(behind[:, None], p1, uv)

    def _match_orb(
        self, prev: NDArray[np.uint8], gray: NDArray[np.uint8]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]] | tuple[None, None]:
        kp1, des1 = self._orb.detectAndCompute(prev, self.mask)
        kp2, des2 = self._orb.detectAndCompute(gray, self.mask)
        if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
            return None, None
        pairs = self._matcher.knnMatch(des1, des2, k=2)
        good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < self.config.orb_ratio * n.distance]
        if len(good) < self.config.min_matches:
            return None, None
        p1 = np.array([kp1[m.queryIdx].pt for m in good])
        p2 = np.array([kp2[m.trainIdx].pt for m in good])
        return p1, p2

    # ------------------------------------------------------------------- pose
    def _recover_pose(
        self, p1: NDArray[np.float64], p2: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]] | None:
        """Relative camera pose ``(R, t, inlier)`` with ``X2 = R X1 + t``, |t| = 1.

        Robust essential matrix, cheirality-checked decomposition, then a
        polish over all matches (:meth:`_refine_pose`). Every call whose
        defaults could drift between OpenCV releases is avoided or has its
        parameters pinned; everything after the RANSAC is numpy. ``inlier``
        is aligned with the caller's ``p1``/``p2`` order.
        """
        # RANSAC draws its samples by index, so the same correspondence set
        # in another order yields another model. Match order is not portable:
        # on the same image ORB returns a bit-identical keypoint set in a
        # different order on Windows and Linux (STL sort tie-breaking), and
        # that order flows through knnMatch into RANSAC. Sorting by
        # coordinates first makes the RANSAC input canonical.
        order = np.lexsort((p2[:, 1], p2[:, 0], p1[:, 1], p1[:, 0]))
        pose = self._solve_pose(p1[order], p2[order])
        if pose is None:
            return None
        R, t, inlier_sorted = pose
        inlier = np.empty_like(inlier_sorted)
        inlier[order] = inlier_sorted
        return R, t, inlier

    def _solve_pose(
        self, p1: NDArray[np.float64], p2: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]] | None:
        """:meth:`_recover_pose` on correspondences already in canonical order."""
        cfg = self.config
        # Classic RANSAC (not USAC): OpenCV seeds it with a fixed internal RNG,
        # so identical correspondences give a bit-identical model on 4.x and
        # 5.x (verified 4.10 vs 5.0). USAC_PARALLEL is thread-order dependent,
        # and its sibling flags' internals changed between 4.x minor releases.
        E, mask = cv2.findEssentialMat(
            p1,
            p2,
            self.K,
            method=cv2.RANSAC,
            prob=cfg.ransac_prob,
            threshold=cfg.ransac_px,
            maxIters=cfg.ransac_max_iters,
        )
        if E is None or E.shape != (3, 3):
            return None
        y1 = np.column_stack([p1, np.ones(len(p1))]) @ self.K_inv.T
        y2 = np.column_stack([p2, np.ones(len(p2))]) @ self.K_inv.T
        ransac_inlier = mask.ravel() > 0
        # The cv2.recoverPose rule (most RANSAC inliers in front of both
        # cameras, first candidate wins ties) with a closed-form numpy
        # triangulation: recoverPose DLT-triangulates every point for all four
        # candidates, ~4.6 ms at 800 matches - more than the whole refinement.
        R_a, R_b, t_e = cv2.decomposeEssentialMat(E)
        t_e = t_e.ravel()
        best: tuple[int, NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]] | None = None
        for R, t in ((R_a, t_e), (R_b, t_e), (R_a, -t_e), (R_b, -t_e)):
            front = self._in_front(R, t, y1, y2) & ransac_inlier
            n_front = int(front.sum())
            if best is None or n_front > best[0]:
                best = (n_front, R, t, front)
        n_in, R, t, inlier = best
        if n_in < cfg.min_inliers:
            return None
        if cfg.pose_refine:
            refined = self._refine_pose(y1, y2, R, t, inlier)
            if refined is not None:
                R_r, t_r, r = refined
                inlier_r = (np.abs(r) < cfg.ransac_px) & self._in_front(R_r, t_r, y1, y2)
                if int(inlier_r.sum()) >= cfg.min_inliers:
                    return R_r, t_r, inlier_r
        return R, t, inlier

    @staticmethod
    def _in_front(
        R: NDArray[np.float64],
        t: NDArray[np.float64],
        y1: NDArray[np.float64],
        y2: NDArray[np.float64],
        max_depth: float = 50.0,
    ) -> NDArray[np.bool_]:
        """Matches triangulating in front of both cameras, closer than ``max_depth``.

        Least-squares depths of ``z2 y2 = z1 R y1 + t`` on normalised rays
        (z-component 1, so z1/z2 are optical-axis depths in units of the unit
        baseline). ``max_depth`` = 50 is the cv2.recoverPose default
        ``distanceThresh``: farther points carry no translation information.
        """
        u = y1 @ R.T
        uu = np.einsum("ij,ij->i", u, u)
        vv = np.einsum("ij,ij->i", y2, y2)
        uv = np.einsum("ij,ij->i", u, y2)
        ut, vt = u @ t, y2 @ t
        det = uu * vv - uv**2
        ok = det > 1e-12 * uu * vv  # parallel rays: depth unobservable
        det = np.where(ok, det, 1.0)
        z1 = (uv * vt - vv * ut) / det
        z2 = (uu * vt - uv * ut) / det
        return ok & (z1 > 0.0) & (z2 > 0.0) & (z1 < max_depth) & (z2 < max_depth)

    def _sampson(
        self, E: NDArray[np.float64], y1: NDArray[np.float64], y2: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Signed Sampson distance [px] on normalised rays (N x 3), plus terms.

        With ``a = E y1``, ``b = E^T y2``, ``e = y2 . a`` the distance is
        ``e / sqrt(Q)``, ``Q = (a0^2 + b0^2) / fx^2 + (a1^2 + b1^2) / fy^2``.
        Returns ``(r, a, b, sqrt(Q))`` for :meth:`_sampson_jacobian`.
        """
        a = y1 @ E.T
        b = y2 @ E
        den = np.sqrt(np.maximum((a[:, :2] ** 2 + b[:, :2] ** 2) @ self._sampson_w, 1e-24))
        return np.einsum("ij,ij->i", y2, a) / den, a, b, den

    def _sampson_jacobian(
        self,
        R: NDArray[np.float64],
        t: NDArray[np.float64],
        B: NDArray[np.float64],
        y1: NDArray[np.float64],
        y2: NDArray[np.float64],
        terms: tuple[NDArray[np.float64], ...],
    ) -> NDArray[np.float64]:
        """Exact N x 5 Jacobian of :meth:`_sampson` for E = [t]x R.

        Parameters (w, d): R <- exp([w]x) R, t <- t + B d. With dE/dw_j =
        [t]x [e_j]x R and dE/dd_k = [B_k]x R: da_i/dp = y1 . dE[i, :],
        db_j/dp = y2 . dE[:, j], de/dp = sum_i y2_i da_i/dp, and
        dr = (de - r dQ / (2 sqrt(Q))) / sqrt(Q). The Q terms matter: forward
        motion puts the epipole inside the image, where Q changes fast, and
        holding Q constant makes the step a non-descent direction there.
        """
        r, a, b, den = terms
        n = len(r)
        SR = (_SKEW_BASIS @ R).reshape(3, 9)
        dE = np.concatenate([(_skew(t) @ SR.reshape(3, 3, 3)).reshape(3, 9), B.T @ SR]).T.reshape(3, 3, 5)
        da = (y1 @ dE.transpose(1, 0, 2).reshape(3, 15)).reshape(n, 3, 5)  # da_i/dp
        db = (y2 @ dE[:, :2, :].reshape(3, 10)).reshape(n, 2, 5)  # db_j/dp, j < 2
        k = (r / den)[:, None] * self._sampson_w
        ca = y2.copy()
        ca[:, :2] -= k * a[:, :2]
        num = np.einsum("ni,nip->np", ca, da) - np.einsum("nj,njp->np", k * b[:, :2], db)
        return num / den[:, None]

    @staticmethod
    def _tangent_basis(t: NDArray[np.float64]) -> NDArray[np.float64]:
        """3 x 2 orthonormal basis of the plane perpendicular to unit ``t``."""
        tx = _skew(t)
        b1 = tx[:, int(np.argmin(np.abs(t)))]  # t x e_i, least-aligned axis: never 0
        b1 = b1 / np.linalg.norm(b1)
        return np.column_stack([b1, tx @ b1])

    @staticmethod
    def _perturb(R: NDArray[np.float64], t: NDArray[np.float64], d: NDArray[np.float64], B: NDArray[np.float64]):
        """(R, t) moved by d = (rotation vector [rad], 2 steps along basis B of t)."""
        w = d[:3]
        th = math.sqrt(float(w @ w))
        W = _skew(w)
        if th < 1e-12:
            dR = _EYE3 + W
        else:
            dR = _EYE3 + (math.sin(th) / th) * W + ((1.0 - math.cos(th)) / th**2) * (W @ W)
        tn = t + B @ d[3:]
        return dR @ R, tn / math.sqrt(float(tn @ tn))

    def _refine_pose(
        self,
        y1: NDArray[np.float64],
        y2: NDArray[np.float64],
        R: NDArray[np.float64],
        t: NDArray[np.float64],
        consensus: NDArray[np.bool_],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]] | None:
        """Polish the RANSAC pose (Gauss-Newton on the Sampson distance, px).

        Why: RANSAC returns the model of the single 5-point sample with the
        most support, unrefined. Which sample wins is a discontinuous function
        of the input - a 1-LSB pixel change (another numpy/OpenCV build,
        another JPEG decoder, aarch64 SIMD) can hand the win to a different
        sample, and on the kart's narrow baselines minimal-sample models
        scatter by several mrad in yaw (sept3: median 1.9, max 8.9 mrad
        between sample orders of the same pair) and by degrees in the
        translation elevation that calibrates the mount pitch. Yaw leaks
        straight into the lateral IPM step (template rotated about a point
        ~4.5 m ahead: 0.01 rad -> 4.5 cm). A fit over all the data is a
        continuous function of it: reproducible across builds, and more
        accurate (3x tighter pitch samples; RMS yaw-rate error -20 % on sept3
        and sept2 zed_right, unchanged on zed_left).

        Two stages: plain least squares over the RANSAC consensus set (the
        classic polish), then a Cauchy fit (scale ``ransac_px``) over every
        match, so no hard inlier cut decides the answer - yaw is weakly
        conditioned against lateral translation (normal-matrix condition
        ~1e4 on ground scenes), and borderline points flipping in or out of
        a hard set moved the optimum by 5 mrad. The 1 px kernel is also what
        keeps it robust: with 20-60 % injected outliers the refined yaw error
        is 35-45 % below raw RANSAC, whereas a wide first kernel (graduated
        non-convexity from 8-10 px) was dragged off by them and did worse
        than RANSAC. Residual caveat: at the noise level a planar ground
        scene can offer two near-equal Cauchy minima (~3.7 mrad apart, both
        about as close to the truth); 0-11 % of RANSAC sample orders on the
        synthetic pairs reach the other one - a change bounded by the
        estimator's own precision, not by the minimal-sample scatter, and
        for a given match set the canonical order of :meth:`_recover_pose`
        makes the choice deterministic.

        Iteratively reweighted Gauss-Newton on the 5-dof manifold (rotation
        vector + tangent plane of the unit ``t``) with the exact Jacobian
        (:meth:`_sampson_jacobian`); step halving guarantees the cost never
        rises; each stage stops at a step below its tolerance or after
        ``pose_refine_max_iters``. Pure numpy, no RNG. Returns ``(R, t,
        sampson_px of every match)``, or None to keep the RANSAC model (the
        solve wandered > 5 deg away).
        """
        cfg = self.config
        R0 = R
        R, t, _ = self._irls(y1[consensus], y2[consensus], R, t, None, cfg.pose_refine_coarse_tol)
        R, t, r = self._irls(y1, y2, R, t, cfg.ransac_px, cfg.pose_refine_tol)
        if float(np.arccos(np.clip((np.trace(R.T @ R0) - 1.0) / 2.0, -1.0, 1.0))) > np.radians(5.0):
            return None
        return R, t, r

    def _irls(
        self,
        y1: NDArray[np.float64],
        y2: NDArray[np.float64],
        R: NDArray[np.float64],
        t: NDArray[np.float64],
        scale: float | None,
        tol: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Minimise sum rho(Sampson) over (R, t); rho = r^2 (``scale`` None) or Cauchy.

        Returns the new ``(R, t)`` and the Sampson residuals [px] there.
        """
        cfg = self.config
        c2 = None if scale is None else scale * scale

        def cost_of(r: NDArray[np.float64]) -> float:
            return float(r @ r) if c2 is None else float(np.sum(np.log1p(r**2 / c2)))

        terms = self._sampson(_skew(t) @ R, y1, y2)
        cost = cost_of(terms[0])
        for _ in range(cfg.pose_refine_max_iters):
            B = self._tangent_basis(t)
            J = self._sampson_jacobian(R, t, B, y1, y2, terms)
            r = terms[0]
            Jw = J if c2 is None else J * (1.0 / (1.0 + r**2 / c2))[:, None]  # IRLS weights
            H = Jw.T @ J
            step = -np.linalg.solve(H + (1e-9 * np.trace(H)) * _EYE5, Jw.T @ r)
            full_step = True
            for _ in range(10):  # halve until the cost drops
                Rn, tn = self._perturb(R, t, step, B)
                trial = self._sampson(_skew(tn) @ Rn, y1, y2)
                cn = cost_of(trial[0])
                if cn <= cost:
                    break
                step = 0.5 * step
                full_step = False
            else:
                break  # no descent left: converged
            if full_step:
                # IRLS converges linearly along the weak yaw/lateral direction;
                # trying the doubled step (one residual pass) about halves the
                # iterations there and keeps the cost monotone.
                R2, t2 = self._perturb(R, t, 2.0 * step, B)
                trial2 = self._sampson(_skew(t2) @ R2, y1, y2)
                c2n = cost_of(trial2[0])
                if c2n < cn:
                    Rn, tn, trial, cn, step = R2, t2, trial2, c2n, 2.0 * step
            R, t, terms, cost = Rn, tn, trial, cn
            if float(np.abs(step).max()) < tol:
                break
        return R, t, terms[0]

    def _warp_rotation_only(self, p1: NDArray[np.float64], R: NDArray[np.float64]) -> NDArray[np.float64]:
        H = self.K @ R @ self.K_inv
        uv = np.column_stack([p1, np.ones(len(p1))]) @ H.T
        return uv[:, :2] / uv[:, 2:3]

    def _base_yaw(self, R: NDArray[np.float64]) -> float:
        """Yaw of the base_link increment implied by a camera rotation."""
        R_b = self.T_base_cam[:3, :3] @ R.T @ self.T_cam_base[:3, :3]
        return float(np.arctan2(R_b[1, 0], R_b[0, 0]))

    def _gate_yaw(self, vision_yaw: float, prior_yaw: float | None) -> tuple[float, bool]:
        if prior_yaw is None:
            return vision_yaw, False
        gate = self.config.yaw_gate_abs + self.config.yaw_gate_rel * abs(prior_yaw)
        if abs(vision_yaw - prior_yaw) <= gate:
            return vision_yaw, False
        return prior_yaw, True

    # ------------------------------------------------- metric ground alignment
    def _ground_normal_cam(self) -> NDArray[np.float64]:
        """Down-pointing ground normal in the optical frame, pitch-corrected."""
        p = self.mount_pitch
        n_base = np.array([np.sin(p), 0.0, -np.cos(p)])
        return self.T_cam_base[:3, :3] @ n_base

    def _ipm_homography(self, x_top: float, y_half: float, dyaw: float) -> NDArray[np.float64]:
        """Homography mapping top-down ground pixels to camera pixels.

        The top-down frame has x (forward) upwards and y (left) leftwards,
        with ``x_top``/``y_half`` at the image top/left edge and ``ipm_res``
        metres per pixel; ``dyaw`` rotates the ground basis so the second
        frame's view is pre-aligned in heading with the first.
        """
        res = self.config.ipm_res
        n = self._ground_normal_cam()
        ez = -n
        f = np.array([0.0, 0.0, 1.0])  # optical axis
        ex = f - (f @ ez) * ez
        ex /= np.linalg.norm(ex)
        ey = np.cross(ez, ex)
        c, s = np.cos(dyaw), np.sin(dyaw)
        ex, ey = c * ex + s * ey, -s * ex + c * ey
        A = np.column_stack([ex, ey, self.cam_height * n])
        G = np.array([[0.0, -res, x_top], [-res, 0.0, y_half], [0.0, 0.0, 1.0]])
        return self.K @ A @ G

    def _ipm_view(
        self, image: NDArray[np.uint8], x_range: tuple[float, float], y_half: float, dyaw: float
    ) -> tuple[NDArray[np.float32], NDArray[np.uint8]]:
        """Top-down view and validity mask (kart + field of view removed)."""
        res = self.config.ipm_res
        nx = int(round((x_range[1] - x_range[0]) / res))
        ny = int(round(2 * y_half / res))
        H = self._ipm_homography(x_range[1], y_half, dyaw)
        if self.camera_model is not None:
            # Distorted stream: the homography gives PINHOLE sample
            # coordinates; the forward distortion maps them onto the actual
            # sensor. Float maps + INTER_LINEAR (the bit-portability dance
            # of the rectified path protects cached research numbers, which
            # a distorted stream never produced). Without this, IPM sampled
            # the distorted image as if pinhole and its scale came out low
            # while its NCC stayed confident (measured: -13% forward scale,
            # trajectory ATE +47% on sept3 with plumb_bob k1=-0.28).
            w_src, h_src = self.image_size
            u = np.arange(ny, dtype=np.float64)
            v = np.arange(nx, dtype=np.float64)[:, None]
            Wh = (H[2, 0] * u + H[2, 2]) + H[2, 1] * v
            Wh = np.where(np.abs(Wh) < 1e-12, 1e-12, Wh)
            px = ((H[0, 0] * u + H[0, 2]) + H[0, 1] * v) / Wh
            py = ((H[1, 0] * u + H[1, 2]) + H[1, 1] * v) / Wh
            dp = self.camera_model.distort_pixels(np.stack([px, py], axis=-1))
            mapx = np.clip(dp[..., 0], -2.0, w_src + 1.0).astype(np.float32)
            mapy = np.clip(dp[..., 1], -2.0, h_src + 1.0).astype(np.float32)
            view = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            valid = cv2.remap(self.mask, mapx, mapy, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        elif self.config.ipm_warp == "auto" and _native_warp_is_portable():
            flags = cv2.WARP_INVERSE_MAP
            border = dict(borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            view = cv2.warpPerspective(image, H, (ny, nx), flags=flags | cv2.INTER_LINEAR, **border)
            valid = cv2.warpPerspective(self.mask, H, (ny, nx), flags=flags | cv2.INTER_NEAREST, **border)
        else:
            xy, frac, near = _remap_tables(H, nx, ny, self.image_size)
            view = cv2.remap(image, xy, frac, cv2.INTER_LINEAR)
            valid = cv2.remap(self.mask, near, None, cv2.INTER_NEAREST)
        sigma = self.config.ipm_highpass_sigma
        ksize = int(round(8.0 * sigma + 1.0)) | 1  # OpenCV's float32 auto-size rule, pinned
        hp = view.astype(np.float32) - cv2.GaussianBlur(
            view.astype(np.float32), (ksize, ksize), sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101
        )
        hp[valid < 128] = 0.0
        return hp, (valid >= 128).astype(np.uint8)

    def _ipm_translation(
        self, prev: NDArray[np.uint8], gray: NDArray[np.uint8], dyaw: float, ncc_gate: float
    ) -> tuple[tuple[float, float], float, float] | None:
        """Planar base_link translation by dense top-down template matching.

        The second frame's yaw-compensated ground band is located inside the
        first frame's larger ground window with masked NCC on high-passed
        views, so the whole texture votes instead of a handful of corners.
        Returns ((forward, left) [m], step sigma [m], NCC strength) or None.
        """
        cfg = self.config
        res = cfg.ipm_res
        search, m1 = self._ipm_view(prev, cfg.ipm_search_fwd, cfg.ipm_search_half, 0.0)
        template, m2 = self._ipm_view(gray, cfg.ipm_template_fwd, cfg.ipm_template_half, -dyaw)
        if int(m2.sum()) < 0.2 * m2.size:
            return None
        scores = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED, mask=m2.astype(np.float32))
        scores = np.nan_to_num(scores, nan=-1.0, posinf=-1.0, neginf=-1.0)
        _, best, _, (u0, v0) = cv2.minMaxLoc(scores)
        if best < ncc_gate:
            return None
        if not (0 < u0 < scores.shape[1] - 1 and 0 < v0 < scores.shape[0] - 1):
            return None  # peak clipped by the search range: step out of bounds
        # Periodic ground texture (expansion joints) aliases the correlation:
        # a rival peak almost as strong means the match is ambiguous.
        r = cfg.ncc_peak_radius_px
        suppressed = scores.copy()
        suppressed[max(0, v0 - r) : v0 + r + 1, max(0, u0 - r) : u0 + r + 1] = -1.0
        if best - float(suppressed.max()) < cfg.ncc_peak_margin:
            return None
        du = self._subpixel(scores[v0, u0 - 1], scores[v0, u0], scores[v0, u0 + 1])
        dv = self._subpixel(scores[v0 - 1, u0], scores[v0, u0], scores[v0 + 1, u0])
        v_zero = (cfg.ipm_search_fwd[1] - cfg.ipm_template_fwd[1]) / res
        u_zero = (cfg.ipm_search_half - cfg.ipm_template_half) / res
        # Forward motion drags the ground towards the camera, so the template
        # lands below its zero-motion row (see docstring of _ipm_homography).
        cam_fwd = (v_zero - (v0 + dv)) * res
        cam_left = (u_zero - (u0 + du)) * res
        # The view is centred on the camera's foot point, not on base_link:
        # remove the lever-arm contribution of the yaw increment.
        c_xy = self.T_base_cam[:2, 3]
        cy, sy = np.cos(dyaw), np.sin(dyaw)
        lever = np.array([[cy - 1.0, -sy], [sy, cy - 1.0]]) @ c_xy
        step = np.array([cam_fwd, cam_left]) - lever
        sigma = float(res * (1.0 + (0.6 - min(best, 0.6)) * 4.0))
        return (float(step[0]), float(step[1])), sigma, float(best)

    @staticmethod
    def _subpixel(a: float, b: float, c: float) -> float:
        denom = a - 2.0 * b + c
        if denom >= -1e-9:
            return 0.0
        return float(np.clip(0.5 * (a - c) / denom, -0.5, 0.5))

    # ----------------------------------------------------- sparse ground fall
    def _sparse_ground_step(
        self,
        q1: NDArray[np.float64],
        q2: NDArray[np.float64],
        R: NDArray[np.float64],
        t: NDArray[np.float64],
        delta_yaw: float,
    ) -> tuple[tuple[float, float], float, float] | None:
        """Fallback: planar translation by GROUND-HOMOGRAPHY consensus.

        With the epipolar rotation ``R`` and unit translation direction
        ``t`` already known, the ground plane (``n1 . X = h`` in frame 1)
        leaves exactly ONE unknown, the step length ``s``:

            x2 ~ (R - (s / h) t n1^T) x1

        so every correspondence yields its own ``s_i`` from a closed-form
        1-D least squares. Points ON the ground agree on one ``s``; walls,
        bushes, wall-base contact bands and the kart's own body do NOT
        (their rays intersect the ground plane elsewhere, so their implied
        ``s_i`` scatter), and a robust weighted-median consensus over
        ``s_i`` ignores them. The previous formulation - a range-weighted
        mean of per-point ground-plane displacements - had no such
        consistency test and was measurably biased by exactly those points
        (forced onto sept3 it recovered only 0.48x of the IPM reference
        scale; near-static wall-base intersections carried the largest
        weights). Refuses pairs whose consensus is too thin or whose
        posterior sigma exceeds the configured bounds.
        """
        cfg = self.config
        h = self.cam_height
        n1 = self._ground_normal_cam()
        C = self.T_base_cam[:3, :3]  # base_link <- camera optical
        _ = (R, t)  # the epipolar pose is deliberately unused (degenerate here)
        # Planar-motion rotation in CAMERA axes from the (IMU-gated) yaw:
        # a world-fixed point's BASE coordinates rotate by -delta_yaw when
        # the kart yaws by +delta_yaw (b2 = Rz(-dyaw)(b1 - m)).
        cy, sy = np.cos(delta_yaw), np.sin(delta_yaw)
        Rz = np.array([[cy, sy, 0.0], [-sy, cy, 0.0], [0.0, 0.0, 1.0]])
        R_pc = C.T @ Rz @ C
        x1_all = np.column_stack([q1, np.ones(len(q1))]) @ self.K_inv.T  # rays, frame 1
        # Only rays that actually LOOK AT nearby ground constrain the step:
        # above-horizon and far-field correspondences have (near-)zero
        # ground sensitivity, and with them included a robust fit locks onto
        # their zero-parallax consensus and crushes the true scale
        # (measured: 0.1x). The cfg ground-range band is the selection.
        g_all = (x1_all @ n1) / h
        slant = np.linalg.norm(x1_all, axis=1) / np.maximum(g_all, 1e-9) * h
        sel = (g_all > 1e-3) & (slant > cfg.min_ground_range_m) & (slant < cfg.max_ground_range_m)
        if int(sel.sum()) < cfg.min_ground_points:
            return None
        x1 = x1_all[sel]
        p2h = np.column_stack([q2[sel], np.ones(int(sel.sum()))])  # pixel homogeneous, frame 2
        # Camera-2 position in frame 1 is C^T m for the base-frame planar
        # step m = (fwd, left, 0); with X2 = R_pc (X1 - C^T m) the ground
        # plane n1 . X1 = h gives  x2 ~ K R_pc (I - C^T m n1^T / h) K^-1 x1,
        # LINEAR in m: p2h ~ A + M m.
        A = x1 @ (self.K @ R_pc).T
        KRC = self.K @ R_pc @ C.T  # columns = pixel response to m_fwd / m_left
        g = g_all[sel]
        M_f = -np.outer(g, KRC[:, 0])
        M_l = -np.outer(g, KRC[:, 1])
        # cross(p2h, A + M m) = 0: rows 0..1 are independent linear equations.
        cA = np.cross(p2h, A)
        cF = np.cross(p2h, M_f)
        cL = np.cross(p2h, M_l)
        Jm = np.stack([cF[:, :2].reshape(-1), cL[:, :2].reshape(-1)], axis=1)  # (2N, 2)
        r0 = cA[:, :2].reshape(-1)  # (2N,)
        # Row scale ~ pixel units: normalise by the homogeneous scale of p2h x
        # (cross rows are ~ fx * pixel error for small errors).
        w = np.ones(len(r0))
        m = np.zeros(2)
        for _round in range(4):
            W = w[:, None]
            JT_J = Jm.T @ (Jm * W)
            JT_r = Jm.T @ (r0 * w)
            try:
                m = np.linalg.solve(JT_J, -JT_r)
            except np.linalg.LinAlgError:
                return None
            resid = (r0 + Jm @ m).reshape(-1, 2)
            per_pt = np.linalg.norm(resid, axis=1)
            scale = float(np.median(per_pt)) * 1.4826 + 1e-9
            wp = 1.0 / (1.0 + (per_pt / (2.5 * scale)) ** 2)  # Cauchy
            w = np.repeat(wp, 2)
        per_pt = np.linalg.norm((r0 + Jm @ m).reshape(-1, 2), axis=1)
        inl = per_pt < 3.0 * (np.median(per_pt) * 1.4826 + 1e-9)
        n_inl = int(inl.sum())
        if n_inl < cfg.min_ground_points:
            return None
        # Posterior sigma of m from the weighted normal equations, mapped to
        # metres via the Jacobian (rows are in pixel-cross units already).
        try:
            cov = np.linalg.inv(Jm.T @ (Jm * w[:, None]))
        except np.linalg.LinAlgError:
            return None
        dof = max(float(np.sum(w)) - 2.0, 1.0)
        s2 = float(np.sum(w * (r0 + Jm @ m) ** 2) / dof)
        sigma_step = float(np.sqrt(max(np.trace(cov) * s2, 0.0)))
        sigma_step = float(np.clip(sigma_step, 1e-3, np.inf))
        step_len = float(np.hypot(m[0], m[1]))
        if sigma_step > max(cfg.max_step_rel_sigma * step_len, cfg.max_step_abs_sigma):
            return None
        strength = float(np.clip(0.5 - sigma_step, 0.0, 0.45))
        return (float(m[0]), float(m[1])), sigma_step, strength

    # ------------------------------------------------------------------- misc
    def _log(self, t0: float, t1: float, status: str, **kw) -> None:
        self.diagnostics.append(VoPairDiagnostics(t0, t1, status, **kw))


def run_visual_odometry(vo: VisualOdometry, frames: Sequence[CameraFrame]) -> list[RelativeMotion]:
    """Process frames in stamp order and collect the emitted increments."""
    out: list[RelativeMotion] = []
    for frame in sorted(frames, key=lambda f: f.stamp):
        motion = vo.process(frame)
        if motion is not None:
            out.append(motion)
    return out
