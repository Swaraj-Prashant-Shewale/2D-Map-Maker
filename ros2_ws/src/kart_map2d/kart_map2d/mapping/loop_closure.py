"""Loop closure: detect trajectory revisits and snap the loop shut.

Dead-reckoning (camera + IMU) drifts by a few percent of distance travelled,
so a lap never quite closes. This module detects revisits from what that
sensor budget actually has - local occupancy patches accumulated along the
trajectory - and corrects the whole trajectory with a small SE2 pose graph.
Everything is classical (correlation + sparse nonlinear least squares).

Why occupancy correlation and not ORB image matching: on an out-and-back
route the revisit happens with *reversed* heading, so the camera sees a
disjoint scene at the same spot and direct descriptor matching starves for
covisibility. The ground occupancy pattern (drivable area with obstacle
contacts) is viewpoint-free once projected into the map plane: a patch built
while driving out and a patch built while driving back overlap wherever the
camera cones intersect, whatever the headings were. Correlation runs over a
full +-180 deg yaw sweep, so no assumption about revisit direction is made.

Detection is verified structurally: the normalized cross-correlation is
masked to jointly observed cells and both patches must lend the overlap a
minimum area of obstacle evidence, so a featureless drivable blob cannot
slide along itself into a false peak. Each match carries an anisotropic
measurement sigma read off the correlation surface (curvature plus the
spread of near-peak candidates): a corridor with straight edges then yields
a laterally tight but along-track loose constraint instead of either a false
certainty or a false rejection. Accepted matches must further survive a
post-solve chi-squared gate - a missed closure costs accuracy, a false one
destroys the map, so the gates are deliberately conservative and a lone
surviving closure additionally needs a high correlation score.

Cost (RTX 4050 laptop, sept3): ~0.7 s per candidate pair, ~35 s for the
50 s run end to end - loop closure is a one-shot (or once-per-lap) batch
step over the accumulated trajectory, not a per-frame stage.

Frames and units
- map frame: planar metres, yaw CCW from the x-axis; ``xyyaw = [x, y, yaw]``.
- base_link / keyframe frame: x forward, y left; patches live in the
  keyframe's own base_link plane.
- All stamps are epoch seconds on the common clock.

Pure functions of their inputs: no pipeline, dataset or model dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.optimize import least_squares
from scipy.signal import fftconvolve

# ------------------------------------------------------------------ SE2 maths


def _wrap(angle):
    """Wrap angle(s) to [-pi, pi)."""
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def se2_compose(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """a . b : first apply b in a's frame, then a (poses as [x, y, yaw])."""
    c, s = np.cos(a[2]), np.sin(a[2])
    return np.array([a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], a[2] + b[2]])


def se2_invert(a: NDArray[np.float64]) -> NDArray[np.float64]:
    c, s = np.cos(a[2]), np.sin(a[2])
    return np.array([-(c * a[0] + s * a[1]), -(-s * a[0] + c * a[1]), -a[2]])


def se2_relative(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """X with b = a . X, i.e. b expressed in a's frame (yaw wrapped)."""
    c, s = np.cos(a[2]), np.sin(a[2])
    dx, dy = b[0] - a[0], b[1] - a[1]
    return np.array([c * dx + s * dy, -s * dx + c * dy, _wrap(b[2] - a[2])])


# ------------------------------------------------------------------ data types


@dataclass(frozen=True)
class KeyframeObservation:
    """One sensor frame's planar evidence, in base_link at ``stamp``.

    ``drivable_xy`` / ``obstacle_xy`` are (N, 2) planar points [m] in the
    vehicle frame (x forward, y left) at the observation time: ground cells
    the sensor saw as drivable and obstacle ground-contact points. How they
    were produced (semantic camera back-projection, LiDAR, ...) is the
    caller's business - this module only rasterizes them.
    """

    stamp: float
    drivable_xy: NDArray[np.float64]
    obstacle_xy: NDArray[np.float64]


@dataclass
class LocalPatch:
    """Local signed occupancy patch in one keyframe's base_link frame.

    ``grid[ix, iy]`` covers keyframe coords ``(idx - (H-1)/2) * cell + center``
    with ``center = (center_forward_m, 0)`` - square, odd-sized, so the array
    centre is an exact cell and rotations pivot cleanly. Field values:
    +1 obstacle evidence, -1 drivable evidence, 0 unseen; ``seen`` marks
    cells with any evidence.
    """

    stamp: float
    xyyaw: NDArray[np.float64]  # estimated map-frame pose used to build it
    grid: NDArray[np.float32]  # (H, H) signed occupancy field
    seen: NDArray[np.bool_]  # (H, H)
    cell: float  # m
    center_forward_m: float
    path_m: float  # estimated path distance at the keyframe


@dataclass
class Closure:
    """One revisit constraint between trajectory keyframes a (early), b (late)."""

    node_a: int
    node_b: int
    stamp_a: float
    stamp_b: float
    xyyaw_ab: NDArray[np.float64]  # measured pose of b in a's frame
    score: float  # masked-NCC peak in [-1, 1]
    overlap_m2: float  # jointly observed area at the peak
    sigma_xyyaw: NDArray[np.float64] | None = None  # anisotropic 1-sigma of the
    # measurement in a's frame [m, m, rad]: correlation-surface curvature plus
    # ambiguity spread, so a corridor ridge yields a laterally tight but
    # along-track loose constraint instead of a false rejection
    residual_xyyaw: NDArray[np.float64] | None = None  # post-solve, a's frame
    chi2: float | None = None  # post-solve, 3 dof
    accepted: bool = False


@dataclass
class ClosureResult:
    corrected_xyyaw: NDArray[np.float64]  # (N, 3), same stamps as the input
    closures: list[Closure]  # accepted and chi2-rejected candidates
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def accepted(self) -> list[Closure]:
        return [c for c in self.closures if c.accepted]


@dataclass
class LoopClosureParams:
    """Tuning knobs; defaults fitted on the September recordings."""

    # keyframes / patches
    spacing_m: float = 1.0  # keyframe spacing along the estimated path
    window_m: float = 7.0  # observations within this path distance join a patch
    half_size_m: float = 12.0  # patch half-extent
    center_forward_m: float = 4.0  # patch centre ahead of the keyframe (camera looks forward)
    cell_m: float = 0.20
    saturate_hits: int = 3  # per-cell evidence saturation (standstill guard)
    # candidate gating
    min_loop_m: float = 18.0  # min estimated path separation of a revisit: closer
    # pairs share observation frames (window is +-window_m) and self-match
    min_loop_same_dir_m: float = 30.0  # same-direction pairs must additionally be
    # farther apart than the whole patch span (2 x half_size): anything closer
    # just re-observes the scene it left, which is odometry, not a revisit
    min_loop_same_dir_s: float = 45.0  # ... and separated in TIME: a same-direction
    # revisit means a lap was driven. Estimated path distance alone can be
    # inflated by an estimator wobble (sept2 logs 28 m of est path in 6 s
    # during a camera gap), and time is the one measure drift cannot stretch
    revisit_min_dyaw_deg: float = 120.0  # heading change below which a pair counts
    # as same-direction for the gates above; a genuine reversed/crossing
    # revisit (out-and-back, lap in the other direction) exceeds it easily
    turn_max_deg_per_m: float = 12.0  # keyframes turning faster than this (over
    # +-2 m of path) are no candidates: relative-pose error inside the patch
    # window smears their content tangentially into high-scoring garbage
    radius_base_m: float = 1.5
    radius_drift_frac: float = 0.20  # of path separation (measured VO drift, not the
    # long-run 2.5% average - scale jitter dominates before it averages out)
    radius_sigma_gain: float = 2.0  # x (sigma_a + sigma_b) when position_sigma given
    radius_cap_m: float = 9.0
    max_candidates_per_kf: int = 4
    # correlation
    yaw_gate_deg: float = 180.0  # full sweep: revisit direction is unknown
    coarse_step_deg: float = 6.0
    fine_step_deg: float = 1.5
    coarse_factor: int = 2  # patch downsampling for the coarse yaw sweep
    blur_m: float = 0.5  # masked pre-correlation blur: thin (cell-wide) structure
    # otherwise decorrelates within ~2 deg of yaw and the coarse sweep steps over it
    min_overlap_m2: float = 25.0
    min_obstacle_m2: float = 1.5  # obstacle evidence each side must lend the overlap;
    # featureless (all-drivable) overlaps cannot match at all, and a couple of
    # chance-aligned blobs in a sliver of overlap must not either
    score_min: float = 0.45
    max_closures: int = 12
    # closure measurement confidence: curvature of the NCC surface at the peak
    # plus the spread of near-peak candidates, per axis of a's frame
    amb_margin: float = 0.06  # candidates within this of the peak count as ambiguity
    obstacle_weight: float = 3.0  # obstacle evidence counts this much more than
    # drivable in the correlation field: structure localises, open area does not
    sigma_xy_unknown_m: float = 1.0  # substitute when the peak's valid island is
    # too small for a curvature fit (boxed in by the overlap gates)
    range_bias_m: float = 1.5  # added (in quadrature) to the forward-axis sigma:
    # opposed viewing directions turn any range-scale error (camera height
    # calibration) into a systematic along-track offset of this order
    curvature_drop: float = 0.10  # sigma = offset at which the fitted NCC drops this much
    sigma_xy_closure_m: float = 0.35  # floor / fallback for hand-built closures
    sigma_xy_max_m: float = 2.5  # clamp; a closure vague on BOTH axes is rejected
    sigma_yaw_closure_deg: float = 2.0  # floor / fallback
    sigma_yaw_max_deg: float = 8.0
    yaw_amb_reject_deg: float = 20.0  # near-peak yaw spread beyond this: no idea
    # of the revisit direction, so the match is noise
    # odometry edge confidence
    odo_sigma_floor_m: float = 0.03
    odo_drift_frac: float = 0.15  # per metre of edge length (measured RPE level)
    odo_sigma_yaw_deg_per_m: float = 0.35
    odo_sigma_yaw_floor_deg: float = 0.2
    # acceptance
    chi2_gate: float = 16.27  # chi2(3 dof) 0.999 quantile
    single_score_min: float = 0.70  # a lone closure can always bend a long chain
    # into agreement, so graph consistency cannot vet it - only detection can


# ------------------------------------------------------------------ patches


def _path_distance(xyyaw: NDArray[np.float64]) -> NDArray[np.float64]:
    xy = np.asarray(xyyaw, dtype=np.float64)[:, :2]
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]


def _keyframe_indices(path: NDArray[np.float64], spacing_m: float) -> NDArray[np.int64]:
    """Greedy subsampling every ``spacing_m`` of travel; endpoints included.

    Standstill produces no keyframes because path distance does not advance.
    """
    idx = [0]
    for i in range(1, len(path)):
        if path[i] - path[idx[-1]] >= spacing_m:
            idx.append(i)
    if idx[-1] != len(path) - 1:
        idx.append(len(path) - 1)
    return np.asarray(idx, dtype=np.int64)


def _interp_pose(
    stamps: NDArray[np.float64], xyyaw: NDArray[np.float64], t: NDArray[np.float64]
) -> NDArray[np.float64]:
    yaw = np.unwrap(np.asarray(xyyaw, dtype=np.float64)[:, 2])
    return np.column_stack(
        [np.interp(t, stamps, xyyaw[:, 0]), np.interp(t, stamps, xyyaw[:, 1]), np.interp(t, stamps, yaw)]
    )


def build_local_patches(
    trajectory_stamps: NDArray[np.float64],
    trajectory_xyyaw: NDArray[np.float64],
    observations: Sequence[KeyframeObservation],
    params: LoopClosureParams | None = None,
) -> list[LocalPatch]:
    """Rasterize per-frame evidence into per-keyframe local occupancy patches.

    Each observation's points are moved into a keyframe's frame through the
    *relative* pose between observation time and keyframe time taken from the
    estimated trajectory - locally accurate even when the trajectory has
    drifted globally, which is exactly what makes the patches loop-closure
    evidence rather than victims of the drift they are meant to measure.
    (Implementation: every observation is transformed into the map frame
    once, and each keyframe re-expresses its whole window with one inverse
    pose - algebraically identical and O(keyframes + points), not
    O(keyframes x observations).)

    Observations join every keyframe within ``window_m`` of travel, so a patch
    fuses a short bidirectional bundle of frames around its keyframe. Per-cell
    hit counts are one per observation frame and saturate at
    ``saturate_hits``, so a standstill cannot outvote moving coverage.
    """
    p = params or LoopClosureParams()
    stamps = np.asarray(trajectory_stamps, dtype=np.float64)
    xyyaw = np.asarray(trajectory_xyyaw, dtype=np.float64)
    path = _path_distance(xyyaw)
    kf_idx = _keyframe_indices(path, p.spacing_m)
    obs_stamps = np.array([o.stamp for o in observations], dtype=np.float64)
    obs_path = np.interp(obs_stamps, stamps, path)
    obs_pose = _interp_pose(stamps, xyyaw, obs_stamps)

    # every observation once into the map frame, tagged with its frame index
    map_pts: dict[str, list[NDArray[np.float64]]] = {"drv": [], "obst": []}
    map_ids: dict[str, list[NDArray[np.int64]]] = {"drv": [], "obst": []}
    for oi, obs in enumerate(observations):
        c, s = np.cos(obs_pose[oi, 2]), np.sin(obs_pose[oi, 2])
        rot = np.array([[c, -s], [s, c]])
        for key, pts in (("drv", obs.drivable_xy), ("obst", obs.obstacle_xy)):
            pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
            map_pts[key].append(pts @ rot.T + obs_pose[oi, :2])
            map_ids[key].append(np.full(len(pts), oi, dtype=np.int64))
    channels = {
        key: (
            np.concatenate(map_pts[key]) if map_pts[key] else np.empty((0, 2)),
            np.concatenate(map_ids[key]) if map_ids[key] else np.empty(0, dtype=np.int64),
        )
        for key in ("drv", "obst")
    }
    # per-channel points sorted by the owning observation's path position let a
    # keyframe pull its whole window as one contiguous block
    for key, (pts, ids) in channels.items():
        by_path = np.argsort(obs_path[ids], kind="stable") if len(ids) else np.empty(0, np.int64)
        channels[key] = (pts[by_path], ids[by_path])

    half = int(round(p.half_size_m / p.cell_m))
    size = 2 * half + 1
    center = np.array([p.center_forward_m, 0.0])
    patches: list[LocalPatch] = []
    for k in kf_idx:
        pose_k = np.array([xyyaw[k, 0], xyyaw[k, 1], xyyaw[k, 2]])
        c, s = np.cos(pose_k[2]), np.sin(pose_k[2])
        inv_rot = np.array([[c, s], [-s, c]])
        counts = {}
        for key, (pts, ids) in channels.items():
            path_of = obs_path[ids] if len(ids) else np.empty(0)
            lo = np.searchsorted(path_of, path[k] - p.window_m)
            hi = np.searchsorted(path_of, path[k] + p.window_m)
            block, bids = pts[lo:hi], ids[lo:hi]
            grid_counts = np.zeros(size * size, dtype=np.int32)
            if len(block):
                pk = (block - pose_k[:2]) @ inv_rot.T
                ij = np.rint((pk - center) / p.cell_m).astype(np.int64) + half
                ok = (ij >= 0).all(axis=1) & (ij < size).all(axis=1)
                if ok.any():
                    flat = ij[ok, 0] * size + ij[ok, 1]
                    # one hit per cell per observation frame
                    keys = np.unique(bids[ok] * (size * size) + flat)
                    grid_counts = np.bincount(keys % (size * size), minlength=size * size).astype(np.int32)
            counts[key] = grid_counts.reshape(size, size)
        drv_f = np.minimum(counts["drv"], p.saturate_hits) / p.saturate_hits
        obst_f = np.minimum(counts["obst"], p.saturate_hits) / p.saturate_hits
        patches.append(
            LocalPatch(
                stamp=float(stamps[k]),
                xyyaw=pose_k,
                grid=(obst_f - drv_f).astype(np.float32),
                seen=(counts["drv"] + counts["obst"]) > 0,
                cell=p.cell_m,
                center_forward_m=p.center_forward_m,
                path_m=float(path[k]),
            )
        )
    return patches


def patches_from_grid(
    grid,
    trajectory_stamps: NDArray[np.float64],
    trajectory_xyyaw: NDArray[np.float64],
    params: LoopClosureParams | None = None,
) -> list[LocalPatch]:
    """Cut per-keyframe patches out of an accumulated EvidenceGrid.

    Fallback source when per-frame observations were not kept. Caveat: an
    accumulated grid blends the evidence of *every* visit, so around a
    revisit each patch already contains both passes smeared together; the
    correlation peak washes out once drift approaches the local structure
    scale. Prefer :func:`build_local_patches` when possible.

    ``grid`` is duck-typed: needs ``cell``, ``origin`` and ``classify()``
    (codes 0 unknown / 1 drivable / 2 obstacle, shape (nx, ny)).
    """
    p = params or LoopClosureParams()
    classes = np.asarray(grid.classify())
    origin = np.asarray(grid.origin, dtype=np.float64)
    gcell = float(grid.cell)
    stamps = np.asarray(trajectory_stamps, dtype=np.float64)
    xyyaw = np.asarray(trajectory_xyyaw, dtype=np.float64)
    path = _path_distance(xyyaw)
    kf_idx = _keyframe_indices(path, p.spacing_m)

    half = int(round(p.half_size_m / p.cell_m))
    size = 2 * half + 1
    ax = (np.arange(size) - half) * p.cell_m
    px, py = np.meshgrid(ax + p.center_forward_m, ax, indexing="ij")  # keyframe coords
    patches: list[LocalPatch] = []
    for k in kf_idx:
        x, y, yaw = xyyaw[k]
        c, s = np.cos(yaw), np.sin(yaw)
        wx = x + c * px - s * py
        wy = y + s * px + c * py
        ix = np.floor((wx - origin[0]) / gcell).astype(np.int64)
        iy = np.floor((wy - origin[1]) / gcell).astype(np.int64)
        ok = (ix >= 0) & (ix < classes.shape[0]) & (iy >= 0) & (iy < classes.shape[1])
        cls = np.zeros((size, size), dtype=np.uint8)
        cls[ok] = classes[ix[ok], iy[ok]]
        patches.append(
            LocalPatch(
                stamp=float(stamps[k]),
                xyyaw=xyyaw[k].copy(),
                grid=((cls == 2).astype(np.float32) - (cls == 1).astype(np.float32)),
                seen=cls != 0,
                cell=p.cell_m,
                center_forward_m=p.center_forward_m,
                path_m=float(path[k]),
            )
        )
    return patches


# ------------------------------------------------------------------ correlation


def _downsample(a: NDArray, factor: int) -> NDArray:
    """Block-mean downsampling (truncates the ragged edge)."""
    if factor <= 1:
        return np.asarray(a, dtype=np.float64)
    n = (np.array(a.shape) // factor) * factor
    a = np.asarray(a[: n[0], : n[1]], dtype=np.float64)
    return a.reshape(n[0] // factor, factor, n[1] // factor, factor).mean(axis=(1, 3))


def _masked_blur(field: NDArray[np.float64], mask: NDArray[np.float64], sigma_cells: float) -> NDArray[np.float64]:
    """Gaussian blur that does not bleed the unseen (mask=0) zeros into the field."""
    if sigma_cells <= 0:
        return field
    num = gaussian_filter(field * mask, sigma_cells)
    den = gaussian_filter(mask, sigma_cells)
    return np.where(den > 0.05, num / np.maximum(den, 1e-9), 0.0)


def _rotated(arrays: Sequence[NDArray[np.float64]], yaw: float) -> list[NDArray[np.float64]]:
    """Rotate patch arrays by ``yaw`` about the array centre (bilinear, zero fill).

    A point at patch coords q lands at R(yaw) q in the rotated patch, matching
    the map-frame yaw convention (CCW, array axis 0 = x, axis 1 = y).
    """
    n = arrays[0].shape[0]
    m = 0.5 * (n - 1)
    idx = np.arange(n, dtype=np.float64) - m
    gx, gy = np.meshgrid(idx, idx, indexing="ij")
    c, s = np.cos(-yaw), np.sin(-yaw)  # output cell samples input at R(-yaw)
    coords = np.stack([c * gx - s * gy + m, s * gx + c * gy + m])
    return [map_coordinates(a, coords, order=1, mode="constant", cval=0.0) for a in arrays]


def _masked_ncc(
    field_f: NDArray[np.float64],
    mask_f: NDArray[np.float64],
    pos_f: NDArray[np.float64],
    field_t: NDArray[np.float64],
    mask_t: NDArray[np.float64],
    pos_t: NDArray[np.float64],
    min_overlap: float,
    min_pos: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Masked normalized cross-correlation (Padfield-style) via FFT.

    Returns ``(ncc, overlap)`` in 'full' correlation layout: index k means the
    template is shifted by ``d = k - (H_t - 1)`` cells relative to the fixed
    patch. Windows are invalidated (``-inf``) when the jointly observed area
    is below ``min_overlap`` cells or when either side contributes fewer than
    ``min_pos`` cells of positive (obstacle) evidence to the overlap: a flat
    drivable blob sliding along itself must not look like a match - only
    obstacle structure localizes.
    """
    f, mf = field_f * mask_f, mask_f
    t, mt = (field_t * mask_t)[::-1, ::-1], mask_t[::-1, ::-1]
    conv = lambda a, b: fftconvolve(a, b, mode="full")  # noqa: E731
    ov = conv(mf, mt)
    sf, sf2 = conv(f, mt), conv(f * f, mt)
    st, st2 = conv(mf, t), conv(mf, t * t)
    cc = conv(f, t)
    po_f = conv(pos_f, mt)  # obstacle evidence each side lends the overlap
    po_t = conv(mf, pos_t[::-1, ::-1])
    with np.errstate(invalid="ignore", divide="ignore"):
        ov_safe = np.maximum(ov, 1e-9)
        var_f = sf2 - sf * sf / ov_safe
        var_t = st2 - st * st / ov_safe
        ncc = (cc - sf * st / ov_safe) / np.sqrt(np.maximum(var_f * var_t, 1e-12))
    bad = (ov < min_overlap) | (po_f < min_pos) | (po_t < min_pos)
    bad |= (var_f < 1e-6 * ov_safe) | (var_t < 1e-6 * ov_safe)
    ncc[bad] = -np.inf
    return ncc, ov


def _gate_mask(size: int, cell: float, d_pred: NDArray[np.float64], radius: float) -> NDArray[np.bool_]:
    """Translation gate: allow peaks within ``radius`` of the predicted shift."""
    d = (np.arange(2 * size - 1) - (size - 1)) * cell
    dx = d[:, None] - d_pred[0]
    dy = d[None, :] - d_pred[1]
    return dx * dx + dy * dy <= radius * radius


@dataclass
class _Match:
    xyyaw_ab: NDArray[np.float64]
    score: float
    overlap_m2: float
    sigma_xyyaw: NDArray[np.float64]


def _quadratic_sigmas(ncc: NDArray[np.float64], ki: int, kj: int, cell: float, drop: float) -> NDArray[np.float64]:
    """Per-axis 1-sigma of the peak from a local 2D quadratic fit [m].

    Fits ``ncc ~ b0 + b.u + 0.5 u^T H u`` on the neighbourhood and reports,
    per axis, the offset at which the fit falls ``drop`` below the peak -
    infinite along a flat ridge, small across it, which is exactly the
    anisotropy the pose graph needs. NaN means the surface has too few valid
    cells to measure curvature at all (a tiny valid island boxed in by the
    overlap gates), which is *not* flatness - the caller substitutes a
    moderate default instead of rejecting.
    """
    k = max(3, int(round(1.0 / cell)))
    sl = ncc[max(0, ki - k) : ki + k + 1, max(0, kj - k) : kj + k + 1]
    ui = (np.arange(max(0, ki - k), min(ki + k + 1, ncc.shape[0])) - ki) * cell
    vi = (np.arange(max(0, kj - k), min(kj + k + 1, ncc.shape[1])) - kj) * cell
    u, v = np.meshgrid(ui, vi, indexing="ij")
    good = np.isfinite(sl)
    if good.sum() < 12:
        return np.array([np.nan, np.nan])
    A = np.column_stack([np.ones(good.sum()), u[good], v[good], u[good] ** 2, u[good] * v[good], v[good] ** 2])
    beta, *_ = np.linalg.lstsq(A, sl[good], rcond=None)
    H = -np.array([[2 * beta[3], beta[4]], [beta[4], 2 * beta[5]]])  # negative Hessian
    out = np.full(2, np.inf)
    if H[0, 0] > 0 and H[1, 1] > 0 and np.linalg.det(H) > 1e-12:
        Hinv = np.linalg.inv(H)
        for i in range(2):
            if Hinv[i, i] > 0:
                out[i] = float(np.sqrt(2.0 * drop * Hinv[i, i]))
    return out


def _correlate_pair(
    patch_a: LocalPatch,
    patch_b: LocalPatch,
    x_pred: NDArray[np.float64],
    radius: float,
    p: LoopClosureParams,
    debug: dict | None = None,
) -> _Match | None:
    """Best gated alignment of patch_b onto patch_a over a full yaw sweep.

    Coarse stage: downsampled patches over the yaw gate at ``coarse_step_deg``,
    keeping every near-peak candidate as ambiguity evidence. Fine stage: full
    resolution around the coarse winner, sub-cell/sub-step refinement, and an
    anisotropic measurement sigma from surface curvature plus the candidate
    spread - a corridor with straight edges localises tightly across-track
    but poorly along-track, and the sigma says so instead of pretending. A
    pair whose near-peak yaws disagree wildly, or that localises on no axis
    at all, returns None. The measured relative pose maps b's frame into
    a's: ``p_a = R(yaw) p_b + t``.
    """
    center = np.array([patch_a.center_forward_m, 0.0])
    yaw_gate = np.radians(p.yaw_gate_deg)

    def d_to_t(yaw: float, d: NDArray[np.float64]) -> NDArray[np.float64]:
        c, s = np.cos(yaw), np.sin(yaw)
        return center - np.array([c * center[0] - s * center[1], s * center[0] + c * center[1]]) + d

    def gated(ncc, yaw, cell, size):
        d_pred = x_pred[:2] - center + np.array(
            [np.cos(yaw) * center[0] - np.sin(yaw) * center[1], np.sin(yaw) * center[0] + np.cos(yaw) * center[1]]
        )
        return np.where(_gate_mask(size, cell, d_pred, radius), ncc, -np.inf)

    # ------------------------------------------------------------- coarse
    factor = max(1, int(p.coarse_factor))
    fa = _downsample(patch_a.grid, factor)
    ma = _downsample(patch_a.seen.astype(np.float64), factor)
    fb = _downsample(patch_b.grid, factor)
    mb = _downsample(patch_b.seen.astype(np.float64), factor)
    # binary obstacle-area masks (downsampled to fractional coverage) so the
    # min-area gate counts square metres, undiluted by hit-count saturation
    pos_a_c = _downsample((patch_a.grid > 0.0).astype(np.float64), factor)
    pos_b_c = _downsample((patch_b.grid > 0.0).astype(np.float64), factor)
    ma, mb = (ma >= 0.5).astype(np.float64), (mb >= 0.5).astype(np.float64)
    cell_c = patch_a.cell * factor
    fa = np.where(fa > 0.0, fa * p.obstacle_weight, fa)
    fb = np.where(fb > 0.0, fb * p.obstacle_weight, fb)
    fa = _masked_blur(fa, ma, p.blur_m / cell_c)
    fb = _masked_blur(fb, mb, p.blur_m / cell_c)
    step = np.radians(p.coarse_step_deg)
    n_steps = int(np.ceil(yaw_gate / step))
    yaws = _wrap(x_pred[2] + np.arange(-n_steps, n_steps + 1) * step)
    min_ov = p.min_overlap_m2 / (cell_c * cell_c)
    min_pos = p.min_obstacle_m2 / (cell_c * cell_c)
    size_c = fa.shape[0]
    per_yaw: list[tuple[float, float, NDArray[np.float64]]] = []  # (score, yaw, d)
    for yaw in yaws:
        rf, rm, rp = _rotated([fb, mb, pos_b_c], yaw)
        ncc, _ = _masked_ncc(fa, ma, pos_a_c, rf, (rm >= 0.5).astype(np.float64), rp, min_ov, min_pos)
        ncc = gated(ncc, yaw, cell_c, size_c)
        k = np.argmax(ncc)
        if np.isfinite(ncc.flat[k]):
            ki, kj = np.unravel_index(k, ncc.shape)
            per_yaw.append((float(ncc.flat[k]), float(yaw), (np.array([ki, kj]) - (size_c - 1)) * cell_c))
    if not per_yaw:
        if debug is not None:
            debug["reason"] = "no valid coarse window (overlap / obstacle-area gates)"
        return None
    score_c, yaw_c, d_c = max(per_yaw, key=lambda item: item[0])

    # ambiguity: every coarse candidate nearly as good as the winner. Wild
    # yaw disagreement means the match is noise; translation disagreement
    # widens the measurement sigma instead of faking certainty.
    near = [(sc, yw, d_to_t(yw, d)) for sc, yw, d in per_yaw if sc >= score_c - p.amb_margin]
    dyaw = np.array([_wrap(yw - yaw_c) for _, yw, _ in near])
    spread_yaw = float(np.sqrt(np.mean(dyaw**2))) if len(near) > 1 else 0.0
    if debug is not None:
        debug.update(score_coarse=score_c, spread_yaw_deg=float(np.degrees(spread_yaw)), n_near=len(near))
    if spread_yaw > np.radians(p.yaw_amb_reject_deg):
        if debug is not None:
            debug["reason"] = "yaw ambiguity"
        return None
    t_near = np.array([t for _, _, t in near])
    spread_t = t_near.std(axis=0) if len(near) > 1 else np.zeros(2)

    # --------------------------------------------------------------- fine
    fa1, ma1 = patch_a.grid.astype(np.float64), patch_a.seen.astype(np.float64)
    fb1, mb1 = patch_b.grid.astype(np.float64), patch_b.seen.astype(np.float64)
    pos_a1, pos_b1 = (fa1 > 0.0).astype(np.float64), (fb1 > 0.0).astype(np.float64)
    fa1 = np.where(fa1 > 0.0, fa1 * p.obstacle_weight, fa1)
    fb1 = np.where(fb1 > 0.0, fb1 * p.obstacle_weight, fb1)
    fa1 = _masked_blur(fa1, ma1, p.blur_m / patch_a.cell)
    fb1 = _masked_blur(fb1, mb1, p.blur_m / patch_a.cell)
    fine = np.radians(p.fine_step_deg)
    n_fine = int(np.ceil(step / fine))
    yaws_f = yaw_c + np.arange(-n_fine, n_fine + 1) * fine
    min_ov1 = p.min_overlap_m2 / (patch_a.cell**2)
    min_pos1 = p.min_obstacle_m2 / (patch_a.cell**2)
    size = fa1.shape[0]
    best = (-np.inf, 0.0, np.zeros(2))
    best_ncc = None
    ov_at_peak = 0.0
    scores_by_yaw = []
    for yaw in yaws_f:
        rf, rm, rp = _rotated([fb1, mb1, pos_b1], yaw)
        ncc, ov = _masked_ncc(fa1, ma1, pos_a1, rf, (rm >= 0.5).astype(np.float64), rp, min_ov1, min_pos1)
        ncc = gated(ncc, yaw, patch_a.cell, size)
        k = np.argmax(ncc)
        scores_by_yaw.append(float(ncc.flat[k]))
        if ncc.flat[k] > best[0]:
            ki, kj = np.unravel_index(k, ncc.shape)
            best = (float(ncc.flat[k]), float(yaw), (np.array([ki, kj]) - (size - 1)) * patch_a.cell)
            best_ncc = ncc
            ov_at_peak = float(ov[ki, kj])
    score, yaw_b, d_b = best
    if not np.isfinite(score):
        if debug is not None:
            debug["reason"] = "no valid fine window"
        return None

    # sub-cell peak + curvature sigmas from a local quadratic surface fit
    ki = int(round(d_b[0] / patch_a.cell)) + size - 1
    kj = int(round(d_b[1] / patch_a.cell)) + size - 1
    d_ref = d_b.astype(np.float64).copy()
    for axis, kk in ((0, ki), (1, kj)):
        if not 0 < kk < 2 * size - 2:
            continue
        sl = best_ncc[kk - 1 : kk + 2, kj] if axis == 0 else best_ncc[ki, kk - 1 : kk + 2]
        if np.isfinite(sl).all():
            denom = sl[0] - 2 * sl[1] + sl[2]
            if denom < -1e-12:
                d_ref[axis] += 0.5 * (sl[0] - sl[2]) / denom * patch_a.cell
    sig_curv = _quadratic_sigmas(best_ncc, ki, kj, patch_a.cell, p.curvature_drop)
    # unmeasurable curvature (tiny valid island) is not flatness
    sig_curv = np.where(np.isnan(sig_curv), p.sigma_xy_unknown_m, sig_curv)

    i_best = int(np.argmax(scores_by_yaw))
    yaw_ref = yaw_b
    sig_yaw_curv = np.inf
    if 0 < i_best < len(scores_by_yaw) - 1:
        s0, s1, s2 = scores_by_yaw[i_best - 1 : i_best + 2]
        if np.isfinite([s0, s1, s2]).all():
            denom = s0 - 2 * s1 + s2
            if denom < -1e-12:
                yaw_ref += 0.5 * (s0 - s2) / denom * fine
                sig_yaw_curv = float(np.sqrt(2.0 * p.curvature_drop * fine**2 / -denom))

    sig_xy = np.clip(np.maximum(sig_curv, spread_t), p.sigma_xy_closure_m, p.sigma_xy_max_m)
    # the camera looks along +x of a's frame, so a range-scale error (camera
    # height calibration) biases the match along x; the lateral axis is clean
    sig_xy[0] = min(np.hypot(sig_xy[0], p.range_bias_m), p.sigma_xy_max_m)
    if debug is not None:
        debug.update(sig_curv=sig_curv.tolist(), spread_t=spread_t.tolist(), score=score)
    if (sig_xy >= p.sigma_xy_max_m - 1e-9).all():
        if debug is not None:
            debug["reason"] = "localises on no axis"
        return None  # localises on no axis: worthless as a constraint
    sig_yaw = float(
        np.clip(max(sig_yaw_curv, spread_yaw), np.radians(p.sigma_yaw_closure_deg), np.radians(p.sigma_yaw_max_deg))
    )

    # p_a = R(yaw) p_b + t with t = c0 - R(yaw) c0 + d (c0 = patch centre offset)
    t = d_to_t(yaw_ref, d_ref)
    return _Match(
        xyyaw_ab=np.array([t[0], t[1], _wrap(yaw_ref)]),
        score=score,
        overlap_m2=ov_at_peak * patch_a.cell**2,
        sigma_xyyaw=np.array([sig_xy[0], sig_xy[1], sig_yaw]),
    )


# ------------------------------------------------------------------ pose graph


def _closure_sigma(c: Closure, p: LoopClosureParams) -> NDArray[np.float64]:
    """A closure's measurement sigma, falling back to the params floor."""
    if c.sigma_xyyaw is not None:
        return np.asarray(c.sigma_xyyaw, dtype=np.float64)
    return np.array([p.sigma_xy_closure_m, p.sigma_xy_closure_m, np.radians(p.sigma_yaw_closure_deg)])


def _solve_pose_graph(
    nodes_xyyaw: NDArray[np.float64],
    odo_sigma: NDArray[np.float64],
    closures: list[Closure],
    p: LoopClosureParams,
) -> NDArray[np.float64]:
    """Sparse nonlinear least squares over SE2 nodes; node 0 stays fixed.

    Odometry edges connect consecutive nodes with the input trajectory's
    relative poses; closure edges add the measured revisit constraints, both
    weighted by their sigmas. Residuals and the (analytic, sparse) Jacobian
    are fully vectorised - a few hundred nodes solve in milliseconds.

    Robustness is IRLS-Huber applied to the *closure* edges only (odometry
    edges between consecutive EKF poses are never outliers): each round
    inflates a closure's sigma by its Huber weight and re-solves with a plain
    quadratic loss. scipy's built-in robust losses crawl on this problem
    (hundreds of iterations without converging); plain linear solves with an
    exact Jacobian converge in a handful of evaluations.
    """
    n = len(nodes_xyyaw)
    odo_meas = np.array([se2_relative(nodes_xyyaw[k], nodes_xyyaw[k + 1]) for k in range(n - 1)])

    ea = np.r_[np.arange(n - 1), [c.node_a for c in closures]].astype(np.int64)
    eb = np.r_[np.arange(1, n), [c.node_b for c in closures]].astype(np.int64)
    meas = np.vstack([odo_meas, *[c.xyyaw_ab[None, :] for c in closures]]) if closures else odo_meas
    sig0 = (
        np.vstack([odo_sigma, *[_closure_sigma(c, p)[None, :] for c in closures]]) if closures else odo_sigma
    )
    sig = sig0.copy()
    n_edges = len(ea)

    def unpack(x):
        poses = np.empty((n, 3))
        poses[0] = nodes_xyyaw[0]
        poses[1:] = x.reshape(n - 1, 3)
        return poses

    def residuals(x):
        poses = unpack(x)
        pa, pb = poses[ea], poses[eb]
        c, s = np.cos(pa[:, 2]), np.sin(pa[:, 2])
        dx, dy = pb[:, 0] - pa[:, 0], pb[:, 1] - pa[:, 1]
        rel = np.column_stack([c * dx + s * dy, -s * dx + c * dy, pb[:, 2] - pa[:, 2]])
        r = rel - meas
        r[:, 2] = _wrap(r[:, 2])
        return (r / sig).reshape(-1)

    # analytic Jacobian: 3x3 blocks per edge endpoint, node 0 excluded
    def jacobian(x):
        poses = unpack(x)
        pa = poses[ea]
        c, s = np.cos(pa[:, 2]), np.sin(pa[:, 2])
        dx, dy = poses[eb][:, 0] - pa[:, 0], poses[eb][:, 1] - pa[:, 1]
        rel_x = c * dx + s * dy
        rel_y = -s * dx + c * dy
        rows, cols, vals = [], [], []
        base = 3 * np.arange(n_edges)
        for node_idx, entries in (
            (ea, [(0, 0, -c), (0, 1, -s), (0, 2, rel_y), (1, 0, s), (1, 1, -c), (1, 2, -rel_x), (2, 2, -np.ones(n_edges))]),
            (eb, [(0, 0, c), (0, 1, s), (1, 0, -s), (1, 1, c), (2, 2, np.ones(n_edges))]),
        ):
            free = node_idx > 0  # node 0 is fixed
            col0 = 3 * (node_idx - 1)
            for ri, ci, v in entries:
                rows.append(base[free] + ri)
                cols.append(col0[free] + ci)
                vals.append((v[free] if isinstance(v, np.ndarray) else np.full(free.sum(), v)) / sig[free, ri])
        return sparse.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(3 * n_edges, 3 * (n - 1)),
        )

    x0 = nodes_xyyaw[1:].reshape(-1)
    for _ in range(3):  # IRLS-Huber rounds on the closure edges
        sol = least_squares(residuals, x0, jac=jacobian, method="trf", max_nfev=100)
        x0 = sol.x
        if not closures:
            break
        r = sol.fun.reshape(-1, 3)[n - 1 :]  # closure rows, already sigma-scaled
        norm = np.linalg.norm(r * sig[n - 1 :] / sig0[n - 1 :], axis=1)  # undo inflation
        w = np.minimum(1.0, 2.0 / np.maximum(norm, 1e-9))  # Huber weight, delta = 2 sigma
        sig[n - 1 :] = sig0[n - 1 :] / np.sqrt(w)[:, None]
    return unpack(sol.x)


def _apply_correction(
    stamps: NDArray[np.float64],
    xyyaw: NDArray[np.float64],
    node_stamps: NDArray[np.float64],
    nodes_old: NDArray[np.float64],
    nodes_new: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Warp the full trajectory through the per-node SE2 corrections.

    Each sample is re-expressed relative to its two bracketing nodes' *old*
    poses, re-attached to their *new* poses, and the two candidates blended
    by time fraction - piecewise-rigid with a smooth seam, so local shape
    (what RPE measures) is preserved while the global error redistributes.
    """
    yaw_u = np.unwrap(xyyaw[:, 2])
    seg = np.clip(np.searchsorted(node_stamps, stamps, side="right") - 1, 0, len(node_stamps) - 2)
    t0, t1 = node_stamps[seg], node_stamps[seg + 1]
    alpha = np.clip((stamps - t0) / np.maximum(t1 - t0, 1e-9), 0.0, 1.0)

    def attach(node_idx):
        old, new = nodes_old[node_idx], nodes_new[node_idx]
        c_o, s_o = np.cos(old[:, 2]), np.sin(old[:, 2])
        dx, dy = xyyaw[:, 0] - old[:, 0], xyyaw[:, 1] - old[:, 1]
        lx, ly = c_o * dx + s_o * dy, -s_o * dx + c_o * dy
        dyaw = yaw_u - old[:, 2]
        c_n, s_n = np.cos(new[:, 2]), np.sin(new[:, 2])
        return np.column_stack(
            [new[:, 0] + c_n * lx - s_n * ly, new[:, 1] + s_n * lx + c_n * ly, new[:, 2] + dyaw]
        )

    a = attach(seg)
    b = attach(seg + 1)
    out = np.empty_like(xyyaw)
    out[:, :2] = (1.0 - alpha[:, None]) * a[:, :2] + alpha[:, None] * b[:, :2]
    out[:, 2] = a[:, 2] + alpha * _wrap(b[:, 2] - a[:, 2])
    return out


# ------------------------------------------------------------------ main entry


def detect_and_close(
    trajectory_stamps: NDArray[np.float64],
    trajectory_xyyaw: NDArray[np.float64],
    source,
    position_sigma: NDArray[np.float64] | None = None,
    params: LoopClosureParams | None = None,
) -> ClosureResult:
    """Detect revisits along an estimated trajectory and close the loops.

    Parameters
    - trajectory_stamps, trajectory_xyyaw: (N,), (N, 3) estimated trajectory
      (map frame, m / rad).
    - source: what the revisit detector correlates - one of
      * a list of :class:`LocalPatch` (from :func:`build_local_patches`),
      * a list of :class:`KeyframeObservation` (patches get built here),
      * an accumulated EvidenceGrid-like object (see :func:`patches_from_grid`).
    - position_sigma: optional (N,) planar 1-sigma position uncertainty per
      sample [m] (e.g. from the EKF covariance). Sharpens the candidate
      search radius and the odometry edge weighting; without it a drift-rate
      model on path distance is used.
    - params: :class:`LoopClosureParams`; defaults fit the September runs.

    Returns a :class:`ClosureResult` with the corrected trajectory (same
    stamps), every considered closure (accepted or chi2-rejected) and
    diagnostics. With no accepted closure the input trajectory is returned
    unchanged - false closures must stay rarer than missed ones.
    """
    p = params or LoopClosureParams()
    stamps = np.asarray(trajectory_stamps, dtype=np.float64)
    xyyaw = np.asarray(trajectory_xyyaw, dtype=np.float64)

    if hasattr(source, "classify") and hasattr(source, "cell"):
        patches = patches_from_grid(source, stamps, xyyaw, p)
    else:
        source = list(source)
        if source and isinstance(source[0], KeyframeObservation):
            patches = build_local_patches(stamps, xyyaw, source, p)
        else:
            patches = source

    diag: dict[str, Any] = {"keyframes": len(patches)}
    if len(patches) < 3:
        return ClosureResult(xyyaw.copy(), [], diag | {"reason": "too few keyframes"})

    kf_pose = np.array([pt.xyyaw for pt in patches])
    kf_path = np.array([pt.path_m for pt in patches])
    kf_stamp = np.array([pt.stamp for pt in patches])
    kf_sigma = (
        np.interp(kf_stamp, stamps, np.asarray(position_sigma, dtype=np.float64))
        if position_sigma is not None
        else None
    )

    # ---------------------------------------------------------- candidates
    def radius(i: int, j: int) -> float:
        sep = kf_path[j] - kf_path[i]
        r = p.radius_base_m + p.radius_drift_frac * sep
        if kf_sigma is not None:
            r = p.radius_base_m + p.radius_sigma_gain * (kf_sigma[i] + kf_sigma[j])
        return float(min(r, p.radius_cap_m))

    # keyframes captured while turning hard have tangentially smeared patches
    # (relative-pose error inside the window scales with yaw rate), which
    # correlate into confident garbage - keep them out of the matching.
    path_full = _path_distance(xyyaw)
    yaw_full = np.unwrap(xyyaw[:, 2])
    kf_s = np.interp(kf_stamp, stamps, path_full)
    turn = np.abs(
        np.interp(kf_s + 2.0, path_full, yaw_full) - np.interp(kf_s - 2.0, path_full, yaw_full)
    ) / np.radians(4.0 * p.turn_max_deg_per_m)
    steady = turn < 1.0

    n = len(patches)
    gap = np.linalg.norm(kf_pose[None, :, :2] - kf_pose[:, None, :2], axis=2)
    candidates: list[tuple[int, int]] = []
    for j in range(n):
        if not steady[j]:
            continue
        def is_revisit_candidate(i: int) -> bool:
            sep = kf_path[j] - kf_path[i]
            if abs(_wrap(kf_pose[j, 2] - kf_pose[i, 2])) >= np.radians(p.revisit_min_dyaw_deg):
                return sep > p.min_loop_m
            return sep > p.min_loop_same_dir_m and kf_stamp[j] - kf_stamp[i] > p.min_loop_same_dir_s

        cand = [i for i in range(j) if steady[i] and is_revisit_candidate(i) and gap[i, j] <= radius(i, j)]
        cand.sort(key=lambda i: gap[i, j])
        candidates += [(i, j) for i in cand[: p.max_candidates_per_kf]]
    diag["candidates"] = len(candidates)
    diag["turning_keyframes"] = int((~steady).sum())

    # ---------------------------------------------------------- correlation
    tentative: list[Closure] = []
    evaluated = 0
    for j in sorted({j for _, j in candidates}):
        best: Closure | None = None
        for i, jj in candidates:
            if jj != j:
                continue
            evaluated += 1
            x_pred = se2_relative(kf_pose[i], kf_pose[j])
            match = _correlate_pair(patches[i], patches[j], x_pred, radius(i, j), p)
            if match is None or match.score < p.score_min:
                continue
            if best is None or match.score > best.score:
                best = Closure(
                    node_a=i,
                    node_b=j,
                    stamp_a=float(kf_stamp[i]),
                    stamp_b=float(kf_stamp[j]),
                    xyyaw_ab=match.xyyaw_ab,
                    score=match.score,
                    overlap_m2=match.overlap_m2,
                    sigma_xyyaw=match.sigma_xyyaw,
                )
        if best is not None:
            tentative.append(best)
    tentative.sort(key=lambda c: -c.score)
    tentative = tentative[: p.max_closures]
    diag["evaluated_pairs"] = evaluated
    diag["tentative"] = len(tentative)
    if not tentative:
        return ClosureResult(xyyaw.copy(), [], diag | {"accepted": 0})

    # ------------------------------------------------------------- solving
    sep = np.maximum(np.diff(kf_path), 0.0)
    sig_xy = np.maximum(p.odo_sigma_floor_m, p.odo_drift_frac * sep)
    if kf_sigma is not None:
        growth = np.sqrt(np.maximum(kf_sigma[1:] ** 2 - kf_sigma[:-1] ** 2, 0.0))
        sig_xy = np.maximum(sig_xy, growth)
    sig_yaw = np.radians(np.maximum(p.odo_sigma_yaw_floor_deg, p.odo_sigma_yaw_deg_per_m * sep))
    odo_sigma = np.column_stack([sig_xy, sig_xy, sig_yaw])

    active = list(tentative)
    nodes_new = kf_pose
    while active:
        nodes_new = _solve_pose_graph(kf_pose, odo_sigma, active, p)
        worst, worst_chi2 = None, -np.inf
        for c in active:
            rel = se2_relative(nodes_new[c.node_a], nodes_new[c.node_b])
            r = np.array([*(rel[:2] - c.xyyaw_ab[:2]), _wrap(rel[2] - c.xyyaw_ab[2])])
            c.residual_xyyaw = r
            c.chi2 = float(np.sum((r / _closure_sigma(c, p)) ** 2))
            if c.chi2 > worst_chi2:
                worst, worst_chi2 = c, c.chi2
        if worst_chi2 <= p.chi2_gate:
            break
        worst.accepted = False
        active.remove(worst)
    diag["rejected_chi2"] = len(tentative) - len(active)
    if len(active) == 1 and active[0].score < p.single_score_min:
        diag["rejected_single"] = 1
        active = []
    for c in active:
        c.accepted = True
    diag["accepted"] = len(active)
    if not active:
        return ClosureResult(xyyaw.copy(), tentative, diag)

    corrected = _apply_correction(stamps, xyyaw, kf_stamp, kf_pose, nodes_new)
    diag["max_node_shift_m"] = float(np.max(np.linalg.norm(nodes_new[:, :2] - kf_pose[:, :2], axis=1)))
    return ClosureResult(corrected, tentative, diag)
