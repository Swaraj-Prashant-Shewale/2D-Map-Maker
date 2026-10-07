"""Classical drivable-area / obstacle estimation - the zero-network fallback.

SegFormer (:mod:`kart_map2d.perception.segmentation`) is the stack's single
allowed neural model; this module answers the same two per-frame questions -
"can the kart drive here?" and "is this an obstacle?" - with plain numpy/cv2
so that even that one model becomes optional (README limitation 6) and a
CPU-only deployment keeps a low latency floor. This module deliberately does
NOT import torch or transformers.

Approach (one vectorised pass per frame, at a reduced working resolution):

1. **Seed**: a trapezoid of ground directly ahead of the kart - base_link
   x in [1.7, 3.4] m, y in [-0.9, +0.9] m on the ground plane, projected
   through the camera intrinsics/mount when provided (fixed image fractions
   otherwise) - minus a dilated copy of the kart body mask. Whenever driving
   is possible at all, this patch is ground, so it defines what "ground"
   looks like in the current frame without any training.
2. **Appearance model**: robust per-channel statistics (median + MAD) of the
   seed in CIELab plus a local texture measure (box-filtered Sobel energy).
   Lab is the shadow-tolerant choice: a cast shadow moves pavement pixels
   mostly along L (darker) and slightly along b (towards skylight blue) while
   barely touching a, so the classifier down-weights L and b against a.
   Shadowed pavement then still matches the sunlit seed model, while grass
   and hedges (a shifted green) or painted kerbs (b shifted yellow) fail.
   L is first normalised by the scene median so the stored model survives
   the camera's auto-exposure swings between frames.
3. **Classification**: chi-square-like colour distance + texture gate below
   the horizon gives a candidate-ground mask; connected components of that
   mask are kept only when they touch the seed (a flood fill with spatial
   coherence), after subtracting a thin contact-edge barrier - strong
   vertical luminance gradient or a chroma step - so the fill cannot leak
   across a ground/obstacle contact line when wall and pavement happen to
   share colour. Morphological closing then heals joint lines and small
   holes that the barrier or the colour gate punched out.
4. **Categories**: drivable = the grown region; obstacle = below-horizon
   scene pixels that are not drivable (they either fail the ground model or
   sit between the drivable boundary and the horizon - building bases,
   people, kerbs, the kart's own unmasked upper body); neutral = everything
   above the horizon (sky, treetops, upper building floors - never mapped);
   the kart body is excluded exactly as in the neural path.

Robustness: when an obstacle sits directly ahead it contaminates the seed;
that is detected as a variance explosion (or the fitted model explaining too
little of its own seed), in which case the EMA model accumulated over the
previous healthy frames is used instead and ``quality`` drops so downstream
consumers can gate on it like any other channel quality.

Frames/units: image pixels (any resolution; tuned on the 960x600 exports);
the seed trapezoid is metres in base_link on the ground plane z = ground_z.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple, Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

#: Class-map value for the kart's own body; identical to segmentation.KART_ID
#: (kept literal here so this module never has to import the torch-dependent
#: neural module).
KART_ID = 255

# Cityscapes train ids used for the synthesised class map so that any code
# reading class maps through segmentation._CATEGORY_LUT categorises a
# classical result exactly like a neural one.
_DRIVABLE_ID = 0  # "road"
_OBSTACLE_ID = 2  # "building"
_NEUTRAL_ID = 10  # "sky"

#: Robust-sigma factor: MAD of a normal distribution times 1.4826 equals its
#: standard deviation.
_MAD_TO_SIGMA = 1.4826


@lru_cache(maxsize=8)
def kart_body_mask(
    height: int,
    width: int,
    bottom_fraction: float = 0.15,
    corner_width_fraction: float = 0.30,
    corner_height_fraction: float = 0.38,
) -> NDArray[np.bool_]:
    """Static kart-body mask (True = kart pixel).

    Mirrors ``segmentation.default_kart_mask`` exactly (same closed form and
    defaults; pinned together by a unit test) but lives here so the classical
    path stays importable without torch.
    """
    rows = np.arange(height, dtype=np.float64).reshape(-1, 1)
    cols = np.arange(width, dtype=np.float64).reshape(1, -1)
    mask = np.broadcast_to(rows >= (1.0 - bottom_fraction) * height, (height, width)).copy()
    rise = (height - 1 - rows) / max(corner_height_fraction * height, 1.0)
    reach = corner_width_fraction * width
    mask |= cols < reach * (1.0 - rise)  # lower-left triangle
    mask |= (width - 1 - cols) < reach * (1.0 - rise)  # lower-right triangle
    mask.flags.writeable = False
    return mask


@dataclass(frozen=True)
class ClassicalSegmentationResult:
    """Per-frame classical segmentation; duck-types SegmentationResult.

    The three category masks and the kart mask partition the image. Unlike
    the neural result there is no per-pixel class detail, so ``class_map``
    is synthesised from the masks using one representative Cityscapes train
    id per category (road / building / sky / KART_ID) - any consumer that
    only distinguishes drivable / obstacle / neutral / kart sees no
    difference.
    """

    drivable_mask: NDArray[np.bool_]
    obstacle_mask: NDArray[np.bool_]
    neutral_mask: NDArray[np.bool_]
    kart_mask: NDArray[np.bool_]
    #: 0..1 confidence in this frame's ground model: 1 = healthy seed, lower
    #: when the seed was contaminated and a previous frame's model (or, with
    #: no history, the contaminated fit itself) had to be used.
    quality: float

    @property
    def class_map(self) -> NDArray[np.uint8]:
        """Coarse uint8 class map compatible with the Cityscapes train ids."""
        out = np.full(self.drivable_mask.shape, _NEUTRAL_ID, dtype=np.uint8)
        out[self.drivable_mask] = _DRIVABLE_ID
        out[self.obstacle_mask] = _OBSTACLE_ID
        out[self.kart_mask] = KART_ID
        return out

    @property
    def drivable_fraction(self) -> float:
        """Drivable share of the scene pixels (kart body excluded)."""
        n = self.drivable_mask.size - int(np.count_nonzero(self.kart_mask))
        return float(np.count_nonzero(self.drivable_mask)) / n if n else 0.0


class _Geometry(NamedTuple):
    """Per-image-size precomputation (work-resolution grids and masks)."""

    work_size: tuple[int, int]  # (ww, wh)
    seed_mask: NDArray[np.bool_]  # (wh, ww) seed trapezoid minus kart
    seed_idx: NDArray[np.intp]  # flat indices of seed pixels
    scene: NDArray[np.bool_]  # (wh, ww) below horizon and not kart
    kart_small: NDArray[np.bool_]
    kart_full: NDArray[np.bool_]  # (h, w)
    horizon_row: int  # full-res row: rows above are neutral
    ring_full: NDArray[np.bool_]  # (h, w) non-kart pixels near the kart mask


class ClassicalGroundSegmenter:
    """Drivable/obstacle estimation with zero neural networks.

    Same call interface as :class:`segmentation.Segmenter` (``segment``,
    ``frames_processed`` / ``inference_seconds`` / ``ms_per_frame``); returns
    :class:`ClassicalSegmentationResult`.

    Parameters
    ----------
    K, T_base_cam, ground_z:
        Camera intrinsics (matching the incoming image resolution) and the
        base_link <- camera-optical mount; when given, the horizon row and
        the seed trapezoid are derived from them, otherwise from the
        ``horizon_frac`` / ``seed_corners_frac`` image fractions (defaults
        fitted to the kart's right ZED at 960x600).
    horizon_pad_px:
        The obstacle band extends this many pixels above the geometric
        horizon (at 960x600 scale) to absorb kart pitch and gentle slopes.
    work_width:
        Internal processing width; masks are computed at this scale and
        upsampled, which is what keeps the CPU cost in single-digit ms.
    weights_lab / sigma_floor_lab:
        Per-channel (L, a, b) weight and minimum sigma of the colour model.
        The low L weight plus generous L/b floors are the shadow tolerance.
    color_tau:
        Threshold on the weighted squared colour distance.
    dark_slack:
        Extra sigma multiplier applied to L deviations *below* the model
        (one-sided): cast shadow darkens ground arbitrarily far, so darker
        is judged leniently while brighter-than-ground (white walls, glare)
        keeps the full penalty.
    texture_k / texture_floor:
        Ground texture gate: seed median + k * MAD, floored (Sobel-energy
        units, exposure-normalised).
    barrier_thresh / chroma_barrier_thresh:
        |d/dy| levels (on L, and on a+b) treated as a ground/obstacle
        contact edge. The chroma one matters most: a material change (wall
        base, kerb, hedge) shifts chroma across the contact line while a
        shadow boundary barely does, so it cuts leaks without severing
        shadowed ground from the seed.
    kart_ring_px:
        Height (full-res pixels at 600 image rows, scaled with the image)
        of the band DIRECTLY ABOVE the kart mask, per column, in which a
        non-drivable pixel is NEUTRAL instead of obstacle. The band hugging
        the kart's silhouette cannot be classified reliably: suspension
        travel makes the silhouette (hence any learned kart mask) breathe
        by tens of pixels, the kart's contact shadow hugs it, and
        anti-aliased boundary pixels flicker. Whatever the cause, those
        pixels move WITH the camera, so labelling them obstacle fences the
        map in with a false ring (measured on sept3: corridor pinched to
        ~0.2 m and the bogus contacts' column occlusion clipped the
        drivable carpet). Neutral is the honest answer: no contact, no
        occlusion. A real obstacle whose base row falls in the band is
        still mapped - its pixels above the band still yield a (slightly
        far) contact, and its base was seen outside the band while
        approaching. Vertical-only on purpose: an isotropic ring reaches
        metres to the side of the wheel bump and erases real evidence. The
        band exists only over a CUSTOM (learned) ``kart_mask``: it absorbs
        the learned silhouette's breathing, while the geometric fallback's
        corner triangles are already oversized guesses - padding them too
        would erase real obstacles near the image corners (and change
        behaviour for every fallback consumer, e.g. the live node, that
        has no learned mask yet). 0 disables.
    ema:
        Blend factor for new healthy seed statistics into the stored model
        (1.0 = use only the current frame).
    """

    def __init__(
        self,
        K: NDArray[np.float64] | None = None,
        T_base_cam: NDArray[np.float64] | None = None,
        ground_z: float = -0.10,
        horizon_frac: float = 0.483,
        horizon_pad_px: float = 15.0,
        seed_x_m: tuple[float, float] = (1.7, 3.4),
        seed_y_m: tuple[float, float] = (-0.9, 0.9),
        seed_corners_frac: tuple[tuple[float, float], ...] = (
            (0.21, 0.675),
            (0.60, 0.675),
            (0.53, 0.583),
            (0.335, 0.583),
        ),
        work_width: int = 320,
        apply_kart_mask: bool = True,
        kart_mask: NDArray[np.bool_] | None = None,
        color_tau: float = 9.0,
        weights_lab: tuple[float, float, float] = (0.2, 1.0, 0.55),
        sigma_floor_lab: tuple[float, float, float] = (10.0, 2.5, 5.0),
        sigma_max_lab: tuple[float, float, float] = (60.0, 8.0, 14.0),
        dark_slack: float = 1.5,
        texture_k: float = 24.0,
        texture_floor: float = 30.0,
        barrier_thresh: float = 90.0,
        chroma_barrier_thresh: float = 22.0,
        kart_ring_px: float = 60.0,
        ema: float = 0.45,
    ) -> None:
        self.K = None if K is None else np.asarray(K, dtype=np.float64)
        self.T_base_cam = None if T_base_cam is None else np.asarray(T_base_cam, dtype=np.float64)
        self.ground_z = float(ground_z)
        self.horizon_frac = float(horizon_frac)
        self.horizon_pad_px = float(horizon_pad_px)
        self.seed_x_m = seed_x_m
        self.seed_y_m = seed_y_m
        self.seed_corners_frac = seed_corners_frac
        self.work_width = int(work_width)
        self.apply_kart_mask = apply_kart_mask
        self._custom_kart_mask = kart_mask
        self.color_tau = float(color_tau)
        self.weights_lab = weights_lab
        self.sigma_floor_lab = sigma_floor_lab
        self.sigma_max_lab = sigma_max_lab
        self.dark_slack = float(dark_slack)
        self.texture_k = float(texture_k)
        self.texture_floor = float(texture_floor)
        self.barrier_thresh = float(barrier_thresh)
        self.chroma_barrier_thresh = float(chroma_barrier_thresh)
        self.kart_ring_px = float(kart_ring_px)
        self.ema = float(ema)
        # Temporal ground model: exposure-normalised (median, sigma) per
        # channel, blended over healthy frames; None until the first one.
        self._model: dict[str, tuple[float, float]] | None = None
        self._geom: dict[tuple[int, int], _Geometry] = {}
        # Wall-clock accounting, mirroring Segmenter so validation scripts
        # can report ms/frame the same way for either backend.
        self.frames_processed = 0
        self.inference_seconds = 0.0

    # ---------------------------------------------------------- geometry
    def _seed_corners_px(self, h: int, w: int) -> NDArray[np.float64]:
        """Seed trapezoid corners (u, v) in full-res pixels."""
        if self.K is not None and self.T_base_cam is not None:
            R_cb = self.T_base_cam[:3, :3].T
            t = self.T_base_cam[:3, 3]
            (x0, x1), (y0, y1) = self.seed_x_m, self.seed_y_m
            pts_base = np.array(
                [[x0, y0], [x0, y1], [x1, y1], [x1, y0]], dtype=np.float64
            )
            pts = np.column_stack([pts_base, np.full(4, self.ground_z)])
            cam = (pts - t) @ R_cb.T  # camera optical frame
            uv = (self.K @ (cam / cam[:, 2:3]).T).T[:, :2]
            return uv
        return np.asarray(self.seed_corners_frac, dtype=np.float64) * (w, h)

    def _horizon_row_px(self, h: int, w: int) -> float:
        """Row of the ground plane's vanishing line, full-res pixels."""
        if self.K is not None and self.T_base_cam is not None:
            # Vanishing point of the base_link forward direction: where level
            # ground meets the sky for a level kart.
            d = self.T_base_cam[:3, :3].T @ np.array([1.0, 0.0, 0.0])
            return float(self.K[1, 2] + self.K[1, 1] * d[1] / d[2])
        return self.horizon_frac * h

    def _geometry(self, h: int, w: int) -> _Geometry:
        geom = self._geom.get((h, w))
        if geom is not None:
            return geom
        ww = min(self.work_width, w)
        wh = max(int(round(h * ww / w)), 1)
        sx, sy = ww / w, wh / h

        if not self.apply_kart_mask:
            kart_full = np.zeros((h, w), dtype=bool)
        elif self._custom_kart_mask is not None:
            kart_full = np.asarray(self._custom_kart_mask, dtype=bool)
        else:
            kart_full = kart_body_mask(h, w)
        kart_small = (
            cv2.resize(kart_full.astype(np.uint8), (ww, wh), interpolation=cv2.INTER_NEAREST) > 0
        )

        # Pad scaled with the image so behaviour is resolution-independent.
        pad = self.horizon_pad_px * h / 600.0
        horizon = float(np.clip(self._horizon_row_px(h, w), 1.0, h - 2.0))
        band_top_small = int(np.clip(round((horizon - pad) * sy), 0, wh - 1))
        rows_small = np.arange(wh).reshape(-1, 1)
        scene = np.broadcast_to(rows_small >= band_top_small, (wh, ww)) & ~kart_small

        corners = self._seed_corners_px(h, w) * (sx, sy)
        canvas = np.zeros((wh, ww), dtype=np.uint8)
        cv2.fillConvexPoly(canvas, np.round(corners).astype(np.int32), 1)
        # Keep the seed clear of the kart even if the body pokes above the
        # static mask (the front wheel does): subtract a dilated copy.
        kart_grown = cv2.dilate(kart_small.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        seed = (canvas > 0) & ~kart_grown & scene
        # Pixels within kart_ring_px directly ABOVE the kart mask (per
        # column), at FULL resolution: composition happens at full res
        # (work-res rescaling aliases exactly at the mask boundary, and the
        # boundary is where the band matters most). The vertical kernel
        # anchored at its top row extends the mask upward only; column
        # dilation is O(1)/pixel and this runs once per image size. Only a
        # CUSTOM (learned) mask gets the band - see kart_ring_px.
        ring_full = np.zeros((h, w), dtype=bool)
        ring_w = int(round(self.kart_ring_px * h / 600.0))
        if ring_w > 0 and self._custom_kart_mask is not None and kart_full.any():
            kernel = np.ones((ring_w + 1, 1), np.uint8)
            up = cv2.dilate(kart_full.astype(np.uint8), kernel, anchor=(0, 0)) > 0
            ring_full = up & ~kart_full
        geom = _Geometry(
            work_size=(ww, wh),
            seed_mask=seed,
            seed_idx=np.flatnonzero(seed.ravel()),
            scene=scene,
            kart_small=kart_small,
            kart_full=kart_full,
            horizon_row=int(np.clip(round(horizon - pad), 0, h - 1)),
            ring_full=ring_full,
        )
        self._geom[(h, w)] = geom
        return geom

    # --------------------------------------------------------- inference
    def segment(self, image_bgr: NDArray[np.uint8]) -> ClassicalSegmentationResult:
        """Segment one BGR uint8 image (any resolution)."""
        start = time.perf_counter()
        h, w = image_bgr.shape[:2]
        g = self._geometry(h, w)
        ww, wh = g.work_size

        small = cv2.resize(image_bgr, (ww, wh), interpolation=cv2.INTER_AREA)
        lab = cv2.cvtColor(small, cv2.COLOR_BGR2Lab).astype(np.float32)
        L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]

        # Exposure normalisation: anchor the scene median so the stored
        # model and the texture thresholds survive auto-exposure changes.
        scene_l = L[g.scene]
        med_scene = float(np.median(scene_l)) if scene_l.size else 128.0
        gain = float(np.clip(128.0 / max(med_scene, 1.0), 0.25, 4.0))
        Ln = L * gain

        gx = cv2.Sobel(Ln, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(Ln, cv2.CV_32F, 0, 1, ksize=3)
        tex = cv2.boxFilter(np.abs(gx) + np.abs(gy), -1, (5, 5))

        stats, quality = self._fit_seed(Ln, a, b, tex, g)

        # Weighted squared colour distance to the ground model. L is
        # one-sided: darker than the model (cast shadow) is judged with a
        # dark_slack-times wider sigma than brighter (walls, glare).
        wl, wa, wb = self.weights_lab
        (mL, sL), (ma, sa), (mb, sb) = stats["L"], stats["a"], stats["b"]
        dL = Ln - mL
        dL_scaled = np.where(dL < 0.0, dL / (self.dark_slack * sL), dL / sL)
        d2 = (
            wl * np.square(dL_scaled)
            + wa * np.square((a - ma) / sa)
            + wb * np.square((b - mb) / sb)
        )
        t_med, t_mad = stats["t"]
        t_gate = max(t_med + self.texture_k * t_mad, self.texture_floor)
        cand = (d2 < self.color_tau) & (tex < t_gate) & g.scene

        # Grow from the seed: components of the candidate mask that touch it,
        # with contact edges removed first so the fill cannot cross a
        # ground/obstacle contact line. Two edge cues: strong |dL/dy|, and a
        # chroma step |da/dy|+|db/dy| (material changes shift chroma even at
        # matched brightness - the cream-wall-on-bright-pavement case - while
        # shadow boundaries barely do, so shadows stay connected).
        gy_a = cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3)
        gy_b = cv2.Sobel(b, cv2.CV_32F, 0, 1, ksize=3)
        barrier = (cv2.boxFilter(np.abs(gy), -1, (3, 3)) > self.barrier_thresh) | (
            cv2.boxFilter(np.abs(gy_a) + np.abs(gy_b), -1, (3, 3)) > self.chroma_barrier_thresh
        )
        cand_cut = cand & ~barrier
        n_labels, labels = cv2.connectedComponents(cand_cut.astype(np.uint8), connectivity=8)
        touch = np.zeros(n_labels, dtype=bool)
        touch[labels[g.seed_mask & cand_cut]] = True
        touch[0] = False
        drv = touch[labels]

        m = drv.astype(np.uint8)
        # Reclaim candidate pixels the thin barrier cut off (joint lines,
        # shadow edges inside the pavement), then heal small holes and drop
        # specks. Closing after component selection cannot re-admit a
        # rejected component, only bridge a few pixels at its border.
        m |= cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) & cand
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        drv = (m > 0) & g.scene

        # Full-resolution composition from the geometric horizon row so the
        # four masks partition the image exactly (work-res upsampling would
        # alias at the kart-mask boundary): above the padded horizon is
        # neutral, below it everything that is not drivable is obstacle -
        # except in the ring around the kart mask (see kart_ring_px), where
        # a non-drivable pixel cannot be classified reliably and moves with
        # the camera: there it is "cannot verify" (neutral), never a contact.
        drivable = cv2.resize(drv.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        below = np.zeros((h, w), dtype=bool)
        below[g.horizon_row :] = True
        drivable &= below & ~g.kart_full
        obstacle = below & ~drivable & ~g.kart_full
        neutral = ~below & ~g.kart_full
        ring_neutral = obstacle & g.ring_full
        if ring_neutral.any():
            obstacle &= ~ring_neutral
            neutral |= ring_neutral

        self.inference_seconds += time.perf_counter() - start
        self.frames_processed += 1
        return ClassicalSegmentationResult(
            drivable_mask=drivable,
            obstacle_mask=obstacle,
            neutral_mask=neutral,
            kart_mask=g.kart_full.copy(),
            quality=quality,
        )

    # ------------------------------------------------------- seed model
    def _fit_seed(
        self,
        Ln: NDArray[np.float32],
        a: NDArray[np.float32],
        b: NDArray[np.float32],
        tex: NDArray[np.float32],
        g: _Geometry,
    ) -> tuple[dict[str, tuple[float, float]], float]:
        """Robust seed statistics with contamination fallback.

        Returns the (median, sigma) model per channel and a quality score.
        A healthy seed updates the temporal EMA model; a contaminated one
        (per-channel spread beyond ``sigma_max_lab``, i.e. an obstacle sits
        in the trapezoid) falls back to the stored model at reduced quality.
        """
        raw: dict[str, tuple[float, float]] = {}
        for name, ch in (("L", Ln), ("a", a), ("b", b), ("t", tex)):
            v = ch.ravel()[g.seed_idx]
            med = float(np.median(v))
            sigma = float(np.median(np.abs(v - med))) * _MAD_TO_SIGMA
            raw[name] = (med, sigma)

        floors = dict(zip("Lab", self.sigma_floor_lab))
        ceilings = dict(zip("Lab", self.sigma_max_lab))
        healthy = g.seed_idx.size >= 64 and all(
            raw[c][1] <= ceilings[c] for c in "Lab"
        )
        if healthy:
            if self._model is None:
                self._model = dict(raw)
            else:
                k = self.ema
                self._model = {
                    c: (
                        (1 - k) * self._model[c][0] + k * raw[c][0],
                        (1 - k) * self._model[c][1] + k * raw[c][1],
                    )
                    for c in raw
                }
            model, quality = self._model, 1.0
        elif self._model is not None:
            # Obstacle in the seed: trust what ground looked like before.
            model, quality = self._model, 0.35
        else:
            model, quality = raw, 0.2
        floored = {
            c: (m, max(s, floors[c])) for c, (m, s) in model.items() if c in floors
        }
        floored["t"] = model["t"]
        return floored, quality

    @property
    def ms_per_frame(self) -> float:
        """Mean wall-clock milliseconds per segmented frame so far."""
        return 1e3 * self.inference_seconds / self.frames_processed if self.frames_processed else 0.0


# ================================================================ kart mask
#: Bump when the learning algorithm changes: cached masks with an older
#: version are relearned instead of trusted.
KART_MASK_LEARN_VERSION = 4

#: Default cache location for learned per-camera kart masks (PNG + JSON
#: sidecar), relative to the working directory like every other cache.
KART_MASK_CACHE_DIR = Path("outputs/cache/kart_masks")


def learn_kart_mask(
    images: Sequence[NDArray[np.uint8]],
    drivable_masks: Sequence[NDArray[np.bool_]] | None = None,
    obstacle_masks: Sequence[NDArray[np.bool_]] | None = None,
    work_width: int = 320,
    min_motion_frames: int = 12,
    motion_thresh: float = 6.0,
    still_sigma_max: float = 14.0,
    max_drivable_frac: float = 0.10,
    ext_sigma_max: float = 28.0,
    ext_obstacle_frac: float = 0.85,
    ext_grow_px: int = 20,
    max_fraction: float = 0.45,
    anchor_side_frac: float = 0.45,
    close_px: int = 2,
    open_px: int = 1,
    dilate_px: int = 2,
) -> NDArray[np.bool_] | None:
    """Learn which pixels show the kart's own body, for ANY rigidly mounted
    camera, from frames taken while the kart drives.

    Principle (the same one the drive stack's ``StaticBodyMask`` uses live):
    the body is fixed in the image while the world streams past, so over
    frames with scene motion a body pixel's appearance barely changes. Raw
    stillness alone is not sufficient - smooth texture-free pavement is also
    temporally quiet - so a pixel is a body CANDIDATE when either

    - **still and never ground**: robust temporal sigma (median/MAD over the
      selected frames, exposure-normalised grey) at most ``still_sigma_max``
      AND classified drivable in at most ``max_drivable_frac`` of the frames
      (``drivable_masks``, from any per-frame classifier; the kart is never
      drivable while pavement is, which separates glossy body panels from
      quiet pavement) - or
    - **persistently obstacle and quiet, next to confirmed body**: classified
      obstacle in at least ``ext_obstacle_frac`` of the frames AND sigma at
      most ``ext_sigma_max`` (``obstacle_masks``), reached by growing the
      confirmed still components at most ``ext_grow_px`` work-resolution
      pixels. This catches the steering front wheel: its tyre rotates and
      swings a few pixels, so it misses the strict stillness cap, but it
      never looks like anything except an obstacle, and it is attached to
      still body parts - while the "always obstacle" band near the horizon
      is far from any confirmed body and stays out;

    and becomes body only when its connected still-component is
    **anchored**: it touches the bottom image edge or the lower
    ``anchor_side_frac`` of the left/right edges - a rigid body always
    enters the frame across a border, while quiet sky or the
    vanishing-point region floats mid-image.

    Frames should span the WHOLE recording (turns included): only frames
    whose exposure-normalised mean absolute difference to the previously
    kept frame reaches ``motion_thresh`` grey levels are used, and learning
    reports failure (returns None) with fewer than ``min_motion_frames``
    such frames or when the learned mask covers more than ``max_fraction``
    of the image (parked kart / degenerate scene). An empty mask is a valid
    result: this camera simply does not see the kart.

    Returns the FULL-resolution bool mask (True = kart) or None when
    learning was not possible; the caller then falls back to the geometric
    :func:`kart_body_mask`. CPU cost is a few ms per used frame.
    """
    if not len(images):
        return None
    h0, w0 = images[0].shape[:2]
    ww = min(int(work_width), w0)
    wh = max(int(round(h0 * ww / w0)), 1)

    def small_mask(masks, i):
        m = np.asarray(masks[i], dtype=np.uint8)
        return cv2.resize(m, (ww, wh), interpolation=cv2.INTER_NEAREST) > 0

    kept_g: list[NDArray[np.float32]] = []
    kept_drv: list[NDArray[np.bool_]] = []
    kept_obs: list[NDArray[np.bool_]] = []
    last: NDArray[np.float32] | None = None
    for i, img in enumerate(images):
        small = cv2.resize(img, (ww, wh), interpolation=cv2.INTER_AREA)
        g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        med = float(np.median(g))
        g *= 128.0 / max(med, 1.0)  # survive auto-exposure swings
        if last is not None and float(np.mean(np.abs(g - last))) < motion_thresh:
            continue  # the scene did not move: a still pixel proves nothing
        kept_g.append(g)
        last = g
        if drivable_masks is not None:
            kept_drv.append(small_mask(drivable_masks, i))
        if obstacle_masks is not None:
            kept_obs.append(small_mask(obstacle_masks, i))
    if len(kept_g) < int(min_motion_frames):
        return None

    stack = np.stack(kept_g)  # (N, wh, ww) float32
    med = np.median(stack, axis=0)
    sigma = np.median(np.abs(stack - med), axis=0) * _MAD_TO_SIGMA
    still = sigma <= float(still_sigma_max)
    if kept_drv:
        drv_frac = np.mean(np.stack(kept_drv), axis=0)
        still &= drv_frac <= float(max_drivable_frac)

    m = still.astype(np.uint8)
    if close_px > 0:
        k = 2 * int(close_px) + 1
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    if open_px > 0:
        k = 2 * int(open_px) + 1
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))

    # Anchor: keep components that reach the bottom edge or the lower part
    # of a side edge; quiet sky / clouds / the vanishing region do not.
    n_labels, labels = cv2.connectedComponents(m, connectivity=8)
    keep = np.zeros(n_labels, dtype=bool)
    bottom = labels[-1, :]
    keep[bottom[m[-1, :] > 0]] = True
    side_lo = int((1.0 - float(anchor_side_frac)) * wh)
    for col in (0, ww - 1):
        edge = labels[side_lo:, col]
        keep[edge[m[side_lo:, col] > 0]] = True
    keep[0] = False
    mask_small = keep[labels]

    # Constrained growth into "persistently obstacle and quiet" pixels: only
    # from confirmed body, only ext_grow_px steps. The steering front wheel
    # (attached to still chassis) is absorbed; the far always-obstacle band
    # near the horizon is out of reach.
    if kept_obs and mask_small.any() and ext_grow_px > 0:
        obs_frac = np.mean(np.stack(kept_obs), axis=0)
        allowed = mask_small | (
            (obs_frac >= float(ext_obstacle_frac)) & (sigma <= float(ext_sigma_max))
        )
        allowed = cv2.morphologyEx(allowed.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        grown = mask_small.astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        for _ in range(int(ext_grow_px)):
            grown = cv2.dilate(grown, kernel) * allowed
        mask_small = grown > 0

    if dilate_px > 0 and mask_small.any():
        k = 2 * int(dilate_px) + 1
        mask_small = cv2.dilate(mask_small.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
    if float(mask_small.mean()) > float(max_fraction):
        return None  # implausibly large: refuse rather than mask the scene
    return cv2.resize(mask_small.astype(np.uint8), (w0, h0), interpolation=cv2.INTER_NEAREST) > 0


def kart_mask_cache_path(cache_root: Path, run_name: str, camera: str) -> Path:
    """PNG holding the learned mask for one run + camera (JSON sidecar beside it)."""
    return Path(cache_root) / run_name / f"{camera}.png"


def load_or_learn_kart_mask(
    frames: Sequence,
    run_name: str,
    camera: str,
    K: NDArray[np.float64] | None = None,
    T_base_cam: NDArray[np.float64] | None = None,
    cache_root: Path = KART_MASK_CACHE_DIR,
    max_frames: int = 48,
    include_geometric: bool = True,
    margin_px: int = 20,
    **learn_kwargs,
) -> tuple[NDArray[np.bool_] | None, dict]:
    """Cached per-camera kart mask for a recording: learn once, reuse forever.

    ``frames`` are the recording's CameraFrame list (any subset spanning the
    run); up to ``max_frames`` are sampled evenly across it. The bootstrap
    drivable / obstacle masks for :func:`learn_kart_mask` come from a
    throwaway :class:`ClassicalGroundSegmenter` built with this camera's
    ``K`` / ``T_base_cam`` (image-fraction defaults when omitted) - the
    classifier only has to be right about open ground and about
    "not-ground", which it is regardless of how much of the kart its
    geometric mask misses. With ``include_geometric``
    (default) the learned mask is unioned with :func:`kart_body_mask`, so it
    is never smaller than the static fallback and results cached with the
    fallback stay a strict subset (what makes post-hoc stamping of cached
    neural class maps exact).

    ``margin_px`` (full-resolution pixels at 600 image rows, scaled with the
    image) grows the final mask isotropically: the kart's silhouette ring
    flickers (anti-aliasing, suspension travel, tyre steering) and its
    contact shadow hugs the silhouette, so pixels within the margin are
    kart residue far more often than scene - measured on sept3, leftover
    silhouette-ring pixels labelled obstacle re-created moving false
    contacts at 1-1.5 m that the raw learned mask had just removed. Ground
    hidden by the margin is re-observed from other rows as the kart moves,
    so the cost is nil; the margin never applies to the geometric fallback.

    Returns ``(mask, meta)``: ``mask`` is None when learning failed (the
    caller keeps the geometric fallback; the failure is cached too), and
    ``meta`` records what happened for reports. The cache key is the run +
    camera; a version bump or a changed image size invalidates it.
    """
    cache_root = Path(cache_root)
    png = kart_mask_cache_path(cache_root, run_name, camera)
    meta_path = png.with_suffix(".json")
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="ascii"))
        except ValueError:
            meta = {}
        if meta.get("version") == KART_MASK_LEARN_VERSION and meta.get("include_geometric") == include_geometric:
            if not meta.get("learned", False):
                return None, meta
            cached = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
            if cached is not None:
                return cached > 0, meta

    t0 = time.perf_counter()
    idx = np.unique(np.linspace(0, len(frames) - 1, int(max_frames)).round().astype(int)) if len(frames) else []
    images = []
    for i in idx:
        img = cv2.imread(frames[i].image_path)
        if img is not None:
            images.append(img)
    drivable = obstacle = None
    if images:
        segmenter = ClassicalGroundSegmenter(K=K, T_base_cam=T_base_cam)
        results = [segmenter.segment(img) for img in images]
        drivable = [r.drivable_mask for r in results]
        obstacle = [r.obstacle_mask for r in results]
    mask = learn_kart_mask(images, drivable, obstacle, **learn_kwargs) if images else None
    learned = mask is not None
    if learned and margin_px > 0 and mask.any():
        k = 2 * max(int(round(margin_px * mask.shape[0] / 600.0)), 1) + 1
        mask = cv2.dilate(mask.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
    if learned and include_geometric:
        mask = mask | kart_body_mask(*mask.shape)
    meta = {
        "version": KART_MASK_LEARN_VERSION,
        "learned": learned,
        "include_geometric": include_geometric,
        "frames_sampled": int(len(images)),
        "mask_fraction": round(float(mask.mean()), 4) if learned else None,
        "learn_seconds": round(time.perf_counter() - t0, 2),
        "image_size": list(images[0].shape[:2]) if images else None,
    }
    png.parent.mkdir(parents=True, exist_ok=True)
    if learned:
        cv2.imwrite(str(png), mask.astype(np.uint8) * 255)
    meta_path.write_text(json.dumps(meta, indent=1), encoding="ascii")
    return (mask if learned else None), meta
