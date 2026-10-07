"""Fused drivable-area / obstacle evidence grid.

The grid accumulates per-cell log-odds evidence in the planar map frame
(x/y anchored at the run start, yaw CCW from x). Two independent *static*
channels are kept per cell: *drivable* (this cell is traversable ground) and
*obstacle* (something non-traversable touches the ground here). Camera
semantics are the mandatory evidence source; LiDAR is an optional booster that
sharpens the same two channels, so the map degrades gracefully instead of
failing when LiDAR drops out.

A 2.5D *elevation* layer (wave 3, additive) annotates the planar grid with
per-cell mean ground z and max obstacle z in MAP-frame metres. It is fed only
through :meth:`EvidenceGrid.update_elevation` or the optional caller-z
arguments of the insert methods (``z_map`` / ``ground_z_map`` /
``obstacle_z_map``) - the caller supplies map-frame heights because it alone
knows the true 3D pose - so flat-mode runs never touch it and their saved
grids stay in the wave-2 format.

A third, *dynamic* layer holds obstacle evidence from segmentation classes
that typically move (:data:`DYNAMIC_CLASS_IDS`: people, riders, vehicles).
It decays with a half-life of a few seconds so crossing pedestrians never
bake into the final map; ``classify()``/``render()`` default to the static
view and expose the dynamic layer only on request (live view). LiDAR points
carry no class; a LiDAR obstacle hit goes static unless the cell's own
history proves the ground there was verifiably free before the occupant
arrived, in which case the hit is routed to the dynamic layer too (see
:meth:`EvidenceGrid.insert_lidar`).

Frames and units
- map frame: planar, metres; ``pose_xyyaw = [x, y, yaw]`` is the base_link
  pose in the map frame (yaw in rad, CCW from map x).
- base_link: x forward, y left, z up. The local ground plane sits at
  ``z = ground_z`` (default -0.10 m) in base_link.
- camera OPTICAL frame: z forward, x right, y down; ``T_base_cam`` maps
  optical-frame points into base_link.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

CLASS_UNKNOWN = 0
CLASS_DRIVABLE = 1
CLASS_OBSTACLE = 2
#: Live-view only: a decaying dynamic obstacle (never in the final map).
CLASS_DYNAMIC = 3

#: Cityscapes *train ids* of classes that move: person, rider, car, truck,
#: bus, train, motorcycle, bicycle (indices 11..18 of
#: ``kart_map2d.perception.segmentation.CITYSCAPES_CLASSES``). Hardcoded so
#: this module never imports the segmentation stack (torch); the test suite
#: cross-checks the ids against that tuple. Callers build ``dynamic_mask``
#: for :meth:`EvidenceGrid.insert_camera` as
#: ``np.isin(class_map, tuple(DYNAMIC_CLASS_IDS))``.
DYNAMIC_CLASS_IDS = frozenset({11, 12, 13, 14, 15, 16, 17, 18})

# Extra cells added on every grow so a slowly advancing kart does not trigger
# a reallocation per scan.
_GROW_CELLS = 24

# One ray table per (intrinsics, image size); computing it per frame would
# dominate insert_camera for 960x600 images.
_RAY_CACHE: dict[tuple[bytes, tuple[int, int]], NDArray[np.float64]] = {}


def _pixel_rays(K: NDArray[np.float64], shape: tuple[int, int]) -> NDArray[np.float64]:
    """(H, W, 3) unit-less ray directions in the camera OPTICAL frame."""
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    key = (K.tobytes(), shape)
    if key not in _RAY_CACHE:
        h, w = shape
        u = (np.arange(w, dtype=np.float64) - K[0, 2]) / K[0, 0]
        v = (np.arange(h, dtype=np.float64) - K[1, 2]) / K[1, 1]
        rays = np.empty((h, w, 3), dtype=np.float64)
        rays[..., 0] = u[None, :]
        rays[..., 1] = v[:, None]
        rays[..., 2] = 1.0
        if len(_RAY_CACHE) > 8:  # a handful of cameras at most; never grows unbounded
            _RAY_CACHE.clear()
        _RAY_CACHE[key] = rays
    return _RAY_CACHE[key]


def _logit(p: float) -> float:
    return float(np.log(p / (1.0 - p)))


def _sigmoid(x: NDArray[np.float32]) -> NDArray[np.float64]:
    return 1.0 / (1.0 + np.exp(-x.astype(np.float64)))


class EvidenceGrid:
    """Auto-growing planar log-odds grid with drivable and obstacle channels.

    Cells are addressed as ``(ix, iy)`` with ``ix`` along map x and ``iy``
    along map y; internal arrays are shaped ``(nx, ny)``. The grid starts
    empty and grows to cover whatever the sensors report, so it works on any
    area, not just a pre-sized track.
    """

    def __init__(
        self,
        cell: float = 0.15,
        ground_z: float = -0.10,
        clamp: float = 6.0,
        p_threshold: float = 0.65,
        lo_cam_drivable: float = 0.8,
        lo_cam_obstacle: float = 1.1,
        lo_lidar_drivable: float = 0.7,
        lo_lidar_obstacle: float = 1.2,
        lo_free: float = 0.3,
        dyn_half_life: float = 3.0,
        dyn_route_obs_max: float = -1.0,
        dyn_route_drv_min: float = 1.0,
        dyn_route_static_fraction: float = 0.05,
        dyn_route_min_floor: int = 2,
        dyn_route_solid_min: float = 2.0,
        kart_footprint: tuple[float, float, float] | None = None,
        min_contact_range_m: float = 0.0,
        cam_route_drv_min: float = 1.0,
        cam_route_obs_max: float = 1.0,
    ) -> None:
        self.cell = float(cell)  # m
        self.ground_z = float(ground_z)  # m, ground plane height in base_link
        self.clamp = float(clamp)  # log-odds saturation
        self.p_threshold = float(p_threshold)  # class decision probability
        self.lo_cam_drivable = float(lo_cam_drivable)
        self.lo_cam_obstacle = float(lo_cam_obstacle)
        self.lo_lidar_drivable = float(lo_lidar_drivable)
        self.lo_lidar_obstacle = float(lo_lidar_obstacle)
        self.lo_free = float(lo_free)  # free-space carve strength (obstacle channel)
        self.dyn_half_life = float(dyn_half_life)  # s, dynamic-evidence half-life
        # Confidently-free gate for LiDAR hit routing (see insert_lidar).
        # Thresholds measured on sept3: walker cells are at lo_obs ~ -6 /
        # lo_drv ~ +2..6 when first occupied, wall cells at ~0 / ~0.
        self.dyn_route_obs_max = float(dyn_route_obs_max)
        self.dyn_route_drv_min = float(dyn_route_drv_min)
        # Every routed hit also leaks this fraction into the static channel,
        # consuming the finite "this cell was free" budget: a walker's dozen
        # hits are absorbed, but a persistently occupied cell (hundreds of
        # hits, e.g. a wall base wrongly carved by grazing rays) climbs back
        # above dyn_route_obs_max and re-earns its static obstacle status.
        # Without it the routing would be absorbing and walls could be lost.
        self.dyn_route_static_fraction = float(dyn_route_static_fraction)
        # Routing additionally requires this many DIRECT in-cell floor
        # returns seen in earlier sweeps: ray-carve depth alone also marks
        # wall-base cells crossed by grazing rays (measured 10% wall loss on
        # sept3), while an actual LiDAR ground return inside the cell is
        # unambiguous "the floor was visible here".
        self.dyn_route_min_floor = int(dyn_route_min_floor)
        # A cell whose STATIC obstacle log-odds ever reached this value is
        # permanently ineligible for routing: it proved itself a real
        # obstacle before any free-verification. Raised furniture (bench
        # seats with visible floor beneath, planter lips) sinks below the
        # carve gate mid-run when grazing rays pass underneath, and without
        # this flag its later hits would be misrouted as transients
        # (measured: 7-10% of wall cells lost on sept3).
        self.dyn_route_solid_min = float(dyn_route_solid_min)
        # The kart's own planar footprint in base_link (x_min, x_max,
        # half_width) [m]: rear bumper to nose, half the track width, with a
        # small margin (wheelbase 0.978 m centred on base_link, ~1.06 m
        # wide). A camera can never see ground INSIDE the vehicle it is
        # bolted to, so an obstacle pixel whose ground projection lands here
        # is the kart's own body (or noise) leaking past the kart mask - it
        # is discarded before contact extraction in insert_camera, and its
        # column is then free to take its contact from a real obstacle pixel
        # higher up. Fail-conservative is preserved: the nearest ground the
        # kart cameras can image starts beyond the nose, so a real obstacle's
        # contact always projects outside this rectangle. Default None (off):
        # the guard is specific to forward vehicle-mounted cameras, so the
        # CALLER that knows the rig enables it - the offline pipeline passes
        # kart_map2d.pipeline.KART_FOOTPRINT_M for the kart.
        self.kart_footprint = None if kart_footprint is None else (
            float(kart_footprint[0]), float(kart_footprint[1]), float(kart_footprint[2]),
        )
        # Camera obstacle contacts closer than this are never emitted (the
        # drive stack's CameraGroundCue applies the same guard at 2.0 m).
        # A ground projection that lands within a couple of metres of the
        # camera comes from a pixel a few degrees below the lens: kart body
        # leaking past the kart mask, its bouncing silhouette, its contact
        # shadow, or near-field texture the segmenter cannot verify - and
        # because it moves with the camera it would fence the kart in.
        # A real obstacle is observed at >= this range while approaching
        # (frames arrive every 0.2-0.7 m of travel), so it is still mapped;
        # only something materialising suddenly beside the kart is not,
        # which live safety owns, not the map. Default 0 (off) for the same
        # reason as kart_footprint: a top-down or handheld camera can see
        # genuine contacts arbitrarily close; the offline pipeline passes
        # kart_map2d.pipeline.CAMERA_MIN_CONTACT_RANGE_M for the kart rig.
        self.min_contact_range = float(min_contact_range_m)
        # Camera twin of the LiDAR history routing (insert_lidar): a camera
        # contact landing on a cell the CAMERA itself verified free - strong
        # drivable log-odds (>= cam_route_drv_min) and no meaningful
        # obstacle history (<= cam_route_obs_max) - goes to the decaying
        # dynamic layer instead of the permanent map, with the same
        # dyn_route_static_fraction leak so persistent occupancy re-earns
        # static status. Measured on sept3, two error families land exactly
        # there and used to bake in: unknown-class movers a class list can
        # never cover (a dog wandering the lane), and the systematic 1-3 m
        # short-of-facade contact smear of far building bases (median 7
        # hits per cell - too consistent for thresholds, but their cells
        # are all driven over or re-verified drivable at close range).
        # Real wall/kerb bases never qualify: col_limit keeps drivable
        # splats in front of the contact line, so their cells hold
        # lo_drv < 1. Active only on stamped inserts (decay needs time).
        self.cam_route_drv_min = float(cam_route_drv_min)
        self.cam_route_obs_max = float(cam_route_obs_max)
        self._origin = np.zeros(2, dtype=np.float64)  # world xy of cell (0, 0) corner
        self._lo_drv = np.zeros((0, 0), dtype=np.float32)
        self._lo_obs = np.zeros((0, 0), dtype=np.float32)
        # Decaying dynamic-obstacle layer. Decay is applied lazily per cell
        # (against _dyn_stamp, the epoch stamp of the cell's last update) so
        # an insert never sweeps the whole grid. float64 stamps: epoch seconds
        # (~1.7e9) do not fit float32's resolution.
        self._lo_dyn = np.zeros((0, 0), dtype=np.float32)
        self._dyn_stamp = np.zeros((0, 0), dtype=np.float64)
        # Saturating count of LiDAR floor returns seen inside each cell in
        # PREVIOUS sweeps (this sweep's own returns never arm its hits, so an
        # object with returns near the ground cannot vouch for itself).
        self._floor_seen = np.zeros((0, 0), dtype=np.uint8)
        # Sticky flag: static obstacle log-odds once reached
        # dyn_route_solid_min here (see that parameter's comment).
        self._was_solid = np.zeros((0, 0), dtype=np.uint8)
        # Elevation layer (wave 3, additive): per-cell running-mean ground z
        # (sum + count) and max obstacle z, all in MAP-frame metres. Fed only
        # through update_elevation / the optional caller-z pathways of the
        # insert methods, so flat-mode runs never touch (or save) it.
        self._elev_sum = np.zeros((0, 0), dtype=np.float64)
        self._elev_cnt = np.zeros((0, 0), dtype=np.uint32)
        self._obs_zmax = np.zeros((0, 0), dtype=np.float32)  # NaN where unseen
        self._elev_touched = False
        self._last_stamp = 0.0  # most recent dynamic insert; default query time
        self._trajectory: list[tuple[float, float, float]] = []

    # ------------------------------------------------------------- geometry
    @property
    def shape(self) -> tuple[int, int]:
        return self._lo_drv.shape

    @property
    def origin(self) -> NDArray[np.float64]:
        """World xy of the lower corner of cell (0, 0), metres."""
        return self._origin.copy()

    def world_to_cell(self, xy: NDArray[np.float64]) -> NDArray[np.int64]:
        """Map-frame metres (..., 2) -> integer cell indices (..., 2).

        Cells are snapped to the origin-independent lattice ``floor(xy / cell)``
        (the origin, always cell-aligned, only offsets the indices): a point on
        a cell boundary then maps to the same cell before and after the grid
        grows, instead of flipping with the origin's floating-point value.
        """
        xy = np.asarray(xy, dtype=np.float64)
        offset = np.rint(self._origin / self.cell)
        return (np.floor(xy / self.cell) - offset).astype(np.int64)

    def cell_to_world(self, ij: NDArray[np.int64]) -> NDArray[np.float64]:
        """Cell indices (..., 2) -> map-frame metres of the cell centre."""
        offset = np.rint(self._origin / self.cell)
        return (offset + np.asarray(ij, dtype=np.float64) + 0.5) * self.cell

    def _ensure(self, mn: NDArray[np.float64], mx: NDArray[np.float64]) -> None:
        """Grow the arrays so world points in [mn, mx] (xy, m) index in-bounds."""
        mn = np.asarray(mn, dtype=np.float64) - self.cell
        mx = np.asarray(mx, dtype=np.float64) + self.cell
        nx, ny = self._lo_drv.shape
        if nx == 0:
            self._origin = np.floor(mn / self.cell) * self.cell
            shape = np.floor((mx - self._origin) / self.cell).astype(int) + 1
            self._lo_drv = np.zeros(tuple(shape), dtype=np.float32)
            self._lo_obs = np.zeros(tuple(shape), dtype=np.float32)
            self._lo_dyn = np.zeros(tuple(shape), dtype=np.float32)
            self._dyn_stamp = np.zeros(tuple(shape), dtype=np.float64)
            self._floor_seen = np.zeros(tuple(shape), dtype=np.uint8)
            self._was_solid = np.zeros(tuple(shape), dtype=np.uint8)
            self._elev_sum = np.zeros(tuple(shape), dtype=np.float64)
            self._elev_cnt = np.zeros(tuple(shape), dtype=np.uint32)
            self._obs_zmax = np.full(tuple(shape), np.nan, dtype=np.float32)
            return
        shape = np.array([nx, ny])
        lo = np.maximum(0, np.ceil((self._origin - mn) / self.cell).astype(int))
        hi = np.maximum(0, np.floor((mx - self._origin) / self.cell).astype(int) + 1 - shape)
        if not lo.any() and not hi.any():
            return
        lo[lo > 0] += _GROW_CELLS
        hi[hi > 0] += _GROW_CELLS
        pad = ((int(lo[0]), int(hi[0])), (int(lo[1]), int(hi[1])))
        self._lo_drv = np.pad(self._lo_drv, pad)
        self._lo_obs = np.pad(self._lo_obs, pad)
        self._lo_dyn = np.pad(self._lo_dyn, pad)
        # Stamp 0 on fresh cells is harmless: their dynamic log-odds is 0.
        self._dyn_stamp = np.pad(self._dyn_stamp, pad)
        self._floor_seen = np.pad(self._floor_seen, pad)
        self._was_solid = np.pad(self._was_solid, pad)
        self._elev_sum = np.pad(self._elev_sum, pad)
        self._elev_cnt = np.pad(self._elev_cnt, pad)
        self._obs_zmax = np.pad(self._obs_zmax, pad, constant_values=np.nan)
        self._origin = self._origin - lo * self.cell

    # -------------------------------------------------------------- updates
    def _collapse_cells(
        self, xy: NDArray[np.float64], increments: NDArray[np.float64]
    ) -> tuple[NDArray[np.int64], NDArray[np.float32]] | None:
        """Grow the grid over ``xy`` and collapse points to unique flat cells.

        Duplicates within one call collapse to the strongest single increment
        so a dense pixel/point cloud cannot saturate a cell from one frame;
        saturation needs repeated, independent observations. Returns
        ``(flat_indices, per_cell_increment)`` or None when nothing is finite.
        """
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        inc = np.broadcast_to(np.asarray(increments, dtype=np.float32), (len(xy),))
        good = np.isfinite(xy).all(axis=1) & np.isfinite(inc)
        if not good.any():
            return None
        xy, inc = xy[good], inc[good]
        self._ensure(xy.min(axis=0), xy.max(axis=0))
        ij = self.world_to_cell(xy)
        flat = ij[:, 0] * self._lo_drv.shape[1] + ij[:, 1]
        uniq, inv = np.unique(flat, return_inverse=True)
        acc = np.zeros(len(uniq), dtype=np.float32)
        if float(inc.min()) >= 0.0:
            np.maximum.at(acc, inv, inc)
        else:
            np.minimum.at(acc, inv, inc)
        return uniq, acc

    def _deposit(self, xy: NDArray[np.float64], increments: NDArray[np.float64], channel: str) -> None:
        """Add log-odds ``increments`` at world ``xy`` (N, 2), once per cell."""
        collapsed = self._collapse_cells(xy, increments)
        if collapsed is None:
            return
        uniq, acc = collapsed
        arr = self._lo_drv if channel == "drivable" else self._lo_obs
        arr.flat[uniq] = np.clip(arr.flat[uniq] + acc, -self.clamp, self.clamp)
        if channel == "obstacle":
            solid = arr.flat[uniq] >= self.dyn_route_solid_min
            if solid.any():
                self._was_solid.flat[uniq[solid]] = 1

    def _deposit_dynamic(self, xy: NDArray[np.float64], increments: NDArray[np.float64], stamp: float) -> None:
        """Add evidence to the decaying dynamic layer at epoch time ``stamp``.

        Only the touched cells are decayed (lazily, from their own last-update
        stamp), so per-insert cost stays proportional to the contact points,
        never to the grid size.
        """
        collapsed = self._collapse_cells(xy, increments)
        if collapsed is None:
            return
        uniq, acc = collapsed
        stamp = float(stamp)
        old = self._lo_dyn.flat[uniq]
        old_stamp = self._dyn_stamp.flat[uniq]
        # Cell clocks only move forward: a late-arriving (older-stamped)
        # observation must not rewind _dyn_stamp, or the evidence already in
        # the cell gets decayed twice at the next query (the two cameras are
        # unsynchronized, so a live feed can interleave out of order).
        # Instead the older side - existing evidence or the new increment -
        # is decayed toward the newer of the two stamps; for monotonic
        # stamps this reduces exactly to the previous behaviour.
        newer = np.maximum(old_stamp, stamp)
        decayed = old * np.exp2(-(newer - old_stamp) / self.dyn_half_life).astype(np.float32)
        acc = acc * np.exp2(-(newer - stamp) / self.dyn_half_life).astype(np.float32)
        self._lo_dyn.flat[uniq] = np.clip(decayed + acc, -self.clamp, self.clamp)
        self._dyn_stamp.flat[uniq] = newer
        self._last_stamp = max(self._last_stamp, stamp)

    def update_elevation(
        self,
        xy_map: NDArray[np.float64],
        z_map: NDArray[np.float64] | float,
        kind: str = "ground",
    ) -> None:
        """Feed the elevation layer at MAP-frame points (the caller-z pathway).

        ``xy_map`` is (N, 2) map-frame metres; ``z_map`` is the map-frame
        height per point (scalar or (N,)). ``kind="ground"`` accumulates a
        per-cell running mean (:meth:`elevation`), ``kind="obstacle"`` a
        per-cell max (:meth:`obstacle_height`). The caller supplies z because
        only it knows the true 3D transform - the lidar3d pipeline passes the
        hybrid-SE3-transformed point heights, and stage B (camera3d) will pass
        triangulated / recorded-depth point heights the same way. The grid
        itself stays planar; this layer is a 2.5D annotation on top of it.
        """
        if kind not in ("ground", "obstacle"):
            raise ValueError(f"update_elevation: unknown kind {kind!r}")
        xy = np.asarray(xy_map, dtype=np.float64).reshape(-1, 2)
        z = np.broadcast_to(np.asarray(z_map, dtype=np.float64), (len(xy),))
        good = np.isfinite(xy).all(axis=1) & np.isfinite(z)
        if not good.any():
            return
        xy, z = xy[good], z[good]
        self._ensure(xy.min(axis=0), xy.max(axis=0))
        ij = self.world_to_cell(xy)
        flat = ij[:, 0] * self._lo_drv.shape[1] + ij[:, 1]
        uniq, inv = np.unique(flat, return_inverse=True)
        if kind == "ground":
            self._elev_sum.flat[uniq] += np.bincount(inv, weights=z, minlength=len(uniq))
            self._elev_cnt.flat[uniq] += np.bincount(inv, minlength=len(uniq)).astype(np.uint32)
        else:
            zmax = np.full(len(uniq), -np.inf)
            np.maximum.at(zmax, inv, z)
            # fmax: an existing NaN (never seen) loses against a real height.
            self._obs_zmax.flat[uniq] = np.fmax(
                self._obs_zmax.flat[uniq], zmax.astype(np.float32)
            )
        self._elev_touched = True

    @staticmethod
    def _to_map(xy_base: NDArray[np.float64], pose_xyyaw: NDArray[np.float64]) -> NDArray[np.float64]:
        """base_link planar points (N, 2) -> map frame, given the base pose."""
        x, y, yaw = float(pose_xyyaw[0]), float(pose_xyyaw[1]), float(pose_xyyaw[2])
        c, s = np.cos(yaw), np.sin(yaw)
        out = np.empty_like(xy_base)
        out[:, 0] = x + c * xy_base[:, 0] - s * xy_base[:, 1]
        out[:, 1] = y + s * xy_base[:, 0] + c * xy_base[:, 1]
        return out

    def insert_camera(
        self,
        drivable_mask: NDArray[np.bool_],
        obstacle_mask: NDArray[np.bool_],
        K: NDArray[np.float64],
        T_base_cam: NDArray[np.float64],
        pose_xyyaw: NDArray[np.float64],
        max_range: float = 18.0,
        dynamic_mask: NDArray[np.bool_] | None = None,
        stamp: float | None = None,
        ground_z_map: float | None = None,
        obstacle_z_map: float | None = None,
        rays: NDArray[np.float64] | None = None,
    ) -> None:
        """Fuse one semantic camera frame (the mandatory evidence path).

        Drivable pixels are back-projected onto the ground plane
        ``z = ground_z`` in base_link (rays at or above the horizon, beyond
        ``max_range``, or occluded by an obstacle in the same image column are
        rejected). For obstacles only each column's LOWEST obstacle pixel -
        the ground contact point - is projected; pixels higher up belong to
        the obstacle's body, whose ground projection would wrongly paint
        evidence behind it. Obstacle pixels whose ground projection falls
        inside ``kart_footprint`` are discarded first: they can only be the
        kart's own body leaking past the kart mask (see the constructor).

        ``drivable_mask``/``obstacle_mask`` are (H, W) bool in image
        coordinates (row 0 = top). ``K`` is the 3x3 intrinsic matrix for that
        image size; ``T_base_cam`` maps the camera OPTICAL frame to base_link.
        Evidence is weighted down quadratically with ground range: one pixel
        row near the horizon spans many metres, so a far observation carries
        genuinely less information.

        ``dynamic_mask`` (optional, (H, W) bool) marks pixels whose
        segmentation class typically moves (:data:`DYNAMIC_CLASS_IDS`; a
        subset of ``obstacle_mask``). A contact point whose pixel is dynamic
        goes to the decaying dynamic layer instead of the permanent obstacle
        channel, so people and animals crossing the scene never bake into the
        final map. Passing ``dynamic_mask`` requires ``stamp`` (epoch
        seconds, the frame time) to drive the per-cell decay; a dynamic
        column still occludes the ground behind it in this frame.

        With ``stamp`` (with or without ``dynamic_mask``) the remaining
        contacts are additionally HISTORY-routed like LiDAR hits: one that
        lands on a cell this camera stream already verified free (drivable
        log-odds >= ``cam_route_drv_min``, obstacle <= ``cam_route_obs_max``)
        is a transient occupant or a far-contact smear and goes to the
        decaying dynamic layer too, leaking ``dyn_route_static_fraction`` of
        its weight into the static channel so persistent occupancy re-earns
        static status. Without ``stamp`` every non-dynamic contact stays
        static (legacy behaviour).

        ``ground_z_map`` / ``obstacle_z_map`` (optional, MAP-frame metres)
        feed the elevation layer: every drivable ground splat updates the
        per-cell mean ground z with ``ground_z_map`` (the caller-supplied
        height of the local ground plane in the map frame - e.g. pose z +
        ``ground_z`` when a 3D pose source exists), and every *static*
        obstacle contact updates the per-cell max obstacle z with
        ``obstacle_z_map``. Camera3d (stage B) feeds real per-point heights
        through :meth:`update_elevation` directly; these scalars exist so a
        plane-projected camera frame can participate too. Both default to
        None = legacy behaviour, untouched existing callers.
        """
        drivable_mask = np.asarray(drivable_mask, dtype=bool)
        obstacle_mask = np.asarray(obstacle_mask, dtype=bool)
        if dynamic_mask is not None and stamp is None:
            raise ValueError("insert_camera: dynamic_mask requires stamp (epoch seconds)")
        h, w = drivable_mask.shape
        # ``rays`` (optional, (H, W, 3) in the camera OPTICAL frame) lets a
        # caller pass a precomputed - e.g. lens-distortion-aware - ray table
        # (kart_map2d.camera.model.CameraModel.pixel_rays); the default is
        # the pinhole table from ``K``, exactly as before. Distortion is
        # handled by POINT/ray tables only - full frames are never warped.
        if rays is None:
            rays = _pixel_rays(K, (h, w))
        elif rays.shape[:2] != (h, w):
            raise ValueError(f"insert_camera: rays shape {rays.shape} does not match the masks {(h, w)}")
        rays = rays @ np.asarray(T_base_cam, dtype=np.float64)[:3, :3].T
        cam = np.asarray(T_base_cam, dtype=np.float64)[:3, 3]

        dz = rays[..., 2]
        below = dz < -1e-9  # a ray must point below the horizon to meet the ground
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(below, (self.ground_z - cam[2]) / dz, np.inf)
            gx = cam[0] + t * rays[..., 0]
            gy = cam[1] + t * rays[..., 1]
            rng = np.hypot(gx - cam[0], gy - cam[1])  # planar range from the camera

        # Never emit an obstacle contact that projects inside the kart's own
        # footprint (the camera cannot see ground there, so such a pixel is
        # the kart body or noise that slipped past the kart mask) or closer
        # than min_contact_range (see the constructor). Filtering the PIXELS
        # (not just the contacts) also lets the column's contact fall back
        # to a real obstacle pixel higher up and stops the bogus near-range
        # col_limit from occluding genuinely visible ground.
        if obstacle_mask.any() and (self.kart_footprint is not None or self.min_contact_range > 0.0):
            inside = np.zeros_like(below)
            if self.kart_footprint is not None:
                x0, x1, half_w = self.kart_footprint
                inside |= below & (gx >= x0) & (gx <= x1) & (np.abs(gy) <= half_w)
            if self.min_contact_range > 0.0:
                inside |= below & (rng < self.min_contact_range)
            if inside.any():
                obstacle_mask = obstacle_mask & ~inside

        # Ground contact per obstacle column: lowest obstacle pixel row.
        col_has = obstacle_mask.any(axis=0)
        col_limit = np.full(w, np.inf)
        contact_xy = np.empty((0, 2))
        contact_w = np.empty(0)
        contact_dyn = np.empty(0, dtype=bool)
        if col_has.any():
            cols = np.flatnonzero(col_has)
            rows = (h - 1) - np.argmax(obstacle_mask[::-1, :][:, cols], axis=0)
            ok = below[rows, cols]
            col_limit[cols[ok]] = rng[rows[ok], cols[ok]]
            keep = ok & (rng[rows, cols] <= max_range)
            r, c = rows[keep], cols[keep]
            contact_xy = np.column_stack([gx[r, c], gy[r, c]])
            contact_w = self._range_weight(rng[r, c], max_range)
            if dynamic_mask is not None:
                contact_dyn = np.asarray(dynamic_mask, dtype=bool)[r, c]

        # Camera history routing (see cam_route_drv_min/_obs_max): a contact
        # on a cell this camera stream already verified free is a transient
        # occupant or a far-contact smear, not new structure. Decided on the
        # state BEFORE this frame's own deposits, like insert_lidar (a
        # contact cell must never be vouched for by evidence from the same
        # frame). Requires ``stamp`` for the decay clock.
        contact_map = np.empty((0, 2))
        route = np.zeros(0, dtype=bool)
        dynv = np.zeros(0, dtype=bool)
        if len(contact_xy):
            contact_map = self._to_map(contact_xy, pose_xyyaw)
            dynv = contact_dyn if dynamic_mask is not None else np.zeros(len(contact_map), dtype=bool)
            route = np.zeros(len(contact_map), dtype=bool)
            if stamp is not None:
                lo = self.logodds_at(contact_map)
                route = (
                    ~dynv
                    & (lo[:, 0] >= self.cam_route_drv_min)
                    & (lo[:, 1] <= self.cam_route_obs_max)
                )

        sel = drivable_mask & below & (rng <= max_range) & (rng <= col_limit[None, :])
        if sel.any():
            xy = np.column_stack([gx[sel], gy[sel]])
            weight = self._range_weight(rng[sel], max_range)
            xy_map = self._to_map(xy, pose_xyyaw)
            self._deposit(xy_map, weight * self.lo_cam_drivable, "drivable")
            if ground_z_map is not None:
                self.update_elevation(xy_map, float(ground_z_map), "ground")
        if len(contact_map):
            static = ~dynv & ~route
            if static.any():
                self._deposit(
                    contact_map[static], contact_w[static] * self.lo_cam_obstacle, "obstacle"
                )
                if obstacle_z_map is not None:
                    self.update_elevation(contact_map[static], float(obstacle_z_map), "obstacle")
            transient = dynv | route
            if transient.any():
                self._deposit_dynamic(
                    contact_map[transient],
                    contact_w[transient] * self.lo_cam_obstacle,
                    float(stamp),  # transient implies stamp (dynv validated above, route gated)
                )
            if route.any() and self.dyn_route_static_fraction > 0.0:
                # The same finite free-budget leak as LiDAR routing: a
                # persistently re-observed occupant eventually climbs past
                # cam_route_obs_max and re-earns its static obstacle status.
                self._deposit(
                    contact_map[route],
                    self.dyn_route_static_fraction * contact_w[route] * self.lo_cam_obstacle,
                    "obstacle",
                )
        self._trajectory.append((float(pose_xyyaw[0]), float(pose_xyyaw[1]), float(pose_xyyaw[2])))

    @staticmethod
    def _range_weight(rng: NDArray[np.float64], max_range: float) -> NDArray[np.float64]:
        return 1.0 / (1.0 + (rng / (0.5 * max_range)) ** 2)

    def insert_lidar(
        self,
        points_base: NDArray[np.float64],
        pose_xyyaw: NDArray[np.float64],
        max_range: float = 40.0,
        min_range: float = 0.3,
        ground_band: float = 0.20,
        obstacle_max_height: float = 2.5,
        carve: bool = True,
        carve_max_range: float = 25.0,
        stamp: float | None = None,
        z_map: NDArray[np.float64] | None = None,
    ) -> None:
        """Fuse one LiDAR sweep (optional accuracy booster).

        ``points_base`` is (N, 3) metres in base_link. Points are split by
        height above the local ground plane ``h = z - ground_z``:
        ``|h| < ground_band`` adds drivable evidence at the point's xy, and
        ``ground_band < h < obstacle_max_height`` adds obstacle evidence
        (higher returns are canopy/ceiling the kart passes under). With
        ``carve`` the 2D ray to each *unique obstacle cell* subtracts obstacle
        evidence along the way - radial sampling at one step per cell, one ray
        per cell rather than per point, which keeps a 20k-point sweep cheap.

        With ``stamp`` (epoch seconds) LiDAR obstacle hits are additionally
        routed by the cell's own history: a hit landing in a cell that was
        *verifiably free* goes to the decaying dynamic layer instead of the
        static channel - the world demonstrably changed there, so the
        occupant is treated as transient. "Verifiably free" requires all of
        (a) obstacle log-odds below ``dyn_route_obs_max`` (deep carving),
        (b) drivable log-odds above ``dyn_route_drv_min``, (c) at least
        ``dyn_route_min_floor`` direct in-cell floor returns in earlier
        sweeps - carve depth alone also flags wall-base cells crossed by
        grazing rays, while a beam that hit the floor inside the cell is
        unambiguous - and (d) the cell never previously proved itself a
        solid static obstacle (``dyn_route_solid_min``), which protects
        raised furniture whose cells show both floor and hits (bench seats,
        planter lips). Each routed hit still leaks
        ``dyn_route_static_fraction`` of its weight into the static channel,
        so persistent occupancy eventually re-earns static status while a
        walker's dozen hits are absorbed. This keeps people who wander onto
        the already-driven track (and are still standing there when the
        recording ends, so no later carving can ever erase them) out of the
        final map; measured on sept3, walker cells sat at obstacle log-odds
        -6 with floor sightings when first hit, wall cells at 0 (see
        scripts/validate_dynamic.py). Without ``stamp`` every hit stays
        static (legacy behaviour). Trade-off, documented deliberately: a
        genuinely permanent object placed onto the driven lane mid-run is
        first treated as dynamic - it shows in the live view immediately and
        only slowly re-earns a place in the final map through the leak.
        Caveat: when other obstacles stand behind the occupant, carve rays
        crossing its cell (-``lo_free`` per sweep) can outpace the leak
        (+5% of ``lo_lidar_obstacle``), so in that geometry the occupant may
        stay live-view-only indefinitely; the final map then reports the
        verified ground, not the parked object.

        ``z_map`` (optional, (N,) aligned with ``points_base``) is each
        point's MAP-frame z; when given, this sweep's existing height split
        also feeds the elevation layer - ground-band points update the
        per-cell mean ground z, obstacle points (static-routed only) the
        per-cell max obstacle z. The caller supplies z because base_link z is
        pose-relative while elevation is a map-frame quantity; the lidar3d
        pipeline passes the hybrid-SE3-transformed heights. Without ``z_map``
        nothing changes (zero extra work, legacy behaviour).
        """
        pts = np.asarray(points_base, dtype=np.float64).reshape(-1, 3)
        zm = None
        if z_map is not None:
            zm = np.asarray(z_map, dtype=np.float64).reshape(-1)
            if len(zm) != len(pts):
                raise ValueError(f"z_map length {len(zm)} != points length {len(pts)}")
        if not len(pts):
            return
        r = np.hypot(pts[:, 0], pts[:, 1])
        keep = np.isfinite(pts).all(axis=1) & (r > min_range) & (r < max_range)
        pts = pts[keep]
        if zm is not None:
            zm = zm[keep]
        height = pts[:, 2] - self.ground_z
        drv = np.abs(height) < ground_band
        obs = (height > ground_band) & (height < obstacle_max_height)

        # Route decision uses the state BEFORE this sweep's own deposits.
        obs_map = self._to_map(pts[obs, :2], pose_xyyaw) if obs.any() else np.empty((0, 2))
        to_dyn = np.zeros(len(obs_map), dtype=bool)
        if stamp is not None and len(obs_map):
            lo = self.logodds_at(obs_map)  # off-grid cells read 0 -> stay static
            ij = self.world_to_cell(obs_map)
            nx, ny = self._floor_seen.shape
            inb = (ij[:, 0] >= 0) & (ij[:, 0] < nx) & (ij[:, 1] >= 0) & (ij[:, 1] < ny)
            floor = np.zeros(len(ij), dtype=np.int64)
            solid = np.ones(len(ij), dtype=np.uint8)  # off-grid: never route
            floor[inb] = self._floor_seen[ij[inb, 0], ij[inb, 1]]
            solid[inb] = self._was_solid[ij[inb, 0], ij[inb, 1]]
            to_dyn = (
                (lo[:, 1] < self.dyn_route_obs_max)
                & (lo[:, 0] > self.dyn_route_drv_min)
                & (floor >= self.dyn_route_min_floor)
                & (solid == 0)
            )

        if drv.any():
            drv_map = self._to_map(pts[drv, :2], pose_xyyaw)
            self._deposit(drv_map, self.lo_lidar_drivable, "drivable")
            if zm is not None:
                self.update_elevation(drv_map, zm[drv], "ground")
            # Floor returns are deliberately NOT used to subtract obstacle
            # evidence in their own cell: measured on sept3 it removed no
            # walker cells (they are never re-observed as floor once people
            # arrive) but, combined with the dynamic routing below, drained
            # mixed floor/obstacle boundary cells at planter and wall bases.
            # Register the direct floor sightings for FUTURE routing
            # decisions (to_dyn above was computed before this sweep).
            ij = self.world_to_cell(drv_map)
            flat = np.unique(ij[:, 0] * self._floor_seen.shape[1] + ij[:, 1])
            seen = self._floor_seen.flat[flat]
            self._floor_seen.flat[flat] = np.minimum(seen.astype(np.int64) + 1, 255)
        if len(obs_map):
            if not to_dyn.all():
                self._deposit(obs_map[~to_dyn], self.lo_lidar_obstacle, "obstacle")
                if zm is not None:
                    # Static-routed hits only: a transient occupant's height
                    # must not bake into the elevation layer either.
                    self.update_elevation(obs_map[~to_dyn], zm[obs][~to_dyn], "obstacle")
            if to_dyn.any():
                self._deposit_dynamic(obs_map[to_dyn], self.lo_lidar_obstacle, float(stamp))
                if self.dyn_route_static_fraction > 0.0:
                    # Leak a fraction to static so persistent occupancy can
                    # re-earn static status (see dyn_route_static_fraction).
                    self._deposit(
                        obs_map[to_dyn],
                        self.dyn_route_static_fraction * self.lo_lidar_obstacle,
                        "obstacle",
                    )
            if carve:
                self._carve(obs_map, pose_xyyaw[:2], carve_max_range)
        self._trajectory.append((float(pose_xyyaw[0]), float(pose_xyyaw[1]), float(pose_xyyaw[2])))

    def _carve(self, obs_xy_map: NDArray[np.float64], origin_xy: NDArray[np.float64], max_range: float) -> None:
        """Subtract obstacle evidence along rays from ``origin_xy`` to obstacles."""
        origin = np.asarray(origin_xy, dtype=np.float64)
        cells = np.unique(np.floor(obs_xy_map / self.cell).astype(np.int64), axis=0)
        targets = (cells + 0.5) * self.cell
        vec = targets - origin
        dist = np.hypot(vec[:, 0], vec[:, 1])
        # Stop 1.5 cells short of the hit so the obstacle cell itself is untouched.
        reach = np.minimum(dist - 1.5 * self.cell, max_range)
        n = np.maximum(np.floor(reach / self.cell).astype(np.int64), 0)
        ok = n > 0
        if not ok.any():
            return
        vec, dist, n = vec[ok] / dist[ok, None], dist[ok], n[ok]
        cum = np.cumsum(n)
        step = (np.arange(cum[-1]) - np.repeat(cum - n, n) + 0.5) * self.cell
        samples = origin + step[:, None] * np.repeat(vec, n, axis=0)
        self._deposit(samples, -self.lo_free, "obstacle")

    # -------------------------------------------------------------- queries
    def drivability(self) -> NDArray[np.float64]:
        """(nx, ny) probability that a cell is drivable ground."""
        return _sigmoid(self._lo_drv)

    def occupancy(self) -> NDArray[np.float64]:
        """(nx, ny) probability that a cell contains a *static* obstacle."""
        return _sigmoid(self._lo_obs)

    def dynamic_logodds(self, now: float | None = None) -> NDArray[np.float32]:
        """(nx, ny) dynamic-obstacle log-odds decayed to time ``now``.

        ``now`` defaults to the stamp of the latest dynamic insert. This is a
        query-time full-grid evaluation (a handful per run: renders, live
        classification); inserts themselves never sweep the grid.
        """
        if self._lo_dyn.size == 0:
            return self._lo_dyn.copy()
        now = self._last_stamp if now is None else float(now)
        dt = np.maximum(now - self._dyn_stamp, 0.0)
        return self._lo_dyn * np.exp2(-dt / self.dyn_half_life).astype(np.float32)

    def dynamic_occupancy(self, now: float | None = None) -> NDArray[np.float64]:
        """(nx, ny) probability that a dynamic obstacle is present *now*."""
        return _sigmoid(self.dynamic_logodds(now))

    def classify(self, include_dynamic: bool = False, now: float | None = None) -> NDArray[np.uint8]:
        """(nx, ny) per-cell class: CLASS_UNKNOWN / CLASS_DRIVABLE / CLASS_OBSTACLE.

        Obstacle evidence wins over drivable evidence: painting a wall cell
        green is worse than leaving a ground cell red.

        The DEFAULT (``include_dynamic=False``) is the final map: static
        evidence only, decayed dynamic observations ignored. Downstream
        consumers (corridor extraction, racing line) call ``classify()`` with
        no arguments and therefore already get the static-only view - no
        change is needed in ``corridor.py``. With ``include_dynamic=True``
        (the live view) cells whose *current* decayed dynamic evidence
        crosses the threshold are marked :data:`CLASS_DYNAMIC`; a cell that
        is also a confirmed static obstacle stays ``CLASS_OBSTACLE``.
        """
        thr = _logit(self.p_threshold)
        out = np.zeros(self._lo_drv.shape, dtype=np.uint8)
        out[self._lo_drv > thr] = CLASS_DRIVABLE
        if include_dynamic:
            out[self.dynamic_logodds(now) > thr] = CLASS_DYNAMIC
        out[self._lo_obs > thr] = CLASS_OBSTACLE
        return out

    def logodds_at(self, xy: NDArray[np.float64]) -> NDArray[np.float64]:
        """(N, 2) [drivable, obstacle] log-odds at world points; 0 off-grid."""
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        ij = self.world_to_cell(xy)
        nx, ny = self._lo_drv.shape
        inside = (ij[:, 0] >= 0) & (ij[:, 0] < nx) & (ij[:, 1] >= 0) & (ij[:, 1] < ny)
        out = np.zeros((len(xy), 2), dtype=np.float64)
        out[inside, 0] = self._lo_drv[ij[inside, 0], ij[inside, 1]]
        out[inside, 1] = self._lo_obs[ij[inside, 0], ij[inside, 1]]
        return out

    def classify_points(self, xy: NDArray[np.float64]) -> NDArray[np.uint8]:
        """Per-point class at world xy (N, 2); off-grid points are unknown."""
        lo = self.logodds_at(xy)
        thr = _logit(self.p_threshold)
        out = np.zeros(len(lo), dtype=np.uint8)
        out[lo[:, 0] > thr] = CLASS_DRIVABLE
        out[lo[:, 1] > thr] = CLASS_OBSTACLE
        return out

    @property
    def has_elevation(self) -> bool:
        """Whether the elevation layer ever received data (wave 3)."""
        return self._elev_touched

    def elevation(self) -> NDArray[np.float32]:
        """(nx, ny) per-cell mean ground z, MAP-frame metres; NaN where unseen."""
        out = np.full(self._elev_sum.shape, np.nan, dtype=np.float32)
        seen = self._elev_cnt > 0
        out[seen] = (self._elev_sum[seen] / self._elev_cnt[seen]).astype(np.float32)
        return out

    def obstacle_height(self) -> NDArray[np.float32]:
        """(nx, ny) per-cell max obstacle z, MAP-frame metres; NaN where unseen."""
        return self._obs_zmax.copy()

    def render_elevation(self, path: Path) -> Path:
        """Two-panel PNG (mean ground z, max obstacle z) with colourbars.

        Same orientation as :meth:`render`: row 0 is max map y, columns follow
        map x. Cells without elevation data are blank.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        path = Path(path)
        panels = [("mean ground z [m]", self.elevation()), ("max obstacle z [m]", self.obstacle_height())]
        nx, ny = self._lo_drv.shape
        x0, y0 = (self._origin if nx else np.zeros(2)).tolist()
        extent = (x0, x0 + nx * self.cell, y0, y0 + ny * self.cell)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        for ax, (label, arr) in zip(axes, panels):
            if arr.size == 0 or not np.isfinite(arr).any():
                ax.set_title(label + " - no data")
                continue
            img = arr.T[::-1]  # (nx, ny) -> rows = y flipped, cols = x
            im = ax.imshow(img, origin="upper", extent=extent, cmap="viridis", interpolation="nearest")
            ax.set_xlabel("map x [m]")
            ax.set_ylabel("map y [m]")
            ax.set_title(label)
            fig.colorbar(im, ax=ax, shrink=0.85)
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        return path

    # ------------------------------------------------------------ rendering
    def render(self, scale: int = 1, live: bool = False, now: float | None = None) -> NDArray[np.uint8]:
        """BGR image: unknown grey, drivable green (brighter = more certain),
        obstacle red, trajectory in blue. Row 0 is max map y (north up),
        columns follow map x.

        Default is the final map (static evidence only). ``live=True`` is the
        operator view: cells with above-threshold dynamic evidence at time
        ``now`` (default: latest dynamic insert) are drawn orange.
        """
        nx, ny = self._lo_drv.shape
        if nx == 0 or ny == 0:
            return np.full((16, 16, 3), 110, dtype=np.uint8)
        cls = self.classify(include_dynamic=live, now=now)
        img = np.full((nx, ny, 3), 110, dtype=np.uint8)

        drv = cls == CLASS_DRIVABLE
        conf = np.clip((self.drivability() - 0.5) * 2.0, 0.0, 1.0)
        img[drv, 0] = 40
        img[drv, 1] = (110 + 120 * conf[drv]).astype(np.uint8)
        img[drv, 2] = 40

        obs = cls == CLASS_OBSTACLE
        conf_o = np.clip((self.occupancy() - 0.5) * 2.0, 0.0, 1.0)
        img[obs, 0] = 30
        img[obs, 1] = 30
        img[obs, 2] = (140 + 115 * conf_o[obs]).astype(np.uint8)

        if live:
            dyn = cls == CLASS_DYNAMIC
            conf_d = np.clip((self.dynamic_occupancy(now) - 0.5) * 2.0, 0.0, 1.0)
            img[dyn, 0] = 20
            img[dyn, 1] = (110 + 50 * conf_d[dyn]).astype(np.uint8)  # orange
            img[dyn, 2] = (200 + 55 * conf_d[dyn]).astype(np.uint8)

        # [ix, iy] -> image rows are y (flipped so +y is up), columns are x.
        img = img.transpose(1, 0, 2)[::-1].copy()
        if scale > 1:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        if self._trajectory:
            traj = np.asarray(self._trajectory, dtype=np.float64)[:, :2]
            ij = self.world_to_cell(traj)
            px = np.column_stack([ij[:, 0], ny - 1 - ij[:, 1]])
            px = ((px + 0.5) * scale).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [px], False, (255, 64, 0), max(1, scale // 2))
            cv2.circle(img, tuple(px[-1, 0]), max(2, scale), (255, 255, 255), -1)
        return img

    # ---------------------------------------------------------- persistence
    def save(self, path: Path) -> None:
        # Format v2 = v1 keys (all still present, so older readers that pick
        # keys by name keep working) + the dynamic layer. Format v3 = v2 +
        # the elevation layer, and is only written when that layer holds data
        # - an untouched (flat-mode) grid still saves byte-identical v2 files.
        extra: dict = {}
        if self._elev_touched:
            extra = dict(
                elev_sum=self._elev_sum,
                elev_cnt=self._elev_cnt,
                obs_zmax=self._obs_zmax,
            )
        np.savez_compressed(
            Path(path),
            format=3 if extra else 2,
            cell=self.cell,
            ground_z=self.ground_z,
            clamp=self.clamp,
            p_threshold=self.p_threshold,
            origin=self._origin,
            lo_drivable=self._lo_drv,
            lo_obstacle=self._lo_obs,
            lo_dynamic=self._lo_dyn,
            dyn_stamp=self._dyn_stamp,
            floor_seen=self._floor_seen,
            was_solid=self._was_solid,
            dyn_half_life=self.dyn_half_life,
            last_stamp=self._last_stamp,
            trajectory=np.asarray(self._trajectory, dtype=np.float64).reshape(-1, 3),
            **extra,
        )

    @classmethod
    def load(cls, path: Path) -> "EvidenceGrid":
        data = np.load(Path(path))
        grid = cls(
            cell=float(data["cell"]),
            ground_z=float(data["ground_z"]),
            clamp=float(data["clamp"]),
            p_threshold=float(data["p_threshold"]),
        )
        grid._origin = data["origin"].astype(np.float64)
        grid._lo_drv = data["lo_drivable"].astype(np.float32)
        grid._lo_obs = data["lo_obstacle"].astype(np.float32)
        if "lo_dynamic" in data.files:  # v2; v1 files load with an empty dynamic layer
            grid.dyn_half_life = float(data["dyn_half_life"])
            grid._lo_dyn = data["lo_dynamic"].astype(np.float32)
            grid._dyn_stamp = data["dyn_stamp"].astype(np.float64)
            grid._last_stamp = float(data["last_stamp"])
        else:
            grid._lo_dyn = np.zeros_like(grid._lo_drv)
            grid._dyn_stamp = np.zeros(grid._lo_drv.shape, dtype=np.float64)
        if "floor_seen" in data.files:
            grid._floor_seen = data["floor_seen"].astype(np.uint8)
        else:
            grid._floor_seen = np.zeros(grid._lo_drv.shape, dtype=np.uint8)
        if "was_solid" in data.files:
            grid._was_solid = data["was_solid"].astype(np.uint8)
        else:
            # Conservative reconstruction for older files: currently-strong
            # static obstacle evidence marks the cell solid.
            grid._was_solid = (grid._lo_obs >= grid.dyn_route_solid_min).astype(np.uint8)
        if "elev_sum" in data.files:  # v3; v1/v2 files load with an empty layer
            grid._elev_sum = data["elev_sum"].astype(np.float64)
            grid._elev_cnt = data["elev_cnt"].astype(np.uint32)
            grid._obs_zmax = data["obs_zmax"].astype(np.float32)
            grid._elev_touched = True
        else:
            grid._elev_sum = np.zeros(grid._lo_drv.shape, dtype=np.float64)
            grid._elev_cnt = np.zeros(grid._lo_drv.shape, dtype=np.uint32)
            grid._obs_zmax = np.full(grid._lo_drv.shape, np.nan, dtype=np.float32)
        grid._trajectory = [tuple(row) for row in data["trajectory"]]
        return grid
