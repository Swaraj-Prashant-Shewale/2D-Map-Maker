"""Dynamic-obstacle layer tests (README limitation 5).

Synthetic pinhole scenes with analytically known answers, like
``test_evidence_grid.py``: a wall of obstacle pixels at a known image row
lands at a known ground contact x. Here the wall is variously marked dynamic
(a person) or static (a building) and the two evidence layers are checked
independently. No inputs/ needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from kart_map2d.mapping.evidence_grid import (
    CLASS_DRIVABLE,
    CLASS_DYNAMIC,
    CLASS_OBSTACLE,
    CLASS_UNKNOWN,
    DYNAMIC_CLASS_IDS,
    EvidenceGrid,
)

# Same synthetic camera as test_evidence_grid.py: 0.30 m ahead of base_link,
# optical centre 0.46 m above base_link (0.56 m above ground), level, forward.
W, H = 64, 48
FX = FY = 60.0
CX, CY = 32.0, 24.0
CAM_T = np.array([0.30, 0.0, 0.46])
CAM_HEIGHT = CAM_T[2] + 0.10

#: Epoch-scale stamp (sept3 t0) so the tests exercise real clock magnitudes.
T0 = 1788593585.5


def make_camera() -> tuple[np.ndarray, np.ndarray]:
    K = np.array([[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]])
    R = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = CAM_T
    return K, T


def ground_x(v: float) -> float:
    """Analytic base_link x where pixel row ``v`` meets the ground plane."""
    return CAM_T[0] + CAM_HEIGHT * FY / (v - CY)


def wall_masks(dynamic: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full-width obstacle rows 26..30 over drivable ground; contact row 30."""
    drivable = np.zeros((H, W), bool)
    drivable[31:, :] = True
    obstacle = np.zeros((H, W), bool)
    obstacle[26:31, :] = True
    dyn = obstacle.copy() if dynamic else np.zeros((H, W), bool)
    return drivable, obstacle, dyn


CONTACT_X = ground_x(30)  # 5.9 m


# ------------------------------------------------------------------ constants
def test_dynamic_ids_match_segmentation_taxonomy():
    """The hardcoded ids must track segmentation.py's Cityscapes order."""
    seg = pytest.importorskip("kart_map2d.perception.segmentation")
    names = ("person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle")
    assert DYNAMIC_CLASS_IDS == seg.class_ids(names)
    # Dynamic classes are obstacles (a dynamic_mask is a subset of obstacle_mask).
    assert DYNAMIC_CLASS_IDS <= seg.OBSTACLE_IDS


# ------------------------------------------------------------------ routing
def test_dynamic_contact_goes_to_dynamic_layer_only():
    K, T = make_camera()
    g = EvidenceGrid()
    drv, obs, dyn = wall_masks(dynamic=True)
    g.insert_camera(drv, obs, K, T, np.zeros(3), dynamic_mask=dyn, stamp=T0)

    probe = np.array([[CONTACT_X, 0.0]])
    lo = g.logodds_at(probe)[0]
    assert lo[1] == 0.0  # static obstacle channel untouched
    assert g.dynamic_logodds(now=T0)[tuple(g.world_to_cell(probe)[0])] > 0.0
    # Final map: nothing there. Live view: dynamic.
    assert g.classify_points(probe)[0] == CLASS_UNKNOWN
    cls_live = g.classify(include_dynamic=True, now=T0)
    assert cls_live[tuple(g.world_to_cell(probe)[0])] == CLASS_DYNAMIC


def test_static_contact_unaffected_by_dynamic_mask_argument():
    """A wall not flagged dynamic must land exactly where it always did."""
    K, T = make_camera()
    drv, obs, dyn_empty = wall_masks(dynamic=False)

    g_old = EvidenceGrid()
    g_old.insert_camera(drv, obs, K, T, np.zeros(3))  # legacy call
    g_new = EvidenceGrid()
    g_new.insert_camera(drv, obs, K, T, np.zeros(3), dynamic_mask=dyn_empty, stamp=T0)

    assert np.array_equal(g_old.classify(), g_new.classify())
    probe = [[CONTACT_X, 0.0]]
    assert np.allclose(g_old.logodds_at(probe), g_new.logodds_at(probe))
    assert g_new.classify_points(probe)[0] == CLASS_OBSTACLE
    assert not g_new.dynamic_logodds().any()


def test_mixed_columns_route_independently():
    """Person in the left half, building in the right half of the same wall."""
    K, T = make_camera()
    drv, obs, _ = wall_masks(dynamic=False)
    dyn = np.zeros((H, W), bool)
    dyn[:, : W // 2] = obs[:, : W // 2]  # left columns (base_link +y) are a person

    g = EvidenceGrid()
    g.insert_camera(drv, obs, K, T, np.zeros(3), dynamic_mask=dyn, stamp=T0)

    # Left column u=0 -> +y side; right column u=W-1 -> -y side.
    edge_y = (CAM_HEIGHT * FY / 6.0) * (CX / FX)
    person = [[CONTACT_X, edge_y - 0.05]]
    building = [[CONTACT_X, -(edge_y - 0.05)]]
    assert g.classify_points(person)[0] == CLASS_UNKNOWN  # static view
    assert g.classify_points(building)[0] == CLASS_OBSTACLE
    cls_live = g.classify(include_dynamic=True, now=T0)
    assert cls_live[tuple(g.world_to_cell(np.array(person))[0])] == CLASS_DYNAMIC


def test_dynamic_column_still_occludes_ground_behind():
    """A person hides the ground behind them for this frame's drivable splat."""
    K, T = make_camera()
    drv = np.ones((H, W), bool)  # drivable everywhere, incl. behind the wall
    obs = np.zeros((H, W), bool)
    obs[28:31, 32] = True  # thin dynamic obstacle in column 32 (y = 0)
    dyn = obs.copy()
    g = EvidenceGrid()
    g.insert_camera(drv, obs, K, T, np.zeros(3), dynamic_mask=dyn, stamp=T0)

    far_x = ground_x(26)  # behind the contact point in that column
    assert g.logodds_at([[far_x, 0.0]])[0].tolist() == [0.0, 0.0]
    assert g.logodds_at([[far_x, CAM_HEIGHT * FY / 2.0 / FX]])[0][0] > 0.0


def test_dynamic_mask_requires_stamp():
    K, T = make_camera()
    drv, obs, dyn = wall_masks(dynamic=True)
    with pytest.raises(ValueError, match="stamp"):
        EvidenceGrid().insert_camera(drv, obs, K, T, np.zeros(3), dynamic_mask=dyn)


# -------------------------------------------------------------------- decay
def test_dynamic_evidence_halves_per_half_life():
    K, T = make_camera()
    g = EvidenceGrid(dyn_half_life=3.0)
    drv, obs, dyn = wall_masks(dynamic=True)
    g.insert_camera(drv, obs, K, T, np.zeros(3), dynamic_mask=dyn, stamp=T0)

    ij = tuple(g.world_to_cell([[CONTACT_X, 0.0]])[0])
    lo0 = g.dynamic_logodds(now=T0)[ij]
    assert lo0 > 0.0
    assert g.dynamic_logodds(now=T0 + 3.0)[ij] == pytest.approx(lo0 / 2, rel=1e-5)
    assert g.dynamic_logodds(now=T0 + 6.0)[ij] == pytest.approx(lo0 / 4, rel=1e-5)
    # Long after the walker left, even the live view is clean again.
    cls_late = g.classify(include_dynamic=True, now=T0 + 60.0)
    assert not (cls_late == CLASS_DYNAMIC).any()


def test_lazy_decay_applied_on_redeposit():
    """Re-observing a cell first decays the old evidence from its own stamp."""
    K, T = make_camera()
    g = EvidenceGrid(dyn_half_life=2.0)
    drv, obs, dyn = wall_masks(dynamic=True)
    g.insert_camera(drv, obs, K, T, np.zeros(3), dynamic_mask=dyn, stamp=T0)
    ij = tuple(g.world_to_cell([[CONTACT_X, 0.0]])[0])
    lo0 = float(g.dynamic_logodds(now=T0)[ij])

    g.insert_camera(drv, obs, K, T, np.zeros(3), dynamic_mask=dyn, stamp=T0 + 2.0)
    # One half-life elapsed: old evidence halved, then one fresh increment.
    assert float(g._lo_dyn[ij]) == pytest.approx(lo0 / 2 + lo0, rel=1e-5)
    assert float(g._dyn_stamp[ij]) == pytest.approx(T0 + 2.0)


def test_repeated_observation_sustains_dynamic_cell():
    """A person standing still stays visible in the live view while observed."""
    K, T = make_camera()
    g = EvidenceGrid(dyn_half_life=3.0)
    drv, obs, dyn = wall_masks(dynamic=True)
    for k in range(10):
        g.insert_camera(drv, obs, K, T, np.zeros(3), dynamic_mask=dyn, stamp=T0 + 0.3 * k)
    ij = tuple(g.world_to_cell([[CONTACT_X, 0.0]])[0])
    cls_live = g.classify(include_dynamic=True, now=T0 + 2.7)
    assert cls_live[ij] == CLASS_DYNAMIC
    assert g.classify()[ij] == CLASS_UNKNOWN  # never enters the final map


def test_out_of_order_stamp_does_not_rewind_decay_clock():
    """The two cameras are unsynchronized: a late-arriving older frame must
    not rewind a cell's decay clock and double-decay existing evidence."""
    K, T = make_camera()
    g = EvidenceGrid(dyn_half_life=3.0)
    drv, obs, dyn = wall_masks(dynamic=True)
    g.insert_camera(drv, obs, K, T, np.zeros(3), dynamic_mask=dyn, stamp=T0 + 10.0)
    ij = tuple(g.world_to_cell([[CONTACT_X, 0.0]])[0])
    lo_new = float(g._lo_dyn[ij])
    g.insert_camera(drv, obs, K, T, np.zeros(3), dynamic_mask=dyn, stamp=T0 + 2.0)
    # Cell clock stays at the newer stamp; the older increment is decayed
    # by its own age (8 s) instead of double-decaying the newer evidence.
    assert float(g._dyn_stamp[ij]) == pytest.approx(T0 + 10.0)
    assert float(g._lo_dyn[ij]) == pytest.approx(lo_new * (1.0 + 2.0 ** (-8.0 / 3.0)), rel=1e-5)
    assert g._last_stamp == pytest.approx(T0 + 10.0)


def test_dynamic_layer_survives_grid_growth():
    K, T = make_camera()
    g = EvidenceGrid()
    drv, obs, dyn = wall_masks(dynamic=True)
    g.insert_camera(drv, obs, K, T, np.zeros(3), dynamic_mask=dyn, stamp=T0)
    ij_before = g.world_to_cell([[CONTACT_X, 0.0]])[0]
    lo_before = float(g.dynamic_logodds(now=T0)[tuple(ij_before)])
    # Force growth on the low side of both axes (origin shift).
    g.insert_lidar(np.array([[-30.0, -30.0, -0.10]]), np.zeros(3), carve=False)
    ij_after = g.world_to_cell([[CONTACT_X, 0.0]])[0]
    assert float(g.dynamic_logodds(now=T0)[tuple(ij_after)]) == pytest.approx(lo_before)


# ------------------------------------------------------- lidar transient hits
WALKER = np.array([[3.02, 0.03, 0.6]])  # torso return, 0.7 m above ground
FLOOR = np.array([[3.02, 0.03, -0.10]])  # floor return in the same cell
WALL = np.array([[6.02, 0.03, 0.6]])  # wall behind; rays to it cross the cell
CELL_XY = [[3.02, 0.03]]


def test_lidar_hit_on_verified_free_cell_routes_dynamic():
    """Free-verified ground -> occupant hits go to the dynamic layer."""
    g = EvidenceGrid()
    for _ in range(8):  # floor seen 8x, carved by rays to the wall behind
        g.insert_lidar(np.vstack([FLOOR, WALL]), np.zeros(3))
    assert g.logodds_at(CELL_XY)[0, 1] < g.dyn_route_obs_max
    for k in range(5):  # a person now stands there (stamped sweeps)
        g.insert_lidar(np.vstack([WALKER, WALL]), np.zeros(3), stamp=T0 + 0.05 * k)
    assert g.classify_points(CELL_XY)[0] != CLASS_OBSTACLE  # final map clean
    ij = tuple(g.world_to_cell(CELL_XY)[0])
    assert g.classify(include_dynamic=True)[ij] == CLASS_DYNAMIC  # live view
    assert g.classify_points([[6.02, 0.03]])[0] == CLASS_OBSTACLE  # wall stays


def test_lidar_routing_requires_floor_history():
    """Carve depth alone (no in-cell floor return) never routes: stays static."""
    g = EvidenceGrid()
    for _ in range(8):  # carved but the floor was never SEEN inside the cell
        g.insert_lidar(WALL, np.zeros(3))
    assert g.logodds_at(CELL_XY)[0, 1] < g.dyn_route_obs_max
    for k in range(5):
        g.insert_lidar(np.vstack([WALKER, WALL]), np.zeros(3), stamp=T0 + 0.05 * k)
    assert g.classify_points(CELL_XY)[0] == CLASS_OBSTACLE
    assert not g.dynamic_logodds().any()


def test_lidar_routing_blocked_for_former_solid_obstacle():
    """A bench: solid early, floor visible beneath later, must stay static."""
    g = EvidenceGrid()
    for _ in range(3):  # the bench proves itself a solid obstacle first
        g.insert_lidar(np.vstack([WALKER, WALL]), np.zeros(3))
    for _ in range(30):  # beams under the seat: floor returns + deep carving
        g.insert_lidar(np.vstack([FLOOR, WALL]), np.zeros(3))
    assert g.logodds_at(CELL_XY)[0, 1] < g.dyn_route_obs_max  # looks "free"...
    for k in range(8):  # ...but its hits still must not be misrouted
        g.insert_lidar(np.vstack([WALKER, WALL]), np.zeros(3), stamp=T0 + 0.05 * k)
    assert g.classify_points(CELL_XY)[0] == CLASS_OBSTACLE
    assert not g.dynamic_logodds().any()


def test_lidar_without_stamp_is_legacy_static():
    g = EvidenceGrid()
    for _ in range(8):
        g.insert_lidar(np.vstack([FLOOR, WALL]), np.zeros(3))
    for _ in range(6):  # no stamp -> routing disabled even on free-verified cells
        g.insert_lidar(np.vstack([WALKER, WALL]), np.zeros(3))
    assert g.classify_points(CELL_XY)[0] == CLASS_OBSTACLE
    assert not g.dynamic_logodds().any()


def test_transient_lidar_hits_wash_out_but_walls_persist():
    """< ~1 s of walker hits + later carving flips the cell back to free."""
    g = EvidenceGrid()
    walker = np.array([[3.0, 0.0, 0.6]])  # torso return, 0.7 m above ground
    wall = np.array([[6.0, 0.0, 0.6], [6.0, 0.15, 0.6], [6.0, -0.15, 0.6]])
    # Two sweeps see the walker (a crossing takes ~0.1-0.3 s per cell at 10 Hz).
    for _ in range(2):
        g.insert_lidar(np.vstack([walker, wall]), np.zeros(3), carve=True)
    assert g.classify_points([[3.0, 0.0]])[0] == CLASS_OBSTACLE
    # The walker leaves; rays to the wall behind now carve their old cells.
    for _ in range(12):
        g.insert_lidar(wall, np.zeros(3), carve=True)
    assert g.classify_points([[3.0, 0.0]])[0] != CLASS_OBSTACLE
    # The persistent wall never flips (carving stops short of the hit cell).
    assert g.classify_points([[6.0, 0.0]])[0] == CLASS_OBSTACLE


# ------------------------------------------------------------ render + io
def test_render_modes():
    K, T = make_camera()
    g = EvidenceGrid()
    drv, obs, dyn = wall_masks(dynamic=True)
    g.insert_camera(drv, obs, K, T, np.zeros(3), dynamic_mask=dyn, stamp=T0)

    nx, ny = g.shape
    ij = g.world_to_cell([[CONTACT_X, 0.0]])[0]
    px = (ny - 1 - ij[1], ij[0])
    final = g.render()
    live = g.render(live=True, now=T0)
    b, gr, r = final[px]
    assert b == gr == r == 110  # dynamic cell absent from the final map (grey)
    b, gr, r = live[px]
    assert r > gr > b  # orange: red > green > blue
    assert r >= 200 and b <= 30


def test_save_load_roundtrips_both_layers(tmp_path):
    K, T = make_camera()
    g = EvidenceGrid(dyn_half_life=2.5)
    drv, obs, dyn = wall_masks(dynamic=True)
    g.insert_camera(drv, obs, K, T, np.zeros(3), dynamic_mask=dyn, stamp=T0)
    g.insert_lidar(np.array([[6.0, 0.0, 0.6], [2.0, 0.0, -0.10]]), np.zeros(3))

    path = tmp_path / "grid_v2.npz"
    g.save(path)
    loaded = EvidenceGrid.load(path)
    assert loaded.dyn_half_life == g.dyn_half_life
    assert np.array_equal(loaded._lo_dyn, g._lo_dyn)
    assert np.array_equal(loaded._dyn_stamp, g._dyn_stamp)
    assert np.array_equal(loaded._floor_seen, g._floor_seen)
    assert np.array_equal(loaded._was_solid, g._was_solid)
    assert loaded._last_stamp == g._last_stamp
    assert np.array_equal(loaded.classify(), g.classify())
    assert np.array_equal(
        loaded.classify(include_dynamic=True, now=T0 + 1.0),
        g.classify(include_dynamic=True, now=T0 + 1.0),
    )


def test_load_v1_file_without_dynamic_layer(tmp_path):
    """Pre-dynamic npz files (no lo_dynamic key) still load cleanly."""
    g = EvidenceGrid()
    g.insert_lidar(np.array([[3.0, 0.0, -0.10], [5.0, 0.0, 0.6]]), np.zeros(3))
    path = tmp_path / "grid_v1.npz"
    np.savez_compressed(  # exactly the v1 key set
        path,
        cell=g.cell,
        ground_z=g.ground_z,
        clamp=g.clamp,
        p_threshold=g.p_threshold,
        origin=g._origin,
        lo_drivable=g._lo_drv,
        lo_obstacle=g._lo_obs,
        trajectory=np.asarray(g._trajectory, dtype=np.float64).reshape(-1, 3),
    )
    loaded = EvidenceGrid.load(path)
    assert np.array_equal(loaded.classify(), g.classify())
    assert loaded._lo_dyn.shape == loaded._lo_drv.shape
    assert not loaded._lo_dyn.any()
    # And a v2 save of it roundtrips again.
    loaded.save(tmp_path / "grid_v2.npz")
    again = EvidenceGrid.load(tmp_path / "grid_v2.npz")
    assert np.array_equal(again.classify(), g.classify())


def test_default_classify_is_static_only_for_corridor():
    """corridor.py calls grid.classify() bare; that must be the final map."""
    K, T = make_camera()
    drv, obs, dyn = wall_masks(dynamic=True)
    g_dyn = EvidenceGrid()
    g_dyn.insert_camera(drv, obs, K, T, np.zeros(3), dynamic_mask=dyn, stamp=T0)
    g_clean = EvidenceGrid()
    g_clean.insert_camera(drv, np.zeros((H, W), bool), K, T, np.zeros(3))
    # A walker-only obstacle leaves the bare classify() identical to a scene
    # that never contained it, apart from the occlusion shadow behind them.
    cls = g_dyn.classify()
    assert not (cls == CLASS_OBSTACLE).any()
    assert (cls == CLASS_DRIVABLE).sum() > 0
    assert not (g_clean.classify() == CLASS_OBSTACLE).any()
