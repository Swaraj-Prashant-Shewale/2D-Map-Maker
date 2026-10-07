"""Synthetic-scene tests for the fused evidence grid.

Every scene here has an analytically known answer: a pinhole camera at a known
pose looking at the ground plane, or a LiDAR wall at a known range. No inputs/
required.
"""

from __future__ import annotations

import numpy as np
import pytest

from kart_map2d.mapping.evidence_grid import (
    CLASS_DRIVABLE,
    CLASS_OBSTACLE,
    CLASS_UNKNOWN,
    EvidenceGrid,
)

# Small synthetic pinhole camera, mounted like the kart's ZED: 0.30 m ahead of
# base_link, optical centre 0.46 m above base_link (0.56 m above the ground
# plane at z = -0.10), level, looking forward.
W, H = 64, 48
FX = FY = 60.0
CX, CY = 32.0, 24.0
CAM_T = np.array([0.30, 0.0, 0.46])
CAM_HEIGHT = CAM_T[2] + 0.10  # above the ground plane


def make_camera() -> tuple[np.ndarray, np.ndarray]:
    K = np.array([[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]])
    # Optical (z fwd, x right, y down) -> base_link (x fwd, y left, z up).
    R = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = CAM_T
    return K, T


def ground_x(v: float) -> float:
    """Analytic base_link x where the pixel row ``v`` meets the ground plane."""
    return CAM_T[0] + CAM_HEIGHT * FY / (v - CY)


def nonzero_world(grid: EvidenceGrid) -> np.ndarray:
    """World centres of every cell holding any evidence, (N, 2)."""
    touched = (grid.drivability() != 0.5) | (grid.occupancy() != 0.5)
    return grid.cell_to_world(np.argwhere(touched))


# ------------------------------------------------------------------ transforms
def test_world_cell_round_trip():
    g = EvidenceGrid()
    pts = np.array([[1.0, 1.0, -0.10], [5.0, 3.0, -0.10], [-2.0, -4.0, -0.10]])
    g.insert_lidar(pts, np.zeros(3))
    ij = g.world_to_cell(pts[:, :2])
    centers = g.cell_to_world(ij)
    assert np.all(np.abs(centers - pts[:, :2]) <= g.cell / 2 + 1e-9)
    assert np.array_equal(g.world_to_cell(centers), ij)


def test_growth_preserves_existing_evidence():
    g = EvidenceGrid()
    g.insert_lidar(np.array([[1.0, 1.0, -0.10]]), np.zeros(3))
    before = g.logodds_at([[1.0, 1.0]])[0, 0]
    assert before > 0
    g.insert_lidar(np.array([[30.0, -25.0, -0.10]]), np.zeros(3))
    assert g.logodds_at([[1.0, 1.0]])[0, 0] == pytest.approx(before)
    assert g.logodds_at([[30.0, -25.0]])[0, 0] > 0


# --------------------------------------------------------------------- camera
def test_camera_all_drivable_wedge():
    K, T = make_camera()
    g = EvidenceGrid()
    g.insert_camera(np.ones((H, W), bool), np.zeros((H, W), bool), K, T, np.zeros(3))

    # The bottom-centre pixel lands at an analytically known point.
    near = np.array([ground_x(H - 1), 0.0])  # x = 0.3 + 0.56*60/23 = 1.761
    assert g.classify_points([near])[0] == CLASS_DRIVABLE
    # No obstacle evidence anywhere, nothing behind or under the kart.
    assert not (g.classify() == CLASS_OBSTACLE).any()
    assert g.classify_points([[-1.0, 0.0], [0.5, 0.0]]).tolist() == [CLASS_UNKNOWN] * 2
    # max_range honoured: no cell past 18 m + one cell of slack.
    world = nonzero_world(g)
    assert np.hypot(world[:, 0] - CAM_T[0], world[:, 1]).max() <= 18.0 + g.cell


def test_camera_range_weighting_monotonic():
    K, T = make_camera()
    g = EvidenceGrid()
    g.insert_camera(np.ones((H, W), bool), np.zeros((H, W), bool), K, T, np.zeros(3))
    near = [ground_x(H - 1), 0.0]
    far = [ground_x(31), 0.0]  # x = 5.1 m
    lo = g.logodds_at([near, far])[:, 0]
    assert lo[1] > 0
    assert lo[0] > lo[1]


def test_camera_wall_contact_point_and_occlusion():
    """Full-width wall: obstacle rows 26..30, drivable ground below it."""
    K, T = make_camera()
    drivable = np.zeros((H, W), bool)
    drivable[31:, :] = True
    obstacle = np.zeros((H, W), bool)
    obstacle[26:31, :] = True  # lowest obstacle row = 30 -> contact at x = 5.9

    g = EvidenceGrid()
    g.insert_camera(drivable, obstacle, K, T, np.zeros(3))

    contact_x = ground_x(30)  # 0.3 + 0.56*60/6 = 5.9
    assert g.classify_points([[contact_x, 0.0]])[0] == CLASS_OBSTACLE
    # The contact line is straight (ground x depends only on the pixel row).
    edge_y = (CAM_HEIGHT * FY / 6.0) * (CX / FX)  # leftmost column, u = 0
    assert g.classify_points([[contact_x, edge_y - 0.05]])[0] == CLASS_OBSTACLE
    # Ground in front of the wall is drivable.
    assert g.classify_points([[3.1, 0.0]])[0] == CLASS_DRIVABLE
    # Nothing at all painted behind the contact line: only the lowest obstacle
    # pixel per column projects, and drivable rays are cut at the contact.
    assert nonzero_world(g)[:, 0].max() < contact_x + 2 * g.cell


def test_camera_single_column_occlusion():
    """A thin obstacle occludes only its own column, not its neighbours."""
    K, T = make_camera()
    drivable = np.ones((H, W), bool)
    obstacle = np.zeros((H, W), bool)
    obstacle[28:31, 32] = True  # contact at x = 5.9 in column u = 32 (y = 0)
    g = EvidenceGrid()
    g.insert_camera(drivable, obstacle, K, T, np.zeros(3))

    far_x = ground_x(26)  # 17.1 m: row 26 is behind the contact point
    lo_blocked = g.logodds_at([[far_x, 0.0]])[0]  # column 32 -> occluded
    lo_neighbor = g.logodds_at([[far_x, CAM_HEIGHT * FY / 2.0 / FX]])[0]  # u = 31
    assert lo_blocked[0] == 0.0 and lo_blocked[1] == 0.0
    assert lo_neighbor[0] > 0.0
    assert g.classify_points([[ground_x(30), 0.0]])[0] == CLASS_OBSTACLE


def test_camera_above_horizon_rejected():
    K, T = make_camera()
    drivable = np.zeros((H, W), bool)
    drivable[: int(CY) + 1, :] = True  # rows at or above the horizon only
    g = EvidenceGrid()
    g.insert_camera(drivable, np.zeros((H, W), bool), K, T, np.zeros(3))
    assert g.shape == (0, 0)
    assert g.render().shape == (16, 16, 3)  # empty grid still renders


def test_camera_pose_transform():
    """The same wedge inserted at a rotated pose lands rotated in the map."""
    K, T = make_camera()
    g = EvidenceGrid()
    pose = np.array([10.0, 5.0, np.pi / 2])  # facing +y
    g.insert_camera(np.ones((H, W), bool), np.zeros((H, W), bool), K, T, pose)
    near = ground_x(H - 1)  # 1.761 m ahead in base_link
    assert g.classify_points([[10.0, 5.0 + near]])[0] == CLASS_DRIVABLE
    assert g.classify_points([[10.0 + near, 5.0]])[0] == CLASS_UNKNOWN


def test_camera_logodds_clamped():
    K, T = make_camera()
    g = EvidenceGrid()
    for _ in range(30):
        g.insert_camera(np.ones((H, W), bool), np.zeros((H, W), bool), K, T, np.zeros(3))
    assert float(np.abs(g.logodds_at([[ground_x(H - 1), 0.0]])).max()) <= g.clamp + 1e-6


# ---------------------------------------------------------------------- lidar
def make_wall_scene() -> np.ndarray:
    """Ground patch 0.5..4.5 m ahead plus a vertical wall at x = 5 m."""
    gx, gy = np.meshgrid(np.arange(0.5, 4.51, 0.1), np.arange(-1.0, 1.01, 0.1))
    ground = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, -0.10)])
    wy, wz = np.meshgrid(np.arange(-2.0, 2.01, 0.05), np.array([0.3, 0.8, 1.3]))
    wall = np.column_stack([np.full(wy.size, 5.0), wy.ravel(), wz.ravel()])
    return np.vstack([ground, wall])


def test_lidar_wall_at_5m():
    g = EvidenceGrid()
    g.insert_lidar(make_wall_scene(), np.zeros(3))
    cls = g.classify_points([[5.0, 0.0], [5.0, 1.5], [2.0, 0.0], [6.0, 0.0]])
    assert cls[0] == CLASS_OBSTACLE and cls[1] == CLASS_OBSTACLE
    assert cls[2] == CLASS_DRIVABLE  # ground band |z - ground| < 0.20
    assert cls[3] == CLASS_UNKNOWN  # nothing painted behind the wall
    assert g.logodds_at([[6.0, 0.0]])[0].tolist() == [0.0, 0.0]


def test_lidar_carves_free_space_before_wall():
    g = EvidenceGrid()
    g.insert_lidar(make_wall_scene(), np.zeros(3))
    # Sample the ray towards the wall point at (5, 2): its path crosses cells
    # outside the ground patch (|y| > 1), so any negative obstacle log-odds
    # there can only come from carving.
    unit = np.array([5.0, 2.0]) / np.hypot(5.0, 2.0)
    probes = np.arange(2.5, 4.5, g.cell)[:, None] * unit[None, :]
    lo_obs = g.logodds_at(probes)[:, 1]
    assert lo_obs.min() < 0.0
    # Carving stops short of the wall itself.
    assert g.classify_points([[5.0, 2.0]])[0] == CLASS_OBSTACLE


def test_lidar_height_bands():
    g = EvidenceGrid()
    pts = np.array(
        [
            [3.0, 0.0, -0.10],  # on the ground -> drivable
            [4.0, 0.0, 0.50],  # 0.6 m above ground -> obstacle
            [2.0, 1.0, 3.50],  # 3.6 m above ground: canopy, ignored
        ]
    )
    g.insert_lidar(pts, np.zeros(3), carve=False)
    cls = g.classify_points([[3.0, 0.0], [4.0, 0.0], [2.0, 1.0]])
    assert cls.tolist() == [CLASS_DRIVABLE, CLASS_OBSTACLE, CLASS_UNKNOWN]


def test_lidar_range_gates():
    g = EvidenceGrid()
    pts = np.array([[0.1, 0.0, -0.10], [80.0, 0.0, -0.10]])  # too close / too far
    g.insert_lidar(pts, np.zeros(3))
    assert g.shape == (0, 0)


# --------------------------------------------------------------- fusion + io
def test_camera_and_lidar_agree():
    """LiDAR should boost camera evidence, not fight it."""
    K, T = make_camera()
    g = EvidenceGrid()
    g.insert_camera(np.ones((H, W), bool), np.zeros((H, W), bool), K, T, np.zeros(3))
    cam_only = g.logodds_at([[3.0, 0.0]])[0, 0]
    g.insert_lidar(np.array([[3.0, 0.0, -0.10]]), np.zeros(3))
    assert g.logodds_at([[3.0, 0.0]])[0, 0] > cam_only


def test_save_load_round_trip(tmp_path):
    g = EvidenceGrid()
    K, T = make_camera()
    g.insert_camera(np.ones((H, W), bool), np.zeros((H, W), bool), K, T, np.zeros(3))
    g.insert_lidar(make_wall_scene(), np.array([0.5, -0.25, 0.1]))
    path = tmp_path / "grid.npz"
    g.save(path)
    loaded = EvidenceGrid.load(path)
    assert loaded.cell == g.cell
    assert np.array_equal(loaded.origin, g.origin)
    assert np.array_equal(loaded.classify(), g.classify())
    assert np.allclose(loaded.drivability(), g.drivability())
    assert len(loaded._trajectory) == 2


def test_render_colors_and_trajectory():
    g = EvidenceGrid()
    g.insert_lidar(make_wall_scene(), np.zeros(3))
    img = g.render(scale=2)
    nx, ny = g.shape
    assert img.shape == (2 * ny, 2 * nx, 3) and img.dtype == np.uint8
    cls = g.classify()
    # Pick one known drivable and one known obstacle cell, map to pixels.
    ij_d = g.world_to_cell([[2.0, 0.0]])[0]
    ij_o = g.world_to_cell([[5.0, 1.5]])[0]
    px = lambda ij: (2 * (ny - 1 - ij[1]), 2 * ij[0])
    b, gr, r = img[px(ij_d)]
    assert gr > r and gr > b  # drivable is green
    b, gr, r = img[px(ij_o)]
    assert r > gr and r > b  # obstacle is red
    assert cls[ij_d[0], ij_d[1]] == CLASS_DRIVABLE
