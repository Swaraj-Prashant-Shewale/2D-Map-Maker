"""Regression: cell assignment must not depend on the grid origin.

A world point lying exactly on a cell boundary (e.g. x = 3.0 with 0.15 m
cells) used to deposit into one cell and, after the grid grew (which shifts
the stored origin and its floating-point rounding), query the neighbouring
cell - ``logodds_at`` silently returned unknown for a point just deposited.
``world_to_cell`` now snaps to the origin-independent lattice
``floor(xy / cell)``.
"""

from __future__ import annotations

import numpy as np

from kart_map2d.mapping.evidence_grid import (
    CLASS_DRIVABLE,
    CLASS_OBSTACLE,
    EvidenceGrid,
)


def test_boundary_point_consistent_across_growth():
    g = EvidenceGrid()
    g.insert_lidar(np.array([[3.0, 0.0, -0.10]]), np.zeros(3), carve=False)
    before = g.logodds_at([[3.0, 0.0]])[0, 0]
    assert before > 0
    # Force growth on the low side of both axes (origin shifts by many cells).
    g.insert_lidar(np.array([[-20.0, -20.0, -0.10]]), np.zeros(3), carve=False)
    assert g.logodds_at([[3.0, 0.0]])[0, 0] == before


def test_boundary_point_with_carve_growth():
    # Carving grows the grid back towards the sensor origin mid-insert; the
    # ground point at exactly x = 3.0 must still classify as drivable.
    g = EvidenceGrid()
    g.insert_lidar(np.array([[3.0, 0.0, -0.10], [4.0, 0.0, 0.50]]), np.zeros(3))
    cls = g.classify_points([[3.0, 0.0], [4.0, 0.0]])
    assert cls.tolist() == [CLASS_DRIVABLE, CLASS_OBSTACLE]


def test_world_cell_lattice_matches_floor():
    g = EvidenceGrid()
    g.insert_lidar(np.array([[1.0, 1.0, -0.10]]), np.zeros(3))
    g.insert_lidar(np.array([[-9.0, 7.0, -0.10]]), np.zeros(3))
    pts = np.array([[0.45, -0.3], [1.05, 1.05], [-8.99, 6.99]])
    expect = np.floor(pts / g.cell) - np.rint(g.origin / g.cell)
    assert np.array_equal(g.world_to_cell(pts), expect.astype(np.int64))
    # Round trip through cell centres still lands in the same cell.
    ij = g.world_to_cell(pts)
    assert np.array_equal(g.world_to_cell(g.cell_to_world(ij)), ij)
