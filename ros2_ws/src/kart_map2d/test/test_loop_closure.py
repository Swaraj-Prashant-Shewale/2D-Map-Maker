"""Loop-closure tests on synthetic worlds (no recorded data needed).

The fixtures build a planar world of obstacle blobs and a drivable carpet,
observe it through a forward-looking wedge from ground-truth poses, and hand
the detector a *drifted* trajectory - exactly the situation after a real
camera+IMU lap: locally sound relative poses, globally accumulated error.
"""

from __future__ import annotations

import numpy as np
import pytest

from kart_map2d.mapping.loop_closure import (
    Closure,
    KeyframeObservation,
    LoopClosureParams,
    build_local_patches,
    detect_and_close,
    patches_from_grid,
    se2_compose,
    se2_invert,
    se2_relative,
    _correlate_pair,
    _solve_pose_graph,
)


# ------------------------------------------------------------------ fixtures


def make_scene(rng, n_blobs=40, extent=10.0):
    """Obstacle blobs + drivable carpet with holes, world frame [m]."""
    centers = rng.uniform(-extent, extent, (n_blobs, 2))
    obstacles = np.vstack([c + rng.normal(0, 0.35, (12, 2)) for c in centers])
    xx, yy = np.meshgrid(np.arange(-extent, extent, 0.25), np.arange(-extent, extent, 0.25))
    drivable = np.column_stack([xx.ravel(), yy.ravel()])
    keep = np.ones(len(drivable), bool)
    for c in centers:
        keep &= np.linalg.norm(drivable - c, axis=1) > 0.8
    return drivable[keep], obstacles


def in_frame(pose, world):
    """World points -> the pose's base_link frame (x forward, y left)."""
    c, s = np.cos(pose[2]), np.sin(pose[2])
    d = world - pose[:2]
    return np.column_stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]])


def observe(pose, world, fov_deg=70.0, max_range=12.0):
    """Forward camera wedge: what a ground-projected camera frame would see."""
    p = in_frame(pose, world)
    r = np.hypot(p[:, 0], p[:, 1])
    ok = (p[:, 0] > 0.3) & (r < max_range) & (np.abs(np.arctan2(p[:, 1], p[:, 0])) < np.radians(fov_deg))
    return p[ok]


def square_track(side=30.0, speed=2.0, dt=0.1):
    """Ground-truth poses driving the perimeter of a square, back to start."""
    waypoints = np.array([[0, 0], [side, 0], [side, side], [0, side], [0, 0]], float)
    pts = []
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        n = int(np.linalg.norm(b - a) / (speed * dt))
        pts += [a + (b - a) * k / n for k in range(n)]
    pts.append(waypoints[-1])
    xy = np.array(pts)
    d = np.diff(xy, axis=0)
    yaw = np.r_[np.arctan2(d[:, 1], d[:, 0]), np.arctan2(d[-1, 1], d[-1, 0])]
    return np.arange(len(xy)) * dt, np.column_stack([xy, yaw])


def walls_for_square(rng, side=30.0, offset=3.5):
    """Notched walls flanking every side of the square track."""
    waypoints = np.array([[0, 0], [side, 0], [side, side], [0, side], [0, 0]], float)
    blobs = []
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        t = (b - a) / np.linalg.norm(b - a)
        nvec = np.array([-t[1], t[0]])
        s = 0.0
        while s < np.linalg.norm(b - a):
            for sign in (+1, -1):
                if rng.random() < 0.7:
                    blobs.append(a + t * s + nvec * sign * offset + rng.normal(0, 0.35, (12, 2)))
            s += 2.0 + rng.random() * 3.0
    obstacles = np.vstack(blobs)
    drv = []
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        t = (b - a) / np.linalg.norm(b - a)
        nvec = np.array([-t[1], t[0]])
        ss = np.arange(0, np.linalg.norm(b - a), 0.25)
        for off in np.arange(-2.5, 2.51, 0.25):
            drv.append(a + t * ss[:, None] + nvec * off)
    return np.vstack(drv), obstacles


def drift(true_xyyaw, scale=1.04, yaw_bias_rad=np.radians(0.01)):
    """Re-integrate ground-truth increments with scale + yaw-rate error."""
    est = np.zeros_like(true_xyyaw)
    est[0] = true_xyyaw[0]
    for k in range(1, len(true_xyyaw)):
        rel = se2_relative(true_xyyaw[k - 1], true_xyyaw[k])
        rel[:2] *= scale
        rel[2] += yaw_bias_rad
        est[k] = se2_compose(est[k - 1], rel)
    return est


def fast_params():
    """Full-size patches (the corner structure needs them) but sparser
    keyframes, a coarser sweep and fewer candidates so the suite stays quick."""
    return LoopClosureParams(spacing_m=1.5, coarse_step_deg=8.0, max_candidates_per_kf=2)


# ---------------------------------------------------------------- unit tests


def test_se2_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(20):
        a, b = rng.normal(0, 3, 3), rng.normal(0, 3, 3)
        assert np.allclose(se2_compose(a, se2_relative(a, b))[:2], b[:2], atol=1e-12)
        assert np.allclose(se2_compose(a, se2_invert(a)), [0, 0, 0], atol=1e-12)


@pytest.mark.parametrize(
    "pose_b",
    [
        np.array([2.0, 1.0, 0.0]),  # translation only
        np.array([1.0, -1.0, 0.4]),  # small yaw
        np.array([3.0, -1.5, np.radians(175.0)]),  # reversed heading (out-and-back)
        np.array([-1.0, 2.0, np.radians(-90.0)]),  # quarter turn
    ],
)
def test_correlation_recovers_relative_pose(pose_b):
    """Two views of one scene: the measured SE2 must match the true one.

    The reversed-heading case is the sept3 revisit situation; the full
    +-180 deg yaw sweep must find it without a viewpoint assumption.
    """
    rng = np.random.default_rng(3)
    drv, obst = make_scene(rng)
    pose_a = np.array([0.0, 0.0, 0.0])
    params = LoopClosureParams()
    patches = []
    for pose in (pose_a, pose_b):
        obs = KeyframeObservation(0.0, in_frame(pose, drv), in_frame(pose, obst))
        patches.append(build_local_patches(np.array([0.0, 1.0]), np.vstack([pose, pose]), [obs], params)[0])
    x_true = se2_relative(pose_a, pose_b)
    x_pred = x_true + np.array([0.8, -0.5, np.radians(4.0)])  # drift-corrupted prior
    m = _correlate_pair(patches[0], patches[1], x_pred, 4.0, params)
    assert m is not None
    err = m.xyyaw_ab - x_true
    assert np.hypot(err[0], err[1]) < 0.15
    assert m.sigma_xyyaw is not None and m.sigma_xyyaw.shape == (3,)
    assert abs(np.degrees((err[2] + np.pi) % (2 * np.pi) - np.pi)) < 1.0
    assert m.score > 0.9


def test_solver_pulls_drifted_chain_to_closure():
    """A 4% stretched chain plus one exact end closure lands near the truth."""
    n = 60
    nodes = np.column_stack([np.arange(n) * 1.04, np.zeros(n), np.zeros(n)])
    odo_sigma = np.tile([0.1, 0.1, np.radians(0.3)], (n - 1, 1))
    closure = Closure(0, n - 1, 0.0, 1.0, np.array([59.0, 0.0, 0.0]), 1.0, 50.0)
    out = _solve_pose_graph(nodes, odo_sigma, [closure], LoopClosureParams())
    assert abs(se2_relative(out[0], out[-1])[0] - 59.0) < 0.5


def test_solver_downweights_false_closure():
    """IRLS-Huber: one absurd closure cannot drag the chain off the truth."""
    n = 60
    nodes = np.column_stack([np.arange(n) * 1.0, np.zeros(n), np.zeros(n)])
    odo_sigma = np.tile([0.1, 0.1, np.radians(0.3)], (n - 1, 1))
    false = Closure(
        0, n - 1, 0.0, 1.0, np.array([50.0, 5.0, 0.5]), 1.0, 50.0,
        sigma_xyyaw=np.array([1.0, 1.0, 0.1]),
    )
    out = _solve_pose_graph(nodes, odo_sigma, [false], LoopClosureParams())
    # a quadratic loss would pull the 59 m end ~7 m toward the bogus 50 m
    # demand; the Huber linear region caps the pull at ~1.2 m
    assert abs(se2_relative(out[0], out[-1])[0] - 59.0) < 2.0


def test_square_loop_recovers_ground_truth():
    rng = np.random.default_rng(7)
    stamps, true = square_track()
    drv, obst = walls_for_square(rng)
    obs = [
        KeyframeObservation(float(stamps[k]), observe(true[k], drv), observe(true[k], obst))
        for k in range(0, len(stamps), 3)
    ]
    est = drift(true)
    end_before = np.linalg.norm(est[-1, :2] - true[-1, :2])
    assert end_before > 1.5  # the injected drift is material

    res = detect_and_close(stamps, est, obs, params=fast_params())
    accepted = res.accepted
    assert accepted, res.diagnostics
    # every accepted closure ties the lap end back to the lap start
    for c in accepted:
        assert c.stamp_b - c.stamp_a > 30.0
        assert c.chi2 is not None and c.chi2 <= LoopClosureParams().chi2_gate
    end_after = np.linalg.norm(res.corrected_xyyaw[-1, :2] - true[-1, :2])
    assert end_after < 0.7
    assert end_after < 0.5 * end_before
    # stamps unchanged, local shape preserved (relative pose over ~2 m windows)
    assert res.corrected_xyyaw.shape == est.shape
    k = len(stamps) // 2
    rel_before = se2_relative(est[k], est[k + 10])
    rel_after = se2_relative(res.corrected_xyyaw[k], res.corrected_xyyaw[k + 10])
    assert np.linalg.norm(rel_before[:2] - rel_after[:2]) < 0.15


def test_non_loop_trajectory_yields_no_closures():
    """A straight drive past the same kind of walls must close nothing."""
    rng = np.random.default_rng(11)
    n = 400
    stamps = np.arange(n) * 0.1
    true = np.column_stack([np.arange(n) * 0.2, np.zeros(n), np.zeros(n)])  # 80 m straight
    blobs = []
    s = 0.0
    while s < 82.0:
        for sign in (+1, -1):
            if rng.random() < 0.7:
                blobs.append(np.array([s, sign * 3.5]) + rng.normal(0, 0.35, (12, 2)))
        s += 2.0 + rng.random() * 3.0
    obst = np.vstack(blobs)
    xx, yy = np.meshgrid(np.arange(0, 82, 0.25), np.arange(-2.5, 2.51, 0.25))
    drv = np.column_stack([xx.ravel(), yy.ravel()])
    obs = [
        KeyframeObservation(float(stamps[k]), observe(true[k], drv), observe(true[k], obst))
        for k in range(0, n, 3)
    ]
    est = drift(true)
    res = detect_and_close(stamps, est, obs, params=fast_params())
    assert res.accepted == []
    assert np.allclose(res.corrected_xyyaw, est)


def test_single_closure_needs_high_score():
    """With max_closures=1 the lone survivor must clear the single-score bar."""
    rng = np.random.default_rng(7)
    stamps, true = square_track()
    drv, obst = walls_for_square(rng)
    obs = [
        KeyframeObservation(float(stamps[k]), observe(true[k], drv), observe(true[k], obst))
        for k in range(0, len(stamps), 3)
    ]
    est = drift(true)
    params = fast_params()
    params.max_closures = 1
    res = detect_and_close(stamps, est, obs, params=params)
    assert len(res.accepted) <= 1
    if res.accepted:
        assert res.accepted[0].score >= params.single_score_min
    # an impossibly high bar must reject the same lone closure
    params_hi = fast_params()
    params_hi.max_closures = 1
    params_hi.single_score_min = 1.01
    res_hi = detect_and_close(stamps, est, obs, params=params_hi)
    assert res_hi.accepted == []
    assert np.allclose(res_hi.corrected_xyyaw, est)


class FakeGrid:
    """Duck-typed accumulated grid (EvidenceGrid interface subset)."""

    def __init__(self, classes, origin, cell):
        self._classes = classes
        self.origin = np.asarray(origin, float)
        self.cell = cell

    def classify(self):
        return self._classes


def test_patches_from_grid_orientation():
    """Grid sampling must respect the keyframe pose: an obstacle 5 m ahead
    lands at the same patch cells whatever the keyframe's map-frame yaw."""
    cell = 0.2
    classes = np.ones((200, 200), np.uint8)  # all drivable
    origin = np.array([-20.0, -20.0])
    ij = ((np.array([5.0, 0.0]) - origin) / cell).astype(int)
    classes[ij[0] - 1 : ij[0] + 2, ij[1] - 1 : ij[1] + 2] = 2  # obstacle at world (5, 0)
    grid = FakeGrid(classes, origin, cell)
    stamps = np.array([0.0, 1.0])
    p0 = patches_from_grid(grid, stamps, np.array([[0, 0, 0.0], [0, 0, 0.0]]), LoopClosureParams())[0]

    ij2 = ((np.array([0.0, 5.0]) - origin) / cell).astype(int)
    classes2 = np.ones((200, 200), np.uint8)
    classes2[ij2[0] - 1 : ij2[0] + 2, ij2[1] - 1 : ij2[1] + 2] = 2  # obstacle at world (0, 5)
    grid2 = FakeGrid(classes2, origin, cell)
    yaw90 = np.radians(90.0)
    p1 = patches_from_grid(grid2, stamps, np.array([[0, 0, yaw90], [0, 0, yaw90]]), LoopClosureParams())[0]

    assert (p0.grid > 0).sum() == (p1.grid > 0).sum() > 0
    half = (p0.grid.shape[0] - 1) // 2
    for patch in (p0, p1):
        centre = (np.argwhere(patch.grid > 0).mean(axis=0) - half) * patch.cell + np.array(
            [patch.center_forward_m, 0.0]
        )
        # the obstacle sits 5 m AHEAD in the keyframe frame either way
        assert np.linalg.norm(centre - [5.0, 0.0]) < 0.5


def test_detect_accepts_grid_source():
    """detect_and_close takes an accumulated grid directly (fallback path)."""
    classes = np.ones((300, 300), np.uint8)
    grid = FakeGrid(classes, np.array([-30.0, -30.0]), 0.2)
    n = 100
    stamps = np.arange(n) * 0.1
    xyyaw = np.column_stack([np.arange(n) * 0.3, np.zeros(n), np.zeros(n)])
    res = detect_and_close(stamps, xyyaw, grid, params=fast_params())
    assert res.accepted == []  # featureless drivable plain: nothing can match
    assert np.allclose(res.corrected_xyyaw, xyyaw)
