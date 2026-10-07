"""Rigid-transform helpers and a minimal static TF tree."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def quat_to_matrix(q_xyzw: NDArray[np.float64]) -> NDArray[np.float64]:
    x, y, z, w = np.asarray(q_xyzw, dtype=np.float64) / np.linalg.norm(q_xyzw)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def yaw_from_quat(q_xyzw: NDArray[np.float64]) -> NDArray[np.float64]:
    q = np.atleast_2d(q_xyzw)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def rpy_matrix(roll: float, pitch: float, yaw: float) -> NDArray[np.float64]:
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def make_transform(rotation: NDArray[np.float64], translation: NDArray[np.float64]) -> NDArray[np.float64]:
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = translation
    return T


def invert(T: NDArray[np.float64]) -> NDArray[np.float64]:
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


def se2(x: float, y: float, yaw: float) -> NDArray[np.float64]:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, x], [s, c, y], [0.0, 0.0, 1.0]])


class StaticTfTree:
    """Resolves fixed transforms between frames from an exported /tf_static."""

    def __init__(self, edges: list[dict]) -> None:
        self._parent: dict[str, tuple[str, NDArray[np.float64]]] = {}
        for edge in edges:
            T = make_transform(quat_to_matrix(np.asarray(edge["rotation_xyzw"])), np.asarray(edge["translation"]))
            self._parent[edge["child"]] = (edge["parent"], T)

    @classmethod
    def load(cls, path: Path) -> "StaticTfTree":
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def add(self, parent: str, child: str, T: NDArray[np.float64]) -> None:
        self._parent[child] = (parent, T)

    def frames(self) -> list[str]:
        """Every child frame the tree knows (camera-link discovery)."""
        return sorted(self._parent)

    def _to_root(self, frame: str) -> tuple[str, NDArray[np.float64]]:
        T = np.eye(4)
        while frame in self._parent:
            parent, T_parent_child = self._parent[frame]
            T = T_parent_child @ T
            frame = parent
        return frame, T

    def lookup(self, target: str, source: str) -> NDArray[np.float64]:
        """Transform mapping points in ``source`` into ``target``."""
        root_s, T_root_source = self._to_root(source)
        root_t, T_root_target = self._to_root(target)
        if root_s != root_t:
            raise KeyError(f"No static transform between {target!r} and {source!r}")
        return invert(T_root_target) @ T_root_source
