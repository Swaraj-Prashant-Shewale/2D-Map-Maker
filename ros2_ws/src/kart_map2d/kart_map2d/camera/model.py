"""Camera intrinsics + lens distortion, handled by POINTS and ray tables only.

Compute budget rule (binding for this project): full frames are NEVER
undistorted. Distortion enters the pipeline in exactly two shapes:

- ``undistort_pixels``: matched feature points (a few hundred per VO pair)
  are mapped from measured (distorted) pixel coordinates to the coordinates
  an ideal pinhole ``K`` would have measured, via ``cv2.undistortPoints``.
- ``pixel_rays``: one precomputed (H, W, 3) table of viewing rays per camera
  and working resolution, used by the evidence grid's ground projection and
  the loop-closure ground observer. Computed once, cached, reused per frame.

With zero distortion both paths reduce EXACTLY to the pinhole math the
research pipeline used (no cv2 round-trip), so behaviour on the September
recordings is bit-identical to the proven code.

Supported models (the ``distortion_model`` strings of sensor_msgs/CameraInfo):
``plumb_bob`` (k1 k2 p1 p2 k3), ``rational_polynomial`` (k1..k6 p1 p2) and
``equidistant`` (fisheye k1..k4). An empty/None model or all-zero
coefficients means "already rectified".

Calibration YAML (for cameras without a CameraInfo topic - a V4L2 device, a
GStreamer pipeline or a video file). ``tools/dump_camera_info.py`` writes it
from a live camera_info topic once::

    # kart_map2d camera calibration
    image_width: 1920
    image_height: 1200
    camera_matrix: [fx, 0, cx, 0, fy, cy, 0, 0, 1]   # row-major 3x3
    distortion_model: plumb_bob                       # or rational_polynomial | equidistant | none
    distortion_coefficients: [k1, k2, p1, p2, k3]

Frames follow project conventions: camera OPTICAL frame (z forward, x right,
y down), pixel origin at the top-left corner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

__all__ = ["CameraModel", "SUPPORTED_DISTORTION_MODELS", "load_calibration_yaml", "save_calibration_yaml"]

SUPPORTED_DISTORTION_MODELS = ("none", "plumb_bob", "rational_polynomial", "equidistant")

#: Ray tables per (K bytes, D bytes, model, shape) - a handful of cameras at
#: most, exactly like evidence_grid._RAY_CACHE.
_RAY_CACHE: dict[tuple, NDArray[np.float64]] = {}


@dataclass(frozen=True)
class CameraModel:
    """Pinhole K at a specific resolution + optional lens distortion.

    ``K`` is the 3x3 intrinsic matrix AT ``size`` = (width, height). ``D``
    holds the distortion coefficients of ``model`` (any length OpenCV
    accepts for that model); an empty ``D`` or ``model == "none"`` means the
    image is already rectified.
    """

    K: NDArray[np.float64]
    size: tuple[int, int]  # (width, height) [px]
    D: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    model: str = "none"

    def __post_init__(self) -> None:
        object.__setattr__(self, "K", np.asarray(self.K, dtype=np.float64).reshape(3, 3))
        object.__setattr__(self, "D", np.asarray(self.D, dtype=np.float64).ravel())
        model = (self.model or "none").lower()
        if model in ("", "empty"):
            model = "none"
        if model not in SUPPORTED_DISTORTION_MODELS:
            raise ValueError(
                f"distortion_model {self.model!r} is not supported "
                f"(use one of {', '.join(SUPPORTED_DISTORTION_MODELS)})"
            )
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "size", (int(self.size[0]), int(self.size[1])))

    # ------------------------------------------------------------ properties
    @property
    def has_distortion(self) -> bool:
        return self.model != "none" and self.D.size > 0 and bool(np.any(self.D != 0.0))

    # ------------------------------------------------------------- scaling
    def scaled(self, scale: float) -> "CameraModel":
        """The same physical camera at ``scale`` times the resolution.

        Distortion coefficients are expressed in NORMALIZED coordinates, so
        they are resolution-independent; only K and the size change (the
        exact transform the node applies to CameraInfo intrinsics).
        """
        if scale == 1.0:
            return self
        K = self.K.copy()
        K[:2] *= float(scale)
        w, h = self.size
        return CameraModel(K, (int(round(w * scale)), int(round(h * scale))), self.D.copy(), self.model)

    # -------------------------------------------------------------- points
    def distort_pixels(self, pts: NDArray[np.float64]) -> NDArray[np.float64]:
        """Ideal-pinhole pixel coordinates -> distorted pixel coordinates.

        The FORWARD distortion (closed form, no iteration): where an ideal
        ray actually lands on the distorted sensor. Used to make sampling
        maps (e.g. the VO ground-window IPM) distortion-aware at the cost of
        a few polynomial terms per map pixel. Identity without distortion.
        """
        pts = np.asarray(pts, dtype=np.float64)
        if not self.has_distortion or len(pts) == 0:
            return pts
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        x = (pts[..., 0] - cx) / fx
        y = (pts[..., 1] - cy) / fy
        D = np.zeros(8)
        D[: min(len(self.D), 8)] = self.D[:8]
        k1, k2, p1, p2, k3, k4, k5, k6 = D
        if self.model == "equidistant":
            r = np.sqrt(x * x + y * y)
            theta = np.arctan(r)
            t2 = theta * theta
            theta_d = theta * (1.0 + k1 * t2 + k2 * t2**2 + p1 * t2**3 + p2 * t2**4)
            scale = np.where(r > 1e-12, theta_d / np.maximum(r, 1e-12), 1.0)
            xd, yd = x * scale, y * scale
        else:  # plumb_bob / rational_polynomial
            r2 = x * x + y * y
            radial = 1.0 + k1 * r2 + k2 * r2**2 + k3 * r2**3
            if self.model == "rational_polynomial":
                radial = radial / (1.0 + k4 * r2 + k5 * r2**2 + k6 * r2**3)
            xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
            yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        out = np.empty_like(pts, dtype=np.float64)
        out[..., 0] = xd * fx + cx
        out[..., 1] = yd * fy + cy
        return out

    def undistort_pixels(self, pts: NDArray[np.float64]) -> NDArray[np.float64]:
        """Distorted pixel coordinates -> ideal-pinhole pixel coordinates (same K).

        (N, 2) in, (N, 2) out, float64. With no distortion this is the
        identity (returns the input array itself: zero cost, bit-identical).
        """
        pts = np.asarray(pts, dtype=np.float64)
        if not self.has_distortion or len(pts) == 0:
            return pts
        import cv2

        if self.model == "equidistant":
            out = cv2.fisheye.undistortPoints(
                pts.reshape(-1, 1, 2), self.K, self.D.reshape(-1, 1), P=self.K
            )
        else:
            out = cv2.undistortPoints(pts.reshape(-1, 1, 2), self.K, self.D, P=self.K)
        return np.ascontiguousarray(out.reshape(-1, 2).astype(np.float64))

    def normalized_rays(self, pts: NDArray[np.float64]) -> NDArray[np.float64]:
        """Pixel coordinates -> (N, 3) rays [x, y, 1] in the optical frame."""
        pts = self.undistort_pixels(np.asarray(pts, dtype=np.float64))
        x = (pts[:, 0] - self.K[0, 2]) / self.K[0, 0]
        y = (pts[:, 1] - self.K[1, 2]) / self.K[1, 1]
        return np.column_stack([x, y, np.ones(len(pts))])

    # ---------------------------------------------------------- ray tables
    def pixel_rays(self, shape: tuple[int, int]) -> NDArray[np.float64]:
        """(H, W, 3) viewing-ray table in the camera OPTICAL frame, cached.

        ``shape`` is (rows, cols) of the image the table serves (the working
        resolution). Without distortion this is EXACTLY the pinhole table
        the evidence grid computes itself (same arithmetic, same dtype).
        With distortion, each pixel's ray goes through cv2.undistortPoints
        once at build time; per-frame cost is unchanged.
        """
        key = (self.K.tobytes(), self.D.tobytes(), self.model, tuple(shape))
        cached = _RAY_CACHE.get(key)
        if cached is not None:
            return cached
        h, w = int(shape[0]), int(shape[1])
        if not self.has_distortion:
            u = (np.arange(w, dtype=np.float64) - self.K[0, 2]) / self.K[0, 0]
            v = (np.arange(h, dtype=np.float64) - self.K[1, 2]) / self.K[1, 1]
            rays = np.empty((h, w, 3), dtype=np.float64)
            rays[..., 0] = u[None, :]
            rays[..., 1] = v[:, None]
            rays[..., 2] = 1.0
        else:
            uu, vv = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
            pts = np.column_stack([uu.ravel(), vv.ravel()])
            ideal = self.undistort_pixels(pts)
            rays = np.empty((h, w, 3), dtype=np.float64)
            rays[..., 0] = ((ideal[:, 0] - self.K[0, 2]) / self.K[0, 0]).reshape(h, w)
            rays[..., 1] = ((ideal[:, 1] - self.K[1, 2]) / self.K[1, 1]).reshape(h, w)
            rays[..., 2] = 1.0
        if len(_RAY_CACHE) > 8:
            _RAY_CACHE.clear()
        _RAY_CACHE[key] = rays
        return rays


# ------------------------------------------------------------------- YAML IO
def _require(data: dict, key: str, path: Path):
    if key not in data:
        raise ValueError(f"{path}: calibration YAML is missing {key!r}")
    return data[key]


def load_calibration_yaml(path: Path | str) -> CameraModel:
    """Load a calibration YAML (see the module docstring) into a CameraModel.

    Accepts both the flat format ``tools/dump_camera_info.py`` writes and the
    common ROS ``camera_calibration`` layout where ``camera_matrix`` /
    ``distortion_coefficients`` are ``{rows, cols, data}`` mappings.
    """
    import yaml

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"camera calibration file {path} does not exist - create it with "
            "tools/dump_camera_info.py (from a live camera_info topic) or by hand "
            "(see kart_map2d/camera/model.py for the format)"
        )
    with open(path, "r", encoding="utf-8-sig") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: calibration YAML must be a mapping")

    def matrix(value, n: int, name: str) -> NDArray[np.float64]:
        if isinstance(value, dict) and "data" in value:
            value = value["data"]
        arr = np.asarray(value, dtype=np.float64).ravel()
        if name == "camera_matrix" and arr.size != n:
            raise ValueError(f"{path}: {name} must hold {n} numbers, got {arr.size}")
        return arr

    width = int(_require(data, "image_width", path))
    height = int(_require(data, "image_height", path))
    K = matrix(_require(data, "camera_matrix", path), 9, "camera_matrix").reshape(3, 3)
    model = str(data.get("distortion_model", "none") or "none")
    D = matrix(data.get("distortion_coefficients", []), -1, "distortion_coefficients")
    return CameraModel(K, (width, height), D, model)


def save_calibration_yaml(path: Path | str, model: CameraModel) -> Path:
    """Write ``model`` in the flat calibration-YAML format (ASCII)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    K = model.K.ravel()
    lines = [
        "# kart_map2d camera calibration (see kart_map2d/camera/model.py)",
        f"image_width: {model.size[0]}",
        f"image_height: {model.size[1]}",
        "camera_matrix: [" + ", ".join(f"{v:.10g}" for v in K) + "]",
        f"distortion_model: {model.model}",
        "distortion_coefficients: [" + ", ".join(f"{v:.10g}" for v in model.D) + "]",
        "",
    ]
    path.write_text("\n".join(lines), encoding="ascii")
    return path
