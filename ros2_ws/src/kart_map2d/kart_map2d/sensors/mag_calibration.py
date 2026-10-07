"""Magnetometer calibration: classical ellipsoid fit -> hard/soft-iron YAML.

A raw three-axis magnetometer on a vehicle measures the Earth field PLUS a
constant vehicle-fixed offset (hard iron: magnets, DC currents) inside a
vehicle-fixed linear distortion (soft iron: nearby ferromagnetic material).
Driving a slow full circle sweeps the Earth field through the sensor's xy
plane, so the raw samples trace an ellipse (3D: an ellipsoid); fitting it
recovers both effects with zero learned models:

    corrected = soft_iron @ (raw - hard_iron)

after which ``corrected`` lies on a sphere and the planar heading is
``atan2(-cy, cx)`` (ENU map convention, x = the heading at declination 0)
up to the fixed mount yaw and the local magnetic declination - both are
[magnetometer] keys in kart_master.cfg.

The fit is the classical least-squares ellipsoid (Li/Griffiths style):
solve the general quadric ``x'Ax + b'x + c = 0`` in the 9-parameter form,
recentre, then take the matrix square root that maps the ellipsoid to a
sphere of radius = geometric mean. Degenerate inputs (too few samples, all
coplanar-collinear, fit not positive definite) are refused with the exact
recording procedure, never guessed.

REFUSAL RULE (requirement 13): switching ``magnetometer = 1`` without a
calibration file must stop at startup and print :data:`CALIBRATION_PROCEDURE`.

YAML format (written by tools/calibrate_magnetometer.py)::

    # kart_map2d magnetometer calibration
    hard_iron: [hx, hy, hz]          # same unit as the topic (tesla)
    soft_iron: [[..],[..],[..]]      # 3x3, unitless
    field_norm: 4.8e-05              # fitted sphere radius (diagnostics)
    samples: 1234
    rms_residual: 0.013              # relative to field_norm (diagnostics)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "CALIBRATION_PROCEDURE",
    "MagCalibration",
    "MagCalibrationError",
    "fit_ellipsoid",
    "load_mag_calibration",
    "save_mag_calibration",
]

#: Printed verbatim when magnetometer = 1 has no calibration file (req 13).
CALIBRATION_PROCEDURE = """\
magnetometer = 1 needs a calibration file ([magnetometer] calibration_file in
kart_master.cfg). Record one like this (once per rig; redo after moving any
metal or wiring near the sensor):

  1. Find open ground away from steel structures and buried cables.
  2. Start recording the magnetometer topic:
       ros2 bag record -o mag_cal /your/mag/topic
  3. Drive one SLOW full circle (360 deg, 30-60 s, walking pace). Slow
     matters: tilting and vibration smear the ellipse.
  4. Stop the recording and fit the calibration:
       python tools/calibrate_magnetometer.py --bag mag_cal --topic /your/mag/topic \\
           --out config/mag_calibration.yaml
     (or --csv samples.csv with columns x,y,z in tesla)
  5. Set [magnetometer] calibration_file = config/mag_calibration.yaml and
     declination_deg for your location (e.g. from NOAA's magnetic field
     calculator), and restart."""


class MagCalibrationError(ValueError):
    """A calibration that cannot be produced or trusted; the message says why."""


@dataclass
class MagCalibration:
    """hard/soft-iron correction: ``corrected = soft_iron @ (raw - hard_iron)``."""

    hard_iron: NDArray[np.float64]  # (3,)
    soft_iron: NDArray[np.float64]  # (3, 3)
    field_norm: float = 1.0  # fitted sphere radius (diagnostics only)
    samples: int = 0
    rms_residual: float = 0.0  # relative to field_norm
    meta: dict = field(default_factory=dict)

    def correct(self, raw: NDArray[np.float64]) -> NDArray[np.float64]:
        """(N, 3) or (3,) raw field -> corrected field (same shape)."""
        raw = np.asarray(raw, dtype=np.float64)
        return (raw - self.hard_iron) @ self.soft_iron.T

    def planar_heading(self, raw: NDArray[np.float64]) -> float:
        """Corrected planar magnetic heading [rad], CCW from magnetic north
        expressed in the project's map convention (yaw CCW from +x).

        The sensor is assumed base_link-aligned up to the [magnetometer]
        ``mount_yaw_deg`` key (applied by the adapter, not here). In ENU
        axes with x forward, the magnetic field points towards magnetic
        north, so the vehicle yaw relative to magnetic north is
        ``atan2(-my, mx)``.
        """
        c = self.correct(np.asarray(raw, dtype=np.float64).reshape(3))
        return float(np.arctan2(-c[1], c[0]))


def fit_ellipsoid(samples: NDArray[np.float64], min_samples: int = 50) -> MagCalibration:
    """Least-squares ellipsoid fit of (N, 3) raw magnetometer samples.

    Raises :class:`MagCalibrationError` (with the recording procedure) for
    inputs that cannot constrain the fit: fewer than ``min_samples`` rows,
    near-zero spread on the xy plane (the kart never turned), or a fitted
    quadric that is not an ellipsoid (junk data).
    """
    pts = np.asarray(samples, dtype=np.float64)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < int(min_samples):
        raise MagCalibrationError(
            f"need at least {min_samples} finite 3-axis samples, got {0 if pts.ndim != 2 else len(pts)}.\n\n"
            + CALIBRATION_PROCEDURE
        )
    # Precondition: work in centred, scaled coordinates for numeric health.
    centre = pts.mean(axis=0)
    scale = float(np.abs(pts - centre).max())
    if scale <= 0:
        raise MagCalibrationError("all samples identical - the sensor is stuck.\n\n" + CALIBRATION_PROCEDURE)
    q = (pts - centre) / scale
    spread = q[:, :2].std(axis=0)
    if float(spread.min()) < 1e-3:
        raise MagCalibrationError(
            "the samples have no planar spread - the recording must contain a full slow "
            "360 deg turn.\n\n" + CALIBRATION_PROCEDURE
        )
    x, y, z = q[:, 0], q[:, 1], q[:, 2]
    # General quadric: ax^2+by^2+cz^2+2dxy+2exz+2fyz+2gx+2hy+2iz = 1.
    D = np.column_stack([x * x, y * y, z * z, 2 * x * y, 2 * x * z, 2 * y * z, 2 * x, 2 * y, 2 * z])
    try:
        v, *_ = np.linalg.lstsq(D, np.ones(len(q)), rcond=None)
    except np.linalg.LinAlgError as exc:
        raise MagCalibrationError(f"ellipsoid fit failed ({exc}).\n\n" + CALIBRATION_PROCEDURE) from None
    A = np.array([[v[0], v[3], v[4]], [v[3], v[1], v[5]], [v[4], v[5], v[2]]])
    b = v[6:9]
    try:
        centre_q = -np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        raise MagCalibrationError("degenerate quadric (singular).\n\n" + CALIBRATION_PROCEDURE) from None
    # Recentred form: with x'Ax + 2b'x = 1 and c = -A^-1 b,
    # (x - c)' A (x - c) = 1 + c'Ac  (the cross term cancels: c'A = -b').
    k = 1.0 + centre_q @ A @ centre_q
    if not np.isfinite(k) or abs(k) < 1e-12:
        raise MagCalibrationError("degenerate quadric (zero radius).\n\n" + CALIBRATION_PROCEDURE)
    A_n = A / k
    eigval, eigvec = np.linalg.eigh(A_n)
    if np.any(eigval <= 0):
        raise MagCalibrationError(
            "the fitted quadric is not an ellipsoid - the data does not cover enough "
            "directions or is corrupted.\n\n" + CALIBRATION_PROCEDURE
        )
    # Map the ellipsoid to a sphere of radius r0 = geometric mean of semi-axes.
    semi = 1.0 / np.sqrt(eigval)  # in preconditioned units
    r0 = float(np.exp(np.mean(np.log(semi))))
    soft = eigvec @ np.diag(r0 / semi) @ eigvec.T  # symmetric, unit-determinant-ish
    hard = centre + scale * centre_q
    field_norm = r0 * scale
    corrected = (pts - hard) @ soft.T
    radii = np.linalg.norm(corrected, axis=1)
    rms = float(np.sqrt(np.mean((radii - field_norm) ** 2)) / field_norm)
    if rms > 0.35:
        raise MagCalibrationError(
            f"fit residual {rms:.2f} of the field norm is too large to trust "
            "(vibration, magnetic disturbance or a partial turn).\n\n" + CALIBRATION_PROCEDURE
        )
    return MagCalibration(
        hard_iron=hard,
        soft_iron=soft,
        field_norm=field_norm,
        samples=int(len(pts)),
        rms_residual=rms,
    )


# ------------------------------------------------------------------- YAML IO
def load_mag_calibration(path: Path | str) -> MagCalibration:
    """Load the calibration YAML; a missing file raises with the procedure."""
    import yaml

    path = Path(path)
    if not path.exists():
        raise MagCalibrationError(f"calibration file {path} does not exist.\n\n" + CALIBRATION_PROCEDURE)
    with open(path, "r", encoding="utf-8-sig") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or "hard_iron" not in data or "soft_iron" not in data:
        raise MagCalibrationError(f"{path} is not a magnetometer calibration (needs hard_iron + soft_iron).\n\n" + CALIBRATION_PROCEDURE)
    hard = np.asarray(data["hard_iron"], dtype=np.float64).reshape(3)
    soft = np.asarray(data["soft_iron"], dtype=np.float64).reshape(3, 3)
    return MagCalibration(
        hard_iron=hard,
        soft_iron=soft,
        field_norm=float(data.get("field_norm", 1.0)),
        samples=int(data.get("samples", 0)),
        rms_residual=float(data.get("rms_residual", 0.0)),
        meta={k: v for k, v in data.items() if k not in ("hard_iron", "soft_iron")},
    )


def save_mag_calibration(path: Path | str, cal: MagCalibration) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [", ".join(f"{v:.10g}" for v in row) for row in cal.soft_iron]
    lines = [
        "# kart_map2d magnetometer calibration (tools/calibrate_magnetometer.py)",
        "hard_iron: [" + ", ".join(f"{v:.10g}" for v in cal.hard_iron) + "]",
        "soft_iron:",
        *[f"  - [{row}]" for row in rows],
        f"field_norm: {cal.field_norm:.10g}",
        f"samples: {cal.samples}",
        f"rms_residual: {cal.rms_residual:.6g}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="ascii")
    return path
