"""kart_master.cfg: the ONE control file - parser, validator, translators.

Design rules (all user requirements):

- INI-style ``key = value`` with sections; read ONCE at startup (no file
  watching, no per-frame cost). Live-safe knobs (e.g. [drive] max_speed_mps)
  are additionally ROS parameters changeable with ``ros2 param set``.
- Accepts UTF-8 (with or without BOM) and UTF-16 with BOM (Notepad saves),
  CRLF line endings, full-line and trailing ``#`` comments.
- Validation BEFORE anything runs: an unknown section/key is an error with a
  did-you-mean hint and the line number; an out-of-range value is an error
  naming the allowed range; every [sensors] switch at 0 is an error; a
  combination with no odometric source (IMU only) is a WARNING explaining it
  is dead reckoning; a not-implemented sensor set to 1 (gnss) is an error.
- Launch arguments / CLI overrides can flip any [sensors] switch
  (``lidar:=1``) without editing the file.
- [sensors] gates which inputs exist; the per-sensor sections hold their
  parameters. Shipped defaults: imu = 1, camera = 1, everything else 0 (the
  camera is the ONLY visual input by default; lidar is strictly an
  off-by-default experiment switch).

The module is pure Python (no ROS imports): the live node, the drive node,
the offline replay runner and the launch files all call it.
"""

from __future__ import annotations

import difflib
import io
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "MasterConfig",
    "MasterConfigError",
    "SENSOR_KEYS",
    "combo_name",
    "load_master_config",
    "parse_master_text",
    "to_drive_params_dict",
    "to_mapping_params_dict",
]


class MasterConfigError(ValueError):
    """A configuration that must not run; ``problems`` lists every issue with its fix."""

    def __init__(self, problems: list[str] | str) -> None:
        self.problems = [problems] if isinstance(problems, str) else list(problems)
        super().__init__("\n".join(self.problems))


# ------------------------------------------------------------------ schema
@dataclass(frozen=True)
class KeySpec:
    """One key: type, default, allowed range/choices, unit, safety flag."""

    kind: str  # "int01" | "int" | "float" | "str" | "choice" | "floats3" | "float_or_empty"
    default: object
    lo: float | None = None
    hi: float | None = None
    unit: str = ""
    choices: tuple = ()
    safety: bool = False


def _k01(default: int, safety: bool = False) -> KeySpec:
    return KeySpec("int01", default, 0, 1, safety=safety)


#: The [sensors] switches, in canonical order (also the combo-name order).
#: gnss is declared but NOT implemented: setting it to 1 is a hard error.
SENSOR_KEYS = (
    "imu",
    "camera",
    "camera_2",
    "lidar",
    "odom",
    "livox_imu",
    "magnetometer",
    "speedometer",
    "wheel_ticks",
    "gnss",
)

_NOT_IMPLEMENTED_SENSORS = ("gnss",)

#: Short codes for output-folder combination names (requirement 9).
_COMBO_CODES = {
    "imu": "imu",
    "camera": "cam",
    "camera_2": "cam2",
    "lidar": "lidar",
    "odom": "odom",
    "livox_imu": "livimu",
    "magnetometer": "mag",
    "speedometer": "spd",
    "wheel_ticks": "wheel",
    "gnss": "gnss",
}

_CAMERA_SECTION_KEYS = {
    "name": KeySpec("str", ""),  # defaults to the section name
    "source": KeySpec("choice", "ros2", choices=("ros2", "device", "gstreamer")),
    "image_topic": KeySpec("str", ""),
    "camera_info_topic": KeySpec("str", ""),
    "compressed": _k01(0),
    "frame": KeySpec("str", ""),  # TF frame override; "" = the image header's frame_id
    "mount_xyz": KeySpec("floats3", (0.0, 0.0, 0.5), unit="m"),
    "mount_rpy_deg": KeySpec("floats3", (0.0, 0.0, 0.0), unit="deg"),
    "height_m": KeySpec("float_or_empty", None, 0.05, 5.0, unit="m"),
    "mask_file": KeySpec("str", ""),
    "learn_mask": _k01(1),
    "calibration_file": KeySpec("str", ""),
    "device": KeySpec("str", "0"),
    "gstreamer_pipeline": KeySpec("str", ""),
    "driver": KeySpec("choice", "none", choices=("none", "zed", "usb_cam")),
    "driver_fps": KeySpec("float", 15.0, 1.0, 120.0, unit="Hz"),
    "driver_width": KeySpec("int", 1920, 64, 8192, unit="px"),
    "driver_height": KeySpec("int", 1200, 64, 8192, unit="px"),
    "processing_scale": KeySpec("float", 0.5, 0.05, 1.0),
    "max_process_rate_hz": KeySpec("float", 0.0, 0.0, 120.0, unit="Hz"),
}

SCHEMA: dict[str, dict[str, KeySpec]] = {
    "sensors": {name: _k01(1 if name in ("imu", "camera") else 0) for name in SENSOR_KEYS},
    "camera": dict(_CAMERA_SECTION_KEYS),
    "camera_2": dict(_CAMERA_SECTION_KEYS),
    "imu": {
        "topic": KeySpec("str", "/imu/data"),
        "frame": KeySpec("str", "imu_link"),
        "rate_hz": KeySpec("float", 100.0, 10.0, 400.0, unit="Hz"),
        "mount_xyz": KeySpec("floats3", (0.0, 0.0, 0.0), unit="m"),
        "livox_topic": KeySpec("str", "/livox/imu"),
        "livox_accel_in_g": _k01(1),
        "livox_xyz": KeySpec("floats3", (0.0, 0.0, 0.52), unit="m"),
        "livox_rpy_deg": KeySpec("floats3", (0.0, 0.0, -90.0), unit="deg"),
    },
    "lidar": {
        "topic": KeySpec("str", "/livox/lidar"),
        "xyz": KeySpec("floats3", (0.0, 0.0, 0.52), unit="m"),
        "rpy_deg": KeySpec("floats3", (0.0, 0.0, -90.0), unit="deg"),
    },
    "odom": {
        "topic": KeySpec("str", "/odom"),
        "stride": KeySpec("int", 2, 1, 20),
    },
    "odometry": {
        "orb_features": KeySpec("int", 3000, 200, 20000),
        "max_corners": KeySpec("int", 1500, 100, 10000),
        "feature_spacing_px": KeySpec("int", 10, 1, 100, unit="px"),
        "min_pair_dt_s": KeySpec("float", 0.09, 0.0, 2.0, unit="s"),
    },
    "magnetometer": {
        "topic": KeySpec("str", "/mag"),
        "calibration_file": KeySpec("str", ""),
        "declination_deg": KeySpec("float", 0.0, -45.0, 45.0, unit="deg"),
        "mount_yaw_deg": KeySpec("float", 0.0, -180.0, 180.0, unit="deg"),
        "sigma_deg": KeySpec("float", 8.0, 0.5, 60.0, unit="deg"),
        "rate_hz": KeySpec("float", 20.0, 0.5, 200.0, unit="Hz"),
    },
    "speedometer": {
        "topic": KeySpec("str", "/speed"),
        "type": KeySpec("choice", "float32", choices=("float32", "twist_stamped")),
        "sigma_mps": KeySpec("float", 0.15, 0.005, 5.0, unit="m/s"),
    },
    "wheel_ticks": {
        "topic": KeySpec("str", "/wheel_ticks"),
        "type": KeySpec("choice", "int64", choices=("int64", "joint_state")),
        "joint": KeySpec("str", ""),
        "ticks_per_metre": KeySpec("float", 1000.0, 1.0, 1e6, unit="ticks/m"),
        "publish_rate_hz": KeySpec("float", 50.0, 1.0, 1000.0, unit="Hz"),
        "sigma_floor_mps": KeySpec("float", 0.05, 0.001, 2.0, unit="m/s"),
    },
    "segmentation": {
        "mode": KeySpec("choice", "classical", choices=("classical", "neural", "auto")),
        "rate_hz": KeySpec("float", 2.0, 0.1, 20.0, unit="Hz"),
        "scale": KeySpec("float", 0.5, 0.05, 1.0),
    },
    "mapping": {
        "cell_size_m": KeySpec("float", 0.15, 0.02, 1.0, unit="m"),
        "max_range_m": KeySpec("float", 18.0, 3.0, 60.0, unit="m"),
        "dynamic_half_life_s": KeySpec("float", 3.0, 0.2, 60.0, unit="s"),
        "min_contact_range_m": KeySpec("float", 2.5, 0.0, 10.0, unit="m"),
        "loop_closure": _k01(1),
        "output_dir": KeySpec("str", "outputs/maps_live"),
        "map_name": KeySpec("str", "live"),
        "save_on_shutdown": _k01(1),
        "corridor_spacing_m": KeySpec("float", 0.5, 0.1, 5.0, unit="m"),
        "corridor_close_cells": KeySpec("int", 2, 0, 10, unit="cells"),
    },
    "drive": {
        "map_package": KeySpec("str", ""),
        "max_speed_mps": KeySpec("float", 2.0, 0.0, 12.0, unit="m/s", safety=True),
        "speed_scale": KeySpec("float", 1.0, 0.0, 1.0, safety=True),
        "lookahead_base_m": KeySpec("float", 0.6, 0.1, 10.0, unit="m"),
        "control_rate_hz": KeySpec("float", 50.0, 5.0, 200.0, unit="Hz"),
        "command_topic": KeySpec("str", "/drive"),
        "command_type": KeySpec("choice", "ackermann", choices=("ackermann", "twist")),
        "global_relocalization": _k01(1),
    },
    "safety": {
        "stop_distance_m": KeySpec("float", 0.5, 0.0, 10.0, unit="m", safety=True),
        "pose_timeout_s": KeySpec("float", 0.2, 0.02, 5.0, unit="s", safety=True),
        "deadman_timeout_s": KeySpec("float", 0.0, 0.0, 10.0, unit="s", safety=True),
        "estop_topic": KeySpec("str", "/estop", safety=True),
    },
    "system": {
        "log_level": KeySpec("choice", "info", choices=("debug", "info", "warn", "error")),
        "use_sim_time": _k01(0),
    },
}


# ------------------------------------------------------------------ parsing
@dataclass
class MasterConfig:
    """Validated configuration: ``cfg["section"]["key"]`` holds typed values.

    Every key of every section is present (file value or schema default), so
    consumers never need their own defaulting. ``warnings`` holds non-fatal
    findings (e.g. dead-reckoning-only sensor sets) for the caller to log.
    """

    values: dict[str, dict[str, object]]
    path: str = ""
    warnings: list = field(default_factory=list)

    def __getitem__(self, section: str) -> dict[str, object]:
        return self.values[section]

    def sensors_on(self) -> list[str]:
        """Enabled [sensors] switches in canonical order."""
        return [k for k in SENSOR_KEYS if self.values["sensors"].get(k)]


def _decode(raw: bytes, where: str) -> str:
    """UTF-8 (with/without BOM) or UTF-16 with BOM (Notepad); CRLF tolerated."""
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16")
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise MasterConfigError(
            f"{where}: not UTF-8 or UTF-16-with-BOM text ({exc}); re-save it as UTF-8"
        ) from None


def _strip_comment(line: str) -> str:
    """Remove a trailing # comment (a # inside a value must be preceded by whitespace)."""
    if line.lstrip().startswith(("#", ";")):
        return ""
    for marker in (" #", "\t#"):
        pos = line.find(marker)
        if pos >= 0:
            return line[:pos]
    return line


def _parse_scalar(section: str, key: str, spec: KeySpec, text: str, lineno: int, problems: list[str]):
    where = f"line {lineno}: [{section}] {key}"

    def rng() -> str:
        unit = f" {spec.unit}" if spec.unit else ""
        return f"{spec.lo:g} .. {spec.hi:g}{unit}"

    text = text.strip()
    if spec.kind == "str":
        return text
    if spec.kind == "choice":
        low = text.lower()
        if low not in spec.choices:
            problems.append(f"{where}: {text!r} is not one of {' | '.join(spec.choices)}")
            return spec.default
        return low
    if spec.kind == "int01":
        if text not in ("0", "1"):
            problems.append(f"{where}: must be 0 or 1, got {text!r}")
            return spec.default
        return int(text)
    if spec.kind == "float_or_empty":
        if text == "" or text.lower() in ("none", "auto"):
            return None
        try:
            value = float(text)
        except ValueError:
            problems.append(f"{where}: {text!r} is not a number (or empty for auto)")
            return spec.default
        if spec.lo is not None and not (spec.lo <= value <= spec.hi):
            problems.append(f"{where}: {value:g} is outside the allowed range {rng()}")
        return value
    if spec.kind == "int":
        try:
            value = int(text)
        except ValueError:
            problems.append(f"{where}: {text!r} is not an integer")
            return spec.default
        if spec.lo is not None and not (spec.lo <= value <= spec.hi):
            problems.append(f"{where}: {value} is outside the allowed range {rng()}")
        return value
    if spec.kind == "float":
        try:
            value = float(text)
        except ValueError:
            problems.append(f"{where}: {text!r} is not a number")
            return spec.default
        if spec.lo is not None and not (spec.lo <= value <= spec.hi):
            problems.append(f"{where}: {value:g} is outside the allowed range {rng()}")
        return value
    if spec.kind == "floats3":
        parts = [p for p in text.replace(",", " ").split() if p]
        try:
            values = tuple(float(p) for p in parts)
        except ValueError:
            values = ()
        if len(values) != 3:
            problems.append(f"{where}: needs 3 numbers (x, y, z), got {text!r}")
            return spec.default
        return values
    raise AssertionError(f"unhandled KeySpec kind {spec.kind}")


def parse_master_text(text: str, where: str = "kart_master.cfg") -> MasterConfig:
    """Parse + validate the cfg text; raises :class:`MasterConfigError` listing
    every problem (nothing runs on a bad file). See :func:`load_master_config`."""
    problems: list[str] = []
    warnings: list[str] = []
    values: dict[str, dict[str, object]] = {s: {k: v.default for k, v in keys.items()} for s, keys in SCHEMA.items()}
    seen: dict[tuple[str, str], int] = {}

    section: str | None = None
    for lineno, raw_line in enumerate(io.StringIO(text).read().splitlines(), start=1):
        line = _strip_comment(raw_line).strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            name = line[1:-1].strip().lower()
            if name not in SCHEMA:
                hint = difflib.get_close_matches(name, list(SCHEMA), n=1, cutoff=0.6)
                suffix = f" (did you mean [{hint[0]}]?)" if hint else ""
                problems.append(f"line {lineno}: unknown section [{name}]{suffix}")
                section = None  # keys below report against the unknown section once
            else:
                section = name
            continue
        if "=" not in line:
            problems.append(f"line {lineno}: expected 'key = value', got {line!r}")
            continue
        key, _, value = line.partition("=")
        key = key.strip().lower()
        if section is None:
            continue  # already reported the section
        keys = SCHEMA[section]
        if key not in keys:
            hint = difflib.get_close_matches(key, list(keys), n=1, cutoff=0.6)
            suffix = f" (did you mean {hint[0]!r}?)" if hint else ""
            problems.append(f"line {lineno}: [{section}] has no key {key!r}{suffix}")
            continue
        if (section, key) in seen:
            problems.append(
                f"line {lineno}: [{section}] {key} was already set on line {seen[(section, key)]}"
            )
            continue
        seen[(section, key)] = lineno
        values[section][key] = _parse_scalar(section, key, keys[key], value, lineno, problems)

    if problems:
        raise MasterConfigError([f"{where}: configuration refused - fix these first:"] + problems)
    cfg = MasterConfig(values=values, path=where, warnings=warnings)
    return cfg


def _cross_validate(cfg: MasterConfig, base_dir: Path | None) -> None:
    """Rules that span keys; called after parsing AND after overrides."""
    problems: list[str] = []
    sensors = cfg.values["sensors"]

    for name in _NOT_IMPLEMENTED_SENSORS:
        if sensors.get(name):
            problems.append(
                f"[sensors] {name} = 1, but {name} input is NOT implemented in this package. "
                f"Set {name} = 0; when the hardware arrives, implement its adapter first "
                "(kart_map2d/sensors/adapters.py) - do not expect it to fuse silently."
            )
    if not any(sensors.get(k) for k in SENSOR_KEYS):
        problems.append(
            "[sensors]: every switch is 0 - nothing would be subscribed and nothing can be "
            "mapped. Enable at least imu = 1 and camera = 1 (the default configuration)."
        )
    if sensors.get("camera_2") and not sensors.get("camera"):
        cfg.warnings.append(
            "[sensors] camera_2 = 1 with camera = 0: only the comparison camera is active; "
            "the primary [camera] section is ignored."
        )
    odometric = [k for k in ("camera", "camera_2", "lidar", "odom", "speedometer", "wheel_ticks") if sensors.get(k)]
    if sensors.get("imu") and not odometric:
        cfg.warnings.append(
            "[sensors]: the IMU is the only motion source - this is DEAD RECKONING. "
            "Position drifts without bound (double-integrated accelerometer noise); ZUPTs "
            "only help while standing still. Enable camera = 1 (or any odometric source) "
            "for a usable map."
        )
    if sensors.get("magnetometer"):
        cal = str(cfg.values["magnetometer"]["calibration_file"]).strip()
        if not cal:
            from kart_map2d.sensors.mag_calibration import CALIBRATION_PROCEDURE

            problems.append("[magnetometer] calibration_file is empty but magnetometer = 1.\n" + CALIBRATION_PROCEDURE)
        elif base_dir is not None and not (base_dir / cal).exists() and not Path(cal).is_absolute():
            from kart_map2d.sensors.mag_calibration import CALIBRATION_PROCEDURE

            problems.append(
                f"[magnetometer] calibration_file {cal!r} does not exist (looked under {base_dir}).\n"
                + CALIBRATION_PROCEDURE
            )
        elif Path(cal).is_absolute() and not Path(cal).exists():
            from kart_map2d.sensors.mag_calibration import CALIBRATION_PROCEDURE

            problems.append(f"[magnetometer] calibration_file {cal!r} does not exist.\n" + CALIBRATION_PROCEDURE)

    for switch, section in (("camera", "camera"), ("camera_2", "camera_2")):
        if not sensors.get(switch):
            continue
        cam = cfg.values[section]
        source = cam["source"]
        if source == "ros2":
            if not cam["image_topic"]:
                problems.append(f"[{section}] source = ros2 needs image_topic")
            if not cam["camera_info_topic"] and not cam["calibration_file"]:
                problems.append(
                    f"[{section}]: give camera_info_topic (ROS intrinsics) or calibration_file "
                    "(tools/dump_camera_info.py writes one from a live camera_info topic)"
                )
        elif source == "device":
            if not cam["calibration_file"]:
                problems.append(
                    f"[{section}] source = device has no camera_info topic: set calibration_file "
                    "(tools/dump_camera_info.py, or calibrate and write the YAML by hand)"
                )
        elif source == "gstreamer":
            if not cam["gstreamer_pipeline"]:
                problems.append(f"[{section}] source = gstreamer needs gstreamer_pipeline")
            if not cam["calibration_file"]:
                problems.append(f"[{section}] source = gstreamer needs calibration_file (no camera_info topic exists)")
        if cam["driver"] != "none" and source != "ros2":
            problems.append(
                f"[{section}] driver = {cam['driver']} starts a ROS driver, which only makes sense "
                "with source = ros2 (the driver publishes the topics)"
            )

    if sensors.get("lidar") and sensors.get("livox_imu"):
        # The mapping node models ONE physical Livox unit: the LiDAR points
        # and its built-in IMU share a mount. Different [lidar] and [imu]
        # livox_* values would silently disagree, so they must match.
        li, im = cfg.values["lidar"], cfg.values["imu"]
        if tuple(li["xyz"]) != tuple(im["livox_xyz"]) or tuple(li["rpy_deg"]) != tuple(im["livox_rpy_deg"]):
            problems.append(
                "[lidar] xyz/rpy_deg and [imu] livox_xyz/livox_rpy_deg describe the same physical "
                "Livox unit but differ - set them equal (or disable one of lidar / livox_imu)."
            )

    if problems:
        raise MasterConfigError([f"{cfg.path}: configuration refused - fix these first:"] + problems)


def load_master_config(
    path: Path | str,
    overrides: dict[str, int] | None = None,
) -> MasterConfig:
    """Read + validate ``kart_master.cfg``.

    ``overrides`` maps [sensors] switch names to 0/1 (launch arguments /
    ``--set sensors.lidar=1`` CLI tokens outrank the file). Raises
    :class:`MasterConfigError` with every problem; ``cfg.warnings`` carries
    the non-fatal findings for the caller to log.
    """
    path = Path(path)
    if not path.exists():
        raise MasterConfigError(f"{path}: the master config file does not exist")
    cfg = parse_master_text(_decode(path.read_bytes(), str(path)), where=str(path))
    cfg.path = str(path)
    for key, value in (overrides or {}).items():
        key = key.lower()
        if key not in SENSOR_KEYS:
            hint = difflib.get_close_matches(key, SENSOR_KEYS, n=1, cutoff=0.6)
            suffix = f" (did you mean {hint[0]!r}?)" if hint else ""
            raise MasterConfigError(f"override {key!r} is not a [sensors] switch{suffix}")
        if value not in (0, 1):
            raise MasterConfigError(f"override {key} must be 0 or 1, got {value!r}")
        cfg.values["sensors"][key] = int(value)
    _cross_validate(cfg, base_dir=path.parent)
    return cfg


# ------------------------------------------------------------------ helpers
def combo_name(cfg: MasterConfig) -> str:
    """Short name of the enabled sensor combination, e.g. ``imu+cam`` (req 9)."""
    return "+".join(_COMBO_CODES[k] for k in cfg.sensors_on()) or "none"


def _resolve(base_dir: Path | None, value: str) -> str:
    """A cfg-relative path resolved against the cfg's directory (absolute passes through)."""
    if not value:
        return value
    p = Path(value)
    if p.is_absolute() or base_dir is None:
        return str(p)
    return str((base_dir / p).resolve())


def _camera_entry(cfg: MasterConfig, section: str, base_dir: Path | None) -> tuple[str, dict]:
    cam = cfg.values[section]
    name = str(cam["name"]).strip() or section
    entry = {
        "source": cam["source"],
        "image_topic": cam["image_topic"],
        "info_topic": cam["camera_info_topic"],
        "compressed": bool(cam["compressed"]),
        "frame": cam["frame"],
        "mount_xyz": tuple(cam["mount_xyz"]),
        "mount_rpy_deg": tuple(cam["mount_rpy_deg"]),
        "height_m": cam["height_m"],
        "mask_file": _resolve(base_dir, str(cam["mask_file"])),
        "learn_mask": bool(cam["learn_mask"]),
        "calibration_file": _resolve(base_dir, str(cam["calibration_file"])),
        "device": str(cam["device"]),
        "gstreamer_pipeline": cam["gstreamer_pipeline"],
        # driver_* double as the device-capture width/height/fps request
        # (for source = ros2 they configure the launch-started driver instead).
        "device_width": int(cam["driver_width"]),
        "device_height": int(cam["driver_height"]),
        "device_fps": float(cam["driver_fps"]),
        "processing_scale": float(cam["processing_scale"]),
        "max_process_rate_hz": float(cam["max_process_rate_hz"]),
    }
    return name, entry


def to_mapping_params_dict(cfg: MasterConfig) -> dict:
    """Translate the cfg into the mapping node's params dict
    (:meth:`kart_map2d.ros2.node.NodeParams.from_dict` consumes it)."""
    base_dir = Path(cfg.path).parent if cfg.path else None
    s = cfg.values["sensors"]
    imu, lidar, odom = cfg.values["imu"], cfg.values["lidar"], cfg.values["odom"]
    seg, mapping, system = cfg.values["segmentation"], cfg.values["mapping"], cfg.values["system"]
    vo = cfg.values["odometry"]
    cameras: dict[str, dict] = {}
    if s["camera"]:
        name, entry = _camera_entry(cfg, "camera", base_dir)
        cameras[name] = entry
    if s["camera_2"]:
        name, entry = _camera_entry(cfg, "camera_2", base_dir)
        if name in cameras:
            name = name + "_2"
        cameras[name] = entry
    # One physical Livox unit: with the LiDAR enabled its [lidar] mount is
    # authoritative for the shared livox mount parameter (cross-validated
    # equal to [imu] livox_* when both switches are on).
    livox_xyz = tuple(lidar["xyz"]) if s["lidar"] else tuple(imu["livox_xyz"])
    livox_rpy = tuple(lidar["rpy_deg"]) if s["lidar"] else tuple(imu["livox_rpy_deg"])
    out: dict = {
        "imu_topic": imu["topic"] if s["imu"] else "",
        "imu_rate_hz": float(imu["rate_hz"]),
        "livox_imu_topic": imu["livox_topic"] if s["livox_imu"] else "",
        "livox_accel_in_g": bool(imu["livox_accel_in_g"]),
        "livox_xyz": livox_xyz,
        "livox_rpy_deg": livox_rpy,
        "odom_topic": odom["topic"] if s["odom"] else "",
        "odom_stride": int(odom["stride"]),
        "lidar_topic": lidar["topic"] if s["lidar"] else "",
        "cameras": cameras,
        "seg_mode": seg["mode"],
        "seg_period_s": 1.0 / float(seg["rate_hz"]),
        "image_scale": float(seg["scale"]),
        "vo_orb_features": int(vo["orb_features"]),
        "vo_max_corners": int(vo["max_corners"]),
        "vo_feature_spacing_px": int(vo["feature_spacing_px"]),
        "vo_min_pair_dt_s": float(vo["min_pair_dt_s"]),
        "grid_cell_m": float(mapping["cell_size_m"]),
        "camera_max_range_m": float(mapping["max_range_m"]),
        "dyn_half_life_s": float(mapping["dynamic_half_life_s"]),
        "min_contact_range_m": float(mapping["min_contact_range_m"]),
        "loop_closure": bool(mapping["loop_closure"]),
        "output_dir": _resolve(base_dir, str(mapping["output_dir"])),
        "map_name": str(mapping["map_name"]),
        "save_on_shutdown": bool(mapping["save_on_shutdown"]),
        "corridor_spacing_m": float(mapping["corridor_spacing_m"]),
        "corridor_close_cells": int(mapping["corridor_close_cells"]),
        "use_sim_time": bool(system["use_sim_time"]),
        # all-sensor inputs (requirement 13)
        "mag_topic": cfg.values["magnetometer"]["topic"] if s["magnetometer"] else "",
        "mag_calibration_file": _resolve(base_dir, str(cfg.values["magnetometer"]["calibration_file"])) if s["magnetometer"] else "",
        "mag_declination_deg": float(cfg.values["magnetometer"]["declination_deg"]),
        "mag_mount_yaw_deg": float(cfg.values["magnetometer"]["mount_yaw_deg"]),
        "mag_sigma_deg": float(cfg.values["magnetometer"]["sigma_deg"]),
        "mag_rate_hz": float(cfg.values["magnetometer"]["rate_hz"]),
        "speed_topic": cfg.values["speedometer"]["topic"] if s["speedometer"] else "",
        "speed_type": cfg.values["speedometer"]["type"],
        "speed_sigma_mps": float(cfg.values["speedometer"]["sigma_mps"]),
        "wheel_topic": cfg.values["wheel_ticks"]["topic"] if s["wheel_ticks"] else "",
        "wheel_type": cfg.values["wheel_ticks"]["type"],
        "wheel_joint": str(cfg.values["wheel_ticks"]["joint"]),
        "wheel_ticks_per_metre": float(cfg.values["wheel_ticks"]["ticks_per_metre"]),
        "wheel_sigma_floor_mps": float(cfg.values["wheel_ticks"]["sigma_floor_mps"]),
    }
    return out


def to_drive_params_dict(cfg: MasterConfig) -> dict:
    """Translate the cfg into the drive node's params dict
    (:meth:`kart_map2d.ros2.drive_node.DriveParams.from_dict` consumes it)."""
    base_dir = Path(cfg.path).parent if cfg.path else None
    s = cfg.values["sensors"]
    imu, lidar, odom = cfg.values["imu"], cfg.values["lidar"], cfg.values["odom"]
    drive, safety, system = cfg.values["drive"], cfg.values["safety"], cfg.values["system"]
    cameras: dict[str, dict] = {}
    for switch, section in (("camera", "camera"), ("camera_2", "camera_2")):
        if not s[switch]:
            continue
        cam = cfg.values[section]
        if cam["source"] != "ros2":
            # The drive node consumes ROS image topics only (its VO / cue
            # workers subscribe); a device camera must be bridged by a driver.
            raise MasterConfigError(
                f"[{section}] source = {cam['source']}: the drive node reads cameras from ROS "
                "topics only. Use source = ros2 with a camera driver, or disable the camera."
            )
        name = str(cam["name"]).strip() or section
        entry = {
            "image_topic": cam["image_topic"],
            "info_topic": cam["camera_info_topic"],
            "compressed": bool(cam["compressed"]),
            "mount_xyz": tuple(cam["mount_xyz"]),
            "mount_rpy_deg": tuple(cam["mount_rpy_deg"]),
            "image_scale": float(cam["processing_scale"]),
        }
        if cam["height_m"] is not None:
            entry["height_m"] = float(cam["height_m"])
        if cam["mask_file"]:
            entry["kart_mask_path"] = _resolve(base_dir, str(cam["mask_file"]))
        if not cam["camera_info_topic"] and cam["calibration_file"]:
            from kart_map2d.camera.model import load_calibration_yaml

            model = load_calibration_yaml(_resolve(base_dir, str(cam["calibration_file"])))
            entry["intrinsics"] = (
                float(model.K[0, 0]), float(model.K[1, 1]), float(model.K[0, 2]), float(model.K[1, 2]),
                float(model.size[0]), float(model.size[1]),
            )
        cameras[name] = entry
    out: dict = {
        "map_package": _resolve(base_dir, str(drive["map_package"])),
        "imu_topic": imu["topic"] if s["imu"] else "",
        "imu_rate_hz": float(imu["rate_hz"]),
        "odom_topic": odom["topic"] if s["odom"] else "",
        "odom_stride": int(odom["stride"]),
        "lidar_topic": lidar["topic"] if s["lidar"] else "",
        "lidar_xyz": tuple(lidar["xyz"]),
        "lidar_rpy_deg": tuple(lidar["rpy_deg"]),
        "cameras": cameras,
        "max_speed_mps": float(drive["max_speed_mps"]),
        "speed_scale": float(drive["speed_scale"]),
        "global_relocalization": bool(drive["global_relocalization"]),
        "lookahead_base_m": float(drive["lookahead_base_m"]),
        "control_rate_hz": float(drive["control_rate_hz"]),
        "obstacle_stop_margin_m": float(safety["stop_distance_m"]),
        "pose_timeout_s": float(safety["pose_timeout_s"]),
        "deadman_timeout_s": float(safety["deadman_timeout_s"]),
        "estop_topic": str(safety["estop_topic"]),
    }
    if drive["command_type"] == "twist":
        out["drive_topic"] = ""
        out["twist_topic"] = str(drive["command_topic"])
    else:
        out["drive_topic"] = str(drive["command_topic"])
        out["twist_topic"] = ""
    _ = system  # log level / sim time are launch concerns for the drive node
    return out
