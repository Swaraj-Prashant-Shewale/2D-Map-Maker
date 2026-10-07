"""kart_master.cfg parser/validator: every rule of the requirement.

- typo -> did-you-mean with the line number
- out-of-range -> error naming the allowed range
- all [sensors] zero -> error
- IMU-only -> dead-reckoning warning
- not-implemented sensor (gnss) = 1 -> error
- UTF-8 (with/without BOM) + UTF-16 BOM + CRLF + trailing # comments
- launch/CLI overrides flip [sensors] switches
- shipped defaults: imu = 1, camera = 1, everything else 0
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kart_map2d.config.master_config import (
    SENSOR_KEYS,
    MasterConfigError,
    combo_name,
    load_master_config,
    parse_master_text,
    to_drive_params_dict,
    to_mapping_params_dict,
)

def _find_shipped_cfg() -> Path:
    """Walk up from this file until kart_master.cfg is found.

    Works on the laptop layout (<project>/ros2_ws/src/kart_map2d/test, cfg at
    <project>/kart_master.cfg) AND the kart layout (~/kart_ws/src/kart_map2d/test,
    cfg staged at the package top by tools/upload_to_kart.py).
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "kart_master.cfg"
        if candidate.exists():
            return candidate
    # Fall back to the laptop-layout path so the assertion message stays useful.
    return here.parents[4] / "kart_master.cfg"


SHIPPED_CFG = _find_shipped_cfg()
PROJECT_ROOT = SHIPPED_CFG.parent

MINIMAL = """
[sensors]
imu = 1
camera = 1

[camera]
image_topic = /cam/image
camera_info_topic = /cam/info
"""


def write_cfg(tmp_path: Path, text: str, encoding: str = "utf-8", name: str = "kart_master.cfg") -> Path:
    path = tmp_path / name
    path.write_bytes(text.encode(encoding))
    return path


# ------------------------------------------------------------ shipped file
def test_shipped_cfg_is_valid_with_camera_plus_imu_defaults():
    """GATE: the checked-in kart_master.cfg must parse, and its [sensors]
    defaults are the user's decision: imu + camera ON, everything else OFF."""
    assert SHIPPED_CFG.exists(), "kart_master.cfg must live at the project root"
    cfg = load_master_config(SHIPPED_CFG)
    assert cfg.sensors_on() == ["imu", "camera"]
    assert cfg["sensors"]["livox_imu"] == 0  # user choice: off by default
    assert cfg["sensors"]["lidar"] == 0  # experiment switch, off
    assert cfg["segmentation"]["mode"] == "classical"  # classical by default
    assert cfg.warnings == []
    assert combo_name(cfg) == "imu+cam"


def test_shipped_cfg_translates_to_node_params():
    from kart_map2d.ros2.node import NodeParams, camera_config_problems

    cfg = load_master_config(SHIPPED_CFG)
    params = NodeParams.from_dict(to_mapping_params_dict(cfg))
    assert list(params.cameras) == ["camera"]  # SINGLE camera by default
    assert params.imu_topic == "/imu/data"
    assert params.livox_imu_topic == "" and params.lidar_topic == "" and params.odom_topic == ""
    assert params.mag_topic == "" and params.speed_topic == "" and params.wheel_topic == ""
    assert params.seg_mode == "classical"
    assert camera_config_problems(params.cameras) == []


# ---------------------------------------------------------------- typos
def test_unknown_key_gets_did_you_mean_and_line_number(tmp_path):
    path = write_cfg(tmp_path, MINIMAL + "\n[mapping]\ncell_sizes_m = 0.2\n")
    with pytest.raises(MasterConfigError) as err:
        load_master_config(path)
    text = "\n".join(err.value.problems)
    assert "cell_sizes_m" in text and "did you mean 'cell_size_m'" in text
    assert "line 11" in text


def test_unknown_section_gets_did_you_mean(tmp_path):
    path = write_cfg(tmp_path, MINIMAL + "\n[mappings]\ncell_size_m = 0.2\n")
    with pytest.raises(MasterConfigError) as err:
        load_master_config(path)
    text = "\n".join(err.value.problems)
    assert "[mappings]" in text and "did you mean [mapping]" in text


def test_duplicate_key_is_an_error_with_both_line_numbers(tmp_path):
    path = write_cfg(tmp_path, MINIMAL + "\n[mapping]\ncell_size_m = 0.2\ncell_size_m = 0.3\n")
    with pytest.raises(MasterConfigError) as err:
        load_master_config(path)
    assert "already set on line" in "\n".join(err.value.problems)


# ---------------------------------------------------------------- ranges
def test_out_of_range_error_names_the_allowed_range(tmp_path):
    path = write_cfg(tmp_path, MINIMAL + "\n[mapping]\ncell_size_m = 5.0\n")
    with pytest.raises(MasterConfigError) as err:
        load_master_config(path)
    text = "\n".join(err.value.problems)
    assert "cell_size_m" in text and "0.02 .. 1" in text and "m" in text


def test_switch_values_other_than_0_1_are_refused(tmp_path):
    path = write_cfg(tmp_path, "[sensors]\nimu = true\ncamera = 1\n[camera]\nimage_topic = /a\ncamera_info_topic = /b\n")
    with pytest.raises(MasterConfigError) as err:
        load_master_config(path)
    assert "must be 0 or 1" in "\n".join(err.value.problems)


def test_bad_choice_lists_the_choices(tmp_path):
    path = write_cfg(tmp_path, MINIMAL + "\n[segmentation]\nmode = magic\n")
    with pytest.raises(MasterConfigError) as err:
        load_master_config(path)
    assert "classical | neural | auto" in "\n".join(err.value.problems)


# ------------------------------------------------------- sensor combinations
def test_all_sensors_zero_is_refused(tmp_path):
    lines = "[sensors]\n" + "\n".join(f"{k} = 0" for k in SENSOR_KEYS)
    path = write_cfg(tmp_path, lines)
    with pytest.raises(MasterConfigError) as err:
        load_master_config(path)
    assert "every switch is 0" in "\n".join(err.value.problems)


def test_imu_only_warns_dead_reckoning(tmp_path):
    path = write_cfg(tmp_path, "[sensors]\nimu = 1\ncamera = 0\n")
    cfg = load_master_config(path)
    assert any("DEAD RECKONING" in w for w in cfg.warnings)


def test_gnss_not_implemented_is_refused(tmp_path):
    path = write_cfg(tmp_path, MINIMAL.replace("[sensors]", "[sensors]\ngnss = 1"))
    with pytest.raises(MasterConfigError) as err:
        load_master_config(path)
    text = "\n".join(err.value.problems)
    assert "gnss" in text and "NOT implemented" in text


def test_magnetometer_without_calibration_prints_procedure(tmp_path):
    path = write_cfg(tmp_path, MINIMAL.replace("[sensors]", "[sensors]\nmagnetometer = 1"))
    with pytest.raises(MasterConfigError) as err:
        load_master_config(path)
    text = "\n".join(err.value.problems)
    assert "calibration_file" in text
    assert "360" in text and "calibrate_magnetometer.py" in text  # the exact procedure


def test_magnetometer_with_missing_file_is_refused(tmp_path):
    text = MINIMAL.replace("[sensors]", "[sensors]\nmagnetometer = 1") + "\n[magnetometer]\ncalibration_file = nope.yaml\n"
    with pytest.raises(MasterConfigError) as err:
        load_master_config(write_cfg(tmp_path, text))
    assert "does not exist" in "\n".join(err.value.problems)


# ------------------------------------------------------------ camera rules
def test_ros2_camera_needs_info_or_calibration(tmp_path):
    path = write_cfg(tmp_path, "[sensors]\nimu = 1\ncamera = 1\n[camera]\nimage_topic = /cam/image\n")
    with pytest.raises(MasterConfigError) as err:
        load_master_config(path)
    assert "camera_info_topic" in "\n".join(err.value.problems)


def test_device_camera_needs_calibration_file(tmp_path):
    path = write_cfg(tmp_path, "[sensors]\nimu = 1\ncamera = 1\n[camera]\nsource = device\ndevice = 0\n")
    with pytest.raises(MasterConfigError) as err:
        load_master_config(path)
    assert "calibration_file" in "\n".join(err.value.problems)


def test_gstreamer_camera_needs_pipeline_and_calibration(tmp_path):
    path = write_cfg(tmp_path, "[sensors]\nimu = 1\ncamera = 1\n[camera]\nsource = gstreamer\n")
    with pytest.raises(MasterConfigError) as err:
        load_master_config(path)
    text = "\n".join(err.value.problems)
    assert "gstreamer_pipeline" in text and "calibration_file" in text


# ------------------------------------------------------------- encodings
def test_utf8_bom_utf16_crlf_and_trailing_comments(tmp_path):
    body = (
        "[sensors]\r\nimu = 1  # the dedicated IMU\r\ncamera = 1\r\n"
        "[camera]\r\nimage_topic = /cam/image\r\ncamera_info_topic = /cam/info\r\n"
        "[mapping]\r\ncell_size_m = 0.2 # metres\r\n"
    )
    for encoding in ("utf-8", "utf-8-sig", "utf-16"):
        cfg = load_master_config(write_cfg(tmp_path, body, encoding, name=f"cfg_{encoding}.cfg"))
        assert cfg["mapping"]["cell_size_m"] == 0.2
        assert cfg.sensors_on() == ["imu", "camera"]


def test_topic_values_keep_inline_hash_without_space(tmp_path):
    cfg = parse_master_text("[camera]\nimage_topic = /cam#0/image\n")
    assert cfg["camera"]["image_topic"] == "/cam#0/image"


# ------------------------------------------------------------- overrides
def test_overrides_flip_switches_like_launch_args(tmp_path):
    path = write_cfg(tmp_path, MINIMAL + "\n[magnetometer]\ncalibration_file =\n")
    cfg = load_master_config(path, overrides={"lidar": 1, "camera": 0})
    assert cfg["sensors"]["lidar"] == 1 and cfg["sensors"]["camera"] == 0
    assert combo_name(cfg) == "imu+lidar"


def test_override_of_unknown_switch_gets_hint(tmp_path):
    path = write_cfg(tmp_path, MINIMAL)
    with pytest.raises(MasterConfigError) as err:
        load_master_config(path, overrides={"lidr": 1})
    assert "did you mean 'lidar'" in "\n".join(err.value.problems)


def test_overrides_are_cross_validated_too(tmp_path):
    path = write_cfg(tmp_path, MINIMAL)
    with pytest.raises(MasterConfigError):
        load_master_config(path, overrides={"gnss": 1})


# ------------------------------------------------------------ translation
def test_combo_names_are_stable_and_order_canonical(tmp_path):
    text = MINIMAL.replace("[sensors]", "[sensors]\nwheel_ticks = 1\nlidar = 1")
    cfg = load_master_config(write_cfg(tmp_path, text))
    assert combo_name(cfg) == "imu+cam+lidar+wheel"  # canonical order, not file order


def test_sensor_switch_gates_topics_in_translation(tmp_path):
    cfg = load_master_config(write_cfg(tmp_path, MINIMAL))
    m = to_mapping_params_dict(cfg)
    assert m["lidar_topic"] == "" and m["odom_topic"] == "" and m["livox_imu_topic"] == ""
    cfg2 = load_master_config(write_cfg(tmp_path, MINIMAL, name="c2.cfg"), overrides={"lidar": 1, "odom": 1})
    m2 = to_mapping_params_dict(cfg2)
    assert m2["lidar_topic"] == "/livox/lidar" and m2["odom_topic"] == "/odom"


def test_camera_2_gets_distinct_name_on_collision(tmp_path):
    text = (
        "[sensors]\nimu = 1\ncamera = 1\ncamera_2 = 1\n"
        "[camera]\nname = cam\nimage_topic = /a/image\ncamera_info_topic = /a/info\n"
        "[camera_2]\nname = cam\nimage_topic = /b/image\ncamera_info_topic = /b/info\n"
    )
    cfg = load_master_config(write_cfg(tmp_path, text))
    m = to_mapping_params_dict(cfg)
    assert set(m["cameras"]) == {"cam", "cam_2"}


def test_drive_translation_maps_safety_and_command_type(tmp_path):
    text = MINIMAL + "\n[drive]\ncommand_type = twist\ncommand_topic = /cmd_vel\nmax_speed_mps = 1.5\n[safety]\nstop_distance_m = 0.8\n"
    cfg = load_master_config(write_cfg(tmp_path, text))
    d = to_drive_params_dict(cfg)
    assert d["twist_topic"] == "/cmd_vel" and d["drive_topic"] == ""
    assert d["max_speed_mps"] == 1.5 and d["obstacle_stop_margin_m"] == 0.8


def test_drive_global_relocalization_default_and_override(tmp_path):
    cfg = load_master_config(write_cfg(tmp_path, MINIMAL))
    assert to_drive_params_dict(cfg)["global_relocalization"] is True  # shipped default
    text = MINIMAL + "\n[drive]\nglobal_relocalization = 0\n"
    cfg = load_master_config(write_cfg(tmp_path, text))
    assert to_drive_params_dict(cfg)["global_relocalization"] is False


def test_drive_translation_refuses_device_cameras(tmp_path):
    text = (
        "[sensors]\nimu = 1\ncamera = 1\n"
        "[camera]\nsource = device\ndevice = 0\ncalibration_file = cal.yaml\n"
    )
    (tmp_path / "cal.yaml").write_text(
        "image_width: 64\nimage_height: 48\ncamera_matrix: [50, 0, 32, 0, 50, 24, 0, 0, 1]\n"
        "distortion_model: none\ndistortion_coefficients: []\n",
        encoding="ascii",
    )
    cfg = load_master_config(write_cfg(tmp_path, text))
    with pytest.raises(MasterConfigError, match="ROS topics only"):
        to_drive_params_dict(cfg)


def test_missing_file_is_a_clear_error(tmp_path):
    with pytest.raises(MasterConfigError, match="does not exist"):
        load_master_config(tmp_path / "nope.cfg")


def test_lidar_mount_is_authoritative_and_cross_checked(tmp_path):
    """One physical Livox unit: [lidar] xyz drives the shared mount; enabling
    BOTH lidar and livox_imu with disagreeing mounts is refused."""
    text = MINIMAL + "\n[lidar]\nxyz = 0, 0, 0.6\nrpy_deg = 0, 0, -90\n"
    cfg = load_master_config(write_cfg(tmp_path, text), overrides={"lidar": 1})
    m = to_mapping_params_dict(cfg)
    assert m["livox_xyz"] == (0.0, 0.0, 0.6)
    both = text + "\n[imu]\nlivox_xyz = 0, 0, 0.52\n"
    with pytest.raises(MasterConfigError, match="same physical"):
        load_master_config(write_cfg(tmp_path, both, name="both.cfg"), overrides={"lidar": 1, "livox_imu": 1})
