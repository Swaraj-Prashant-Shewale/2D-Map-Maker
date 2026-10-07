"""pytest bootstrap + shared fixtures for the upload-tool tests.

Puts tools/ on sys.path so ``import upload_to_kart`` works (which in turn
imports tools/_bootstrap.py to reach the kart_map2d package), and builds a
throwaway fake project tree so the collection tests never touch - let alone
seed files into - the real package source.
"""

import sys
from pathlib import Path

import pytest

TOOLS_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = TOOLS_DIR.parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import upload_to_kart  # noqa: E402


#: every file of the fake tree, rel-posix -> should it be in the DEFAULT
#: (classical, with tests) upload set?
FAKE_TREE = {
    # allowlisted package source
    "ros2_ws/src/kart_map2d/package.xml": True,
    "ros2_ws/src/kart_map2d/setup.py": True,
    "ros2_ws/src/kart_map2d/setup.cfg": True,
    "ros2_ws/src/kart_map2d/conftest.py": True,
    "ros2_ws/src/kart_map2d/resource/kart_map2d": True,
    "ros2_ws/src/kart_map2d/kart_map2d/__init__.py": True,
    "ros2_ws/src/kart_map2d/kart_map2d/ros2/node.py": True,
    "ros2_ws/src/kart_map2d/kart_map2d/ros2/fastdds_large_shm.xml": True,
    "ros2_ws/src/kart_map2d/launch/mapping.launch.py": True,
    "ros2_ws/src/kart_map2d/config/mapping_params.example.yaml": True,
    "ros2_ws/src/kart_map2d/test/test_x.py": True,
    "kart_master.cfg": True,
    "requirements/orin-base.txt": True,
    "requirements/orin-neural.txt": True,
    "config/mag_calibration.yaml": True,  # root site-config, allowlisted
    # never / not allowlisted
    "requirements/laptop.txt": False,
    "README.md": False,
    "GUIDE_2D_MAPPING.html": False,
    "KART_UPLOAD.txt": False,
    ".gitignore": False,
    "docs/design.md": False,
    "tools/upload_to_kart.py": False,
    "tools/replay_run.py": False,
    ".venv/Scripts/python.exe": False,
    "docker/Dockerfile": False,
    "ros2_ws/build/kart_map2d/x": False,
    "ros2_ws/install/setup.bash": False,
    "ros2_ws/log/latest/x": False,
    "outputs/run1__imu+cam/grid.npz": False,
    "outputs/cache/models/segformer.bin": False,  # only with neural
    "sept_data.zip": False,
    # never-list junk INSIDE allowed folders
    "ros2_ws/src/kart_map2d/test/__pycache__/test_x.cpython-310.pyc": False,
    "ros2_ws/src/kart_map2d/kart_map2d/stray.pyc": False,
    # seeded doc INSIDE an allowed folder: excluded, never uploaded
    "ros2_ws/src/kart_map2d/kart_map2d/LEFTOVER_NOTES.md": False,
    # recordings behind the junction stand-in: never traversed
    "inputs/sept3_20260905_130304/sept3_0.db3": False,
    "inputs/sept3_20260905_130304/metadata.yaml": False,
}

DEFAULT_EXPECTED = sorted(rel for rel, ship in FAKE_TREE.items() if ship)


@pytest.fixture()
def fake_root(tmp_path):
    for rel in FAKE_TREE:
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x" * 10)
    return tmp_path


@pytest.fixture()
def patched_root(fake_root, monkeypatch):
    """Point the tool at the fake tree and neutralize the real cfg read."""
    monkeypatch.setattr(upload_to_kart, "PROJECT_ROOT", fake_root)
    monkeypatch.setattr(upload_to_kart, "cfg_wants_neural", lambda path: False)
    return fake_root
