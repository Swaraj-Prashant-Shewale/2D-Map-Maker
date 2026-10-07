"""The real project tree: dry-run and staging against the actual checkout.

These are the regression guards for requirement 10 on the living layout:
the real README/guide/inputs/outputs must show up EXCLUDED, no never-list
file may be picked, and the set must stay small (source only, no data).
"""

import re

import pytest

import upload_to_kart
from upload_to_kart import PLACEHOLDER_HOST, main, never_reason

REAL = upload_to_kart.PROJECT_ROOT

pytestmark = pytest.mark.skipif(
    not (REAL / "kart_master.cfg").is_file()
    or not (REAL / "ros2_ws/src/kart_map2d/package.xml").is_file(),
    reason="real project tree not present",
)


@pytest.fixture()
def dry_run_output(capsys):
    rc = main(["--host", PLACEHOLDER_HOST, "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    return out


def _upload_rels(out):
    rels = []
    for line in out.splitlines():
        m = re.match(r"\s+UPLOAD\s+(\S+)", line)
        if m:
            rels.append(m.group(1))
    return rels


def test_real_dry_run_uploads_only_clean_files(dry_run_output):
    rels = _upload_rels(dry_run_output)
    assert rels, "empty upload set?"
    for rel in rels:
        assert never_reason(rel) is None, rel
    assert "ros2_ws/src/kart_map2d/package.xml" in rels
    assert "ros2_ws/src/kart_map2d/setup.py" in rels
    assert "kart_master.cfg" in rels
    assert "requirements/orin-base.txt" in rels
    assert any(rel.startswith("ros2_ws/src/kart_map2d/launch/") for rel in rels)
    assert any(rel.startswith("ros2_ws/src/kart_map2d/kart_map2d/") for rel in rels)


def test_real_dry_run_excludes_the_docs_and_data(dry_run_output):
    out = dry_run_output
    assert re.search(r"EXCLUDED\s+README\.md", out)
    assert re.search(r"EXCLUDED\s+KART_UPLOAD\.txt", out)
    assert re.search(r"EXCLUDED\s+inputs/", out)
    assert "never followed" in out  # the junction is named, not walked
    assert re.search(r"EXCLUDED\s+outputs/", out)
    assert re.search(r"EXCLUDED\s+tools/", out)
    assert re.search(r"EXCLUDED\s+docker/", out)
    assert re.search(r"EXCLUDED\s+\.venv/", out)


def test_real_upload_set_is_small_source_only(dry_run_output):
    m = re.search(r"UPLOAD SET: (\d+) files, ([0-9.]+) MB", dry_run_output)
    assert m, dry_run_output.splitlines()[-3:]
    files, mb = int(m.group(1)), float(m.group(2))
    assert 30 <= files <= 400, "upload set size drifted suspiciously"
    assert mb < 10, "the upload set should be source code, not data"


def test_real_stage_matches_the_manifest(tmp_path, capsys):
    stage_dir = tmp_path / "orin_stage"
    rc = main(["--host", PLACEHOLDER_HOST, "--stage-to", str(stage_dir)])
    out = capsys.readouterr().out
    assert rc == 0
    manifest = len(_upload_rels(out))
    staged = sum(1 for p in stage_dir.rglob("*") if p.is_file())
    assert staged == manifest
    assert (stage_dir / "setup.py").is_file()
    assert (stage_dir / "kart_master.cfg").is_file()
    assert not (stage_dir / "ros2_ws").exists()
    # no never-list name anywhere in the staged tree
    for p in stage_dir.rglob("*"):
        rel = p.relative_to(stage_dir).as_posix()
        assert "__pycache__" not in rel and not rel.endswith((".pyc", ".md", ".html", ".db3", ".zip")), rel
