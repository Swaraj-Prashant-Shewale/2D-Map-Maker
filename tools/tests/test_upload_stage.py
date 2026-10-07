"""Staging: the exact tree that lands on the Orin (used by the container proof)."""

import pytest

import upload_to_kart
from upload_to_kart import PLACEHOLDER_HOST, UploadRefused, collect, main, stage, staged_rel


def test_package_files_are_rerooted_to_the_dest(patched_root, tmp_path):
    c = collect(project_root=patched_root)
    stage_dir = tmp_path / "orin"
    staged = stage(c, stage_dir)
    assert len(staged) == len(c.files)
    # <dest>/ IS the package dir, ready for colcon
    assert (stage_dir / "setup.py").is_file()
    assert (stage_dir / "package.xml").is_file()
    assert (stage_dir / "kart_map2d/ros2/node.py").is_file()
    assert (stage_dir / "launch/mapping.launch.py").is_file()
    assert (stage_dir / "test/test_x.py").is_file()
    # non-package extras keep their project-relative place under <dest>/
    assert (stage_dir / "kart_master.cfg").is_file()
    assert (stage_dir / "requirements/orin-base.txt").is_file()
    assert (stage_dir / "config/mag_calibration.yaml").is_file()
    # and the Windows workspace nesting is gone
    assert not (stage_dir / "ros2_ws").exists()


def test_staged_tree_holds_nothing_else(patched_root, tmp_path):
    c = collect(project_root=patched_root)
    stage_dir = tmp_path / "orin"
    stage(c, stage_dir)
    on_disk = sorted(p.relative_to(stage_dir).as_posix()
                     for p in stage_dir.rglob("*") if p.is_file())
    expected = sorted(staged_rel(rel) for _, rel, _ in c.files)
    assert on_disk == expected


def test_staged_rel_mapping():
    assert staged_rel("ros2_ws/src/kart_map2d/setup.py") == "setup.py"
    assert staged_rel("kart_master.cfg") == "kart_master.cfg"
    assert staged_rel("requirements/orin-base.txt") == "requirements/orin-base.txt"


def test_stage_refuses_a_non_empty_dir(patched_root, tmp_path):
    c = collect(project_root=patched_root)
    stage_dir = tmp_path / "orin"
    stage_dir.mkdir()
    (stage_dir / "stale.txt").write_text("old")
    with pytest.raises(UploadRefused, match="not empty"):
        stage(c, stage_dir)


def test_stage_detects_path_collisions(tmp_path):
    src = tmp_path / "a.txt"
    src.write_text("x")
    c = upload_to_kart.Collection(files=[
        (src, "ros2_ws/src/kart_map2d/foo.txt", 1),
        (src, "foo.txt", 1),
    ])
    with pytest.raises(UploadRefused, match="collision"):
        stage(c, tmp_path / "orin")


def test_main_stage_to_stages_and_skips_ssh(patched_root, tmp_path, capsys):
    stage_dir = tmp_path / "orin_stage"
    rc = main(["--host", PLACEHOLDER_HOST, "--stage-to", str(stage_dir)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "no ssh contact" in out
    assert (stage_dir / "setup.py").is_file()


def test_dry_run_prints_sizes_totals_and_copies_nothing(patched_root, tmp_path, capsys):
    rc = main(["--host", PLACEHOLDER_HOST, "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "dry run: nothing copied" in out
    assert "UPLOAD SET:" in out
    assert "MB" in out and "bytes" in out
    # every upload line carries a size
    upload_lines = [l for l in out.splitlines() if l.strip().startswith("UPLOAD ")]
    assert upload_lines and all("(" in l and "B" in l for l in upload_lines)
    # and the re-rooted target is shown for package files
    assert any("->" in l for l in upload_lines)
