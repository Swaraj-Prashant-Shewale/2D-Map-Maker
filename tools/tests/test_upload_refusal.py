"""The never-list wins over everything, including edits to the allowlist."""

from pathlib import Path
from types import SimpleNamespace

import pytest

import upload_to_kart
from upload_to_kart import PLACEHOLDER_HOST, UploadRefused, collect, main


def _force_allow(monkeypatch, pattern):
    monkeypatch.setattr(upload_to_kart, "ALLOWLIST",
                        upload_to_kart.ALLOWLIST + [(pattern, "forced edit")])


@pytest.mark.parametrize("pattern,never_hit", [
    ("README.md", "*.md"),
    ("inputs/**", "inputs/**"),
    ("inputs/sept3_20260905_130304/**", "inputs/**"),
    ("outputs/maps_live/**", "outputs/**"),
    ("tools/calibrate_magnetometer.py", "tools/**"),
    ("docker/Dockerfile", "docker/**"),
    ("GUIDE.html", "*.html"),
    ("KART_UPLOAD.txt", "KART_UPLOAD.txt"),
    ("ros2_ws/build/**", "ros2_ws/build/**"),
])
def test_forced_allowlist_edit_is_refused(patched_root, monkeypatch, pattern, never_hit):
    _force_allow(monkeypatch, pattern)
    c = collect(project_root=patched_root)
    assert c.violations, f"{pattern} should be a violation"
    assert c.violations[-1][0] == pattern
    assert c.violations[-1][1] == never_hit
    assert c.files == []  # nothing collected once the edit is spotted


def test_forced_edit_makes_main_exit_2_and_copy_nothing(patched_root, monkeypatch, capsys):
    _force_allow(monkeypatch, "README.md")
    rc = main(["--host", PLACEHOLDER_HOST, "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 2
    assert "REFUSED" in out and "never-list" in out
    assert "UPLOAD SET:" not in out  # no manifest, no copying


def test_forced_edit_refused_even_without_dry_run(patched_root, monkeypatch, capsys):
    _force_allow(monkeypatch, "inputs/**")
    rc = main(["--host", "orin@10.0.0.2"])
    assert rc == 2
    assert "REFUSED" in capsys.readouterr().out


def test_final_screen_is_a_hard_stop(tmp_path):
    """Even if the filters regress, a never-list file in the copy set aborts."""
    with pytest.raises(UploadRefused, match="internal guard"):
        upload_to_kart._final_screen([(tmp_path / "x", "README.md", 3)], neural=False)
    upload_to_kart._final_screen([(tmp_path / "x", "kart_master.cfg", 3)], neural=False)


def test_placeholder_host_is_refused_for_a_real_upload(patched_root, capsys):
    rc = main(["--host", PLACEHOLDER_HOST])
    out = capsys.readouterr().out
    assert rc == 2
    assert "EDIT" in out


def test_host_without_user_is_refused(patched_root, capsys):
    rc = main(["--host", "just-a-name"])
    assert rc == 2


def test_spaced_destination_is_refused(patched_root, capsys):
    rc = main(["--host", "orin@10.0.0.2", "--dest", "kart ws/src/kart_map2d"])
    out = capsys.readouterr().out
    assert rc == 2
    assert "space" in out


def test_invalid_cfg_blocks_the_upload(patched_root, monkeypatch, capsys):
    def bad(path):
        raise UploadRefused("kart_master.cfg is invalid - fix it before uploading")
    monkeypatch.setattr(upload_to_kart, "cfg_wants_neural", bad)
    rc = main(["--host", "orin@10.0.0.2", "--dry-run"])
    assert rc == 2
    assert "invalid" in capsys.readouterr().out


def test_missing_scp_is_a_clear_error(patched_root, monkeypatch, capsys):
    monkeypatch.setattr(upload_to_kart.shutil, "which", lambda name: None)
    rc = main(["--host", "orin@10.0.0.2"])
    out = capsys.readouterr().out
    assert rc == 2
    assert "OpenSSH" in out and "Nothing was uploaded" in out


class _Runner:
    def __init__(self):
        self.calls = []

    def __call__(self, cmd, **kwargs):
        self.calls.append([str(c) for c in cmd])
        return SimpleNamespace(returncode=0, stderr="", stdout="")


def _fake_which(name):
    return name  # pretend ssh/scp/tar all exist


def test_default_upload_never_deletes_on_the_target(patched_root, monkeypatch):
    monkeypatch.setattr(upload_to_kart.shutil, "which", _fake_which)
    runner = _Runner()
    c = collect(project_root=patched_root)
    upload_to_kart.upload(c, "orin@10.0.0.2", "kart_ws/src/kart_map2d",
                          use_tar=False, clean_dest=False, runner=runner)
    joined = [" ".join(call) for call in runner.calls]
    assert not any("rm " in j or j.startswith("rm") for j in joined), joined
    assert any("mkdir -p" in j for j in joined)
    assert any(call[0] == "scp" for call in runner.calls)


def test_clean_dest_deletes_only_with_the_flag_and_first(patched_root, monkeypatch):
    monkeypatch.setattr(upload_to_kart.shutil, "which", _fake_which)
    runner = _Runner()
    c = collect(project_root=patched_root)
    upload_to_kart.upload(c, "orin@10.0.0.2", "kart_ws/src/kart_map2d",
                          use_tar=False, clean_dest=True, runner=runner)
    assert "rm -rf -- kart_ws/src/kart_map2d" in " ".join(runner.calls[0])


@pytest.mark.parametrize("dest", ["", ".", "..", "/", "~", "/home", "../x"])
def test_clean_dest_refuses_unsafe_paths(patched_root, monkeypatch, dest):
    monkeypatch.setattr(upload_to_kart.shutil, "which", _fake_which)
    runner = _Runner()
    c = collect(project_root=patched_root)
    with pytest.raises(UploadRefused):
        upload_to_kart.upload(c, "orin@10.0.0.2", dest,
                              use_tar=False, clean_dest=True, runner=runner)
    assert runner.calls == []  # refused before ANY remote command


def test_tar_mode_sends_one_tarball_and_extracts(patched_root, monkeypatch):
    monkeypatch.setattr(upload_to_kart.shutil, "which", _fake_which)
    runner = _Runner()
    c = collect(project_root=patched_root)
    upload_to_kart.upload(c, "orin@10.0.0.2", "kart_ws/src/kart_map2d",
                          use_tar=True, clean_dest=False, runner=runner)
    joined = [" ".join(call) for call in runner.calls]
    assert any(call[0] == "tar" for call in runner.calls)
    assert any("kart2d_upload.tgz" in j and call[0] == "scp"
               for call, j in zip(runner.calls, joined))
    assert any("tar -xzf" in j for j in joined)


def test_scp_failure_reports_and_refuses(patched_root, monkeypatch):
    monkeypatch.setattr(upload_to_kart.shutil, "which", _fake_which)

    def failing(cmd, **kwargs):
        rc = 1 if str(cmd[0]).endswith("scp") else 0
        return SimpleNamespace(returncode=rc, stderr="boom", stdout="")

    c = collect(project_root=patched_root)
    with pytest.raises(UploadRefused, match="scp"):
        upload_to_kart.upload(c, "orin@10.0.0.2", "kart_ws/src/kart_map2d",
                              use_tar=False, clean_dest=False, runner=failing)
