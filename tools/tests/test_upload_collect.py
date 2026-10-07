"""Collection semantics on a throwaway fake project tree (see conftest)."""

from conftest import DEFAULT_EXPECTED, FAKE_TREE

import upload_to_kart
from upload_to_kart import collect


def _rels(collection):
    return [rel for _, rel, _ in collection.files]


def test_default_set_is_exactly_the_allowlist(patched_root):
    c = collect(project_root=patched_root)
    assert _rels(c) == DEFAULT_EXPECTED
    assert not c.violations


def test_sizes_and_total_are_reported(patched_root):
    c = collect(project_root=patched_root)
    assert all(size == 10 for _, _, size in c.files)
    assert c.total_bytes == 10 * len(DEFAULT_EXPECTED)


def test_nothing_picked_matches_the_neverlist(patched_root):
    c = collect(project_root=patched_root)
    for rel in _rels(c):
        assert upload_to_kart.never_reason(rel) is None, rel


def test_seeded_doc_inside_the_package_is_excluded_not_uploaded(patched_root):
    c = collect(project_root=patched_root)
    assert "ros2_ws/src/kart_map2d/kart_map2d/LEFTOVER_NOTES.md" not in _rels(c)
    excluded = {rel: pattern for rel, pattern, _ in c.excluded}
    assert excluded.get("ros2_ws/src/kart_map2d/kart_map2d/LEFTOVER_NOTES.md") == "*.md"


def test_seeded_guide_html_readme_outputs_inputs_are_excluded(patched_root):
    c = collect(project_root=patched_root)
    excluded = {rel: pattern for rel, pattern, _ in c.excluded}
    assert excluded.get("GUIDE_2D_MAPPING.html") == "*.html"
    assert excluded.get("README.md") == "*.md"
    assert excluded.get("KART_UPLOAD.txt") == "KART_UPLOAD.txt"
    assert excluded.get("outputs/") == "outputs/**"
    assert excluded.get("inputs/") == "inputs/**"
    assert excluded.get("tools/") == "tools/**"
    assert excluded.get("docker/") == "docker/**"
    assert excluded.get(".venv/") == ".venv/**"


def test_recordings_are_never_walked(patched_root):
    """No file under inputs/ may appear ANYWHERE - picked or excluded.

    The real inputs/ is a junction onto 37 GB of recordings; the collector
    must list the top-level entry once and never descend."""
    c = collect(project_root=patched_root)
    everywhere = _rels(c) + [rel for rel, _, _ in c.excluded]
    deep = [rel for rel in everywhere if rel.replace("\\", "/").startswith("inputs/") and rel != "inputs/"]
    assert deep == []


def test_junction_is_pruned_even_when_allowlisted(patched_root, monkeypatch, tmp_path):
    """A link/junction inside an allowed folder is never followed."""
    import os
    target = tmp_path / "external"
    target.mkdir()
    (target / "secret.bin").write_bytes(b"x")
    link = patched_root / "ros2_ws/src/kart_map2d/kart_map2d/linked"
    try:
        os.symlink(target, link, target_is_directory=True)
    except OSError:
        import subprocess
        r = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                           capture_output=True)
        if r.returncode != 0:
            import pytest
            pytest.skip("cannot create symlink or junction on this machine")
    c = collect(project_root=patched_root)
    assert not any("linked" in rel for rel in _rels(c))


def test_no_tests_drops_the_test_folder(patched_root):
    c = collect(project_root=patched_root, with_tests=False)
    assert not any(rel.startswith("ros2_ws/src/kart_map2d/test/") for rel in _rels(c))
    rest = [r for r in DEFAULT_EXPECTED if not r.startswith("ros2_ws/src/kart_map2d/test/")]
    assert _rels(c) == rest


def test_neural_adds_exactly_the_weights(patched_root):
    classical = _rels(collect(project_root=patched_root))
    neural = _rels(collect(project_root=patched_root, neural=True))
    assert set(neural) - set(classical) == {"outputs/cache/models/segformer.bin"}


def test_neural_carveout_does_not_unlock_the_rest_of_outputs(patched_root):
    c = collect(project_root=patched_root, neural=True)
    assert "outputs/run1__imu+cam/grid.npz" not in _rels(c)


def test_neural_carveout_still_blocks_docs_inside_models(patched_root):
    (patched_root / "outputs/cache/models/NOTES.md").write_text("x")
    c = collect(project_root=patched_root, neural=True)
    assert "outputs/cache/models/NOTES.md" not in _rels(c)


def test_pycache_junk_is_pruned_and_counted(patched_root):
    c = collect(project_root=patched_root)
    rels = _rels(c)
    assert not any("__pycache__" in rel or rel.endswith(".pyc") for rel in rels)
    assert c.junk >= 1  # the stray .pyc inside kart_map2d/


def test_fake_tree_covers_every_neverlist_family(patched_root):
    """Sanity of the fixture itself: each never pattern family is exercised."""
    tested = set(FAKE_TREE)
    assert any(r.startswith("inputs/") for r in tested)
    assert any(r.startswith("outputs/") for r in tested)
    assert any(r.endswith(".md") for r in tested)
    assert any(r.endswith(".html") for r in tested)
    assert any(r.endswith(".zip") for r in tested)
    assert any(r.endswith(".db3") for r in tested)
    assert any("__pycache__" in r for r in tested)
