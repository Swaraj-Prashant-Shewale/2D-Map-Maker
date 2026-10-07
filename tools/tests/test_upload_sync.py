"""KART_UPLOAD.txt and tools/upload_to_kart.py must state the SAME lists.

KART_UPLOAD.txt is the human contract, the script holds the machine copy;
this test parses the "+", "~" and "-" pattern lines out of the text file
and fails the moment either side drifts. It also pins the plain-English
promises the contract must keep making (the privacy reason, the docs-stay-
on-the-laptop rule, the dry-run/upload instructions, the no-delete rule).
"""

import re
from pathlib import Path

import upload_to_kart

CONTRACT = Path(upload_to_kart.PROJECT_ROOT) / "KART_UPLOAD.txt"
_LINE = re.compile(r"^  ([+~-]) (\S+)\s*$")


def parse_contract():
    got = {"+": [], "~": [], "-": []}
    for line in CONTRACT.read_text(encoding="utf-8").splitlines():
        m = _LINE.match(line)
        if m:
            got[m.group(1)].append(m.group(2))
    return got


def test_contract_file_exists():
    assert CONTRACT.is_file()


def test_allowlist_matches_contract():
    got = parse_contract()
    assert got["+"] == [p for p, _ in upload_to_kart.ALLOWLIST]


def test_neural_list_matches_contract():
    got = parse_contract()
    assert got["~"] == [p for p, _ in upload_to_kart.NEURAL_ALLOWLIST]


def test_neverlist_matches_contract():
    got = parse_contract()
    assert got["-"] == [p for p, _ in upload_to_kart.NEVERLIST]


def test_no_list_is_trivially_empty():
    got = parse_contract()
    assert len(got["+"]) >= 10
    assert len(got["~"]) >= 1
    assert len(got["-"]) >= 15


def test_junk_patterns_are_a_subset_of_the_neverlist():
    never = {p for p, _ in upload_to_kart.NEVERLIST}
    assert set(upload_to_kart.JUNK_PATTERNS) <= never


def test_plain_english_promises_are_still_made():
    text = CONTRACT.read_text(encoding="utf-8")
    # privacy reason for the recordings
    assert "PRIVACY" in text and "identifiable people" in text
    # explanation files stay on the laptop
    assert "stay on the laptop" in text.lower() or "stays on the laptop" in text.lower()
    # how to run it
    assert "--dry-run" in text
    assert "tools/upload_to_kart.py" in text
    assert "scp" in text
    # never deletes without the flag
    assert "--clean-dest" in text and "never deletes" in text
    # space-free target
    assert "SPACE-FREE" in text


def test_every_pattern_has_a_reason_in_the_script():
    for pattern, why in (upload_to_kart.ALLOWLIST
                         + upload_to_kart.NEURAL_ALLOWLIST
                         + upload_to_kart.NEVERLIST):
        assert why.strip(), f"pattern {pattern!r} has no reason text"
