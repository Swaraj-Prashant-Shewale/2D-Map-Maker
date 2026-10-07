"""Upload the kart_map2d package to the Orin - allowlist only, scp transport.

    .venv/Scripts/python.exe tools/upload_to_kart.py --host orin@EDIT-ME-ORIN-IP --dry-run
    .venv/Scripts/python.exe tools/upload_to_kart.py --host orin@192.168.55.1
    .venv/Scripts/python.exe tools/upload_to_kart.py --host orin@192.168.55.1 --tar
    .venv/Scripts/python.exe tools/upload_to_kart.py --host orin@EDIT-ME-ORIN-IP --stage-to C:/tmp/orin_stage

Implements KART_UPLOAD.txt exactly. The ALLOWLIST / NEURAL_ALLOWLIST /
NEVERLIST pattern strings below are the machine-readable copy of that file's
"+", "~" and "-" entries; tools/tests/test_upload_sync.py fails if the two
ever disagree.

Guarantees (each one is tested):
- ONLY allowlisted files are copied; everything else is excluded.
- NEVER-list entries are refused: an allowlist EDIT that names a never-list
  path (e.g. someone adds "README.md" or "inputs/**") aborts the run with
  exit 2 before anything is copied, and a final invariant re-screens the
  copy set so a bug in the filter can never ship a never-list file.
- The inputs/ junction (37 GB of recordings with identifiable people) is
  never traversed, let alone copied.
- --dry-run prints every file that would be copied, with sizes and totals,
  and copies nothing.
- Neural SegFormer weights go up ONLY with --neural (or when kart_master.cfg
  itself says [segmentation] mode = neural, so the kart never boots neural
  without its weights).
- The remote destination must be SPACE-FREE (colcon breaks on spaces).
- Transport is Windows' built-in OpenSSH scp (missing scp/ssh is a clear
  error, not a stack trace); --tar sends one tarball instead of many small
  files. NOTHING on the target is ever deleted unless --clean-dest is given.
"""

import _bootstrap  # noqa: F401  (makes kart_map2d importable from tools/)

import argparse
import fnmatch
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from kart_map2d.config.master_config import MasterConfigError, load_master_config

PROJECT_ROOT = _bootstrap.PROJECT_ROOT

#: The package source prefix; staged/uploaded files under it are re-rooted so
#: that <dest>/ IS the package dir (setup.py at the top, ready for colcon).
PKG_PREFIX = "ros2_ws/src/kart_map2d/"

PLACEHOLDER_HOST = "orin@EDIT-ME-ORIN-IP"  # EDIT: your Orin's user@address
DEFAULT_DEST = "kart_ws/src/kart_map2d"    # relative to the Orin $HOME

# --------------------------------------------------------------------------
# The three lists. Pattern language (same as KART_UPLOAD.txt):
#   - a pattern WITH "/" is project-root-relative; "**" spans directories,
#     "*" stops at "/".
#   - a pattern WITHOUT "/" matches a file or directory NAME anywhere.
# Keep these in sync with KART_UPLOAD.txt - test_upload_sync.py enforces it.
# --------------------------------------------------------------------------

#: (pattern, why). What goes on the kart.
ALLOWLIST: list[tuple[str, str]] = [
    ("ros2_ws/src/kart_map2d/kart_map2d/**", "the package source - everything both nodes run"),
    ("ros2_ws/src/kart_map2d/launch/**", "mapping/drive/replay launch files"),
    ("ros2_ws/src/kart_map2d/config/**", "example params files"),
    ("ros2_ws/src/kart_map2d/package.xml", "colcon build needs it"),
    ("ros2_ws/src/kart_map2d/setup.py", "colcon build needs it"),
    ("ros2_ws/src/kart_map2d/setup.cfg", "colcon build needs it"),
    ("ros2_ws/src/kart_map2d/resource/**", "ament resource marker"),
    ("ros2_ws/src/kart_map2d/conftest.py", "pytest bootstrap for the on-device tests"),
    ("ros2_ws/src/kart_map2d/test/**", "on-device verification tests (skip with --no-tests)"),
    ("kart_master.cfg", "THE control file - the kart team edits this on the kart"),
    ("requirements/orin-base.txt", "pip install list for the Orin"),
    ("requirements/orin-neural.txt", "pip list consulted only if you switch to neural"),
    ("config/**", "site calibration files the cfg points at (mag/camera YAML, mask PNGs)"),
]

#: (pattern, why). Added ONLY with --neural (or cfg [segmentation] mode = neural).
NEURAL_ALLOWLIST: list[tuple[str, str]] = [
    ("outputs/cache/models/**", "SegFormer weights - only when segmentation mode = neural"),
]

#: (pattern, why). Refused, always - even when an allowlist edit names them.
NEVERLIST: list[tuple[str, str]] = [
    ("inputs/**", "PRIVACY: 37 GB of track recordings with identifiable people; never leaves the laptop"),
    ("outputs/**", "laptop test maps/caches/replay artifacts; a stale test map on the kart is a driving hazard"),
    ("results/**", "validation outputs (matrix runs, drive-sim traces, demo video with identifiable people); laptop only"),
    ("*.md", "explanation files (README, docs, reports) stay on the laptop"),
    ("*.html", "the HTML guide stays on the laptop"),
    ("docs/**", "documentation stays on the laptop"),
    ("KART_UPLOAD.txt", "this contract is laptop documentation"),
    ("tools/**", "laptop-side tooling (this uploader, replay runner, exporters) - runs on Windows"),
    (".venv/**", "Windows virtualenv - useless and huge on aarch64"),
    ("docker/**", "laptop test container; the Orin runs bare JetPack"),
    (".git/**", "never ship history"),
    (".gitignore", "git plumbing"),
    ("ros2_ws/build/**", "colcon artifacts are rebuilt ON the kart"),
    ("ros2_ws/install/**", "colcon artifacts are rebuilt ON the kart"),
    ("ros2_ws/log/**", "colcon artifacts are rebuilt ON the kart"),
    ("__pycache__", "bytecode is per-machine"),
    ("*.pyc", "bytecode is per-machine"),
    (".pytest_cache", "pytest cache"),
    ("*.zip", "no archives, no research-workspace snapshots"),
    ("*.db3", "bag data never ships"),
    ("*.mcap", "bag data never ships"),
]

#: never-list entries that are routine build litter INSIDE allowed folders:
#: pruned silently during the walk (a count is reported), everything else on
#: the never-list gets its own EXCLUDED line.
JUNK_PATTERNS = ("__pycache__", "*.pyc", ".pytest_cache")

#: prefix that the neural allowlist is allowed to carve out of outputs/**.
_NEURAL_PREFIX = "outputs/cache/models/"


class UploadRefused(Exception):
    """A never-list violation (or unusable arguments): nothing may be copied."""


# --------------------------------- matching -------------------------------

def _anchored_regex(pattern: str) -> re.Pattern:
    out, i = [], 0
    while i < len(pattern):
        if pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return re.compile("^" + "".join(out) + "$")


def matches(rel: str, pattern: str) -> bool:
    """KART_UPLOAD.txt pattern semantics (see module docstring)."""
    rel = rel.replace("\\", "/")
    if "/" in pattern:
        return _anchored_regex(pattern).match(rel) is not None
    return any(fnmatch.fnmatchcase(part, pattern) for part in rel.split("/"))


def never_reason(rel: str, *, neural: bool = False) -> tuple[str, str] | None:
    """(pattern, why) of the never-list entry hitting ``rel``, else None."""
    rel = rel.replace("\\", "/")
    carve_out = neural and rel.startswith(_NEURAL_PREFIX)
    for pattern, why in NEVERLIST:
        if carve_out and pattern == "outputs/**":
            continue  # the one sanctioned carve-out: the weights zone only
        if matches(rel, pattern):
            return (pattern, why)
    return None


def _is_junk(rel: str) -> bool:
    return any(matches(rel, p) for p in JUNK_PATTERNS)


def pattern_violation(pattern: str, *, neural: bool = False) -> tuple[str, str] | None:
    """Never-list hit for an ALLOWLIST *pattern* itself (a forced allowlist
    edit like adding "README.md" or "inputs/**" is refused here, before any
    file is even considered)."""
    probe = pattern.replace("**", "zz/zz").replace("*", "zz").replace("?", "z")
    return never_reason(probe, neural=neural)


# --------------------------------- collection -----------------------------

@dataclass
class Collection:
    files: list[tuple[Path, str, int]] = field(default_factory=list)   # (abs, rel, bytes)
    excluded: list[tuple[str, str, str]] = field(default_factory=list)  # (rel, pattern, why)
    junk: int = 0
    violations: list[tuple[str, str, str]] = field(default_factory=list)  # (allow pattern, never pattern, why)

    @property
    def total_bytes(self) -> int:
        return sum(size for _, _, size in self.files)


def _walk_files(base: Path):
    """Files under ``base``, never following links/junctions, junk pruned."""
    for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
        keep = []
        for d in dirnames:
            p = Path(dirpath) / d
            if _is_junk(d) or p.is_symlink() or _is_junction(p):
                continue
            keep.append(d)
        dirnames[:] = keep
        for f in filenames:
            yield Path(dirpath) / f


def _is_junction(path: Path) -> bool:
    """Windows directory junction (reparse point that is not a symlink)."""
    try:
        return bool(os.stat(path, follow_symlinks=False).st_file_attributes & 0x400)  # REPARSE_POINT
    except (OSError, AttributeError):
        return False


def collect(*, project_root: Path | None = None, with_tests: bool = True,
            neural: bool = False) -> Collection:
    root = Path(project_root or PROJECT_ROOT)
    result = Collection()
    entries = list(ALLOWLIST) + (list(NEURAL_ALLOWLIST) if neural else [])

    # A forced allowlist edit that names never-list territory: refuse it at
    # the PATTERN level, before touching any file.
    for pattern, _why in entries:
        hit = pattern_violation(pattern, neural=neural)
        if hit is not None:
            result.violations.append((pattern, hit[0], hit[1]))
    if result.violations:
        return result

    picked: dict[str, Path] = {}
    for pattern, _why in entries:
        base = root / pattern.split("**")[0].rstrip("/")
        if "**" in pattern:
            if not base.exists():
                continue  # e.g. root config/ "if present"
            candidates = _walk_files(base)
        else:
            candidates = [base] if base.is_file() else []
        for path in candidates:
            rel = path.relative_to(root).as_posix()
            if not matches(rel, pattern):
                continue
            if not with_tests and rel.startswith(PKG_PREFIX + "test/"):
                continue
            if _is_junk(rel):
                result.junk += 1
                continue
            hit = never_reason(rel, neural=neural)
            if hit is not None:
                result.excluded.append((rel, hit[0], hit[1]))
                continue
            picked[rel] = path

    # Top-level entries that contribute nothing: show WHY they stay behind
    # (this is where a seeded guide.html / README / the inputs junction and
    # the outputs dir become visible in the dry-run).
    picked_top = {rel.split("/", 1)[0] for rel in picked}
    for entry in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        name = entry.name
        if name in picked_top:
            continue
        probe = name + "/zz/zz" if entry.is_dir() else name
        hit = never_reason(probe, neural=neural) or never_reason(name, neural=neural)
        if hit is not None:
            shown = name + "/ (junction - never followed)" if _is_junction(entry) else (
                name + "/" if entry.is_dir() else name)
            result.excluded.append((shown, hit[0], hit[1]))
        else:
            result.excluded.append((name + "/" if entry.is_dir() else name,
                                    "(not in the allowlist)", "only allowlisted files ship"))

    result.files = sorted(
        ((path, rel, path.stat().st_size) for rel, path in picked.items()),
        key=lambda t: t[1],
    )

    # Belt and braces: even if the filtering above ever regresses, a
    # never-list file in the final copy set aborts the run.
    _final_screen(result.files, neural=neural)
    return result


def _final_screen(files: list[tuple[Path, str, int]], *, neural: bool) -> None:
    """Raise if any file of the final copy set matches the never-list."""
    for _, rel, _ in files:
        hit = never_reason(rel, neural=neural)
        if hit is not None:
            raise UploadRefused(
                f"internal guard: {rel} is in the copy set but matches the "
                f"never-list ({hit[0]} - {hit[1]}); refusing to continue"
            )


def staged_rel(rel: str) -> str:
    """Path of ``rel`` inside the staged tree / on the Orin (<dest>/...)."""
    return rel[len(PKG_PREFIX):] if rel.startswith(PKG_PREFIX) else rel


def stage(collection: Collection, stage_dir: Path) -> list[Path]:
    stage_dir = Path(stage_dir)
    if stage_dir.exists() and any(stage_dir.iterdir()):
        raise UploadRefused(f"stage dir {stage_dir} is not empty - delete it first")
    seen: dict[str, str] = {}
    out = []
    for path, rel, _size in collection.files:
        srel = staged_rel(rel)
        if srel in seen:
            raise UploadRefused(f"staging collision: {rel} and {seen[srel]} both map to {srel}")
        seen[srel] = rel
        target = stage_dir / srel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        out.append(target)
    return out


# --------------------------------- reporting ------------------------------

def _human(size: int) -> str:
    if size >= 1_000_000:
        return f"{size / 1_000_000:.2f} MB"
    if size >= 1_000:
        return f"{size / 1_000:.1f} KB"
    return f"{size} B"


def print_manifest(c: Collection, *, neural: bool, with_tests: bool) -> None:
    for _, rel, size in c.files:
        arrow = "" if staged_rel(rel) == rel else f"  ->  {staged_rel(rel)}"
        print(f"  UPLOAD  {rel}{arrow}  ({_human(size)})")
    for rel, pattern, why in c.excluded:
        print(f"  EXCLUDED  {rel}  [{pattern}] {why}")
    if c.junk:
        print(f"  (plus {c.junk} bytecode/cache junk files pruned silently)")
    print(f"UPLOAD SET: {len(c.files)} files, {c.total_bytes / 1_000_000:.2f} MB "
          f"({c.total_bytes:,} bytes); tests {'included' if with_tests else 'SKIPPED'}; "
          f"neural weights {'INCLUDED' if neural else 'not included (classical)'}")


# --------------------------------- transport ------------------------------

def _require_tool(name: str) -> str:
    exe = shutil.which(name)
    if exe is None:
        raise UploadRefused(
            f"'{name}' was not found on PATH. Windows ships OpenSSH: Settings > System > "
            "Optional features > add 'OpenSSH Client' (or run this from a machine that has "
            "ssh/scp). Nothing was uploaded."
        )
    return exe


def _validate_clean_dest(dest: str) -> None:
    bad = {"", ".", "..", "/", "~", "$HOME"}
    if dest.strip() in bad or dest.startswith("/") or dest.startswith(".."):
        raise UploadRefused(f"--clean-dest refuses to delete {dest!r} (unsafe path)")


def upload(collection: Collection, host: str, dest: str, *, use_tar: bool,
           clean_dest: bool, runner=subprocess.run) -> None:
    ssh = _require_tool("ssh")
    scp = _require_tool("scp")
    tar = _require_tool("tar") if use_tar else None
    with tempfile.TemporaryDirectory(prefix="kart2d_upload_") as tmp:
        stage_dir = Path(tmp) / "kart_map2d"
        stage_dir.mkdir()
        stage(collection, stage_dir)
        if clean_dest:
            _validate_clean_dest(dest)
            print(f"--clean-dest: removing {host}:{dest} first")
            r = runner([ssh, host, f"rm -rf -- {dest}"], capture_output=True, text=True)
            if r.returncode != 0:
                raise UploadRefused(f"ssh rm failed: {(r.stderr or '').strip()}")
        r = runner([ssh, host, f"mkdir -p {dest}"], capture_output=True, text=True)
        if r.returncode != 0:
            raise UploadRefused(
                f"ssh mkdir failed (host reachable? key set up?): {(r.stderr or '').strip()}"
            )
        if use_tar:
            ball = Path(tmp) / "kart2d_upload.tgz"
            r = runner([tar, "-czf", str(ball), "-C", str(stage_dir), "."],
                       capture_output=True, text=True)
            if r.returncode != 0:
                raise UploadRefused(f"tar failed: {(r.stderr or '').strip()}")
            r = runner([scp, "-q", str(ball), f"{host}:/tmp/kart2d_upload.tgz"])
            if r.returncode != 0:
                raise UploadRefused("scp of the tarball failed")
            r = runner([ssh, host,
                        f"tar -xzf /tmp/kart2d_upload.tgz -C {dest} && rm -f /tmp/kart2d_upload.tgz"],
                       capture_output=True, text=True)
            if r.returncode != 0:
                raise UploadRefused(f"remote tar extract failed: {(r.stderr or '').strip()}")
        else:
            r = runner([scp, "-r", "-q", str(stage_dir) + "/.", f"{host}:{dest}/"])
            if r.returncode != 0:
                raise UploadRefused("scp failed")
    print(f"uploaded to {host}:{dest} (nothing on the target was deleted"
          + (" beyond --clean-dest)" if clean_dest else ")"))
    print("next on the Orin: see the AFTER THE UPLOAD section of KART_UPLOAD.txt")


# ----------------------------------- main ---------------------------------

def cfg_wants_neural(cfg_path: str) -> bool:
    """True when kart_master.cfg says [segmentation] mode = neural.

    An invalid cfg is an UploadRefused: never ship a control file the
    validator rejects."""
    try:
        cfg = load_master_config(cfg_path)
    except MasterConfigError as exc:
        raise UploadRefused(
            "kart_master.cfg is invalid - fix it before uploading:\n  - "
            + "\n  - ".join(exc.problems)
        ) from None
    return str(cfg["segmentation"]["mode"]) == "neural"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", required=True,
                        help=f"scp target user@address, e.g. {PLACEHOLDER_HOST} "
                             "(EDIT the placeholder to your Orin's address)")
    parser.add_argument("--dest", default=DEFAULT_DEST,
                        help=f"remote path, space-free, relative to the Orin home "
                             f"(default {DEFAULT_DEST})")
    parser.add_argument("--dry-run", action="store_true",
                        help="print every file that would be copied (with sizes and totals); copy nothing")
    parser.add_argument("--stage-to", metavar="DIR",
                        help="stage the exact upload tree into DIR and stop (no ssh); "
                             "used by the container proof")
    parser.add_argument("--neural", action="store_true",
                        help="also upload the SegFormer weights (outputs/cache/models)")
    parser.add_argument("--no-tests", action="store_true", help="skip the test/ folder")
    parser.add_argument("--tar", action="store_true",
                        help="send one tarball instead of many files (faster over slow links)")
    parser.add_argument("--clean-dest", action="store_true",
                        help="DELETE the remote dest first; without this flag nothing is ever "
                             "deleted on the target")
    parser.add_argument("--cfg", default=str(PROJECT_ROOT / "kart_master.cfg"),
                        help="control file to validate + read the segmentation mode from")
    args = parser.parse_args(argv)

    try:
        if " " in args.dest:
            raise UploadRefused(
                f"destination {args.dest!r} contains a space - the Orin workspace path "
                "must be space-free (colcon breaks on spaces)")
        neural = args.neural or cfg_wants_neural(args.cfg)
        if neural and not args.neural:
            print("note: kart_master.cfg says [segmentation] mode = neural -> weights included "
                  "(same as passing --neural)")

        c = collect(with_tests=not args.no_tests, neural=neural)
        if c.violations:
            print("REFUSED - the allowlist names never-list territory; nothing was copied:")
            for allow_pat, never_pat, why in c.violations:
                print(f"  allowlist entry {allow_pat!r} matches never-list {never_pat!r}: {why}")
            print("fix ALLOWLIST in tools/upload_to_kart.py (and KART_UPLOAD.txt) first.")
            return 2

        print_manifest(c, neural=neural, with_tests=not args.no_tests)

        if args.dry_run:
            print("dry run: nothing copied")
            return 0
        if args.stage_to:
            staged = stage(c, Path(args.stage_to))
            print(f"staged {len(staged)} files into {args.stage_to} (no ssh contact)")
            return 0
        if "EDIT" in args.host.upper() or "@" not in args.host:
            raise UploadRefused(
                f"--host {args.host!r} is the placeholder / not user@address; EDIT it to your "
                "Orin's real address (find it with: ssh into the kart's access point)")
        upload(c, args.host, args.dest, use_tar=args.tar, clean_dest=args.clean_dest)
        return 0
    except UploadRefused as exc:
        print(f"REFUSED: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
