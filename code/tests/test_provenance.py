"""
PURPOSE
    Regression tests for the provenance stamp (PROTOCOL rule 8). Two bugs, both
    fixed 2026-08-22, both of which made `dirty` meaningless for regenerated
    results while leaving newly-created ones correct — which is why they hid:

      B1  write_csv stamped AFTER `open(path,"w")` truncated its own tracked
          output, so the file's own truncation registered as an uncommitted
          change.
      B2  the stamp was recomputed per file, so in a script writing several
          tracked outputs, outputs 2..n saw output 1 as an uncommitted change.

    The guard against both: writing to a TRACKED path on a CLEAN tree must
    report dirty=false, and must keep doing so for every subsequent output.

    ISOLATION (PROTOCOL rule 14, added 2026-08-22). Every test here runs against
    a throwaway git repo under `tmp_path`, with `provenance._REPO_ROOT`
    monkeypatched to point at it. **No test writes to a tracked path in the
    thesis tree.** The first version of these tests did — it wrote to
    `results/phase1/*` and perturbed `PROTOCOL.md` — and one restore went
    through `write_text`, which on Windows rewrote LF as CRLF against the
    `* -text` attribute and put a 101-line no-op diff into a commit. A test that
    can corrupt the evidence it exists to protect is the wrong shape of test.

INPUTS
    None from the thesis tree. Skips if git is unavailable.

OUTPUTS
    pytest pass/fail only.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import provenance  # noqa: E402


@pytest.fixture
def temp_repo(tmp_path, monkeypatch):
    """A throwaway git repo with two committed (tracked) outputs.

    Yields (repo_path, tracked_csv, tracked_json). The thesis working tree is
    never touched: provenance._REPO_ROOT is redirected here for the test.
    """
    if shutil.which("git") is None:
        pytest.skip("git unavailable")

    repo = tmp_path / "repo"
    (repo / "results").mkdir(parents=True)

    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True,
                       check=False)

    git("init", "-b", "main")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "provenance test")

    tracked_csv = repo / "results" / "out.csv"
    tracked_json = repo / "results" / "out.json"
    tracked_csv.write_bytes(b"value\n1\n")
    tracked_json.write_bytes(b"{}\n")
    git("add", "-A")
    git("commit", "-m", "seed tracked outputs")

    monkeypatch.setattr(provenance, "_REPO_ROOT", repo)
    provenance.stamp(refresh=True)
    try:
        yield repo, tracked_csv, tracked_json
    finally:
        # Drop the temp repo's state from the process-wide snapshot so later
        # tests re-read the real tree.
        monkeypatch.undo()
        provenance.stamp(refresh=True)


def _header(path: Path) -> list[str]:
    return [l for l in path.read_text().splitlines() if l.startswith("#")]


def test_temp_repo_starts_clean(temp_repo):
    """Precondition: the fixture must hand over a committed, clean tree."""
    _repo, _csv, _json = temp_repo
    assert provenance.is_dirty() is False


def test_writing_tracked_path_on_clean_tree_is_not_dirty(temp_repo):
    """B1: the write must not report the file's own truncation as dirty."""
    _repo, tracked_csv, _json = temp_repo
    provenance.write_csv(pd.DataFrame({"value": [1, 2]}), tracked_csv)
    header = _header(tracked_csv)
    assert any("dirty: False" in line for line in header), header


def test_second_tracked_output_in_same_run_is_not_dirty(temp_repo):
    """B2: writing output 1 must not poison the stamp of output 2."""
    _repo, tracked_csv, tracked_json = temp_repo
    provenance.write_csv(pd.DataFrame({"value": [1]}), tracked_csv)
    # Output 1 is now an uncommitted change on disk; output 2 must still read
    # clean, because provenance describes the run, not the moment of writing.
    provenance.write_json({"k": "v"}, tracked_json)
    stamped = json.loads(tracked_json.read_text())["_provenance"]
    assert stamped["dirty"] is False
    # One run, one snapshot: the two outputs must name the same commit.
    commit_line = next(l for l in _header(tracked_csv)
                       if l.startswith("# commit:"))
    assert stamped["commit"] in commit_line


def test_dirty_is_still_reported_when_the_tree_really_is_dirty(temp_repo):
    """The fix must not make `dirty` unconditionally False."""
    repo, _csv, _json = temp_repo
    probe = repo / "results" / "out.csv"
    original = probe.read_bytes()          # bytes: never round-trip as text
    probe.write_bytes(original + b"2\n")
    assert provenance.is_dirty() is True
    assert provenance.stamp(refresh=True)["dirty"] is True
    probe.write_bytes(original)
    assert provenance.stamp(refresh=True)["dirty"] is False


def test_no_test_writes_into_the_thesis_tree(temp_repo):
    """Rule 14, asserted rather than trusted: the real tree is untouched.

    Compares before/after rather than requiring a clean tree — the invariant is
    that these tests CHANGE nothing, which holds whether or not the developer
    has uncommitted work in progress.
    """
    repo, tracked_csv, tracked_json = temp_repo
    real_root = Path(__file__).resolve().parents[1]

    def status() -> str:
        return subprocess.run(["git", "status", "--porcelain"], cwd=real_root,
                              capture_output=True, text=True).stdout

    before = status()
    provenance.write_csv(pd.DataFrame({"value": [1]}), tracked_csv)
    provenance.write_json({"k": "v"}, tracked_json)
    (repo / "results" / "extra.csv").write_bytes(b"x\n")
    after = status()

    assert after == before, (
        "provenance writes leaked into the thesis tree:\n"
        f"before:\n{before}\nafter:\n{after}"
    )
