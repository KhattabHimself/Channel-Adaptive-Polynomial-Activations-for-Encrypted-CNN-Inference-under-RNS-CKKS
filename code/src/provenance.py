"""
PURPOSE
    Bind every result file to the exact code state that produced it.

    A commit hash written into an output file is worthless if the working tree
    had uncommitted edits when the script ran — the hash then points at code
    that is not the code that ran. This module makes that failure impossible to
    miss: it records the dirty state explicitly, and `require_clean()` refuses
    to run at all unless the tree is committed.

INPUTS
    None (reads git metadata from the repo containing this file).

OUTPUTS
    stamp()          -> dict of provenance fields
    require_clean()  -> raises SystemExit if the tree is dirty
    write_csv()      -> DataFrame to CSV with a provenance comment header
    write_json()     -> dict to JSON with a "_provenance" key

USAGE
    from src.provenance import require_clean, write_csv

    require_clean()                     # top of every script that writes results
    write_csv(df, "results/phase1/invariants.csv")

    Reading a stamped CSV back:
        pandas.read_csv(path, comment="#")
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _git(*args: str) -> str | None:
    """Run a git command in the repo root. Returns None if git is unavailable."""
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def is_dirty() -> bool | None:
    """True if tracked files have uncommitted changes. None if git unavailable."""
    status = _git("status", "--porcelain", "--untracked-files=no")
    if status is None:
        return None
    return bool(status)


_STAMP_CACHE: dict | None = None


def stamp(refresh: bool = False) -> dict:
    """Provenance for the RUN, snapshotted once and reused for every output.

    Snapshotting is the fix, not an optimisation. Provenance answers "what code
    state produced this result", which is a property of the run, not of the
    moment each file happens to be written. Recomputing per file made a script
    that writes several tracked outputs report `dirty=true` on outputs 2..n,
    because output 1 was by then an uncommitted change — the same class of bug
    as the write-after-truncate one fixed earlier the same day, one step
    removed. `require_clean()` primes this cache at script start (rule 8), so
    the snapshot is taken before anything is written.

    All outputs of one run therefore share one commit, one dirty flag and one
    timestamp — which also makes them groupable by run. Pass refresh=True only
    to deliberately re-read git state (tests do this).
    """
    global _STAMP_CACHE
    if _STAMP_CACHE is not None and not refresh:
        return dict(_STAMP_CACHE)
    _STAMP_CACHE = _compute_stamp()
    return dict(_STAMP_CACHE)


def _compute_stamp() -> dict:
    dirty = is_dirty()
    return {
        "commit": _git("rev-parse", "HEAD"),
        "commit_short": _git("rev-parse", "--short", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "describe": _git("describe", "--tags", "--always", "--dirty"),
        "dirty": dirty,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script": os.path.relpath(sys.argv[0], _REPO_ROOT)
        if sys.argv and sys.argv[0]
        else None,
        "argv": sys.argv[1:],
        "hostname": os.environ.get("HOSTNAME", "unknown"),
        "container_image": os.environ.get("IMAGE_TAG", "unset"),
    }


def require_clean(allow_override: bool = True) -> dict:
    """
    Refuse to produce results from uncommitted code.

    Set ALLOW_DIRTY_RUN=1 to bypass during active development. Results produced
    under the bypass are stamped dirty=true and must never be quoted in a report
    or a thesis table.
    """
    info = stamp()

    if info["commit"] is None:
        print(
            "PROVENANCE: not a git repo (or git unavailable). Results produced "
            "here cannot be traced to a code version.",
            file=sys.stderr,
        )
        return info

    if not info["dirty"]:
        return info

    if allow_override and os.environ.get("ALLOW_DIRTY_RUN") == "1":
        print(
            "PROVENANCE WARNING: running with uncommitted changes "
            "(ALLOW_DIRTY_RUN=1). Outputs are stamped dirty=true and are NOT "
            "citable in any report.",
            file=sys.stderr,
        )
        return info

    changed = _git("status", "--porcelain", "--untracked-files=no") or ""
    print(
        "PROVENANCE ERROR: working tree has uncommitted changes, so any commit "
        "hash stamped into these results would be wrong.\n\n"
        f"{changed}\n\n"
        "Commit first, or set ALLOW_DIRTY_RUN=1 for a throwaway run.",
        file=sys.stderr,
    )
    raise SystemExit(2)


def _header_lines(extra: dict | None = None) -> list[str]:
    info = stamp()
    if extra:
        info.update(extra)
    return [f"# {k}: {v}" for k, v in info.items()]


def write_csv(df, path: str | Path, extra: dict | None = None) -> Path:
    """Write a DataFrame to CSV behind a '#'-commented provenance header.

    The header is built BEFORE the file is opened. `open(path, "w")` truncates,
    and if `path` is tracked that truncation is itself an uncommitted change —
    so stamping afterwards made every regenerated tracked result report
    `dirty=true` no matter how clean the tree was. New (untracked) outputs were
    unaffected, which is why the bug hid: `git status --untracked-files=no`
    does not see them. Fixed 2026-08-22; see write_json/save_figure, which
    already ordered it correctly.
    """
    path = Path(path)
    header = "\n".join(_header_lines(extra)) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        fh.write(header)
        df.to_csv(fh, index=False)
    return path


def write_json(obj: dict, path: str | Path, extra: dict | None = None) -> Path:
    """Write a dict to JSON with provenance under the '_provenance' key."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    info = stamp()
    if extra:
        info.update(extra)
    payload = dict(obj)
    payload["_provenance"] = info
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    return path


def save_figure(fig, path: str | Path) -> Path:
    """Save a matplotlib figure with the commit stamped into the image footer."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    info = stamp()
    tag = f"{info['describe'] or 'no-git'} · {info['timestamp_utc'][:19]}Z"
    fig.text(0.005, 0.005, tag, fontsize=5, alpha=0.5)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


if __name__ == "__main__":
    print(json.dumps(stamp(), indent=2))
