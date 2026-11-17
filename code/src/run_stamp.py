"""
Saves results with a timestamp and a hash of the scripts, so I can tell which
version of the code made which result. No git on colab so hashes instead.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]      # the code/ folder
PROJECT_RESULTS = PACKAGE_ROOT.parent / "results"
RESULTS_DIR = PROJECT_RESULTS / "_unsorted"


def use_results(experiment: str) -> Path:
    """send this run's outputs to results/<experiment>/"""
    global RESULTS_DIR
    RESULTS_DIR = PROJECT_RESULTS / experiment
    return RESULTS_DIR

_CACHE: dict | None = None

_HASHED_SUFFIXES = {".py", ".txt"}


def _package_hashes() -> dict[str, str]:
    out: dict[str, str] = {}
    for p in sorted(PACKAGE_ROOT.rglob("*")):
        if p.is_file() and p.suffix in _HASHED_SUFFIXES \
                and RESULTS_DIR not in p.parents:
            h = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
            out[str(p.relative_to(PACKAGE_ROOT)).replace("\\", "/")] = h
    return out


def _git(*args: str) -> str | None:
    try:
        r = subprocess.run(["git", *args], cwd=PACKAGE_ROOT,
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


def stamp(refresh: bool = False) -> dict:
    """One snapshot per process, so every output of a run agrees."""
    global _CACHE
    if _CACHE is not None and not refresh:
        return dict(_CACHE)
    _CACHE = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "colab": "COLAB_RELEASE_TAG" in os.environ,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "argv": " ".join(sys.argv),
        "package_sha256": _package_hashes(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain") or ""),
    }
    return dict(_CACHE)


def write_result(name: str, payload: dict, verdict: str) -> Path:
    """write a stamped JSON result plus a one-line VERDICT file"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    doc = dict(payload)
    doc["VERDICT"] = verdict
    doc["_provenance"] = stamp()
    out = RESULTS_DIR / f"{name}.json"
    out.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    (RESULTS_DIR / f"VERDICT_{name}.txt").write_text(verdict + "\n",
                                                     encoding="utf-8")
    return out


def write_table(name: str, df, payload: dict,
                verdict: str | None = None) -> Path:
    """CSV of the rows, with a short `# key: value` header in the repo's own style (read back with pandas.read_csv(..., comment="#"))"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    st = stamp()
    csv = RESULTS_DIR / f"{name}.csv"
    with csv.open("w", encoding="utf-8", newline="") as f:
        f.write(f"# timestamp_utc: {st['timestamp_utc']}\n")
        f.write(f"# argv: {st['argv']}\n")
        f.write(f"# hostname: {st['hostname']}  colab: {st['colab']}\n")
        for k, v in payload.items():
            if isinstance(v, (str, int, float, bool)):
                f.write(f"# {k}: {v}\n")
        df.to_csv(f, index=False)
    if verdict is not None:
        write_result(name, dict(payload, rows_csv=csv.name), verdict)
    return csv


def banner(lines: list[str]) -> None:
    """Print a block that stays legible in a truncated log."""
    width = max(len(l) for l in lines) + 4
    print("=" * width, flush=True)
    for l in lines:
        print(f"| {l.ljust(width - 4)} |", flush=True)
    print("=" * width, flush=True)
