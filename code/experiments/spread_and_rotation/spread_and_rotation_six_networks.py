"""
Second network check, work in progress.

Spread and rotation on more networks. Only needs the weights, no data.
The two old networks (resnet20, imagenet mobilenetv2) run first and should
give exactly the same numbers as before, otherwise the copied code is wrong
and nothing else should be read.

run:  python exp2_second_network.py

TODO
- check the chenyaofo hub names are right
- thresholds are a guess for now (spread < 2, rotation <= 0.5), decide before
  running
- mobilenetv2 spread was already 2.09, so the narrow spread point is really a
  resnet thing
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.network_tools import (channel_table, degrees_at, depthwise_regions,  # noqa: E402
                  dispersion, load_model, rotation)
import src.run_stamp as _stamp  # noqa: E402
from src.run_stamp import banner, write_result, write_table  # noqa: E402

BUDGETS = [0.3, 0.1, 0.03, 0.01, 0.003]          # as src/phase3/gate_b.py

NETWORKS = {
    "cifar10_resnet20": "control",
    "imagenet_mobilenetv2": "control",
    "cifar10_resnet56": "new",
    "cifar100_resnet20": "new",
    "cifar10_mobilenetv2_x1_0": "new",
    "imagenet_mnasnet1_0": "new",
}

SPREAD_THRESHOLD = 2.0
ROTATION_THRESHOLD = 0.5

CONTROL_SPREAD = {
    "cifar10_resnet20": (21, 1.4721542355821913),
    "imagenet_mobilenetv2": (52, 2.093598456172688),
}
CONTROL_ROTATION = {
    "imagenet_mobilenetv2": (16, {
        ("A", 2): 0.0875, ("A", 4): 0.15, ("A", 8): 0.2125,
        ("B", 2): 0.0625, ("B", 4): 0.1625, ("B", 8): 0.275,
    }),
}
TOL = 1e-6


def analyse(net: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    model = load_model(net)
    table = channel_table(model, net)
    regions = depthwise_regions(model)

    disp, rot = [], []
    for B in BUDGETS:
        df = degrees_at(table, B)
        disp.append(dispersion(df, B))
        if regions:
            rot.append(rotation(df, regions, B))
    d = pd.concat(disp, ignore_index=True)
    r = pd.concat(rot, ignore_index=True) if rot else pd.DataFrame()

    first = d[d.budget_B == BUDGETS[0]]
    summary = {
        "n_sites": int(first.site.nunique()),
        "median_spread": float(first.s_c_spread_p90_p10.median()),
        "frac_sites_spread_ge_2": float(
            (first.s_c_spread_p90_p10 >= SPREAD_THRESHOLD).mean()),
        "n_regions": len(regions),
    }
    summary["spread_narrow"] = summary["median_spread"] < SPREAD_THRESHOLD
    if len(r):
        rates = r.groupby(["delta_model", "G"]).rotates.mean()
        summary["rotation_rate"] = {f"{m}_G{g}": float(v)
                                    for (m, g), v in rates.items()}
        summary["max_rotation_rate"] = float(rates.max())
        summary["rotation_rare"] = bool(rates.max() <= ROTATION_THRESHOLD)
    return d, r, summary


def check_controls(summaries: dict, rotations: dict) -> list[str]:
    """Every mismatch against the archived numbers, as a readable line."""
    misses = []
    for net, (n_sites, median) in CONTROL_SPREAD.items():
        s = summaries.get(net)
        if s is None or "error" in s:
            misses.append(f"{net}: did not run")
            continue
        if s["n_sites"] != n_sites:
            misses.append(f"{net}: {s['n_sites']} sites, archived {n_sites}")
        if abs(s["median_spread"] - median) > TOL:
            misses.append(f"{net}: median spread {s['median_spread']!r}, "
                          f"archived {median!r}")
    for net, (n_regions, rates) in CONTROL_ROTATION.items():
        s = summaries.get(net)
        if s is None or "error" in s:
            continue                                # already reported above
        if s["n_regions"] != n_regions:
            misses.append(f"{net}: {s['n_regions']} regions found, archived "
                          f"{n_regions} (region detection differs)")
        r = rotations.get(net, pd.DataFrame())
        got = r.groupby(["delta_model", "G"]).rotates.mean() if len(r) else {}
        for key, want in rates.items():
            have = float(got[key]) if key in getattr(got, "index", {}) else None
            if have is None or abs(have - want) > TOL:
                misses.append(f"{net}: rotation {key} = {have}, archived "
                              f"{want}")
    return misses


def main() -> None:
    _stamp.use_results("spread_and_rotation")
    ap = argparse.ArgumentParser(description="EXP 2: second-network check")
    ap.add_argument("--nets", nargs="+", default=list(NETWORKS),
                    help="networks to analyse; controls are always required "
                         "for a verdict on new networks")
    args = ap.parse_args()

    banner(["EXP 2 - SECOND FRONT ON MORE NETWORKS (weights only, CPU)",
            f"spread narrow: median p90/p10 < {SPREAD_THRESHOLD}   "
            f"rotation rare: every rate <= {ROTATION_THRESHOLD}",
            "controls must reproduce the archive before new nets are read"])

    all_disp, all_rot = [], []
    summaries: dict[str, dict] = {}
    rotations: dict[str, pd.DataFrame] = {}
    for net in args.nets:
        role = NETWORKS.get(net, "new")
        print(f"\n--- {net} ({role}) ---", flush=True)
        try:
            d, r, s = analyse(net)
        except Exception as e:  # noqa: BLE001  one bad checkpoint must not
            traceback.print_exc()          # sink the controls or the others
            summaries[net] = {"role": role, "error": f"{type(e).__name__}: {e}"}
            continue
        s["role"] = role
        summaries[net] = s
        rotations[net] = r
        all_disp.append(d)
        if len(r):
            all_rot.append(r)
        line = (f"  sites {s['n_sites']}  median spread "
                f"{s['median_spread']:.3f}  narrow={s['spread_narrow']}  "
                f"regions {s['n_regions']}")
        if "max_rotation_rate" in s:
            line += (f"  max rotation {s['max_rotation_rate']:.3f}  "
                     f"rare={s['rotation_rare']}")
        print(line, flush=True)

    misses = check_controls(summaries, rotations)
    ran_new = {n: s for n, s in summaries.items()
               if s.get("role") == "new" and "error" not in s}
    failed = [n for n, s in summaries.items() if "error" in s]

    if misses:
        verdict = ("CONTROL FAILED - new networks NOT interpreted: "
                   + "; ".join(misses))
    else:
        narrow = [n for n, s in ran_new.items() if s["spread_narrow"]]
        with_rot = {n: s for n, s in ran_new.items() if "rotation_rare" in s}
        rare = [n for n, s in with_rot.items() if s["rotation_rare"]]
        verdict = (f"CONTROLS REPRODUCED. New networks: spread narrow on "
                   f"{len(narrow)}/{len(ran_new)}; rotation rare on "
                   f"{len(rare)}/{len(with_rot)} depthwise.")
        if failed:
            verdict += f" Not run: {', '.join(failed)}."

    payload = {
        "experiment": "EXP 2 second-network check (spread, rotation)",
        "budgets": BUDGETS,
        "spread_threshold": SPREAD_THRESHOLD,
        "rotation_threshold": ROTATION_THRESHOLD,
        "per_network": summaries,
        "control_mismatches": misses,
    }
    if all_disp:
        write_table("exp2_spread", pd.concat(all_disp, ignore_index=True),
                    {"experiment": payload["experiment"]})
    if all_rot:
        write_table("exp2_rotation", pd.concat(all_rot, ignore_index=True),
                    {"experiment": payload["experiment"]})
    out = write_result("exp2_second_network", payload, verdict)

    banner([f"VERDICT: {verdict}"])
    print(f"wrote {out}", flush=True)
    raise SystemExit(1 if misses else 0)


if __name__ == "__main__":
    main()
