"""
PURPOSE
    Phase 3 / Gate B analyses, on the invariants produced by invariants.py:

      1. DISPERSION, swept over the budget B, reported per site and summarised.
         Both the classic statistic (spread in s_c) and the operative one:
         spread in PLATEAU CLASS. Dispersion inside one plateau costs nothing,
         because delta is a step function and a bucket's level cost only moves
         when its maximum degree crosses a plateau boundary
         [LAB_NOTEBOOK Sec. 4.6]. Reported under both delta models.

      2. GATE C, folded in (REVISED_PLAN): sweep a dead-channel threshold tau
         over percentiles of s_c and recompute dispersion on survivors only. If
         the dispersion lives on channels a pruner would delete, it is an
         artifact. The fine-tune half of the original Gate C still needs a GPU
         and is not done here.

      3. CONJUNCT (3): does the argmax bucket ROTATE between the activation
         sites of one mixing-free region? With n = 2 sites the level bound is
         (n-1)*(f_max-f_min) = one site's worth, and it is realised only if the
         bucket carrying the maximum changes between the sites. This is the
         unmeasured probability p that sets MobileNetV2's expected saving
         [IMPOSSIBILITY_AND_ESCAPE Sec. 4.4]. Measured on MobileNetV2, whose
         inverted residuals are the only regions of length 2 available.

INPUTS
    results/phase3/invariants_<net>.csv (from invariants.py). Regenerated
    internally per budget value, so the CSV is a convenience, not a dependency.

OUTPUTS
    results/phase3/dispersion_<net>.csv     per (budget, site) dispersion
    results/phase3/tau_sweep_<net>.csv      Gate C: dispersion vs prune depth
    results/phase3/conjunct3_<net>.csv      per (budget, G, region) rotation
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.provenance import require_clean, write_csv  # noqa: E402
from src.levels import DELTA_MODELS  # noqa: E402
from src.channel_degrees import invariants  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO.parent / "results"
BUDGETS = [0.3, 0.1, 0.03, 0.01, 0.003]
TAUS = [0, 1, 5, 10, 20]          # percentile of s_c defining the dead set
BUCKET_COUNTS = [2, 4, 8]


def _spread(series: pd.Series) -> float:
    """p90/p10 ratio, the dispersion statistic the gate is stated in."""
    lo = np.percentile(series, 10)
    hi = np.percentile(series, 90)
    return float(hi / lo) if lo > 0 else float("inf")


def dispersion(df: pd.DataFrame, budget: float) -> pd.DataFrame:
    rows = []
    for (site_index, site), g in df.groupby(["site_index", "site"]):
        row = {
            "budget_B": budget, "site_index": site_index, "site": site,
            "n_channels": len(g),
            "s_c_spread_p90_p10": _spread(g.s_c),
            "degree_min": int(g.degree.min()),
            "degree_median": float(g.degree.median()),
            "degree_max": int(g.degree.max()),
        }
        for name, model in DELTA_MODELS.items():
            classes = g.degree.map(model)
            row[f"plateau_min_{name}"] = int(classes.min())
            row[f"plateau_max_{name}"] = int(classes.max())
            # f_max - f_min: the term the region bound multiplies by (n-1).
            row[f"plateau_spread_{name}"] = int(classes.max() - classes.min())
            row[f"n_plateau_classes_{name}"] = int(classes.nunique())
        rows.append(row)
    return pd.DataFrame(rows)


def tau_sweep(df: pd.DataFrame, budget: float) -> pd.DataFrame:
    """Gate C: is the dispersion carried by channels a pruner would remove?"""
    rows = []
    for tau in TAUS:
        cut = np.percentile(df.s_c, tau) if tau > 0 else -np.inf
        kept = df[df.s_c > cut]
        for name, model in DELTA_MODELS.items():
            per_site = []
            for _, g in kept.groupby("site_index"):
                if len(g) < 2:
                    continue
                classes = g.degree.map(model)
                per_site.append({
                    "s_spread": _spread(g.s_c),
                    "plateau_spread": int(classes.max() - classes.min()),
                })
            if not per_site:
                continue
            p = pd.DataFrame(per_site)
            rows.append({
                "budget_B": budget, "tau_percentile": tau,
                "delta_model": name,
                "channels_kept": int(len(kept)),
                "channels_pruned": int(len(df) - len(kept)),
                "median_s_spread": float(p.s_spread.median()),
                "median_plateau_spread": float(p.plateau_spread.median()),
                "sites_with_plateau_spread_ge1":
                    int((p.plateau_spread >= 1).sum()),
                "n_sites": int(len(p)),
            })
    return pd.DataFrame(rows)


# ── conjunct (3): argmax bucket rotation within a mixing-free region ───────
_CNA_BN = re.compile(r"^features\.(\d+)\.conv\.(\d+)\.1$")


def regions_mobilenetv2(df: pd.DataFrame) -> dict[int, list[str]]:
    """Activation sites per inverted-residual block.

    A BN named features.i.conv.j.1 sits inside a Conv2dNormActivation, so it
    precedes an activation. The project convolution's BN (features.i.conv.3)
    does not — the linear bottleneck — and is correctly excluded. Blocks with
    two such sites are the n = 2 regions; the t = 1 block has one and is
    excluded by the same rule.
    """
    out: dict[int, list[str]] = {}
    for site in df.site.unique():
        m = _CNA_BN.match(site)
        if m:
            out.setdefault(int(m.group(1)), []).append(site)
    return {k: sorted(v) for k, v in out.items() if len(v) >= 2}


def conjunct3(df: pd.DataFrame, budget: float) -> pd.DataFrame:
    rows = []
    for block, sites in regions_mobilenetv2(df).items():
        s1 = df[df.site == sites[0]].sort_values("channel_idx")
        s2 = df[df.site == sites[1]].sort_values("channel_idx")
        if len(s1) != len(s2) or len(s1) == 0:
            continue
        d1 = s1.degree.to_numpy()
        d2 = s2.degree.to_numpy()
        # One permutation is shared across the region (the residual/placement
        # constraint), so order by the per-channel max over its sites — what a
        # solver optimising the region as a whole would do.
        order = np.argsort(-np.maximum(d1, d2), kind="stable")
        for G in BUCKET_COUNTS:
            if len(order) < G:
                continue
            buckets = np.array_split(order, G)
            for name, model in DELTA_MODELS.items():
                f1 = [max(model(int(x)) for x in d1[b]) for b in buckets]
                f2 = [max(model(int(x)) for x in d2[b]) for b in buckets]
                arg1, arg2 = int(np.argmax(f1)), int(np.argmax(f2))
                rows.append({
                    "budget_B": budget, "block": block, "G": G,
                    "delta_model": name,
                    "site_1": sites[0], "site_2": sites[1],
                    "n_channels": len(d1),
                    "argmax_bucket_site1": arg1,
                    "argmax_bucket_site2": arg2,
                    "rotates": bool(arg1 != arg2),
                    "f_spread_site1": int(max(f1) - min(f1)),
                    "f_spread_site2": int(max(f2) - min(f2)),
                })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--net", default="resnet20",
                    choices=["resnet20", "mobilenetv2"])
    args = ap.parse_args()
    require_clean()

    disp, taus, c3 = [], [], []
    for B in BUDGETS:
        df = invariants(args.net, B)
        disp.append(dispersion(df, B))
        taus.append(tau_sweep(df, B))
        if args.net == "mobilenetv2":
            c3.append(conjunct3(df, B))

    d = pd.concat(disp, ignore_index=True)
    t = pd.concat(taus, ignore_index=True)
    write_csv(d, RESULTS / f"spread_and_rotation/dispersion_{args.net}.csv")
    write_csv(t, RESULTS / f"spread_and_rotation/tau_sweep_{args.net}.csv")

    print(f"=== {args.net}: dispersion, median over sites ===")
    print(d.groupby("budget_B").agg(
        median_s_spread=("s_c_spread_p90_p10", "median"),
        median_plateau_spread_A=("plateau_spread_A", "median"),
        median_plateau_spread_B=("plateau_spread_B", "median"),
        median_degree_max=("degree_max", "median")).to_string())

    print(f"\n=== {args.net}: Gate C, dispersion vs prune depth ===")
    print(t[t.delta_model == "B"][
        ["budget_B", "tau_percentile", "channels_pruned", "median_s_spread",
         "median_plateau_spread", "sites_with_plateau_spread_ge1", "n_sites"]
    ].to_string(index=False))

    if c3:
        c = pd.concat(c3, ignore_index=True)
        write_csv(c, RESULTS / f"spread_and_rotation/conjunct3_{args.net}.csv")
        print("\n=== conjunct (3): argmax bucket rotation rate p ===")
        print(c.groupby(["delta_model", "G", "budget_B"]).rotates.agg(
            ["mean", "sum", "count"]).to_string())


if __name__ == "__main__":
    main()
