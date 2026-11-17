"""What the measured time saving becomes if the degrees are limited.

    python code/scripts/time_saving_with_degree_limits.py     (from the project folder)

The accuracy proxy (experiments/grouping_accuracy_proxy) shows that on a
MobileNetV2 the degrees the rule gives are usable only when no channel runs
below about degree 15 and none above about 31. The time saving of thesis
section 6.14 was measured with degrees from 1 to 63. This recomputes it, from
the same saved group degrees and the same five timing runs, with every degree
raised to a floor and lowered to a cap first.

No new measurement is made. The grouping is the saved one, computed for the
unlimited degrees; it is not recomputed for the limited ones, so a grouping
chosen for the limited degrees could do at least as well in multiplications.
The row floor 1, cap 63 is the control: it must reproduce
summary_time_saving.csv.

Writes results/grouping_time/repeat_runs/summary_time_saving_degree_limits.csv
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
GT = ROOT / "results" / "grouping_time"
RUNS = sorted((GT / "repeat_runs").glob("run_*/"), key=lambda p: int(p.name.split("_")[1]))
assert RUNS, "no runs found"
LIMITS = [(1, 63), (1, 31), (7, 63), (7, 31), (15, 63), (15, 31)]


def read(p):
    return pd.read_csv(p, comment="#")


deg = read(GT / "exp4_bucket_degrees.csv")
deg = deg[~deg.site.str.match(r"^features\.\d+\.conv\.[23]$")]   # no activation there
rows = []
for floor, cap in LIMITS:
    placed_d = deg.degree_placed.clip(floor, cap)
    layer_d = deg.degree_layerwise.clip(floor, cap)
    per_run = {}
    for r in RUNS:
        t = read(r / "exp4_degree_timing.csv")
        t = t[t.chain_depth == t.chain_depth.min()]
        sec = dict(zip(t.degree, t.seconds_median))
        for B in sorted(deg.budget_B.unique()):
            m = deg.budget_B == B
            placed = sum(sec[d] for d in placed_d[m])
            layer = sum(sec[d] for d in layer_d[m])
            per_run.setdefault(B, []).append((layer, placed))
    for B, v in per_run.items():
        s = pd.Series([100 * (1 - p / l) for l, p in v])
        rows.append({"degree_floor": floor, "degree_cap": cap, "budget_B": B,
                     "saving_pct_mean": s.mean(), "saving_pct_std": s.std(),
                     "layerwise_s_mean": pd.Series([l for l, _ in v]).mean(),
                     "grouped_s_mean": pd.Series([p for _, p in v]).mean(),
                     "group_activations": int((deg.budget_B == B).sum()),
                     "groups_changed_by_grouping": int(
                         (placed_d[deg.budget_B == B] != layer_d[deg.budget_B == B]).sum())})
out = pd.DataFrame(rows)

ref = read(GT / "repeat_runs" / "summary_time_saving.csv").set_index("budget_B")["mean"]
ctl = out[(out.degree_floor == 1) & (out.degree_cap == 63)].set_index("budget_B").saving_pct_mean
assert (ctl - ref).abs().max() < 1e-9, "control does not reproduce summary_time_saving.csv"

dst = GT / "repeat_runs" / "summary_time_saving_degree_limits.csv"
out.to_csv(dst, index=False)
print("control reproduces summary_time_saving.csv")
print(out.pivot(index="budget_B", columns=["degree_floor", "degree_cap"],
                values="saving_pct_mean").round(1).to_string())
print()
print(out[(out.degree_floor == 15) & (out.degree_cap == 31)]
      [["budget_B", "saving_pct_mean", "saving_pct_std", "layerwise_s_mean",
        "grouped_s_mean", "group_activations", "groups_changed_by_grouping"]].round(2).to_string(index=False))
print("wrote", dst)
