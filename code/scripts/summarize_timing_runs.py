"""Mean and standard deviation over the repeated OpenFHE timing runs.

    python code/scripts/summarize_timing_runs.py          (from the project folder)

Reads every results/grouping_time/repeat_runs/run_*/ folder (made by
repeat_timing_runs.sh) and the saved group degrees, and writes three tables
to results/grouping_time/repeat_runs/:

  summary_time_saving.csv   time saved by grouping per budget: one column per
                            run, then mean and standard deviation
  summary_per_degree.csv    seconds per activation by degree: mean, std, levels
  summary_direct_check.csv  the direct check per run, with mean and std

Only the activations are timed. Convolutions, bootstraps and everything else
in an encrypted network are not part of these numbers.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
GT = ROOT / "results" / "grouping_time"
RUNS = sorted((GT / "repeat_runs").glob("run_*/"), key=lambda p: int(p.name.split("_")[1]))
assert RUNS, "no runs found"


def read(p):
    return pd.read_csv(p, comment="#")


deg = read(GT / "exp4_bucket_degrees.csv")
# drop the three sites where no activation happens (projection batch norms)
deg = deg[~deg.site.str.match(r"^features\.\d+\.conv\.[23]$")]

saving, seconds, direct, per_degree, levels = {}, {}, {}, {}, {}
for r in RUNS:
    t = read(r / "exp4_degree_timing.csv")
    t = t[t.chain_depth == t.chain_depth.min()]
    sec = dict(zip(t.degree, t.seconds_median))
    per_degree[r.name] = t.set_index("degree").seconds_median
    levels[r.name] = t.set_index("degree").levels_used
    rows, secs = {}, {}
    for B, g in deg.groupby("budget_B"):
        placed = sum(sec[d] for d in g.degree_placed)
        layer = sum(sec[d] for d in g.degree_layerwise)
        rows[B] = 100 * (1 - placed / layer)
        secs[B] = (layer, placed)
    saving[r.name] = pd.Series(rows)
    seconds[r.name] = pd.DataFrame(secs, index=["layerwise_s", "grouped_s"]).T
    d = read(r / "exp4_direct_regions.csv").pivot(index="budget_B", columns="layout", values="seconds")
    direct[r.name] = 100 * (1 - d.placed / d.layerwise)

s = pd.DataFrame(saving)
out = s.copy()
out["mean"], out["std"], out["min"], out["max"] = s.mean(1), s.std(1), s.min(1), s.max(1)
lw = pd.DataFrame({k: v.layerwise_s for k, v in seconds.items()})
gr = pd.DataFrame({k: v.grouped_s for k, v in seconds.items()})
out["layerwise_s_mean"], out["layerwise_s_std"] = lw.mean(1), lw.std(1)
out["grouped_s_mean"], out["grouped_s_std"] = gr.mean(1), gr.std(1)
out["saved_s_mean"], out["saved_s_std"] = (lw - gr).mean(1), (lw - gr).std(1)
out.index.name = "budget_B"
out.to_csv(GT / "repeat_runs" / "summary_time_saving.csv")

p = pd.DataFrame(per_degree)
lv = pd.DataFrame(levels)
pd_out = pd.DataFrame({"seconds_mean": p.mean(1), "seconds_std": p.std(1),
                       "seconds_min": p.min(1), "seconds_max": p.max(1),
                       "levels_used": lv.iloc[:, 0],
                       "levels_same_in_all_runs": lv.nunique(axis=1) == 1})
pd_out.to_csv(GT / "repeat_runs" / "summary_per_degree.csv")

d = pd.DataFrame(direct)
d_out = d.copy()
d_out["mean"], d_out["std"] = d.mean(1), d.std(1)
d_out.to_csv(GT / "repeat_runs" / "summary_direct_check.csv")

# the same saving with the degrees of the exact interval formula
# (results/interval_formula_check), if that experiment has been run
BN = ROOT / "results" / "interval_formula_check" / "exp4b_bucket_degrees_bnout.csv"
if BN.exists():
    degb = read(BN)
    degb = degb[~degb.site.str.match(r"^features\.\d+\.conv\.[23]$")]
    sb = {}
    for r in RUNS:
        sec = per_degree[r.name].to_dict()
        sb[r.name] = pd.Series({B: 100 * (1 - sum(sec[d] for d in g.degree_placed)
                                          / sum(sec[d] for d in g.degree_layerwise))
                                for B, g in degb.groupby("budget_B")})
    sb = pd.DataFrame(sb)
    sb_out = sb.copy()
    sb_out["mean"], sb_out["std"] = sb.mean(1), sb.std(1)
    sb_out.index.name = "budget_B"
    sb_out.to_csv(GT / "repeat_runs" / "summary_time_saving_exact_interval.csv")

pd.set_option("display.width", 200)
print(f"{len(RUNS)} runs")
if BN.exists():
    print("\nSame, with the exact interval formula (mean, std):")
    print(sb_out[["mean", "std"]].round(2).T.to_string())
print("\nTime saved by grouping, percent of activation time, by budget:")
print(out[["mean", "std", "min", "max", "layerwise_s_mean", "grouped_s_mean", "saved_s_mean", "saved_s_std"]].round(2).to_string())
print("\nDirect check (percent saved):")
print(d_out.round(2).to_string())
print("\nSeconds per activation at selected degrees (mean, std):")
print((1000 * pd_out.loc[[1, 2, 7, 15, 31, 63], ["seconds_mean", "seconds_std"]]).round(1).to_string())
print("levels identical in every run:", bool(pd_out.levels_same_in_all_runs.all()))
cv = (pd_out.seconds_std / pd_out.seconds_mean)
print("relative std of per-degree time: median %.1f%%, max %.1f%%" % (100 * cv.median(), 100 * cv.max()))
