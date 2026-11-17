"""Confidence intervals and tests for the main comparisons, from saved results.

    python code/scripts/statistics_existing_results.py      (from the project folder)

Makes no new measurement. Reads result files and writes
results/statistics/*.csv. What each interval covers is stated in the output,
because the sources of variation differ:

  accuracies      binomial sampling of the 10,000 test images (Wilson interval).
                  The networks are fixed checkpoints, so this is NOT variation
                  over training runs.
  rotation rates  the 80 cases are 16 regions at 5 budgets and are not
                  independent. Two intervals: exact binomial on the 80 (too
                  narrow), and a bootstrap that resamples whole regions.
  time savings    five timing runs on one machine on one day (t interval).
  proxy           21 sites (bootstrap over sites, sign test).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportion_confint

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
OUT = RES / "statistics"
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(0)


def read(p):
    return pd.read_csv(p, comment="#")


def wilson(acc_pct, n):
    k = round(acc_pct / 100 * n)
    lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson")
    return 100 * lo, 100 * hi


# ---------------------------------------------------------------- accuracy
rows = []
g = read(RES / "range_vs_degree" / "exp3_range_grid_train.csv")
for r in g[g.rule.isin(["minmax", "sigma6", "sigma3"])].itertuples():
    lo, hi = wilson(r.top1, r.n_images)
    rows.append({"source": "Table 6.11 (interval rule against degree)",
                 "what": f"{r.target} {r.rule} degree {r.degree}", "top1": r.top1,
                 "ci95_lo": lo, "ci95_hi": hi, "n": r.n_images})
c = read(RES / "activation_accuracy" / "oa_composite.csv")
for r in c.itertuples():
    lo, hi = wilson(r.top1, 10000)
    blo, bhi = wilson(r.baseline_top1, 10000)
    # unpaired interval for the difference: conservative, the images are the same
    se = np.sqrt(r.top1 * (100 - r.top1) / 10000 + r.baseline_top1 * (100 - r.baseline_top1) / 10000)
    rows.append({"source": "Table 6.10 (composite sequences)", "what": r.sequence,
                 "top1": r.top1, "ci95_lo": lo, "ci95_hi": hi, "n": 10000,
                 "points_lost": -r.delta_points,
                 "points_lost_ci95_hi_unpaired": -r.delta_points + 1.96 * se,
                 "within_1_point": r.within_1_point})
acc = pd.DataFrame(rows)
acc.to_csv(OUT / "accuracy_intervals.csv", index=False)
comp = acc[acc.source.str.startswith("Table 6.10")]
n_within = int(comp.within_1_point.sum())
n_robust = int((comp.points_lost_ci95_hi_unpaired <= 1.0).sum())
print(f"accuracy: a Wilson interval on 10,000 images is about +/- "
      f"{(acc.ci95_hi - acc.ci95_lo).median() / 2:.2f} points (median half-width)")
print(f"composite: {n_within} of {len(comp)} sequences are within 1 point; for {n_robust} the "
      "upper 95% bound of the loss (unpaired, conservative) is also within 1 point")
s = comp[comp.what == "7-15-15"].iloc[0]
print(f"  7-15-15: lost {s.points_lost:.2f}, unpaired upper bound {s.points_lost_ci95_hi_unpaired:.2f}")

# ---------------------------------------------------------------- rotation
rot = read(RES / "spread_and_rotation" / "exp2_rotation.csv")
rows = []
for (net, G, model), d in rot.groupby(["net", "G", "delta_model"]):
    k, n = int(d.rotates.sum()), len(d)
    lo, hi = proportion_confint(k, n, alpha=0.05, method="beta")
    ref = 1 - 1 / G
    p_ref = stats.binomtest(k, n, ref, alternative="less").pvalue
    blocks = d.block.unique()
    piv = d.pivot_table(index="block", columns="budget_B", values="rotates").to_numpy(dtype=float)
    boot = [np.nanmean(piv[rng.integers(0, len(blocks), len(blocks))]) for _ in range(10000)]
    rows.append({"net": net, "G": G, "delta_model": model, "rotating": k, "cases": n,
                 "rate_pct": 100 * k / n,
                 "exact_binomial_ci95_lo": 100 * lo, "exact_binomial_ci95_hi": 100 * hi,
                 "region_bootstrap_ci95_lo": 100 * np.percentile(boot, 2.5),
                 "region_bootstrap_ci95_hi": 100 * np.percentile(boot, 97.5),
                 "regions": len(blocks),
                 "regions_that_rotate_at_some_budget": int((np.nansum(piv, 1) > 0).sum()),
                 "reference_unrelated_pct": 100 * ref,
                 "p_value_rate_below_reference": p_ref})
rt = pd.DataFrame(rows)
rt.to_csv(OUT / "rotation_intervals.csv", index=False)
print(f"rotation: {len(rt)} (network, G, model) cells; largest upper bound of the "
      f"region-bootstrap interval {rt.region_bootstrap_ci95_hi.max():.1f}%; "
      f"largest p-value against the unrelated reference {rt.p_value_rate_below_reference.max():.1e}")
worst = rt.loc[rt.rate_pct.idxmax()]
print(f"  highest rate: {worst.net} G={worst.G} model {worst.delta_model}: {worst.rate_pct:.1f}% "
      f"[{worst.region_bootstrap_ci95_lo:.1f}, {worst.region_bootstrap_ci95_hi:.1f}] "
      f"against {worst.reference_unrelated_pct:.1f}%")

# -------------------------------------------------------------------- time
rows = []
for name, f in [("time saving, interval fixed in advance", "summary_time_saving.csv"),
                ("time saving, exact interval", "summary_time_saving_exact_interval.csv"),
                ("direct check", "summary_direct_check.csv")]:
    t = read(RES / "grouping_time" / "repeat_runs" / f)
    runs = [c for c in t.columns if c.startswith("run_")]
    for r in t.itertuples():
        v = np.array([getattr(r, c) for c in runs])
        half = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
        rows.append({"what": name, "budget_B": r.budget_B, "runs": len(v), "mean_pct": v.mean(),
                     "std_pct": v.std(ddof=1), "ci95_lo": v.mean() - half, "ci95_hi": v.mean() + half,
                     "p_value_saving_is_zero": stats.ttest_1samp(v, 0).pvalue})
lim = read(RES / "grouping_time" / "repeat_runs" / "summary_time_saving_degree_limits.csv")
tm = pd.DataFrame(rows)
tm.to_csv(OUT / "time_saving_intervals.csv", index=False)
r = tm[(tm.what.str.startswith("time saving, interval fixed")) & (tm.budget_B == 0.01)].iloc[0]
print(f"time: saving at B=0.01 is {r.mean_pct:.1f}% with a 95% t interval "
      f"[{r.ci95_lo:.1f}, {r.ci95_hi:.1f}] over {r.runs} runs")

# ------------------------------------------------------------------- proxy
px = read(RES / "sensitivity_proxy" / "proxy_validation_resnet20.csv")
rho = px.groupby("site").apply(lambda d: stats.spearmanr(d.v_norm, d.influence).statistic)
boot = [rho.sample(len(rho), replace=True, random_state=int(i)).mean() for i in range(10000)]
k = int((rho > 0).sum())
proxy = pd.DataFrame([{
    "sites": len(rho), "mean_rank_correlation": rho.mean(), "median": rho.median(),
    "bootstrap_ci95_lo": np.percentile(boot, 2.5), "bootstrap_ci95_hi": np.percentile(boot, 97.5),
    "sites_positive": k,
    "sign_test_p_value": stats.binomtest(k, len(rho), 0.5, alternative="greater").pvalue,
    "wilcoxon_p_value": stats.wilcoxon(rho, alternative="greater").pvalue}])
proxy.to_csv(OUT / "proxy_intervals.csv", index=False)
p = proxy.iloc[0]
print(f"proxy: mean within-site rank correlation {p.mean_rank_correlation:+.2f} "
      f"[{p.bootstrap_ci95_lo:+.2f}, {p.bootstrap_ci95_hi:+.2f}], positive on {k} of {len(rho)} "
      f"sites (sign test p = {p.sign_test_p_value:.3f})")

# ------------------------------------------------------------------ spread
sp = read(RES / "spread_and_rotation" / "exp2_spread.csv")
rows = []
for net, d in sp.groupby("net"):
    per_site = d.groupby("site").s_c_spread_p90_p10.first()
    boot = [np.median(per_site.sample(len(per_site), replace=True, random_state=int(i)))
            for i in range(10000)]
    rows.append({"net": net, "sites": len(per_site), "median_spread": per_site.median(),
                 "bootstrap_ci95_lo": np.percentile(boot, 2.5),
                 "bootstrap_ci95_hi": np.percentile(boot, 97.5),
                 "share_of_bootstrap_medians_below_2": float(np.mean(np.array(boot) < 2))})
spd = pd.DataFrame(rows)
spd.to_csv(OUT / "spread_intervals.csv", index=False)
print("spread (median over sites, bootstrap over sites):")
for r in spd.itertuples():
    print(f"  {r.net:<28} {r.median_spread:8.2f}  [{r.bootstrap_ci95_lo:.2f}, {r.bootstrap_ci95_hi:.2f}]"
          f"  below 2 in {100 * r.share_of_bootstrap_medians_below_2:.0f}% of resamples")
print("wrote", OUT)
