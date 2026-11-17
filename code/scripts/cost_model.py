"""A small cost model for one polynomial activation, checked against measurements.

    python code/scripts/cost_model.py        (from the project folder)

Three parts. Each says what it was checked against, and what it was not.

1. Levels. A closed form for the levels OpenFHE's Chebyshev evaluator uses at
   degree d, checked against the measured levels at all 63 degrees. It is
   fitted to the measurements, not derived from the library's source.
2. Time. Four one-line models of the time of one activation as a function of
   the degree, fitted to the five-run means (12-level chain, N = 2^16), with
   their errors, and a check of how well each predicts the measured saving of
   grouping (thesis Table 6.14).
3. Blocks and bootstraps. The block formula of thesis section 4.3 turned into
   a count of bootstraps for a ResNet with a given chain. This part is a
   model only: no bootstrapped network was run, so it is not checked.

Writes results/cost_model/*.csv
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
GT = ROOT / "results" / "grouping_time"
OUT = ROOT / "results" / "cost_model"
OUT.mkdir(exist_ok=True)


def read(p):
    return pd.read_csv(p, comment="#")


# ------------------------------------------------------------------ levels
def delta_A(d):
    return math.ceil(math.log2(d + 1))


def delta_B(d):
    return math.ceil(math.log2(d)) + 1


def levels_model(d: int) -> int:
    """delta_A, plus one for the degrees at the top of each power-of-two range.
    With k = ceil(log2(d + 1)), the extra level applies when
    d >= 2^k - 2^(ceil(k/2) - 1)."""
    if d <= 2:
        return d
    k = math.ceil(math.log2(d + 1))
    return k + (d >= 2 ** k - 2 ** (math.ceil(k / 2) - 1))


m = read(GT / "repeat_runs" / "summary_per_degree.csv")
lv = pd.DataFrame({"degree": m.degree, "measured": m.levels_used,
                   "model": [levels_model(int(d)) for d in m.degree],
                   "delta_A": [delta_A(int(d)) for d in m.degree],
                   "delta_B": [delta_B(int(d)) for d in m.degree]})
lv.to_csv(OUT / "levels_model.csv", index=False)
print(f"levels: the closed form matches the measurement at {(lv.model == lv.measured).sum()} of "
      f"{len(lv)} degrees (delta_A at {(lv.delta_A == lv.measured).sum()}, "
      f"delta_B at {(lv.delta_B == lv.measured).sum()})")

# -------------------------------------------------------------------- time
d = m.degree.to_numpy(float)
t = m.seconds_mean.to_numpy()
keep = d >= 2                      # degree 1 is a straight line: 3.5 ms, modelled apart
features = {
    "a + b * sqrt(d)": np.sqrt(d),
    "a + b * ceil(sqrt(2d))  (the multiplication count)": np.ceil(np.sqrt(2 * d)),
    "a + b * log2(d)": np.log2(d),
    "a + b * d": d,
    "relative fit of a + b * sqrt(d)": np.sqrt(d),
    "power law a * d^b (a in s, b the exponent)": d,
}
T1 = float(t[d == 1][0])
rows, fits = [], {}
for name, f in features.items():
    A = np.vstack([np.ones(keep.sum()), f[keep]]).T
    coef, *_ = np.linalg.lstsq(A, t[keep], rcond=None)
    pred = np.where(d >= 2, coef[0] + coef[1] * f, T1)
    if name.startswith("relative"):
        # the same straight line in sqrt(d), fitted to relative instead of absolute error
        w = 1 / t[keep]
        coef, *_ = np.linalg.lstsq(A * w[:, None], t[keep] * w, rcond=None)
        pred = np.where(d >= 2, coef[0] + coef[1] * f, T1)
    if name.startswith("power"):
        pw, *_ = np.linalg.lstsq(np.vstack([np.ones(keep.sum()), np.log(d[keep])]).T,
                                 np.log(t[keep]), rcond=None)
        coef = np.array([np.exp(pw[0]), pw[1]])
        pred = np.where(d >= 2, coef[0] * d ** coef[1], T1)
    rel = np.abs(pred[keep] - t[keep]) / t[keep]
    ss = 1 - ((pred[keep] - t[keep]) ** 2).sum() / ((t[keep] - t[keep].mean()) ** 2).sum()
    fits[name] = dict(zip(d.astype(int), pred))
    rows.append({"model": name, "a_ms": 1000 * coef[0], "b_ms": 1000 * coef[1], "R2": ss,
                 "median_relative_error_pct": 100 * np.median(rel),
                 "largest_relative_error_pct": 100 * rel.max()})
tm = pd.DataFrame(rows)

# how well each model predicts the saving of grouping
deg = read(GT / "exp4_bucket_degrees.csv")
deg = deg[~deg.site.str.match(r"^features\.\d+\.conv\.[23]$")]
meas = read(GT / "repeat_runs" / "summary_time_saving.csv").set_index("budget_B")["mean"]
pred_rows = []
for name, table in fits.items():
    for B, g in deg.groupby("budget_B"):
        placed = sum(table[int(x)] for x in g.degree_placed)
        layer = sum(table[int(x)] for x in g.degree_layerwise)
        pred_rows.append({"model": name, "budget_B": B, "predicted_saving_pct": 100 * (1 - placed / layer),
                          "measured_saving_pct": meas[B]})
pr = pd.DataFrame(pred_rows)
pr["error_points"] = pr.predicted_saving_pct - pr.measured_saving_pct
tm = tm.merge(pr.groupby("model").error_points.apply(lambda e: e.abs().max())
              .rename("largest_error_on_saving_points").reset_index(), on="model")
tm.to_csv(OUT / "time_models.csv", index=False)
pr.to_csv(OUT / "time_model_saving_check.csv", index=False)
print("time of one activation, degree 2 to 63 (degree 1 fixed at its measured "
      f"{1000 * T1:.1f} ms):")
print(tm.round(2).to_string(index=False))
best = tm.loc[tm.largest_error_on_saving_points.idxmin()]
print(f"best at predicting the saving: {best.model}: median error on one activation "
      f"{best.median_relative_error_pct:.1f}%, largest {best.largest_relative_error_pct:.0f}%, "
      f"saving predicted within {best.largest_error_on_saving_points:.1f} points at every budget")
print(pr[pr.model == best.model][["budget_B", "predicted_saving_pct", "measured_saving_pct"]]
      .round(1).to_string(index=False))

# ----------------------------------------------------- blocks and bootstraps
BOOT_S = 11.3      # one bootstrap of a full ciphertext, measured once (thesis 6.14)
rows = []
# a(d) = 1 + delta(d) for a single interpolant, as in thesis section 4.3; the
# composite depths are those of thesis Table 6.10
for label, act in [("degree 7 interpolant", 5), ("degree 15", 6), ("degree 31", 7), ("degree 63", 8),
                   ("composite 7-15-15 (cheapest within 1 point)", 14),
                   ("composite 15-15-15-15", 20)]:
    for convs, name in ((2, "standard convolutions"), (3, "earlier system's convolutions")):
        block = 2 * act + convs
        for chain in (24, 43):
            per = chain // block
            rows.append({"activation": label, "levels_per_activation": act, "block": name,
                         "levels_per_block": block, "chain_levels": chain,
                         "blocks_per_chain": per,
                         "bootstrap_points_resnet20": (math.ceil(9 / per) - 1) if per else None,
                         "note": "" if per else "one block does not fit in the chain"})
bm = pd.DataFrame(rows)
bm.to_csv(OUT / "block_and_bootstrap_model.csv", index=False)
print("blocks per chain and bootstrap points for the 9 blocks of ResNet-20 (model, not measured):")
print(bm[(bm.block == "standard convolutions")][
    ["activation", "levels_per_block", "chain_levels", "blocks_per_chain",
     "bootstrap_points_resnet20"]].to_string(index=False))
print("wrote", OUT)
