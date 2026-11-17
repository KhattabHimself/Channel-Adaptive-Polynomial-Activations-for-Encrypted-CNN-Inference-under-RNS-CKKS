"""
Does the interval formula change the weights-only numbers?

The weights-only degree model (archive src/phase3/invariants.py, and the
earlier system's compute_channel_intervals) gives channel c the interval
[b_c - 3|a_c|, b_c + 3|a_c|], with a_c = gamma/std and b_c = beta - gamma*mu/std
the folded BatchNorm. That is the activation's input for a conv output x in
[-3, 3]. The conv output actually has mean mu and std std (the running
statistics), so the activation's input has mean beta and std |gamma|, and its
3-std interval is [beta - 3|gamma|, beta + 3|gamma|]. The two agree only when
mu = 0 and std = 1.

This redoes spread (exp2), rotation (exp2) and the time saving (exp4) with
the BN-output interval:  s_c = |gamma_c| * ||V[:,c]||,  t0 = -beta / (3|gamma|).
To compare like with like, the budgets are rescaled per network so the median
s_c is the same under both formulas; only the per-channel structure changes.

Stage 1 of exp4 is reused with the new table. Its control check against the
archive is EXPECTED to report MISMATCH here (the degrees are different on
purpose); its output goes to exp4b_bucket_degrees_bnout, not exp4's file.

run:  python exp4b_interval_scale.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import src.network_tools as core  # noqa: E402
import experiments.grouping_time.grouping_time as e4  # noqa: E402
from src.network_tools import (bn_sites, _consumer_norms, channel_table, degrees_at,  # noqa: E402
                  depthwise_regions, load_model, rotation, spread)
from experiments.spread_and_rotation.spread_and_rotation_six_networks import BUDGETS as ROT_BUDGETS  # noqa: E402
import src.run_stamp as _stamp  # noqa: E402
from src.run_stamp import banner, write_result, write_table  # noqa: E402

NETWORKS = ["cifar10_resnet20", "imagenet_mobilenetv2", "cifar10_resnet56",
            "cifar100_resnet20", "cifar10_mobilenetv2_x1_0",
            "imagenet_mnasnet1_0"]
ORIGINAL_TABLE = channel_table
ORIGINAL_DEGREES = e4.DEGREES_CSV          # exp4's own file, read not written


def table_bnout(model, net: str, sigma: float = core.SIGMA) -> pd.DataFrame:
    """channel_table with the activation input's own mean and std"""
    rows = []
    for site_index, (name, bn, consumer) in enumerate(bn_sites(model)):
        gamma = bn.weight.detach().cpu().numpy()
        beta = bn.bias.detach().cpu().numpy()
        v = _consumer_norms(consumer, bn.num_features)
        for c in range(bn.num_features):
            half = sigma * abs(gamma[c])
            t0 = math.inf if half <= 0 else -beta[c] / half
            vc = float(v[c]) if c < len(v) else float("nan")
            rows.append({"net": net, "site": name, "site_index": site_index,
                         "channel_idx": c, "a_c": float(gamma[c]),
                         "b_c": float(beta[c]), "v_norm": vc,
                         "s_c": abs(gamma[c]) * vc,
                         "t0": max(-1.5, min(1.5, t0))})
    return pd.DataFrame(rows)


def rescaled(model, net) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    old = ORIGINAL_TABLE(model, net)
    new = table_bnout(model, net)
    k = float(new.s_c.median() / old.s_c.median())
    new["s_c"] = new.s_c / k          # same median s_c: same budget scale
    return old, new, k


def site_spread(t: pd.DataFrame) -> float:
    return float(t.groupby("site_index").s_c.apply(spread).median())


def max_rot(t: pd.DataFrame, regions) -> float:
    r = pd.concat([rotation(degrees_at(t, B), regions, B) for B in ROT_BUDGETS],
                  ignore_index=True)
    return float(r.groupby(["delta_model", "G"]).rotates.mean().max())


def time_saving(deg: pd.DataFrame, timing: pd.DataFrame) -> pd.DataFrame:
    """exp4's combine, activations scope, shortest chain"""
    deg = deg[~deg.site.str.match(r"^features\.\d+\.conv\.[23]$")]
    tg = timing[timing.chain_depth == timing.chain_depth.min()]
    t = dict(zip(tg.degree, tg.seconds_median))
    rows = []
    for B, g in deg.groupby("budget_B"):
        placed = sum(t[d] for d in g.degree_placed)
        layer = sum(t[d] for d in g.degree_layerwise)
        rows.append({"budget_B": B,
                     "median_region_degree_max": float(
                         g.groupby("region").degree_layerwise.max().median()),
                     "time_saving_pct": 100 * (1 - placed / layer)})
    return pd.DataFrame(rows)


def main() -> None:
    _stamp.use_results("interval_formula_check")
    banner(["EXP 4b - BN-output interval vs the archive's folded-BN interval"])
    rows = []
    for net in NETWORKS:
        model = load_model(net)
        old, new, k = rescaled(model, net)
        regions = depthwise_regions(model)
        row = {"net": net, "scale_factor": k,
               "spread_archive": site_spread(old), "spread_bnout": site_spread(new)}
        if regions:
            row["max_rotation_archive"] = max_rot(old, regions)
            row["max_rotation_bnout"] = max_rot(new, regions)
        for B in [0.01, 0.1]:
            da, dn = degrees_at(old, B).degree, degrees_at(new, B).degree
            row[f"degree_changed_share_B{B:g}"] = float((da != dn).mean())
        rows.append(row)
        print({k2: (round(v, 3) if isinstance(v, float) else v)
               for k2, v in row.items()}, flush=True)
    summ = pd.DataFrame(rows)

    # exp4 stage 1 on the new table, then exp4's combine on the stored timing
    model = load_model("mobilenetv2")
    _, _, k = rescaled(model, "mobilenetv2")
    core.channel_table = lambda m, n, sigma=core.SIGMA: rescaled(m, n)[1]
    e4.DEGREES_CSV = "exp4b_bucket_degrees_bnout"
    e4.stage_degrees()
    core.channel_table = ORIGINAL_TABLE
    timing = pd.read_csv(_stamp.PROJECT_RESULTS / "grouping_time" / f"{e4.TIMING_CSV}.csv", comment="#")
    new_deg = pd.read_csv(_stamp.RESULTS_DIR / "exp4b_bucket_degrees_bnout.csv",
                          comment="#")
    old_deg = pd.read_csv(_stamp.PROJECT_RESULTS / "grouping_time" / f"{ORIGINAL_DEGREES}.csv", comment="#")
    ts = time_saving(old_deg, timing).merge(
        time_saving(new_deg, timing), on="budget_B",
        suffixes=("_archive", "_bnout"))
    print(summ.round(3).to_string(index=False))
    print(ts.round(2).to_string(index=False))

    r = summ.set_index("net")
    t001 = ts.set_index("budget_B").loc[0.01]
    verdict = (
        "BN-output interval vs archive: median spread "
        + ", ".join(f"{n} {r.loc[n, 'spread_archive']:.2f}->{r.loc[n, 'spread_bnout']:.2f}"
                    for n in ["cifar10_resnet20", "imagenet_mobilenetv2"])
        + "; max rotation "
        + ", ".join(f"{n} {r.loc[n, 'max_rotation_archive']:.3f}->{r.loc[n, 'max_rotation_bnout']:.3f}"
                    for n in r.index if not np.isnan(r.loc[n].get('max_rotation_archive', np.nan)))
        + f"; MobileNetV2 time saving at B=0.01 {t001.time_saving_pct_archive:.1f}%"
          f" -> {t001.time_saving_pct_bnout:.1f}%")
    write_table("exp4b_interval_scale", summ, {"experiment": "EXP 4b"}, None)
    write_table("exp4b_time_saving", ts, {"experiment": "EXP 4b time",
                                          "mobilenetv2_scale_factor": k})
    write_result("exp4b_interval_scale", {"rows_csv": "exp4b_interval_scale.csv"},
                 verdict)
    banner([verdict[i:i + 100] for i in range(0, len(verdict), 100)])


if __name__ == "__main__":
    main()
