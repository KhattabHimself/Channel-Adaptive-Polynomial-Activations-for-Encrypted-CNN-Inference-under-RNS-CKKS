"""
Why is the CIFAR MobileNetV2 spread 75,033? (exp2 loose end)

spread = p90/p10 of s_c = |a_c| * ||V[:,c]||, per activation site. If more
than 10% of a site's channels have s_c ~ 0 (dead channels: BN gamma ~ 0, or
nothing downstream reads them), p10 is ~0 and the ratio blows up. This counts
dead channels per site and recomputes the spread over live channels only.
Weights only, no data, CPU.

run:  python exp2b_dead_channels.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.network_tools import channel_table, load_model, spread  # noqa: E402
import src.run_stamp as _stamp  # noqa: E402
from src.run_stamp import banner, write_table  # noqa: E402

NETWORKS = ["cifar10_resnet20", "imagenet_mobilenetv2", "cifar10_resnet56",
            "cifar100_resnet20", "cifar10_mobilenetv2_x1_0",
            "imagenet_mnasnet1_0"]
# dead: s_c below this fraction of the site's median s_c
DEAD_REL = 1e-3
# there is no clean line between dead and live, so also sweep the cut
SWEEP = [1e-3, 1e-2, 1e-1]


def activation_sites(model) -> set[str]:
    """BN layers whose next leaf module is an activation (the linear
    bottleneck's projection BN is not one: nothing is approximated there)"""
    import torch.nn as nn
    acts = (nn.ReLU, nn.ReLU6, nn.Hardswish, nn.SiLU, nn.GELU)
    leaves = [(n, m) for n, m in model.named_modules()
              if not list(m.children())]
    return {n for (n, m), (_, nxt) in zip(leaves, leaves[1:])
            if isinstance(m, nn.BatchNorm2d) and isinstance(nxt, acts)}


def main() -> None:
    _stamp.use_results("dead_channels")
    banner(["EXP 2b - dead channels and the spread statistic"])
    rows = []
    for net in NETWORKS:
        model = load_model(net)
        act = activation_sites(model)
        t = channel_table(model, net)
        for (si, site), g in t.groupby(["site_index", "site"]):
            s = g.s_c.to_numpy()
            med = np.median(s)
            dead = s < DEAD_REL * med if med > 0 else s <= 0
            live = pd.Series(s[~dead])
            rows.append({
                "net": net, "site_index": si, "site": site,
                "n_channels": len(s), "n_dead": int(dead.sum()),
                "dead_frac": float(dead.mean()),
                "min_abs_gamma_dead": float(np.min(np.abs(
                    g.a_c.to_numpy()[dead]))) if dead.any() else np.nan,
                "spread_all": spread(g.s_c),
                "spread_live": spread(live) if len(live) >= 2 else np.nan,
                "activation_site": site in act,
                **{f"below_{c:g}_frac": float((s < c * med).mean())
                   for c in SWEEP},
                **{f"spread_above_{c:g}": spread(pd.Series(s[s >= c * med]))
                   for c in SWEEP},
            })
    df = pd.DataFrame(rows)
    summ = df.groupby("net").agg(
        sites=("site", "count"),
        sites_with_dead=("n_dead", lambda x: int((x > 0).sum())),
        dead_channels=("n_dead", "sum"),
        channels=("n_channels", "sum"),
        median_spread_all=("spread_all", "median"),
        median_spread_live=("spread_live", "median"))
    summ["dead_pct"] = 100 * summ.dead_channels / summ.channels
    print(summ.round(3).to_string())
    v = summ.loc["cifar10_mobilenetv2_x1_0"]
    verdict = (f"cifar10_mobilenetv2_x1_0: {int(v.dead_channels)} of "
               f"{int(v.channels)} channels dead ({v.dead_pct:.1f}%) on "
               f"{int(v.sites_with_dead)}/{int(v.sites)} sites; median spread "
               f"{v.median_spread_all:.4g} with them, "
               f"{v.median_spread_live:.3f} without")
    write_table("exp2b_dead_channels", df, {
        "experiment": "EXP 2b dead channels", "dead_rel": DEAD_REL}, verdict)
    write_table("exp2b_dead_channels_summary", summ.reset_index(), {
        "experiment": "EXP 2b summary"})
    sweep_cols = [c for c in df.columns if c.startswith(("below_", "spread_above_"))]
    sw = df[df.activation_site].groupby("net")[sweep_cols].median()
    print("\nactivation sites only, medians over sites:")
    print(sw.round(3).to_string())
    write_table("exp2b_threshold_sweep", sw.reset_index(), {
        "experiment": "EXP 2b threshold sweep, activation sites only"})
    banner([verdict])


if __name__ == "__main__":
    main()
