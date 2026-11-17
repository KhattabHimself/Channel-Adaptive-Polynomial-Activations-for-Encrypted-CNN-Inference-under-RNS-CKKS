"""
Can ANY grouping save a level in a two-site region? (extension)

The rotation rate of the thesis is measured for one grouping, the sorted one.
A grouping chosen for levels could do better, and the thesis lists that as a
limit. This removes the limit with an exact test that holds for every
grouping.

Setting, as in thesis section 4.9: a region with two activation sites, n
channels, G groups of at most cap = ceil(n/G) channels. A group uses, at each
site, the levels of the largest degree it holds there. The region costs
    L_group = max over groups of (levels at site 1 + levels at site 2)
against L_layer = (largest at site 1) + (largest at site 2) without grouping.

Let T1 be the set of channels whose degree at site 1 is in the top level
class of site 1, and T2 the same at site 2.

Fact 1. A grouping saves at least one level if and only if no group contains
        a channel of T1 and a channel of T2.
        (L_group = L_layer exactly when some group is top at both sites.)
Fact 2. Such a grouping exists if and only if T1 and T2 share no channel and
        ceil(|T1|/cap) + ceil(|T2|/cap) <= G.
        (Put T1 into its own groups, T2 into its own, the rest anywhere.)

So "separable" below is an exact yes or no for all groupings at once, read
directly from the degrees. The script also records what the sorted grouping
of the thesis actually saves, and, as a reference, the probability that two
random sets of the same sizes would be disjoint.

Weights only, no images. Same degree rule, budgets and depth models as the
rotation experiment, so the two can be compared case by case.

run (from code/):
    python -m experiments.spread_and_rotation.any_grouping_bound
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import src.run_stamp as _stamp  # noqa: E402
from src.network_tools import (DELTA_MODELS, channel_table, degrees_at,  # noqa: E402
                               depthwise_regions, load_model)
from src.run_stamp import banner, write_table  # noqa: E402

NETS = ["imagenet_mobilenetv2", "imagenet_mnasnet1_0", "cifar10_mobilenetv2_x1_0"]
BUDGETS = [0.003, 0.01, 0.03, 0.1, 0.3]
GROUPS = [2, 4, 8]


def p_disjoint(n: int, k1: int, k2: int) -> float:
    """probability that two random subsets of sizes k1, k2 of n items are disjoint"""
    if k1 + k2 > n:
        return 0.0
    lc = lambda a, b: gammaln(a + 1) - gammaln(b + 1) - gammaln(a - b + 1)  # noqa: E731
    return float(np.exp(lc(n - k1, k2) - lc(n, k2)))


def main() -> None:
    _stamp.use_results("spread_and_rotation")
    banner(["CAN ANY GROUPING SAVE A LEVEL? exact test for two-site regions"])
    rows = []
    for net in NETS:
        model = load_model(net)
        table = channel_table(model, net)
        regions = depthwise_regions(model)
        for B in BUDGETS:
            df = degrees_at(table, B)
            for block, (s1, s2) in regions.items():
                a = df[df.site == s1].sort_values("channel_idx")
                b = df[df.site == s2].sort_values("channel_idx")
                if len(a) != len(b) or len(a) == 0:
                    continue
                d1, d2 = a.degree.to_numpy(), b.degree.to_numpy()
                n = len(d1)
                order = np.argsort(-np.maximum(d1, d2), kind="stable")
                for mname, delta in DELTA_MODELS.items():
                    l1 = np.array([delta(int(x)) for x in d1])
                    l2 = np.array([delta(int(x)) for x in d2])
                    T1, T2 = l1 == l1.max(), l2 == l2.max()
                    k1, k2, both = int(T1.sum()), int(T2.sum()), int((T1 & T2).sum())
                    for G in GROUPS:
                        if n < G:
                            continue
                        cap = math.ceil(n / G)
                        separable = both == 0 and math.ceil(k1 / cap) + math.ceil(k2 / cap) <= G
                        buckets = np.array_split(order, G)
                        f1 = [l1[bk].max() for bk in buckets]
                        f2 = [l2[bk].max() for bk in buckets]
                        saved = int(l1.max() + l2.max() - max(x + y for x, y in zip(f1, f2)))
                        rows.append({
                            "net": net, "budget_B": B, "block": block, "delta_model": mname,
                            "G": G, "n_channels": n, "top_class_site1": k1, "top_class_site2": k2,
                            "top_at_both_sites": both, "levels_site1": int(l1.max()),
                            "levels_site2": int(l2.max()),
                            "separable_by_some_grouping": bool(separable),
                            "sorted_grouping_rotates": bool(np.argmax(f1) != np.argmax(f2)),
                            "levels_saved_by_sorted_grouping": saved,
                            "p_disjoint_if_unrelated": p_disjoint(n, k1, k2),
                        })
    out = pd.DataFrame(rows)
    s = out.groupby(["net", "delta_model", "G"]).agg(
        cases=("separable_by_some_grouping", "size"),
        any_grouping_can_save_pct=("separable_by_some_grouping", lambda x: 100 * x.mean()),
        share_a_channel_is_top_at_both_pct=("top_at_both_sites", lambda x: 100 * (x > 0).mean()),
        sorted_rotates_pct=("sorted_grouping_rotates", lambda x: 100 * x.mean()),
        sorted_saves_a_level_pct=("levels_saved_by_sorted_grouping", lambda x: 100 * (x > 0).mean()),
        unrelated_reference_pct=("p_disjoint_if_unrelated", lambda x: 100 * x.mean()),
        median_channels=("n_channels", "median"),
        median_top_class_size_site1=("top_class_site1", "median"),
        median_top_class_size_site2=("top_class_site2", "median"),
    ).reset_index()
    print(s.round(1).to_string(index=False), flush=True)
    hi = s.any_grouping_can_save_pct.max()
    verdict = (f"Share of (region, budget) cases in which SOME grouping can save a level: at most "
               f"{hi:.1f}% over the {len(s)} (network, depth model, G) cells; a channel is in the "
               f"top level class at both sites in {s.share_a_channel_is_top_at_both_pct.min():.0f} to "
               f"{s.share_a_channel_is_top_at_both_pct.max():.0f}% of cases. The sorted grouping "
               f"saves a level in at most {s.sorted_saves_a_level_pct.max():.1f}% of cases.")
    write_table("ext_any_grouping_cases", out, {"experiment": "any-grouping bound (extension)"}, verdict)
    write_table("ext_any_grouping_summary", s, {"experiment": "any-grouping bound (extension)"}, verdict)
    banner([verdict])


if __name__ == "__main__":
    main()
