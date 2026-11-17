"""
PURPOSE
    Turn the grouped WRN-28-10 design question into a measured claim.

    LAB_NOTEBOOK Sec. 5.4's reversal argues that region LENGTH, not G, ranks
    conjunct (2), and that grouped WRN-28-10 therefore dominates both
    MobileNetV2 (same G, 3x the region) and grouped WRN-16-10 (same G and
    width, 3x the region). Until now that was prose plus a capacity row. This
    runs the Phase 5 solver over the real stage geometry to produce numbers.

    SYNTHETIC, AND LABELLED SO. No pretrained WRN-28-10 checkpoint exists in
    this tree, so per-channel degrees are drawn rather than fitted. They are
    NOT arbitrary: the draw is calibrated to what Gate B actually measured
    [reports/PHASE_3_REPORT.md] —

      * plateau spread of ONE class within a site (median, both delta models),
      * p90/p10 spread in the underlying invariant of ~2x (MobileNetV2, the
        one architecture that passes both conjuncts),
      * argmax-bucket rotation between sites at the measured rate p.

    So the output is "what the solver returns on a network shaped like grouped
    WRN-28-10 whose degree statistics look like the ones we measured
    elsewhere" — a conditional claim with its condition stated, not a
    measurement of WRN-28-10. Replacing the draw with a fitted distribution
    needs a trained checkpoint and is the obvious follow-up.

INPUTS
    results/phase2/G_table.csv (stage geometry: C, G, cap per stage)

OUTPUTS
    results/phase5/grouped_wrn_synthetic.csv — per (network, stage, G, trial):
        placement cost, layerwise baseline, uniform baseline, saving,
        region levels under independent vs realigned accounting
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.provenance import require_clean, write_csv  # noqa: E402
from src.levels import DELTA_MODELS  # noqa: E402
from src.grouping_solver import (  # noqa: E402
    G_TABLE, KAPPA_MODELS, baseline_layerwise, baseline_uniform,
    dp_place_shared, load_stages,
)

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO.parent / "results"

# Region length in activation sites, from capacity.py's mixfree_region model.
# Both networks are grouped, so a region runs the length of the stage; the
# entry block is excluded where a projection shortcut mixes.
REGION_SITES = {"wrn28_10_g4": 6, "wrn16_10_g4": 2, "mobilenetv2": 2}

# Degree alphabet: one plateau class of spread is what Gate B measured, so the
# draw spans exactly that — 5..8 sits inside delta=4, 9..16 inside delta=5.
LOW, HIGH = 7, 15


def draw_site(n_channels: int, rng: random.Random, p_high: float) -> list[int]:
    """Per-channel degrees for one activation site."""
    return [HIGH if rng.random() < p_high else LOW for _ in range(n_channels)]


def draw_region(n_channels: int, n_sites: int, rng: random.Random,
                p_high: float, p_rotate: float) -> list[list[int]]:
    """One mixing-free region: n_sites correlated draws over the same channels.

    Sites within a region share channel identity (a depthwise or grouped
    convolution preserves it), so their degrees are correlated. p_rotate is the
    measured probability that a site re-draws which channels are the demanding
    ones — the conjunct-(3) rate from Gate B.
    """
    first = draw_site(n_channels, rng, p_high)
    sites = [first]
    for _ in range(n_sites - 1):
        if rng.random() < p_rotate:
            sites.append(draw_site(n_channels, rng, p_high))   # rotated
        else:
            sites.append(list(first))                          # same argmax
    return sites


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--networks", nargs="+",
                    default=["wrn28_10_g4", "wrn16_10_g4", "mobilenetv2"])
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--p-high", type=float, nargs="+",
                    default=[0.02, 0.05, 0.10, 0.25],
                    help="fraction of channels needing the higher degree; "
                         "swept, because concentration is what decides "
                         "whether region length converts into a saving")
    ap.add_argument("--p-rotate", type=float, default=0.28,
                    help="conjunct-(3) rotation rate; Gate B measured "
                         "0.06 (G=2) to 0.28 (G=8)")
    ap.add_argument("--kappa", choices=sorted(KAPPA_MODELS),
                    default="sqrt2d")
    ap.add_argument("--seed", type=int, default=20260822)
    args = ap.parse_args()
    require_clean()

    kappa = KAPPA_MODELS[args.kappa]
    rng = random.Random(args.seed)
    rows = []

    for net in args.networks:
        n_sites = REGION_SITES[net]
        for stage in load_stages(G_TABLE, net, 65536):
            if stage.G < 2:
                continue
            for p_high, trial in ((p, t) for p in args.p_high
                                  for t in range(args.trials)):
                sites = draw_region(stage.channels, n_sites, rng,
                                    p_high, args.p_rotate)
                placed = dp_place_shared(sites, stage.G, stage.cap, kappa)
                lay = baseline_layerwise(sites, stage.G, kappa)
                uni = baseline_uniform(sites, stage.G, kappa)
                row = {
                    "network": net, "stage": stage.name, "C": stage.channels,
                    "G": stage.G, "region_sites": n_sites, "trial": trial,
                    "kappa": args.kappa, "p_high": p_high,
                    "p_rotate": args.p_rotate,
                    "placement_cost": placed.cost,
                    "layerwise_cost": lay, "uniform_cost": uni,
                    "mult_saving_vs_layerwise":
                        (lay - placed.cost) / lay if lay else 0.0,
                }
                # Region-level accounting over ALL n_sites. The two-site
                # helpers in dp_solver take act1/act2 only, which would throw
                # away exactly the region length this experiment is about, so
                # the region form is computed here:
                #
                #   placement : max_j  sum_s delta(m[j][s])
                #       buckets evolve independently inside the region, so the
                #       region costs what its DEEPEST bucket costs.
                #   layerwise : sum_s  max_j delta(m[j][s])
                #       one degree per site, so every bucket pays the site max.
                #
                # The difference is the region saving, bounded above by
                # (n_sites - 1) * (f_max - f_min) [IMPOSSIBILITY Sec. 4.2].
                bm = placed.bucket_maxes            # [bucket][site] -> degree
                for name, model in DELTA_MODELS.items():
                    lv = [[model(bm[j][s]) for s in range(n_sites)]
                          for j in range(len(bm))]
                    place = max(sum(lv[j]) for j in range(len(lv)))
                    layer = sum(max(lv[j][s] for j in range(len(lv)))
                                for s in range(n_sites))
                    row[f"region_levels_placement_{name}"] = place
                    row[f"region_levels_layerwise_{name}"] = layer
                    row[f"region_level_saving_{name}"] = layer - place
                rows.append(row)

    df = pd.DataFrame(rows)
    out = RESULTS / "grouping_multiplications/grouped_wrn_synthetic.csv"
    write_csv(df, out, extra={"SYNTHETIC": "degrees drawn, not fitted; "
                              "calibrated to Gate B statistics",
                              "seed": args.seed})

    print("=== solver on grouped WRN geometry, SYNTHETIC degrees ===")
    print(df.groupby(["network", "region_sites", "stage"]).agg(
        G=("G", "first"),
        mult_saving=("mult_saving_vs_layerwise", "mean"),
        lvl_place_B=("region_levels_placement_B", "mean"),
        lvl_layer_B=("region_levels_layerwise_B", "mean"),
        lvl_saved_B=("region_level_saving_B", "mean")).to_string())
    print("\nPER NETWORK — the quantity region length controls:")
    print(df.groupby(["network", "region_sites"]).agg(
        mult_saving=("mult_saving_vs_layerwise", "mean"),
        region_levels_saved_A=("region_level_saving_A", "mean"),
        region_levels_saved_B=("region_level_saving_B", "mean")).to_string())
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
