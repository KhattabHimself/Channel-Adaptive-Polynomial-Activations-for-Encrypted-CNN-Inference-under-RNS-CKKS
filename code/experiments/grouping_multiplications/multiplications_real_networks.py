"""
PURPOSE
    Run the placement solver on REAL per-channel degrees from Gate B, not on a
    calibrated synthetic distribution.

    The multiplication saving is now the last positive claim the thesis makes
    (O-8 removed the level and bootstrap claims), and until this script it
    rested on drawn degrees [src/phase5/synthetic_grouped_wrn.py]. A headline
    number resting on a distribution we chose is not a measurement, however
    carefully the draw was calibrated.

METHOD
    1. Real degrees: invariants.py assigns each channel the minimal degree
       meeting the decided budget, from the pretrained checkpoint's BN
       statistics and next-layer weight norms.
    2. Real geometry: one forward pass with hooks records every activation
       site's actual (C, H, W). G is then computed per site with capacity.py's
       multiplexed model — occupied = base_grid * ceil(C*H*W / base_grid),
       G = ceil(occupied / slots) — rather than read from a stage table, so
       every site gets its own G instead of a stage-level approximation.
    3. Placement: sites are grouped into mixing-free regions (a region shares
       one permutation). dp_place_shared solves each region; the layerwise and
       uniform baselines are computed on the same degrees.
    4. Swept over the budget B, because B is a free parameter and a saving
       that only exists at one B is not a result.

    Degrees are real; the ARCHITECTURE choice is still ours, and ResNet-20 has
    G = 1 everywhere (Gate A), so its placement problem is degenerate by
    construction and is reported as the null it is.

INPUTS
    Pretrained checkpoints via torch.hub / torchvision.

OUTPUTS
    results/phase5/real_degrees_placement.csv — per (net, budget, region, G):
        placement_cost, layerwise_cost, uniform_cost, mult_saving_vs_layerwise,
        n_channels, n_sites, degree_min/median/max
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.provenance import require_clean, write_csv  # noqa: E402
from experiments.spread_and_rotation.spread_first_networks import regions_mobilenetv2  # noqa: E402
from src.channel_degrees import bn_sites, invariants, load_model  # noqa: E402
from src.grouping_solver import (  # noqa: E402
    KAPPA_MODELS, baseline_layerwise, baseline_uniform, dp_place_shared,
)

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO.parent / "results"
BUDGETS = [10.0, 3.0, 1.0, 0.3, 0.1, 0.03, 0.01]
SLOTS = 32768                       # N = 2^16
BASE_GRID = {"resnet20": 32 * 32, "mobilenetv2": 112 * 112}
INPUT_SHAPE = {"resnet20": (1, 3, 32, 32), "mobilenetv2": (1, 3, 224, 224)}


def site_shapes(net: str) -> dict[str, tuple[int, int, int]]:
    """(C, H, W) at every BN site, from one real forward pass."""
    model = load_model(net)
    shapes: dict[str, tuple[int, int, int]] = {}
    handles = []
    for name, bn, _ in bn_sites(model):
        def hook(_m, _i, out, name=name):
            shapes[name] = tuple(out.shape[1:])
        handles.append(bn.register_forward_hook(hook))
    with torch.no_grad():
        model(torch.zeros(*INPUT_SHAPE[net]))
    for h in handles:
        h.remove()
    return shapes


def site_G(net: str, shape: tuple[int, int, int]) -> int:
    C, H, W = shape
    occupied = BASE_GRID[net] * math.ceil(C * H * W / BASE_GRID[net])
    return math.ceil(occupied / SLOTS)


def regions_for(net: str, df: pd.DataFrame) -> dict[str, list[str]]:
    """Mixing-free regions as {region_id: [site names]}.

    MobileNetV2: the two Conv2dNormActivation sites of an inverted residual,
    per capacity.py's mixfree_region model. ResNet-20: every dense convolution
    mixes, so each site is its own region of length 1 — the n = 1 case where
    the level saving is identically zero and only multiplications remain.
    """
    if net == "mobilenetv2":
        regions = {f"block{b}": sites
                   for b, sites in regions_mobilenetv2(df).items()}
        covered = {s for v in regions.values() for s in v}
        for s in df.site.unique():
            if s not in covered:
                regions[f"solo_{s}"] = [s]
        return regions
    return {f"solo_{s}": [s] for s in df.site.unique()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nets", nargs="+",
                    default=["mobilenetv2", "resnet20"])
    ap.add_argument("--budgets", type=float, nargs="+", default=BUDGETS,
                    help="network-wide B; swept because B is free "
                         "and a saving that exists at one B is not "
                         "a result")
    ap.add_argument("--kappa", choices=sorted(KAPPA_MODELS), default="sqrt2d")
    args = ap.parse_args()
    require_clean()

    kappa = KAPPA_MODELS[args.kappa]
    rows = []
    for net in args.nets:
        shapes = site_shapes(net)
        for B in args.budgets:
            df = invariants(net, B)
            deg = defaultdict(dict)
            for r in df.itertuples():
                deg[r.site][r.channel_idx] = int(r.degree)
            for rid, site_names in regions_for(net, df).items():
                sites = [[deg[s][c] for c in sorted(deg[s])]
                         for s in site_names if s in deg]
                if not sites or len({len(x) for x in sites}) != 1:
                    continue                     # ragged region: skip, report
                C = len(sites[0])
                G = max(site_G(net, shapes[s]) for s in site_names
                        if s in shapes)
                cap = math.ceil(C / G) if G else C
                flat = [d for s in sites for d in s]
                row = {
                    "net": net, "budget_B": B, "region": rid,
                    "n_sites": len(sites), "C": C, "G": G, "cap": cap,
                    "degree_min": min(flat),
                    "degree_median": float(pd.Series(flat).median()),
                    "degree_max": max(flat),
                    "kappa": args.kappa,
                }
                if G < 2:
                    # G = 1: one bucket, so the bucket max IS the layer max and
                    # placement is identically the layerwise baseline. Recorded
                    # rather than skipped — it is Gate A's FAIL, per site.
                    row.update(placement_cost=float("nan"),
                               layerwise_cost=float("nan"),
                               uniform_cost=float("nan"),
                               mult_saving_vs_layerwise=0.0,
                               degenerate=True)
                else:
                    placed = dp_place_shared(sites, G, cap, kappa)
                    lay = baseline_layerwise(sites, G, kappa)
                    uni = baseline_uniform(sites, G, kappa)
                    row.update(
                        placement_cost=placed.cost, layerwise_cost=lay,
                        uniform_cost=uni,
                        mult_saving_vs_layerwise=
                            (lay - placed.cost) / lay if lay else 0.0,
                        degenerate=False)
                rows.append(row)

    out = pd.DataFrame(rows)
    write_csv(out, RESULTS / "grouping_multiplications/real_degrees_placement.csv",
              extra={"degrees": "REAL, from pretrained BN statistics",
                     "slots": SLOTS})

    live = out[~out.degenerate]
    print("=== REAL degrees: multiplication saving vs layerwise ===")
    if len(live):
        print(live.groupby(["net", "budget_B"]).agg(
            regions=("region", "count"),
            mean_saving=("mult_saving_vs_layerwise", "mean"),
            median_saving=("mult_saving_vs_layerwise", "median"),
            max_saving=("mult_saving_vs_layerwise", "max"),
            median_G=("G", "median"),
            median_dmax=("degree_max", "median")).round(4).to_string())
    print("\ndegenerate (G=1, placement == layerwise) regions per net:")
    print(out.groupby("net").degenerate.agg(["sum", "count"]).to_string())


if __name__ == "__main__":
    main()
