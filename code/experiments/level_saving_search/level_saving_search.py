"""
PURPOSE
    Settle IMPOSSIBILITY_AND_ESCAPE Sec. 4.7: is the non-conversion of the
    region bound S(R) STRUCTURAL, or an artifact of optimising the wrong
    objective?

    Result IV measured S(R) = 0 using dp_place_shared, which minimises
        multiplication objective:  sum_j kappa(max_j degree)
    But the level cost of a region is
        level objective:           max_j sum_s delta(m[j][s])
    and minimising a sum of maxima is not the same as minimising a maximum of
    sums. The measured zero may therefore be the saving of a placement that was
    never trying to obtain it.

    ONE POSITIVE INSTANCE REFUTES "STRUCTURAL". So this exhaustively enumerates
    every partition of small channel sets and maximises S(R) directly.

METHOD
    For C channels, G buckets, cap on bucket size, n sites:
        L_place(P) = max_j  sum_s delta(max degree in block j at site s)
        L_layer    = sum_s  max over ALL channels delta(degree at site s)
        S(R)(P)    = L_layer - L_place(P)
    Enumerate every set partition into exactly G non-empty blocks respecting
    the cap; report max_P S(R), and separately what the multiplication-optimal
    placement achieves on the same instance.

    Two instance families:
      random      degrees drawn per site, sweeping concentration
      adversarial constructed to be ANTI-CORRELATED across sites — channel A
                  demanding at site 1 and cheap at site 2, channel B the
                  reverse. If any structure realises the bound, this is it.

INPUTS
    None.

OUTPUTS
    results/phase5/level_objective_probe.csv — per instance:
        family, C, G, cap, n_sites, seed, L_layer,
        best_level_placement, best_S, mult_optimal_S, bound
"""

from __future__ import annotations

import argparse
import itertools
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.provenance import require_clean, write_csv  # noqa: E402
from src.levels import DELTA_MODELS  # noqa: E402
from src.grouping_solver import KAPPA_MODELS, dp_place_shared  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO.parent / "results"


def partitions_into(n_items: int, n_blocks: int, cap: int):
    """Every partition of range(n_items) into exactly n_blocks non-empty
    blocks of size <= cap. Restricted-growth strings, canonical so each
    partition appears once."""
    assign = [0] * n_items

    def rec(i: int, used: int):
        if i == n_items:
            if used == n_blocks:
                blocks = [[] for _ in range(n_blocks)]
                for c, b in enumerate(assign):
                    blocks[b].append(c)
                if all(1 <= len(b) <= cap for b in blocks):
                    yield blocks
            return
        # remaining items cannot open enough new blocks
        if n_blocks - used > n_items - i:
            return
        for b in range(min(used + 1, n_blocks)):
            assign[i] = b
            yield from rec(i + 1, used + (1 if b == used else 0))

    yield from rec(0, 0)


def level_cost(blocks, sites, delta_fn) -> int:
    """max_j sum_s delta(max degree in block j at site s)."""
    return max(sum(delta_fn(max(sites[s][c] for c in blk))
                   for s in range(len(sites)))
               for blk in blocks)


def layerwise_cost(sites, delta_fn) -> int:
    """sum_s max over all channels delta(degree at site s)."""
    return sum(delta_fn(max(site)) for site in sites)


def probe(sites, G: int, cap: int, delta_fn, kappa):
    C = len(sites[0])
    L_layer = layerwise_cost(sites, delta_fn)
    best_S, best_blocks = -1, None
    for blocks in partitions_into(C, G, cap):
        S = L_layer - level_cost(blocks, sites, delta_fn)
        if S > best_S:
            best_S, best_blocks = S, blocks
    # what the multiplication-objective solver gives on the same instance
    placed = dp_place_shared(sites, G, cap, kappa)
    order, sizes = placed.order, placed.sizes
    mult_blocks, k = [], 0
    for sz in sizes:
        mult_blocks.append(list(order[k:k + sz]))
        k += sz
    mult_S = L_layer - level_cost(mult_blocks, sites, delta_fn)
    return L_layer, best_S, mult_S, best_blocks


def adversarial(C: int, n_sites: int, low: int, high: int):
    """Anti-correlated demand: each channel is demanding at exactly one site."""
    return [[high if c % n_sites == s else low for c in range(C)]
            for s in range(n_sites)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-channels", type=int, default=9)
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--seed", type=int, default=20260823)
    ap.add_argument("--kappa", choices=sorted(KAPPA_MODELS), default="sqrt2d")
    args = ap.parse_args()
    require_clean()

    kappa = KAPPA_MODELS[args.kappa]
    rng = random.Random(args.seed)
    rows = []

    for name, delta_fn in DELTA_MODELS.items():
        for C in range(4, args.max_channels + 1):
            for G in (2, 3):
                if G > C:
                    continue
                cap = -(-C // G) + 1          # ceil(C/G), one slot of slack
                for n_sites in (2, 3):
                    # adversarial family — one instance per shape
                    sites = adversarial(C, n_sites, 1, 15)
                    L, bS, mS, _ = probe(sites, G, cap, delta_fn, kappa)
                    rows.append({"family": "adversarial", "delta_model": name,
                                 "C": C, "G": G, "cap": cap,
                                 "n_sites": n_sites, "trial": 0,
                                 "L_layer": L, "best_S": bS,
                                 "mult_optimal_S": mS})
                    # random family
                    for t in range(args.trials):
                        sites = [[rng.choice([1, 3, 7, 15])
                                  for _ in range(C)] for _ in range(n_sites)]
                        L, bS, mS, _ = probe(sites, G, cap, delta_fn, kappa)
                        rows.append({"family": "random", "delta_model": name,
                                     "C": C, "G": G, "cap": cap,
                                     "n_sites": n_sites, "trial": t,
                                     "L_layer": L, "best_S": bS,
                                     "mult_optimal_S": mS})

    df = pd.DataFrame(rows)
    write_csv(df, RESULTS / "level_saving_search/level_objective_probe.csv",
              extra={"question": "IMPOSSIBILITY 4.7 structural vs unoptimised",
                     "seed": args.seed})

    print("=== Sec. 4.7: can a LEVEL-objective placement realise S(R) > 0? ===")
    print(f"instances enumerated: {len(df)}")
    print(f"\ninstances with best_S > 0 (level objective): "
          f"{(df.best_S > 0).sum()} / {len(df)}")
    print(f"instances with mult_optimal_S > 0: "
          f"{(df.mult_optimal_S > 0).sum()} / {len(df)}")
    print("\nby family:")
    print(df.groupby(["family", "delta_model"]).agg(
        n=("best_S", "size"),
        frac_level_positive=("best_S", lambda s: (s > 0).mean()),
        max_level_S=("best_S", "max"),
        mean_level_S=("best_S", "mean"),
        frac_mult_positive=("mult_optimal_S", lambda s: (s > 0).mean()),
        max_mult_S=("mult_optimal_S", "max")).round(3).to_string())

    verdict = "REFUTED — level-objective placement realises S(R) > 0" \
        if (df.best_S > 0).any() else "NOT REFUTED on these instances"
    print(f"\n>>> 'non-conversion is structural': {verdict}")


if __name__ == "__main__":
    main()
