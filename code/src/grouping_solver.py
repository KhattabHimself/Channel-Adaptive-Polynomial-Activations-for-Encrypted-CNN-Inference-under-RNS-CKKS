#!/usr/bin/env python3
"""
PURPOSE
    Phase 5 core — the channel-placement solver, network-independent.

    Channels sharing a ciphertext must share one polynomial, so a bucket costs
    what its MAXIMUM required degree costs. Channel order is a free
    compile-time plaintext permutation, so the question is which channels
    co-reside. This module answers that exactly: sort by required degree, then
    DP over cut points.

    `fit_degrees.py` is NOT in this module and is NOT written yet — it needs
    D-GateA (which network) and the error-budget sign-off (which budget), both
    open [LAB_NOTEBOOK Sec.6 O-6, O-7]. Everything here is validated on
    SYNTHETIC degree distributions and refuses to invent real ones.

OPTIMALITY — the exchange argument, with its assumptions

    Setting. Channels c = 1..C carry required degrees d_c. A placement is a
    partition into buckets B_1..B_G (one ciphertext each) with |B_j| <= cap.
    Cost  J(B) = sum_j kappa( max_{c in B_j} d_c ).

    Assumptions.
      A1  kappa is non-decreasing on the degrees.
      A2  A bucket's cost depends on its contents ONLY through the maximum
          degree. (Mixing degrees inside a ciphertext is ruled out
          independently: under Paterson-Stockmeyer sum_g m(d_g) >= m(d_max),
          so masking always loses — LAB_NOTEBOOK Sec.1.3.)
      A3  Bucket capacities are fixed in advance and do not depend on
          contents (slot capacity is set by (N, H, W), not by degree).
      A4  Exactly one activation site is being placed; its permutation is not
          shared with any other site.
      A5  Channel order is a free compile-time plaintext permutation.

    Claim. Under A1-A5 some optimal placement has every bucket equal to a
    contiguous interval of the degree-sorted channel sequence.

    Proof. Take an optimal placement B that is not sorted-consistent: there
    are buckets p != q and channels x in B_p, y in B_q with d_x < d_y and
    M_p := max(B_p) > M_q := max(B_q). Swap x and y; bucket sizes are
    preserved, so the placement stays feasible by A3.
        B_p' = B_p \\ {x} u {y}: every remaining element is <= M_p, and
                                 d_y <= M_q < M_p, so M_p' <= M_p.
        B_q' = B_q \\ {y} u {x}: every remaining element is <= M_q, and
                                 d_x < d_y <= M_q, so M_q' <= M_q.
    By A1 and A2 neither bucket's cost rises, so J does not increase. The swap
    strictly decreases the number of pairs placed against sorted order, a
    non-negative integer, so the process terminates. A placement with no such
    pairs puts every channel of a lower-max bucket below every channel of a
    higher-max bucket, i.e. the buckets are intervals of the sorted order. QED

    Consequence. The permutation is settled by sorting; the ONLY remaining
    freedom is where to cut, and DP over cut points is exact. `dp_place`
    implements that; `brute_force_place` checks it (test_dp_vs_bruteforce).

    Where the argument FAILS — and these are the interesting cases:
      * A4 fails at a residual junction. Branches meeting at a shortcut are
        added slot-wise, so they share ONE permutation. The objective becomes
        sum_s sum_j kappa(max_{B_j} d^s_c) over the sites s of the group, and
        no single sort key is optimal in general: a swap that helps one site
        can hurt another. `dp_place_shared` is therefore exact only in the
        cuts, given an order; the order is a heuristic and its cost is an
        UPPER BOUND on the constrained optimum. `brute_force_shared` measures
        the slack. This gap is RQ4.
      * A1 fails for a non-monotone kappa. All three models here are monotone
        and `test_all_kappa_models_are_monotone` asserts it.
      * A2 fails under slot masking, which is ruled out above.

PLATEAU CLASSES
    delta(d) is a step function [LAB_NOTEBOOK Sec.4.6]. Collapsing each degree
    to the top of its delta-plateau BEFORE solving
      * preserves the LEVEL objective exactly (delta(rep(d)) == delta(d), and
        rep is non-decreasing so rep(max) == max(rep)); and
      * turns the MULTIPLICATION objective into an upper bound, because
        kappa(rep(d)) >= kappa(d) strictly inside a plateau.
    So collapse is free for levels and lossy for kappa. Dispersion inside one
    plateau is worth exactly zero for levels.

OPEN — O-8, raised by this module, NOT resolved here
    Under multiplexed packing a convolution mixes every input channel into
    every output channel, so the G ciphertexts of a layer must be aligned to a
    common level before the next convolution; the backend mod_drops the deeper
    ones down [LAB_NOTEBOOK Sec.4.4 V2]. If that is what happens, the level
    consumed at a site is a(max over ALL channels of the layer) whatever the
    placement, and PLACEMENT BUYS EXACTLY ZERO LEVELS — the level lever is
    layerwise, which is what OLA-approx already takes [SLR row 76], and
    placement acts only on the multiplication count sum_j kappa. Both level
    models are computed here and reported side by side; no bootstrap claim is
    made until O-8 is closed. See reports/PHASE_5_REPORT.md Sec.5.

INPUTS
    results/phase2/G_table.csv   per-stage G, channels, headroom (--model)
    a degrees CSV                layer, channel, degree   (--degrees)
                                 does not exist yet; the CLI refuses rather
                                 than fabricating one (PROTOCOL rule 2)

OUTPUTS
    results/phase5/synthetic_validation.csv
        trial, generator, seed, n_channels, n_buckets, cap, d_max,
        kappa_model, delta_model, collapse, dp_cost, brute_cost, agree,
        uniform_cost, layerwise_cost, placed_unconstrained,
        placed_constrained_ub, constrained_optimum, heuristic_slack,
        rq4_gap_ub, levels_realigned, levels_independent_min,
        levels_independent_max, bootstraps_realigned, bootstraps_independent
    library API otherwise; no real-network result files until D-GateA closes.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.provenance import require_clean, write_csv  # noqa: E402
from src.levels import (  # noqa: E402
    C1_BASELINE_CONV, C2_SVD_CONV, C_BN, DELTA_MODELS, S_SHORTCUT, a,
)

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO.parent / "results"
G_TABLE = RESULTS / "ciphertexts_per_layer" / "G_table.csv"


# ── kappa: non-scalar multiplications per polynomial evaluation ─────────────
# PLUGGABLE (06_PHASE5_solver.md). The solver's correctness needs only
# monotonicity (assumption A1); which model the thesis quotes is a reporting
# decision and is NOT made here.
#
# UNVERIFIED: Paterson & Stockmeyer (1973) does not resolve to an SLR row or a
# corpus PDF [corpus/slr_index.json, 166 records, no match on "paterson" or
# "stockmeyer"]. The sqrt(2d) form is quoted from this project's own Phase 5
# brief [prompts/06_PHASE5_solver.md], not from a corpus source, and none of
# these counts has been checked against a reference implementation
# (PROTOCOL rule 8: theory is not verification).

def kappa_sqrt2d(d: int) -> int:
    """The ~sqrt(2d) proxy named in the Phase 5 brief. Asymptotic only."""
    if d < 1:
        raise ValueError(f"degree must be >= 1, got {d}")
    return math.ceil(math.sqrt(2 * d))


def kappa_bsgs_upper(d: int) -> int:
    """Explicit baby-step/giant-step UPPER bound, minimised over the baby step.

    Construction, so the number is auditable rather than asserted: write
    p(x) = sum_{i < 2^j} q_i(x) * x^(i*k) with deg q_i <= k-1 and
    k * 2^j >= d+1, then count non-scalar multiplications as
        baby steps  x^2 .. x^(k-1)   max(0, k-2)
        giant steps x^k, x^2k, ...   j
        recombine   binary tree      2^j - 1
    This OVERCOUNTS whenever a block q_i is constant, so it is an upper bound
    on Paterson-Stockmeyer, not PS itself.
    """
    if d < 1:
        raise ValueError(f"degree must be >= 1, got {d}")
    best = None
    for k in range(1, d + 2):
        j = 0
        while k * (2 ** j) < d + 1:
            j += 1
        cost = max(0, k - 2) + j + (2 ** j - 1)
        best = cost if best is None else min(best, cost)
    return best


def kappa_linear(d: int) -> int:
    """The archive's naive Chebyshev recurrence: one non-scalar multiply per
    degree step [encrypted_activation.py:L156]. Mirrors delta_impl — for
    archive-vs-model comparison only, NEVER a thesis cost model
    (PROTOCOL rule 6)."""
    if d < 1:
        raise ValueError(f"degree must be >= 1, got {d}")
    return d


KAPPA_MODELS = {
    "sqrt2d": kappa_sqrt2d,
    "bsgs": kappa_bsgs_upper,
    "linear": kappa_linear,
}


# ── plateau collapse ───────────────────────────────────────────────────────

def plateau_top(d: int, delta_fn, ceiling: int = 1 << 16) -> int:
    """Largest degree sharing d's delta-plateau. Model-agnostic: found by
    walking up, not by assuming a closed form — delta_A and delta_B partition
    the degree line differently [dk_derivation Sec.10.5]."""
    target = delta_fn(d)
    hi = d
    while hi < ceiling and delta_fn(hi + 1) == target:
        hi += 1
    return hi


def collapse_to_plateaus(degrees: list[int], delta_fn) -> list[int]:
    """Map every degree to the top of its plateau. Exact for the level
    objective, an upper bound for the kappa objective (module docstring)."""
    cache: dict[int, int] = {}
    out = []
    for d in degrees:
        if d not in cache:
            cache[d] = plateau_top(d, delta_fn)
        out.append(cache[d])
    return out


# ── placement ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Placement:
    """One solved placement. Buckets are consecutive runs of `order`."""
    cost: float
    order: tuple[int, ...]      # channel indices, in placement order
    sizes: tuple[int, ...]      # bucket sizes, in placement order
    bucket_maxes: tuple         # per bucket: an int, or a tuple over sites

    @property
    def n_buckets(self) -> int:
        return len(self.sizes)


def _check_feasible(n_channels: int, n_buckets: int, cap: int) -> None:
    if n_buckets < 1:
        raise ValueError("n_buckets must be >= 1")
    if cap < 1:
        raise ValueError("cap must be >= 1")
    if n_buckets > n_channels:
        raise ValueError(
            f"{n_buckets} non-empty buckets cannot hold {n_channels} channels")
    if n_buckets * cap < n_channels:
        raise ValueError(
            f"capacity {n_buckets}x{cap} cannot hold {n_channels} channels")


def _reconstruct(back, n_buckets: int, n: int) -> list[int]:
    sizes, i = [], n
    for j in range(n_buckets, 0, -1):
        ip = back[j][i]
        sizes.append(i - ip)
        i = ip
    sizes.reverse()
    return sizes


def dp_place(degrees: list[int], n_buckets: int, cap: int, kappa) -> Placement:
    """Exact optimum for ONE site (assumptions A1-A5).

    Sorts — optimal by the exchange argument above — then DPs over cut points:
        dp[j][i] = min over i' in [j-1, i-1], i - i' <= cap
                       dp[j-1][i'] + kappa(sorted[i-1])
    because an ascending contiguous run has its maximum at its last element,
    which makes the bucket term independent of i'. O(G * C * cap).
    """
    n = len(degrees)
    _check_feasible(n, n_buckets, cap)
    order = tuple(sorted(range(n), key=lambda i: degrees[i]))
    srt = [degrees[i] for i in order]

    INF = float("inf")
    dp = [[INF] * (n + 1) for _ in range(n_buckets + 1)]
    back = [[-1] * (n + 1) for _ in range(n_buckets + 1)]
    dp[0][0] = 0.0
    for j in range(1, n_buckets + 1):
        for i in range(j, n - (n_buckets - j) + 1):
            cost_last = kappa(srt[i - 1])       # bucket max == last element
            lo = max(j - 1, i - cap)
            best, arg = INF, -1
            for ip in range(lo, i):
                if dp[j - 1][ip] < best:
                    best, arg = dp[j - 1][ip], ip
            if best < INF:
                dp[j][i] = best + cost_last
                back[j][i] = arg

    if dp[n_buckets][n] == INF:
        raise ValueError("no feasible placement — check n_buckets and cap")

    sizes = _reconstruct(back, n_buckets, n)
    maxes, pos = [], 0
    for sz in sizes:
        maxes.append(max(srt[pos:pos + sz]))
        pos += sz
    return Placement(dp[n_buckets][n], order, tuple(sizes), tuple(maxes))


def _restricted_growth(n: int, max_blocks: int):
    """Enumerate set partitions of {0..n-1} with at most max_blocks blocks as
    restricted growth strings — canonical, so no partition appears twice."""
    assign = [0] * n

    def rec(i: int, used: int):
        if i == n:
            yield tuple(assign)
            return
        for b in range(min(used + 1, max_blocks)):
            assign[i] = b
            yield from rec(i + 1, max(used, b + 1))
    yield from rec(0, 0)


def _blocks_of(assign, n_buckets: int, cap: int):
    """Feasible partition as a list of index lists, or None."""
    blocks: dict[int, list[int]] = {}
    for c, b in enumerate(assign):
        blocks.setdefault(b, []).append(c)
    if len(blocks) != n_buckets:
        return None
    vals = list(blocks.values())
    if any(len(v) > cap for v in vals):
        return None
    return vals


def brute_force_place(degrees: list[int], n_buckets: int, cap: int,
                      kappa) -> Placement:
    """Every feasible partition, no sorting assumed. The check on `dp_place`."""
    n = len(degrees)
    _check_feasible(n, n_buckets, cap)
    best_cost, best = float("inf"), None
    for assign in _restricted_growth(n, n_buckets):
        blocks = _blocks_of(assign, n_buckets, cap)
        if blocks is None:
            continue
        cost = sum(kappa(max(degrees[c] for c in v)) for v in blocks)
        if cost < best_cost:
            best_cost, best = cost, blocks
    if best is None:
        raise ValueError("no feasible placement")
    keyed = sorted(best, key=lambda v: max(degrees[c] for c in v))
    order = tuple(c for v in keyed for c in v)
    return Placement(best_cost, order, tuple(len(v) for v in keyed),
                     tuple(max(degrees[c] for c in v) for v in keyed))


# ── residual constraint: one permutation shared by several sites ───────────

def shared_order(sites: list[list[int]]) -> tuple[int, ...]:
    """Heuristic permutation for a residual group: sort by the pointwise max
    degree across the group's sites, with the sum as a tie-break.

    HEURISTIC, not optimal — assumption A4 fails for a shared permutation, so
    no single key can be optimal in general (`test_shared_order_is_not_optimal`
    exhibits an instance). Any order is feasible, so `dp_place_shared` on this
    order is an UPPER BOUND on the constrained optimum, and the RQ4 gap taken
    from it is an upper bound on the price of residual alignment.
    """
    n = len(sites[0])
    keys = [(max(s[c] for s in sites), sum(s[c] for s in sites), c)
            for c in range(n)]
    keys.sort()
    return tuple(k[2] for k in keys)


def dp_place_shared(sites: list[list[int]], n_buckets: int, cap: int, kappa,
                    order: tuple[int, ...] | None = None) -> Placement:
    """Exact DP over cuts for a FIXED permutation, objective summed over the
    sites of one residual group. O(G * C^2 * |sites|)."""
    if not sites:
        raise ValueError("need at least one site")
    n = len(sites[0])
    if any(len(s) != n for s in sites):
        raise ValueError("all sites in a residual group must have the same "
                         "channel count — that is what sharing a permutation "
                         "means")
    _check_feasible(n, n_buckets, cap)
    order = shared_order(sites) if order is None else order

    seg = [[0.0] * (n + 1) for _ in range(n + 1)]   # seg[i][k] = cost of [i,k)
    for i in range(n):
        running = [0] * len(sites)
        for k in range(i + 1, n + 1):
            c = order[k - 1]
            for si, s in enumerate(sites):
                if s[c] > running[si]:
                    running[si] = s[c]
            seg[i][k] = sum(kappa(m) for m in running)

    INF = float("inf")
    dp = [[INF] * (n + 1) for _ in range(n_buckets + 1)]
    back = [[-1] * (n + 1) for _ in range(n_buckets + 1)]
    dp[0][0] = 0.0
    for j in range(1, n_buckets + 1):
        for i in range(j, n - (n_buckets - j) + 1):
            lo = max(j - 1, i - cap)
            for ip in range(lo, i):
                if dp[j - 1][ip] == INF:
                    continue
                cand = dp[j - 1][ip] + seg[ip][i]
                if cand < dp[j][i]:
                    dp[j][i], back[j][i] = cand, ip

    if dp[n_buckets][n] == INF:
        raise ValueError("no feasible placement")

    sizes = _reconstruct(back, n_buckets, n)
    maxes, pos = [], 0
    for sz in sizes:
        block = order[pos:pos + sz]
        maxes.append(tuple(max(s[c] for c in block) for s in sites))
        pos += sz
    return Placement(dp[n_buckets][n], order, tuple(sizes), tuple(maxes))


def brute_force_shared(sites: list[list[int]], n_buckets: int, cap: int,
                       kappa) -> float:
    """True constrained optimum over all partitions — the reference the
    heuristic order is measured against."""
    n = len(sites[0])
    _check_feasible(n, n_buckets, cap)
    best = float("inf")
    for assign in _restricted_growth(n, n_buckets):
        blocks = _blocks_of(assign, n_buckets, cap)
        if blocks is None:
            continue
        cost = sum(kappa(max(s[c] for c in v)) for v in blocks for s in sites)
        best = min(best, cost)
    return best


# ── baselines, from the same degree input ─────────────────────────────────

def baseline_uniform(sites: list[list[int]], n_buckets: int, kappa) -> float:
    """One degree everywhere: the maximum over every channel of every site."""
    d = max(max(s) for s in sites)
    return len(sites) * n_buckets * kappa(d)


def baseline_layerwise(sites: list[list[int]], n_buckets: int,
                       kappa) -> float:
    """Per-site maximum — OLA-approx style [SLR row 76]. This is what the
    solver degenerates to at G=1, which is Gate A's whole point."""
    return sum(n_buckets * kappa(max(s)) for s in sites)


# ── levels and bootstraps: BOTH models, no claim made (O-8) ───────────────

def block_levels(act1_degree: int, act2_degree: int, delta_fn,
                 svd: bool = True, fused: bool = False) -> int:
    """Per-block level consumption when the two activation sites may carry
    different degrees:  l = a(d1) + a(d2) + c1 + c2 + 2b + s.
    Reduces to ell_from_derivation when d1 == d2 [dk_derivation Sec.4.1]."""
    c2 = C2_SVD_CONV if svd else C1_BASELINE_CONV
    return (a(act1_degree, delta_fn, fused) + a(act2_degree, delta_fn, fused)
            + C1_BASELINE_CONV + c2 + 2 * C_BN + S_SHORTCUT)


def levels_realigned(bucket_maxes_act1: list[int],
                     bucket_maxes_act2: list[int], delta_fn,
                     **kw) -> list[int]:
    """Every ciphertext charged the LAYER maximum, because a convolution mixes
    all channels and the operands must be aligned before it [Sec.4.4 V2].
    Placement-invariant by construction — that is O-8."""
    d1, d2 = max(bucket_maxes_act1), max(bucket_maxes_act2)
    ell = block_levels(d1, d2, delta_fn, **kw)
    return [ell] * len(bucket_maxes_act1)


def levels_independent(bucket_maxes_act1: list[int],
                       bucket_maxes_act2: list[int], delta_fn,
                       **kw) -> list[int]:
    """Each ciphertext charged its own bucket maxima. An OPTIMISTIC BOUND: it
    holds only if the G ciphertexts never have to be aligned to each other,
    which multiplexed packing does not currently give (O-8)."""
    return [block_levels(m1, m2, delta_fn, **kw)
            for m1, m2 in zip(bucket_maxes_act1, bucket_maxes_act2)]


def bootstrap_count(levels_per_bucket: list[int], n_blocks: int,
                    headroom: int) -> int:
    """Bootstrap operations over a chain of n_blocks blocks, summed over the G
    ciphertexts. Each ciphertext refreshes when its own consumption exhausts
    the usable headroom."""
    if headroom < 1:
        raise ValueError("headroom must be >= 1")
    return sum(math.ceil(ell * n_blocks / headroom)
               for ell in levels_per_bucket)


# ── network spec, read from Phase 2 — nothing hardcoded ───────────────────

STAGE_RE = re.compile(r"^s(\d+)_(\d+)x(\d+)x(\d+)$")


@dataclass(frozen=True)
class Stage:
    name: str
    channels: int
    height: int
    width: int
    G: int
    headroom: int
    cap: int          # channels per ciphertext


def load_stages(path: Path, network: str, N: int, packing: str = "multiplexed",
                dup: int = 1) -> list[Stage]:
    """Per-stage G straight out of results/phase2/G_table.csv.

    cap = ceil(C / G): the smallest uniform per-ciphertext channel count that
    lets G ciphertexts hold the stage. Stated as an assumption because the CSV
    carries occupied_slots, not a channels-per-ciphertext column; override with
    --cap. Blocks per stage are NOT in the CSV and are NOT invented here — they
    come from --blocks-per-stage or the bootstrap columns stay empty.
    """
    if not path.exists():
        raise SystemExit(f"missing {path} — Phase 2 output (PROTOCOL rule 2)")
    with open(path) as fh:
        rows = list(csv.DictReader(
            line for line in fh if not line.startswith("#")))
    out: list[Stage] = []
    for r in rows:
        if (r["network"] != network or int(r["N"]) != N
                or r["packing"] != packing or int(r["dup"]) != dup):
            continue
        m = STAGE_RE.match(r["stage"])
        if not m:
            raise SystemExit(f"unparseable stage name {r['stage']!r}")
        _idx, C, H, W = (int(g) for g in m.groups())
        G = int(r["G"])
        out.append(Stage(r["stage"], C, H, W, G, int(r["headroom"]),
                         math.ceil(C / G)))
    if not out:
        raise SystemExit(
            f"no rows for network={network} N={N} packing={packing} dup={dup} "
            f"in {path} (PROTOCOL rule 2: nothing is invented here)")
    return out


# ── synthetic validation ──────────────────────────────────────────────────

# name -> callable(rng, n, dmax) -> list[int]. SYNTHETIC. These carry no claim
# about real BN dispersion; Gate B measures that (Phase 3).
GENERATORS = {
    "uniform": lambda rng, n, dmax: [rng.randint(1, dmax) for _ in range(n)],
    "log_uniform": lambda rng, n, dmax: [
        min(dmax, 1 << rng.randint(0, max(1, dmax.bit_length()) - 1))
        for _ in range(n)],
    "bimodal": lambda rng, n, dmax: [
        rng.randint(1, max(1, dmax // 8)) if rng.random() < 0.7
        else rng.randint(max(1, dmax // 2), dmax) for _ in range(n)],
    "heavy_tail": lambda rng, n, dmax: [
        max(1, min(dmax, int(dmax / (rng.random() * 8 + 1))))
        for _ in range(n)],
    "flat_plateau": lambda rng, n, dmax: [
        rng.choice([5, 6, 7, 8]) for _ in range(n)],
}


def run_synthetic(seed: int, trials: int, max_channels: int,
                  n_blocks: int | None, headroom: int = 27) -> list[dict]:
    """The acceptance gate as data: DP vs brute force on random synthetic
    distributions, plus baselines, the shared-permutation slack and both level
    models. One row per (trial, kappa model, delta model, collapse)."""
    rng = random.Random(seed)
    rows: list[dict] = []
    gen_names = list(GENERATORS)
    for t in range(trials):
        gname = gen_names[t % len(gen_names)]
        n = rng.randint(3, max_channels)
        n_buckets = rng.randint(2, min(4, n))
        cap = rng.randint(math.ceil(n / n_buckets), n)
        dmax = rng.choice([7, 15, 31])
        site1 = GENERATORS[gname](rng, n, dmax)
        site2 = GENERATORS[gname](rng, n, dmax)
        for kname, kappa in KAPPA_MODELS.items():
            for dname, delta_fn in DELTA_MODELS.items():
                for collapse in (False, True):
                    d1 = (collapse_to_plateaus(site1, delta_fn) if collapse
                          else site1)
                    d2 = (collapse_to_plateaus(site2, delta_fn) if collapse
                          else site2)
                    p1 = dp_place(d1, n_buckets, cap, kappa)
                    p2 = dp_place(d2, n_buckets, cap, kappa)
                    bf = brute_force_place(d1, n_buckets, cap, kappa)
                    sh = dp_place_shared([d1, d2], n_buckets, cap, kappa)
                    shb = brute_force_shared([d1, d2], n_buckets, cap, kappa)
                    m1, m2 = list(p1.bucket_maxes), list(p2.bucket_maxes)
                    lr = levels_realigned(m1, m2, delta_fn)
                    li = levels_independent(m1, m2, delta_fn)
                    rows.append({
                        "trial": t, "generator": gname, "seed": seed,
                        "n_channels": n, "n_buckets": n_buckets, "cap": cap,
                        "d_max": dmax, "kappa_model": kname,
                        "delta_model": dname, "collapse": collapse,
                        "dp_cost": p1.cost, "brute_cost": bf.cost,
                        "agree": p1.cost == bf.cost,
                        "uniform_cost": baseline_uniform([d1, d2], n_buckets,
                                                         kappa),
                        "layerwise_cost": baseline_layerwise([d1, d2],
                                                             n_buckets, kappa),
                        "placed_unconstrained": p1.cost + p2.cost,
                        "placed_constrained_ub": sh.cost,
                        "constrained_optimum": shb,
                        "heuristic_slack": sh.cost - shb,
                        "rq4_gap_ub": sh.cost - (p1.cost + p2.cost),
                        "levels_realigned": lr[0],
                        "levels_independent_min": min(li),
                        "levels_independent_max": max(li),
                        "bootstraps_realigned": (
                            bootstrap_count(lr, n_blocks, headroom)
                            if n_blocks else ""),
                        "bootstraps_independent": (
                            bootstrap_count(li, n_blocks, headroom)
                            if n_blocks else ""),
                    })
    return rows


# ── CLI ───────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Phase 5 placement solver core (network-independent).")
    p.add_argument("--model", help="network name as it appears in "
                                   "results/phase2/G_table.csv")
    p.add_argument("--g-table", type=Path, default=G_TABLE)
    p.add_argument("--N", type=int, default=65536)
    p.add_argument("--packing", default="multiplexed")
    p.add_argument("--dup", type=int, default=1)
    p.add_argument("--cap", type=int, default=None,
                   help="override channels per ciphertext")
    p.add_argument("--degrees", type=Path, default=None,
                   help="CSV: layer, channel, degree (does not exist yet)")
    p.add_argument("--blocks-per-stage", type=int, default=None,
                   help="not in G_table.csv; required for bootstrap counts")
    p.add_argument("--kappa", choices=sorted(KAPPA_MODELS), default="sqrt2d")
    p.add_argument("--constrained", action="store_true",
                   help="residual constraint: one shared permutation")
    p.add_argument("--collapse", dest="collapse", action="store_true")
    p.add_argument("--no-collapse", dest="collapse", action="store_false")
    p.set_defaults(collapse=None)
    p.add_argument("--validate-synthetic", action="store_true")
    p.add_argument("--trials", type=int, default=60)
    p.add_argument("--max-channels", type=int, default=10)
    p.add_argument("--seed", type=int, default=20260822)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    if args.validate_synthetic:
        require_clean()
        import pandas as pd
        rows = run_synthetic(args.seed, args.trials, args.max_channels,
                             args.blocks_per_stage)
        bad = [r for r in rows if not r["agree"]]
        out = (args.out
               or RESULTS / "grouping_multiplications" / "synthetic_validation.csv")
        write_csv(pd.DataFrame(rows), out, extra={
            "acceptance_gate": "dp_cost == brute_cost on every row",
            "disagreements": len(bad),
            "note": "SYNTHETIC degree distributions; no claim about real BN "
                    "dispersion (that is Gate B / Phase 3)",
        })
        print(f"{len(rows)} rows -> {out}")
        if bad:
            print(f"ACCEPTANCE GATE FAILED: {len(bad)} rows where DP != brute "
                  f"force. Nothing downstream is valid.", file=sys.stderr)
            return 1
        print("acceptance gate: DP == brute force on all rows")
        return 0

    if not args.model:
        p.error("give --model or --validate-synthetic")

    stages = load_stages(args.g_table, args.model, args.N, args.packing,
                         args.dup)
    print(f"{args.model} @ N={args.N} {args.packing} dup={args.dup}")
    for s in stages:
        cap = args.cap or s.cap
        print(f"  {s.name}: C={s.channels} G={s.G} cap={cap} "
              f"headroom={s.headroom}")
    if all(s.G == 1 for s in stages):
        print("  NOTE: G=1 on every stage — one bucket per site, so the "
              "placement solver is identically the layerwise baseline "
              "[LAB_NOTEBOOK Sec.5.1]. This is Gate A's FAIL, not a bug.")

    if args.degrees is None or not args.degrees.exists():
        print("\nNo per-channel degrees available. fit_degrees.py is blocked "
              "on D-GateA [O-6] and the error-budget sign-off [O-7], and "
              "PROTOCOL rule 2 forbids inventing a distribution. Run "
              "--validate-synthetic to exercise the solver.", file=sys.stderr)
        return 2

    print("\ndegrees file supplied, but the real-network path is deliberately "
          "not wired: the evaluation plan is PROPOSED, not approved "
          "(DECISIONS.md).", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
