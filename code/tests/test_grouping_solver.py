"""
PURPOSE
    Phase 5 acceptance tests for the placement solver core.

    `test_dp_vs_bruteforce` IS THE ACCEPTANCE GATE (PHASE5_PROMPT Part 1). If
    it fails, stop and report — nothing downstream is valid. Everything else
    here defends a specific claim made in `dp_solver`'s docstring: the
    exchange argument, the plateau collapse, the assumptions that fail at a
    residual junction, and O-8.

INPUTS
    results/phase2/G_table.csv for the loader test; skipped if absent (rule 2).

OUTPUTS
    pytest pass/fail only.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.levels import (  # noqa: E402
    DELTA_MODELS, delta_A, delta_B, ell_from_derivation,
)
from src.grouping_solver import (  # noqa: E402
    GENERATORS, KAPPA_MODELS, Stage, baseline_layerwise, baseline_uniform,
    block_levels, bootstrap_count, brute_force_place, brute_force_shared,
    collapse_to_plateaus, dp_place, dp_place_shared, kappa_bsgs_upper,
    kappa_linear, kappa_sqrt2d, levels_independent, levels_realigned,
    load_stages, plateau_top, shared_order,
)

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO.parent / "results"
G_TABLE = RESULTS / "ciphertexts_per_layer" / "G_table.csv"


# ── THE ACCEPTANCE GATE ───────────────────────────────────────────────────

def test_dp_vs_bruteforce():
    """ACCEPTANCE GATE: sorted-order DP == exhaustive partition search on
    random synthetic distributions up to 12 channels, under every kappa model.

    Brute force makes no sorting assumption, so agreement is a check on the
    exchange argument itself, not just on the DP recurrence.
    """
    rng = random.Random(5_2026)
    gens = list(GENERATORS)
    cases: list[tuple[int, int]] = [
        (rng.randint(3, 9), rng.randint(2, 4)) for _ in range(40)
    ] + [(10, 3), (11, 3), (12, 3)]

    for trial, (n, n_buckets) in enumerate(cases):
        n_buckets = min(n_buckets, n)
        cap = rng.randint(-(-n // n_buckets), n)
        dmax = rng.choice([7, 15, 31])
        degrees = GENERATORS[gens[trial % len(gens)]](rng, n, dmax)
        for kname, kappa in KAPPA_MODELS.items():
            dp = dp_place(degrees, n_buckets, cap, kappa)
            bf = brute_force_place(degrees, n_buckets, cap, kappa)
            assert dp.cost == bf.cost, (
                f"DP != brute force: kappa={kname} degrees={degrees} "
                f"G={n_buckets} cap={cap} -> {dp.cost} vs {bf.cost}")
            assert sum(dp.sizes) == n and len(dp.sizes) == n_buckets
            assert all(1 <= s <= cap for s in dp.sizes)


# ── the exchange argument's assumptions ───────────────────────────────────

def test_all_kappa_models_are_monotone():
    """Assumption A1. Without it the exchange argument does not run, and the
    DP is not justified for that cost model."""
    for name, kappa in KAPPA_MODELS.items():
        vals = [kappa(d) for d in range(1, 129)]
        assert all(b >= a for a, b in zip(vals, vals[1:])), name


def test_kappa_models_are_distinct_and_ordered_where_it_matters():
    """The three models are not interchangeable; the archive's linear count is
    far above both bounded ones at the degrees the thesis cares about."""
    assert kappa_bsgs_upper(1) == 0            # a*x + b is scalar-only
    assert kappa_bsgs_upper(2) == 1            # x^2
    assert kappa_bsgs_upper(3) == 2            # x^2, x^3
    assert kappa_bsgs_upper(7) == 4            # ~sqrt(2*7) = 3.74
    assert kappa_sqrt2d(7) == 4
    assert kappa_linear(7) == 7
    for d in (7, 15, 31):
        assert kappa_linear(d) > kappa_bsgs_upper(d)


def test_sorted_order_attains_the_optimum():
    """The exchange argument's conclusion, stated directly: some optimal
    placement is an interval partition of the sorted order."""
    rng = random.Random(11)
    for _ in range(30):
        n = rng.randint(4, 9)
        degrees = [rng.randint(1, 15) for _ in range(n)]
        g = rng.randint(2, 3)
        cap = rng.randint(-(-n // g), n)
        dp = dp_place(degrees, g, cap, kappa_sqrt2d)
        srt = sorted(degrees)
        pos, maxes = 0, []
        for sz in dp.sizes:
            maxes.append(max(srt[pos:pos + sz]))
            pos += sz
        assert tuple(maxes) == dp.bucket_maxes
        assert dp.cost == brute_force_place(degrees, g, cap, kappa_sqrt2d).cost


def test_more_buckets_never_helps_the_kappa_objective():
    """Why the bucket count is G and not a free variable: splitting a bucket
    adds a kappa term and cannot lower the one it splits."""
    degrees = [1, 2, 3, 9, 12, 15]
    two = dp_place(degrees, 2, 3, kappa_sqrt2d).cost
    three = dp_place(degrees, 3, 2, kappa_sqrt2d).cost
    assert three >= two


# ── plateau classes ───────────────────────────────────────────────────────

def test_plateau_top_is_model_specific():
    """delta_A and delta_B partition the degree line differently
    [dk_derivation Sec.10.5], so the collapse must take delta as an argument."""
    # delta_A classes: {1} {2,3} {4..7} {8..15} ...
    # delta_B classes: {1} {2}   {3,4}  {5..8}  {9..16} ...
    # NOTE: dk_derivation Sec.10.5 prints these two lists against the wrong
    # labels ("delta_B gives {1},{2,3},{4-7},{8-15}"). The formulas and the
    # code are right; the doc line is transposed. Corrected there.
    assert plateau_top(5, delta_A) == 7
    assert plateau_top(5, delta_B) == 8
    assert plateau_top(1, delta_A) == 1
    assert plateau_top(1, delta_B) == 1
    assert plateau_top(2, delta_A) == 3
    assert plateau_top(2, delta_B) == 2
    assert plateau_top(3, delta_A) == 3
    assert plateau_top(3, delta_B) == 4


def test_collapse_preserves_the_level_objective_exactly():
    """Dispersion inside one plateau is worth exactly zero for levels
    [LAB_NOTEBOOK Sec.4.6]: collapsing cannot change any delta."""
    rng = random.Random(7)
    for delta_fn in DELTA_MODELS.values():
        for _ in range(50):
            degrees = [rng.randint(1, 40) for _ in range(rng.randint(3, 12))]
            coll = collapse_to_plateaus(degrees, delta_fn)
            assert all(delta_fn(c) == delta_fn(d)
                       for c, d in zip(coll, degrees))
            assert delta_fn(max(coll)) == delta_fn(max(degrees))


def test_collapse_is_only_an_upper_bound_for_kappa():
    """kappa is not constant on a delta-plateau, so collapsing is lossy for
    the multiplication objective — and the loss is real, not hypothetical."""
    degrees = [4, 5, 6, 7]          # one delta_A plateau, four kappa values
    exact = dp_place(degrees, 2, 2, kappa_linear).cost
    coarse = dp_place(collapse_to_plateaus(degrees, delta_A), 2, 2,
                      kappa_linear).cost
    assert coarse >= exact
    assert coarse > exact           # strictly, on this instance


# ── residual constraint: where assumption A4 fails ────────────────────────

def test_shared_dp_reduces_to_single_site_dp():
    """One site in the group means A4 holds again and the shared solver must
    agree with the single-site optimum."""
    rng = random.Random(3)
    for _ in range(20):
        n = rng.randint(4, 9)
        degrees = [rng.randint(1, 15) for _ in range(n)]
        g = rng.randint(2, 3)
        cap = rng.randint(-(-n // g), n)
        assert (dp_place_shared([degrees], g, cap, kappa_sqrt2d).cost
                == dp_place(degrees, g, cap, kappa_sqrt2d).cost)


def test_shared_order_is_not_optimal():
    """Assumption A4 fails for a shared permutation, and no single sort key
    fixes it. Concrete instance: two sites whose degree orders are opposed."""
    site1 = [1, 2, 3, 9]
    site2 = [9, 3, 2, 1]
    heuristic = dp_place_shared([site1, site2], 2, 2, kappa_linear).cost
    optimum = brute_force_shared([site1, site2], 2, 2, kappa_linear)
    assert heuristic == 24          # order (1,2,0,3): buckets {1,2} {0,3}
    assert optimum == 22            # buckets {0,1} {2,3}
    assert heuristic > optimum      # the heuristic is an upper bound only


def test_shared_heuristic_stays_an_upper_bound():
    """It is allowed to be loose; it is not allowed to be optimistic. Anything
    below the true constrained optimum would make the RQ4 gap a fiction."""
    rng = random.Random(23)
    for _ in range(40):
        n = rng.randint(4, 8)
        s1 = [rng.randint(1, 15) for _ in range(n)]
        s2 = [rng.randint(1, 15) for _ in range(n)]
        g = rng.randint(2, 3)
        cap = rng.randint(-(-n // g), n)
        for kappa in KAPPA_MODELS.values():
            ub = dp_place_shared([s1, s2], g, cap, kappa).cost
            opt = brute_force_shared([s1, s2], g, cap, kappa)
            assert ub >= opt


def test_rq4_gap_is_non_negative():
    """The price of residual alignment cannot be a discount: constraining two
    sites to one permutation can never beat solving them independently."""
    rng = random.Random(101)
    for _ in range(40):
        n = rng.randint(4, 9)
        s1 = [rng.randint(1, 15) for _ in range(n)]
        s2 = [rng.randint(1, 15) for _ in range(n)]
        g = rng.randint(2, 3)
        cap = rng.randint(-(-n // g), n)
        unconstrained = (dp_place(s1, g, cap, kappa_sqrt2d).cost
                         + dp_place(s2, g, cap, kappa_sqrt2d).cost)
        constrained = brute_force_shared([s1, s2], g, cap, kappa_sqrt2d)
        assert constrained >= unconstrained


def test_shared_order_is_a_permutation():
    s1, s2 = [3, 1, 4, 1, 5], [9, 2, 6, 5, 3]
    assert sorted(shared_order([s1, s2])) == list(range(5))


# ── baselines ─────────────────────────────────────────────────────────────

def test_baseline_ordering():
    """uniform >= layerwise >= placed. Any violation means a baseline is
    mis-specified, and a solver that beats a mis-specified baseline is the
    weak-baseline pattern the SLR condemns [prompts/10_SLR_GROUNDING.md]."""
    rng = random.Random(31)
    for _ in range(40):
        n = rng.randint(4, 10)
        s1 = [rng.randint(1, 15) for _ in range(n)]
        s2 = [rng.randint(1, 15) for _ in range(n)]
        g = rng.randint(2, 3)
        cap = rng.randint(-(-n // g), n)
        for kappa in KAPPA_MODELS.values():
            uni = baseline_uniform([s1, s2], g, kappa)
            lay = baseline_layerwise([s1, s2], g, kappa)
            placed = (dp_place(s1, g, cap, kappa).cost
                      + dp_place(s2, g, cap, kappa).cost)
            constrained = dp_place_shared([s1, s2], g, cap, kappa).cost
            assert uni >= lay >= placed
            assert lay >= constrained >= placed


def test_at_g_equals_one_the_solver_is_the_layerwise_baseline():
    """Gate A's whole point [LAB_NOTEBOOK Sec.5.1]: one bucket means the
    bucket maximum IS the layer maximum, so placement buys nothing at all."""
    rng = random.Random(59)
    for _ in range(20):
        n = rng.randint(3, 12)
        s = [rng.randint(1, 31) for _ in range(n)]
        for kappa in KAPPA_MODELS.values():
            assert (dp_place(s, 1, n, kappa).cost
                    == baseline_layerwise([s], 1, kappa))


# ── levels: O-8 made concrete ─────────────────────────────────────────────

def test_block_levels_reduces_to_the_hand_derivation():
    """Equal degrees on both activation sites must reproduce
    ell_from_derivation exactly [dk_derivation Sec.4.1]."""
    for delta_fn in DELTA_MODELS.values():
        for d in (1, 2, 3, 4, 7, 8, 15):
            assert (block_levels(d, d, delta_fn)
                    == ell_from_derivation(d, delta_fn))
            assert (block_levels(d, d, delta_fn, svd=False)
                    == ell_from_derivation(d, delta_fn, svd=False))


def test_levels_are_placement_invariant_under_realignment():
    """O-8. If the G ciphertexts of a layer must be aligned before the next
    convolution, the level cost is set by the LAYER maximum, which no
    permutation can change. Placement then buys exactly zero levels."""
    degrees = [1, 1, 2, 3, 7, 7, 15, 15]
    for delta_fn in DELTA_MODELS.values():
        good = dp_place(degrees, 2, 4, kappa_sqrt2d)
        bad_maxes = [15, 15]                    # the worst possible placement
        assert (levels_realigned(list(good.bucket_maxes),
                                 list(good.bucket_maxes), delta_fn)
                == levels_realigned(bad_maxes, bad_maxes, delta_fn))
        # ... while the multiplication objective does separate them.
        assert good.cost < 2 * kappa_sqrt2d(15)


def test_independent_levels_are_a_strict_optimistic_bound():
    """The optimistic model differs from the realigned one exactly when the
    buckets differ in plateau class — which is the whole hypothesis."""
    for delta_fn in DELTA_MODELS.values():
        m1 = m2 = [1, 15]
        ind = levels_independent(m1, m2, delta_fn)
        real = levels_realigned(m1, m2, delta_fn)
        assert min(ind) < real[0] and max(ind) == real[0]
        assert bootstrap_count(ind, 9, 27) <= bootstrap_count(real, 9, 27)


def test_bootstrap_count_is_per_ciphertext():
    assert bootstrap_count([7, 7], 9, 27) == 2 * 3      # ceil(63/27) = 3 each
    assert bootstrap_count([7], 9, 63) == 1


# ── the model comes from Phase 2, not from the source ─────────────────────

def test_no_architecture_is_hardcoded():
    """PHASE5_PROMPT: 'No architecture hardcoded anywhere.' The only route to
    a network's shape is results/phase2/G_table.csv."""
    src = (REPO / "src" / "grouping_solver.py").read_text()
    for token in ("resnet20", "resnet18", "wrn16", "wrn28", "BasicBlock"):
        assert token not in src, f"{token!r} hardcoded in dp_solver.py"


@pytest.mark.skipif(not G_TABLE.exists(),
                    reason="missing results/phase2/G_table.csv (rule 2)")
def test_load_stages_reads_g_from_phase2():
    stages = load_stages(G_TABLE, "resnet20", 65536)
    assert [s.name for s in stages] == [
        "s1_16x32x32", "s2_32x16x16", "s3_64x8x8"]
    assert [s.channels for s in stages] == [16, 32, 64]
    # Gate A: every ResNet-20 stage is G=1 at 2^16 [LAB_NOTEBOOK Sec.5.2].
    assert all(s.G == 1 for s in stages)
    assert all(s.cap == s.channels for s in stages)
    assert all(s.headroom == 27 for s in stages)


@pytest.mark.skipif(not G_TABLE.exists(),
                    reason="missing results/phase2/G_table.csv (rule 2)")
def test_load_stages_recovers_the_escape_routes():
    """The two live candidates in the proposed evaluation plan, straight from
    Phase 2 — C keeps a within-network G=1 control, D does not."""
    c = load_stages(G_TABLE, "wrn16_8", 65536)
    d = load_stages(G_TABLE, "wrn16_10", 65536)
    assert [s.G for s in c] == [4, 2, 1]
    assert [s.G for s in d] == [5, 3, 2]
    assert [s.cap for s in c] == [32, 128, 512]
    assert isinstance(c[0], Stage)


@pytest.mark.skipif(not G_TABLE.exists(),
                    reason="missing results/phase2/G_table.csv (rule 2)")
def test_load_stages_refuses_an_unknown_network():
    with pytest.raises(SystemExit):
        load_stages(G_TABLE, "resnet50", 65536)
