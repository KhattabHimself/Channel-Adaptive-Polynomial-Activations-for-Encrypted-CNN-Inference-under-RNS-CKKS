"""
PURPOSE
    Level-consumption checker for CKKS network graphs (Phase 1, Task 2).

    Walks a network description charging every operation its level cost, with
    delta(d) = ceil(log2 d) + 1 for a degree-d polynomial activation
    (PROTOCOL.md rule 6), and predicts:
      - the level at every site,
      - the gap sequence d_k observed at block boundaries,
      - where the chain exhausts for a given usable depth L.

    It also carries the PARITY INVARIANT (`require_parity`, added 2026-08-22
    when O-2 closed): l == c1 + c2 + s (mod 2) for any single-degree block,
    whatever the degree and whatever the delta model. A measured l of the
    wrong parity is a mixed-degree or mis-scoped measurement and must never be
    reconciled by adjusting the model.

    The per-op cost table is PLUGGABLE and deliberately not defaulted: the
    authoritative per-block charging is Khattab's hand derivation
    (docs/dk_derivation.md, Phase 1 Task 1). This module must reproduce his
    numbers once they exist; until then, only arithmetic that is independent of
    the derivation (delta, recurrence fitting, exhaustion index) is exercised
    by tests.

INPUTS
    None from disk. Callers supply a cost table (dict op-name -> levels).

OUTPUTS
    Library only; no result files. Used by tests/test_level_checker.py and by
    the Phase 5 solver.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


def delta_A(d: int) -> int:
    """Paterson-Stockmeyer LOWER bound on polynomial depth: ceil(log2(d+1)).

    delta_A(7) = 3. What an optimal PS evaluator achieves.
    """
    if d < 1:
        raise ValueError(f"degree must be >= 1, got {d}")
    return math.ceil(math.log2(d + 1))


def delta_B(d: int) -> int:
    """Practical UPPER bound on polynomial depth: ceil(log2 d) + 1.

    delta_B(7) = 4. What an implementation costs when the coefficient
    multiply is not folded into the last multiplication.
    """
    if d < 1:
        raise ValueError(f"degree must be >= 1, got {d}")
    return math.ceil(math.log2(d)) + 1


def delta_impl(d: int) -> int:
    """What the ARCHIVE actually costs: the naive Chebyshev recurrence is
    linear in d [encrypted_activation.py:L156], so depth = d and
    a_impl(7) = 1 + 7 = 8 levels including the domain transform.

    NEVER a thesis cost model (PROTOCOL rule 6) — for archive-vs-model
    comparison only. See dk_derivation.md Sec. 6 and Sec. 11.
    """
    if d < 1:
        raise ValueError(f"degree must be >= 1, got {d}")
    return d


#: The two thesis models. Every result must be reported under both.
DELTA_MODELS = {"A": delta_A, "B": delta_B}


# ── Khattab's hand derivation (docs/dk_derivation.md §4, 2026-08-22) ────────
# l(d) = 2a(d) + c1 + c2 + 2b + s,  a(d) = 1 + delta(d),
# c1 = 1 (baseline conv1), c2 = 2 (SVD conv2), b = 0 (BN folded), s = 0
# (mod_drop discards the identity branch's surplus at the residual add).
C1_BASELINE_CONV = 1
C2_SVD_CONV = 2
C_BN = 0
S_SHORTCUT = 0
DOMAIN_TRANSFORM = 1  # fixed Chebyshev interval map, one per activation


def plateau_class(d: int, delta_fn) -> int:
    """Equivalence class for Gate B histogramming (LAB_NOTEBOOK Sec. 4.6).

    Degrees sharing a class cost the same levels, so dispersion inside one
    class is worth exactly zero. Which degrees share a class depends on the
    delta model — see dk_derivation.md Sec. 10.5.
    """
    return delta_fn(d)


def a(d: int, delta_fn, fused: bool = False) -> int:
    """Levels per activation site: domain transform + delta_fn(d).

    delta_fn is REQUIRED (PROTOCOL rule 6). Pass delta_impl to model the
    archive. fused=True folds the domain transform into the preceding
    plaintext affine (derivation Sec. 8).
    """
    return (0 if fused else DOMAIN_TRANSFORM) + delta_fn(d)


def ell_from_derivation(d: int, delta_fn, svd: bool = True,
                        fused: bool = False) -> int:
    """Per-block consumption. l(1) = 7 under both thesis models (measured)."""
    c2 = C2_SVD_CONV if svd else C1_BASELINE_CONV
    return 2 * a(d, delta_fn, fused) + C1_BASELINE_CONV + c2 \
        + 2 * C_BN + S_SHORTCUT


class ParityViolation(ValueError):
    """A measured l whose parity is impossible under the block model."""


def expected_parity(svd: bool = True, s: int = S_SHORTCUT) -> int:
    """l = 2a(d) + c1 + c2 + 2b + s, and 2a, 2b are even, so

        l  ==  c1 + c2 + s   (mod 2)

    independent of degree and of the delta model. With b = 0 (V1: BN folded,
    beta a level-free scalar add) and s = 0 (V2: mod_drop discards the
    identity branch's surplus) both CONFIRMED against artifacts, this is a
    hard invariant, not a modelling choice.
    """
    c2 = C2_SVD_CONV if svd else C1_BASELINE_CONV
    return (C1_BASELINE_CONV + c2 + s) % 2


def check_parity(ell: int, svd: bool = True, s: int = S_SHORTCUT) -> bool:
    """True iff a measured per-block l can come from a single-degree block."""
    return ell % 2 == expected_parity(svd, s)


def require_parity(ell: int, svd: bool = True, s: int = S_SHORTCUT,
                   label: str = "measurement") -> int:
    """PERMANENT INVARIANT (PHASE5_PROMPT, O-2 closed 2026-08-22).

    Any measured l whose parity disagrees with c1 + c2 + s is a MIXED-DEGREE
    or MIS-SCOPED measurement, not a data point — solving 2a(d) = l - c1 - c2
    for it gives a non-integer a. This is how O-2 closed: Vector D ran SVD off
    (c1 + c2 = 2, even, so l must be even) and measured l = 9, giving
    a = 3.5. Cause: two clusters at D_avg 4.5-6.8, a mixed-degree
    configuration the single-degree block model never applied to
    [dk_derivation Sec.7; LAB_NOTEBOOK Sec.6 O-2].

    Never fit the model to such a number. Raise, and re-scope the measurement.
    """
    if not check_parity(ell, svd, s):
        want = "odd" if expected_parity(svd, s) else "even"
        c2 = C2_SVD_CONV if svd else C1_BASELINE_CONV
        raise ParityViolation(
            f"{label}: l = {ell} is impossible for a single-degree block with "
            f"SVD {'on' if svd else 'off'} (c1={C1_BASELINE_CONV}, c2={c2}, "
            f"s={s} -> l must be {want}). Solving 2a(d) = l - c1 - c2 - s "
            f"gives a = {(ell - C1_BASELINE_CONV - c2 - s) / 2}, not an "
            f"integer. This is a mixed-degree or mis-scoped measurement; do "
            f"not fit the model to it.")
    return ell


@dataclass
class Site:
    """One level-consuming site in the walk."""
    name: str
    op: str
    cost: int
    level_after: int


@dataclass
class WalkResult:
    sites: list[Site] = field(default_factory=list)
    block_gaps: list[int] = field(default_factory=list)  # d_k at block ends
    exhausted_at: str | None = None  # site name where level would go < 0


# ResNet-20 for CIFAR-10: stem, then 3 stages x 3 basic blocks, then head.
# Each basic block: conv1-bn1-act-conv2-bn2-(shortcut add)-act.
# Blocks 3 and 6 (first of stages 2 and 3) have a projection shortcut conv.
RESNET20_BLOCKS = 9


def walk_resnet20(costs: dict[str, int], start_level: int,
                  act_degree: int = 1, delta_fn=delta_B) -> WalkResult:
    """Charge one op sequence per basic block; record the gap to start_level
    at every block boundary (the instrumentation in the as-run pipeline
    measured gaps against FRESH ciphertexts at the top level — see
    results/phase1/log_events.csv, ct_b=43 in every mismatch pair).

    `costs` must provide: conv, bn, and optionally shortcut_conv, plus
    anything the hand derivation charges. Activation is charged via
    delta(act_degree) unless costs["act"] overrides it.
    """
    act_cost = costs.get("act", delta_fn(act_degree))
    result = WalkResult()
    level = start_level

    def charge(name: str, op: str, cost: int) -> None:
        nonlocal level
        level -= cost
        result.sites.append(Site(name, op, cost, level))
        if level < 0 and result.exhausted_at is None:
            result.exhausted_at = name

    charge("stem.conv", "conv", costs["conv"])
    charge("stem.bn", "bn", costs["bn"])
    charge("stem.act", "act", act_cost)
    for k in range(RESNET20_BLOCKS):
        blk = f"block{k}"
        charge(f"{blk}.conv1", "conv", costs["conv"])
        charge(f"{blk}.bn1", "bn", costs["bn"])
        charge(f"{blk}.act1", "act", act_cost)
        charge(f"{blk}.conv2", "conv", costs["conv"])
        charge(f"{blk}.bn2", "bn", costs["bn"])
        if k in (3, 6) and "shortcut_conv" in costs:
            charge(f"{blk}.shortcut", "conv", costs["shortcut_conv"])
        charge(f"{blk}.act2", "act", act_cost)
        result.block_gaps.append(start_level - level)
    return result


def fit_recurrence(gaps: list[int]) -> tuple[float, float, float]:
    """Least-squares fit gaps[k] = d0 + k*l. Returns (d0, l, max_residual)."""
    n = len(gaps)
    if n < 2:
        raise ValueError("need at least two gap observations")
    ks = list(range(n))
    kbar = sum(ks) / n
    gbar = sum(gaps) / n
    l = sum((k - kbar) * (g - gbar) for k, g in zip(ks, gaps)) / \
        sum((k - kbar) ** 2 for k in ks)
    d0 = gbar - l * kbar
    resid = max(abs(g - (d0 + l * k)) for k, g in zip(ks, gaps))
    return d0, l, resid


def exhaustion_index(depth: int, d0: float, l: float) -> int:
    """k* = floor((L - d0) / l): last block index the chain can absorb."""
    return math.floor((depth - d0) / l)


def first_unphysical_index(gaps: list[int], max_gap: int) -> int | None:
    """Index of the first gap that exceeds the physically possible maximum
    (a gap cannot exceed the chain's top level). Returns None if all valid."""
    for k, g in enumerate(gaps):
        if g > max_gap:
            return k
    return None
