"""
Time gain of sorted placement (Lydia's question, 2026-10-15: "can we measure
the time gain?").

The multiplication saving (R8) was a COUNT, from a cost model. This measures
wall-clock time in OpenFHE instead. Three stages, run in this order:

  degrees   (laptop, torch)   real per-channel degrees on MobileNetV2, the
            sorted placement and the layerwise baseline for every region.
            Control: must reproduce results/phase5/real_degrees_placement.csv
            exactly, or the copied code is wrong and nothing else is read.
  time      (Linux, openfhe)  (a) seconds to evaluate one degree-d Chebyshev
            series on a full N = 2^16 ciphertext, d = 1..63, and the levels it
            used. (b) DIRECT: actually evaluate every bucket of every region at
            two budgets, placed vs layerwise, and time the whole thing.
            (c) optional: one bootstrap, for scale.
  combine   (either)          time saving per budget from (a), checked
            against (b).

run:  python exp4_time_gain.py degrees
      python exp4_time_gain.py time            # in docker, python 3.10
      python exp4_time_gain.py time --bootstrap
      python exp4_time_gain.py combine

Placement can only change how long the ACTIVATIONS take. Levels and
bootstraps are the same for every placement (R2, exp1), and the convolutions
do not depend on the degree. So the number here is a saving on the activation
part of inference, not on the whole of it.
"""

from __future__ import annotations

import argparse
import math
import re
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import src.run_stamp as _stamp  # noqa: E402
from src.run_stamp import banner, write_result, write_table  # noqa: E402

BUDGETS = [10.0, 3.0, 1.0, 0.3, 0.1, 0.03, 0.01]   # as src/phase5/real_degrees.py
DIRECT_BUDGETS = [1.0, 0.3]        # the deployable window (degree_max <= 16)
SLOTS = 32768                      # N = 2^16
BASE_GRID = 112 * 112              # mobilenetv2, as real_degrees.py
# the archived R8 run; only the degrees stage reads it, on the laptop
CONTROL = (_stamp.PROJECT_RESULTS / "grouping_multiplications"
           / "real_degrees_placement.csv")
DEGREES_CSV = "exp4_bucket_degrees"
TIMING_CSV = "exp4_degree_timing"
DIRECT_CSV = "exp4_direct_regions"
_CNA_BN = re.compile(r"^features\.(\d+)\.conv\.(\d+)\.1$")


# placement, copied from src/phase5/dp_solver.py (same logic, shorter errors)
def kappa_sqrt2d(d: int) -> int:
    return math.ceil(math.sqrt(2 * d))


def _check_feasible(n_channels: int, n_buckets: int, cap: int) -> None:
    if n_buckets < 1 or cap < 1:
        raise ValueError("need n_buckets >= 1 and cap >= 1")
    if n_buckets > n_channels:
        raise ValueError("more buckets than channels")
    if n_buckets * cap < n_channels:
        raise ValueError("buckets cannot hold every channel")


def _reconstruct(back, n_buckets: int, n: int) -> list[int]:
    sizes, i = [], n
    for j in range(n_buckets, 0, -1):
        ip = back[j][i]
        sizes.append(i - ip)
        i = ip
    return sizes[::-1]


def shared_order(sites: list[list[int]]) -> tuple[int, ...]:
    n = len(sites[0])
    keys = [(max(s[c] for s in sites), sum(s[c] for s in sites), c)
            for c in range(n)]
    keys.sort()
    return tuple(k[2] for k in keys)


def dp_place_shared(sites, n_buckets, cap, kappa):
    """exact DP over cuts of the shared order; returns (cost, bucket maxes)"""
    n = len(sites[0])
    _check_feasible(n, n_buckets, cap)
    order = shared_order(sites)
    seg = [[0.0] * (n + 1) for _ in range(n + 1)]
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
    sizes = _reconstruct(back, n_buckets, n)
    maxes, pos = [], 0
    for sz in sizes:
        block = order[pos:pos + sz]
        maxes.append(tuple(max(s[c] for c in block) for s in sites))
        pos += sz
    return dp[n_buckets][n], maxes


# stage 1: degrees
def stage_degrees() -> int:
    import torch
    from src.network_tools import bn_sites, channel_table, degrees_at, load_model

    model = load_model("mobilenetv2")
    shapes = {}
    hooks = []
    for name, bn, _ in bn_sites(model):
        def hook(_m, _i, out, name=name):
            shapes[name] = tuple(out.shape[1:])
        hooks.append(bn.register_forward_hook(hook))
    with torch.no_grad():
        model(torch.zeros(1, 3, 224, 224))
    for h in hooks:
        h.remove()

    def site_G(shape) -> int:
        C, H, W = shape
        occupied = BASE_GRID * math.ceil(C * H * W / BASE_GRID)
        return math.ceil(occupied / SLOTS)

    table = channel_table(model, "mobilenetv2")
    rows, control = [], []
    for B in BUDGETS:
        df = degrees_at(table, B)
        deg = {s: g.sort_values("channel_idx").degree.astype(int).tolist()
               for s, g in df.groupby("site")}
        blocks: dict[int, list[str]] = {}
        for s in deg:
            m = _CNA_BN.match(s)
            if m:
                blocks.setdefault(int(m.group(1)), []).append(s)
        regions = {f"block{b}": sorted(v) for b, v in blocks.items()
                   if len(v) >= 2}
        covered = {s for v in regions.values() for s in v}
        for s in df.site.unique():
            if s not in covered:
                regions[f"solo_{s}"] = [s]
        for rid, names in regions.items():
            sites = [deg[s] for s in names]
            if len({len(x) for x in sites}) != 1:
                continue
            C = len(sites[0])
            G = max(site_G(shapes[s]) for s in names)
            if G < 2:
                continue               # G = 1: placement is the baseline
            cap = math.ceil(C / G)
            cost, maxes = dp_place_shared(sites, G, cap, kappa_sqrt2d)
            lay = sum(G * kappa_sqrt2d(max(s)) for s in sites)
            control.append({"budget_B": B, "region": rid,
                            "placement_cost": cost, "layerwise_cost": lay})
            for si, s in enumerate(sites):
                for j in range(G):
                    rows.append({
                        "budget_B": B, "region": rid, "site": names[si],
                        "site_pos": si, "G": G, "C": C, "bucket": j,
                        "degree_placed": int(maxes[j][si]),
                        "degree_layerwise": int(max(s)),
                    })

    ours = pd.DataFrame(control)
    ref = pd.read_csv(CONTROL, comment="#")
    ref = ref[(ref.net == "mobilenetv2") & (~ref.degenerate)]
    m = ref.merge(ours, on=["budget_B", "region"], suffixes=("_ref", ""))
    ok = (len(m) == len(ref) == len(ours)
          and np.allclose(m.placement_cost_ref, m.placement_cost)
          and np.allclose(m.layerwise_cost_ref, m.layerwise_cost))
    verdict = (f"CONTROL {'REPRODUCED' if ok else 'MISMATCH'}: {len(ours)} "
               f"live regions vs {len(ref)} in the archive")
    write_table(DEGREES_CSV, pd.DataFrame(rows), {
        "experiment": "EXP 4 stage 1: bucket degrees, MobileNetV2",
        "control": str(CONTROL.name), "control_ok": ok}, verdict)
    banner([verdict])
    return 0 if ok else 1


# stage 2: time
def cheb_relu(d: int) -> list[float]:
    """Chebyshev coefficients of ReLU on [-1,1], interpolated at degree d"""
    f = lambda t: np.maximum(0.0, t)  # noqa: E731
    return list(np.polynomial.chebyshev.Chebyshev.interpolate(
        f, d, domain=[-1, 1]).coef)


class Backend:
    def __init__(self, depth: int):
        from openfhe import (CCParamsCKKSRNS, GenCryptoContext,
                             PKESchemeFeature, ScalingTechnique, SecurityLevel)
        p = CCParamsCKKSRNS()
        p.SetMultiplicativeDepth(depth)
        p.SetScalingModSize(40)
        p.SetFirstModSize(60)
        p.SetRingDim(1 << 16)
        p.SetBatchSize(SLOTS)
        p.SetScalingTechnique(ScalingTechnique.FIXEDAUTO)
        p.SetSecurityLevel(SecurityLevel.HEStd_128_classic)
        self.cc = GenCryptoContext(p)
        for f in (PKESchemeFeature.PKE, PKESchemeFeature.KEYSWITCH,
                  PKESchemeFeature.LEVELEDSHE, PKESchemeFeature.ADVANCEDSHE):
            self.cc.Enable(f)
        self.keys = self.cc.KeyGen()
        self.cc.EvalMultKeyGen(self.keys.secretKey)
        self.depth = depth
        self.ring_dim = self.cc.GetRingDimension()
        rng = np.random.default_rng(0)
        self.x = rng.uniform(-1, 1, SLOTS)
        self.fresh = self.cc.Encrypt(
            self.keys.publicKey, self.cc.MakeCKKSPackedPlaintext(self.x.tolist()))

    def evaluate(self, d: int):
        """one activation at degree d on a full ciphertext; (seconds, ct)"""
        coeffs = cheb_relu(d)
        # openfhe evaluates c0/2 + sum c_k T_k, so the constant goes in doubled
        # (without this every output is off by c0/2 = 1/(2 pi) for ReLU)
        ofhe = [2 * coeffs[0]] + coeffs[1:]
        t = time.perf_counter()
        if d == 1:                     # affine: no ciphertext product at all
            out = self.cc.EvalAdd(self.cc.EvalMult(self.fresh, coeffs[1]),
                                  coeffs[0])
        else:
            out = self.cc.EvalChebyshevSeries(self.fresh, ofhe, -1.0, 1.0)
        return time.perf_counter() - t, out

    @staticmethod
    def levels_used(ct) -> int:
        """GetLevel plus the rescale FIXEDAUTO is still holding back"""
        return int(ct.GetLevel()) + int(ct.GetNoiseScaleDeg()) - 1

    def max_error(self, d: int, ct) -> float:
        pt = self.cc.Decrypt(ct, self.keys.secretKey)
        pt.SetLength(SLOTS)
        got = np.array(pt.GetRealPackedValue())
        want = np.polynomial.chebyshev.chebval(self.x, cheb_relu(d))
        return float(np.max(np.abs(got - want)))


def stage_time(reps: int, depths: list[int], bootstrap: bool) -> int:
    try:
        import openfhe  # noqa: F401
    except ImportError as e:
        print(f"openfhe not importable ({e}). Needs Linux and Python 3.10.",
              file=sys.stderr)
        return 2

    # (a) seconds per degree
    rows = []
    for depth in depths:
        be = Backend(depth)
        be.evaluate(7)                 # warm-up, not recorded
        print(f"\n--- chain depth {depth}, ring dim {be.ring_dim} ---",
              flush=True)
        for d in range(1, 64):
            ts, ct = [], None
            for _ in range(reps):
                dt, ct = be.evaluate(d)
                ts.append(dt)
            row = {"chain_depth": depth, "degree": d,
                   "seconds_median": statistics.median(ts),
                   "seconds_min": min(ts), "seconds_max": max(ts),
                   "reps": reps, "levels_used": Backend.levels_used(ct),
                   "delta_A": math.ceil(math.log2(d + 1)),
                   "delta_B": math.ceil(math.log2(d)) + 1,
                   "max_abs_error_vs_plain": be.max_error(d, ct)}
            rows.append(row)
            print(f"  d={d:2d}  {row['seconds_median']*1000:8.1f} ms  "
                  f"levels {row['levels_used']}  "
                  f"err {row['max_abs_error_vs_plain']:.1e}", flush=True)
    timing = pd.DataFrame(rows)
    write_table(TIMING_CSV, timing, {
        "experiment": "EXP 4 stage 2a: seconds per activation degree",
        "ring_dim": 1 << 16, "slots": SLOTS, "scale_bits": 40,
        "security": "HEStd_128_classic", "reps": reps})

    # (b) direct: every bucket of every live region, placed vs layerwise
    deg = pd.read_csv(_stamp.RESULTS_DIR / f"{DEGREES_CSV}.csv", comment="#")
    be = Backend(depths[0])
    be.evaluate(7)
    direct = []
    for B in DIRECT_BUDGETS:
        sub = deg[deg.budget_B == B]
        for col in ("degree_placed", "degree_layerwise"):
            total = 0.0
            for d in sub[col]:
                dt, _ = be.evaluate(int(d))
                total += dt
            direct.append({"budget_B": B, "layout": col.split("_")[1],
                           "activations": len(sub), "seconds": total})
            print(f"  direct B={B}  {col:17s} {len(sub)} activations  "
                  f"{total:7.2f} s", flush=True)
    write_table(DIRECT_CSV, pd.DataFrame(direct), {
        "experiment": "EXP 4 stage 2b: direct region timing",
        "chain_depth": depths[0]})

    # (c) one bootstrap, for scale only
    if bootstrap:
        try:
            boot = time_bootstrap()
            write_result("exp4_bootstrap", boot,
                         f"one bootstrap at N=2^16: {boot['seconds']:.1f} s")
            print(f"  bootstrap {boot['seconds']:.1f} s", flush=True)
        except Exception as e:  # noqa: BLE001  scale only, never a result
            print(f"  bootstrap not timed: {e}", flush=True)
    return 0


def time_bootstrap(after: int = 10, spare: int = 2) -> dict:
    """One bootstrap of a full ciphertext. For scale only: security is not
    set (the python wheel has no GetBootstrapDepth, so the chain is sized by
    hand), which does not change the time for a fixed ring dim and chain."""
    from openfhe import (CCParamsCKKSRNS, GenCryptoContext,
                         PKESchemeFeature, ScalingTechnique, SecretKeyDist,
                         SecurityLevel)
    level_budget = [4, 4]
    p = CCParamsCKKSRNS()
    dist = SecretKeyDist.UNIFORM_TERNARY
    p.SetSecretKeyDist(dist)
    p.SetSecurityLevel(SecurityLevel.HEStd_NotSet)
    p.SetRingDim(1 << 16)
    p.SetBatchSize(SLOTS)
    # 59 / 60 bits as OpenFHE's own bootstrapping example; with 50 / 60
    # EvalBootstrap refuses ("degree must be <= the correction factor")
    p.SetScalingModSize(59)
    p.SetFirstModSize(60)
    p.SetScalingTechnique(ScalingTechnique.FLEXIBLEAUTO)
    # 4 + 4 for the two linear transforms, 14 for EvalMod (uniform ternary),
    # as OpenFHE's C++ GetBootstrapDepth, plus a spare
    p.SetMultiplicativeDepth(after + 4 + 4 + 14 + spare)
    cc = GenCryptoContext(p)
    for f in (PKESchemeFeature.PKE, PKESchemeFeature.KEYSWITCH,
              PKESchemeFeature.LEVELEDSHE, PKESchemeFeature.ADVANCEDSHE,
              PKESchemeFeature.FHE):
        cc.Enable(f)
    cc.EvalBootstrapSetup(level_budget)
    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    cc.EvalBootstrapKeyGen(keys.secretKey, SLOTS)
    ct = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(
        [0.5] * SLOTS, 1, p.GetMultiplicativeDepth() - 1))
    t = time.perf_counter()
    cc.EvalBootstrap(ct)
    return {"seconds": time.perf_counter() - t, "level_budget": level_budget,
            "levels_after_bootstrap": after, "ring_dim": cc.GetRingDimension(),
            "slots": SLOTS}


# stage 3: combine
def stage_combine() -> int:
    deg = pd.read_csv(_stamp.RESULTS_DIR / f"{DEGREES_CSV}.csv", comment="#")
    timing = pd.read_csv(_stamp.RESULTS_DIR / f"{TIMING_CSV}.csv", comment="#")
    # The archived R8 run made every BN site not inside a block a "solo"
    # region, which includes the projection BN of the linear bottleneck
    # (features.i.conv.2 / conv.3). No activation follows those, so nothing is
    # evaluated there. "archive" keeps them (to compare with R8 and with the
    # direct run); "activations" drops them and is the headline.
    no_act = deg.site.str.match(r"^features\.\d+\.conv\.[23]$")
    scopes = {"archive": deg, "activations": deg[~no_act]}
    rows = []
    for depth, tg in timing.groupby("chain_depth"):
        t = dict(zip(tg.degree, tg.seconds_median))
        for scope, dsc in scopes.items():
            for B, g in dsc.groupby("budget_B"):
                placed = sum(t[d] for d in g.degree_placed)
                layer = sum(t[d] for d in g.degree_layerwise)
                kp = sum(kappa_sqrt2d(d) for d in g.degree_placed)
                kl = sum(kappa_sqrt2d(d) for d in g.degree_layerwise)
                rows.append({
                    "chain_depth": depth, "scope": scope, "budget_B": B,
                    "regions": g.region.nunique(), "activations": len(g),
                    "median_region_degree_max": float(
                        g.groupby("region").degree_layerwise.max().median()),
                    "seconds_layerwise": layer, "seconds_placed": placed,
                    "seconds_saved": layer - placed,
                    "time_saving_pct": 100 * (1 - placed / layer),
                    "mult_saving_pct_model": 100 * (1 - kp / kl),
                })
    out = pd.DataFrame(rows).sort_values(["chain_depth", "scope", "budget_B"])
    print(out.round(3).to_string(index=False))

    check = ""
    direct_path = _stamp.RESULTS_DIR / f"{DIRECT_CSV}.csv"
    if direct_path.exists():
        dr = pd.read_csv(direct_path, comment="#")
        piv = dr.pivot(index="budget_B", columns="layout", values="seconds")
        piv["direct_saving_pct"] = 100 * (1 - piv.placed / piv.layerwise)
        print("\ndirect measurement:\n" + piv.round(3).to_string())
        check = "; direct: " + ", ".join(
            f"B={b} {r.direct_saving_pct:.1f}%" for b, r in piv.iterrows())

    d0 = out[(out.chain_depth == out.chain_depth.min())
             & (out.scope == "activations")]
    win = d0[d0.budget_B.isin(DIRECT_BUDGETS)]
    verdict = ("Activation time saved by sorted placement on MobileNetV2, "
               "deployable budgets (median region degree <= 16): " + ", ".join(
                   f"B={r.budget_B} {r.time_saving_pct:.1f}% "
                   f"({r.seconds_saved:.1f} s)" for r in win.itertuples())
               + check)
    write_table("exp4_time_gain", out, {
        "experiment": "EXP 4: time gain of sorted placement",
        "scope": "activations only; levels, bootstraps, convolutions "
                 "unchanged by placement"}, verdict)
    banner([verdict])
    return 0


def main() -> None:
    _stamp.use_results("grouping_time")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("stage", choices=["degrees", "time", "combine"])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--depths", type=int, nargs="+", default=[12, 24],
                    help="chain depths to time at; the first is used for the "
                         "direct run")
    ap.add_argument("--bootstrap", action="store_true")
    args = ap.parse_args()
    banner([f"EXP 4 - TIME GAIN OF SORTED PLACEMENT, stage: {args.stage}"])
    if args.stage == "degrees":
        code = stage_degrees()
    elif args.stage == "time":
        code = stage_time(args.reps, args.depths, args.bootstrap)
    else:
        code = stage_combine()
    raise SystemExit(code)


if __name__ == "__main__":
    main()
