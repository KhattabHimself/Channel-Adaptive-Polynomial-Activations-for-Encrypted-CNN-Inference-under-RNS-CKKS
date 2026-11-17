"""
Realignment check (R2), work in progress.

Put the same 8 channels into two ciphertexts in two different ways, run each
bucket at its highest degree, then add the buckets together like a conv does.
If both groupings end at the same output level, grouping doesn't save levels.
If they end at different levels, R2 is wrong.

run:  python exp1_realignment.py
      python exp1_realignment.py --logN 16 --depth 30

TODO
- run on colab, openfhe doesn't install on windows
- run the logN 16 version too, small N only checks the level logic
- openfhe's GetLevel counts levels used not levels left, double check the
  printout reads the right way round
- exit code 2 means openfhe is missing, not a result
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import src.run_stamp as _stamp  # noqa: E402
from src.run_stamp import banner, write_result  # noqa: E402

# degrees per channel, spread out on purpose so grouping could matter
CHANNEL_DEGREES = [15, 15, 7, 7, 3, 3, 1, 1]

# balanced: expensive channels split up, both buckets expensive
# sorted: expensive channels together, one cheap bucket (what a solver would do)
PARTITIONS = {
    "balanced": [[0, 2, 4, 6], [1, 3, 5, 7]],
    "sorted": [[0, 1, 2, 3], [4, 5, 6, 7]],
}

FALSIFICATION = ("FALSIFICATION: if the two partitions give DIFFERENT output "
                 "levels, R2 is REFUTED and Condition 2, R3 and R4 re-open.")


def activation_levels(degree: int) -> int:
    """levels for one activation: domain transform + ceil(log2 d) + 1"""
    return 1 + (math.ceil(math.log2(degree)) + 1 if degree > 1 else 1)


def run(logN: int, depth: int, scale_bits: int) -> dict:
    try:
        from openfhe import (CCParamsCKKSRNS, GenCryptoContext,
                             PKESchemeFeature, ScalingTechnique, SecurityLevel)
    except ImportError as e:
        # exit 2 so a missing library never looks like a result
        print("\nOpenFHE is not importable. Run:  pip install openfhe\n"
              "(The PyPI wheel ships a Linux binary; it does not import on "
              "Windows. Use Colab or any Linux host.)\n", file=sys.stderr)
        print(f"backend unavailable: {e}", file=sys.stderr)
        raise SystemExit(2)

    params = CCParamsCKKSRNS()
    params.SetMultiplicativeDepth(depth)
    params.SetScalingModSize(scale_bits)
    params.SetRingDim(1 << logN)
    # FIXEDAUTO so openfhe lines up the levels itself when adding
    params.SetScalingTechnique(ScalingTechnique.FIXEDAUTO)
    # not a secure parameter set, it's only testing levels
    params.SetSecurityLevel(SecurityLevel.HEStd_NotSet)

    cc = GenCryptoContext(params)
    cc.Enable(PKESchemeFeature.PKE)
    cc.Enable(PKESchemeFeature.LEVELEDSHE)
    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)

    slots = (1 << logN) // 2
    per_partition = {}
    for name, buckets in PARTITIONS.items():
        cts, degrees, consumed = [], [], []
        for blk in buckets:
            # each bucket runs at the highest degree inside it
            d = max(CHANNEL_DEGREES[c] for c in blk)
            n = activation_levels(d)
            degrees.append(d)
            consumed.append(n)
            pt = cc.MakeCKKSPackedPlaintext([0.5] * slots)
            ct = cc.Encrypt(keys.publicKey, pt)
            for _ in range(n):          # FIXEDAUTO rescales for us
                ct = cc.EvalMult(ct, ct)
            cts.append(ct)

        pre = [c.GetLevel() for c in cts]

        # the mixing step, adding every bucket together like a conv
        acc = cts[0]
        for ct in cts[1:]:
            acc = cc.EvalAdd(acc, ct)

        per_partition[name] = {
            "buckets": buckets, "bucket_degrees": degrees,
            "levels_per_bucket": consumed,
            "bucket_levels_before_add": pre,
            "output_level": int(acc.GetLevel()),
        }
        print(f"  {name:<9} degrees {degrees} -> bucket levels {pre} "
              f"-> OUTPUT LEVEL {acc.GetLevel()}", flush=True)
    return per_partition


def main() -> None:
    _stamp.use_results("level_realignment")
    ap = argparse.ArgumentParser(
        description="EXP 1: realignment / partition invariance (CPU)")
    ap.add_argument("--logN", type=int, default=12,
                    help="12 = small-N semantics test (default); 16 = "
                         "deployment size")
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--scale-bits", type=int, default=40)
    args = ap.parse_args()

    banner([
        "EXP 1 - REALIGNMENT (R2): does the partition change the output level?",
        f"N = 2^{args.logN}   depth = {args.depth}   scale = 2^{args.scale_bits}",
        FALSIFICATION,
    ])
    if args.logN < 14:
        print("NOTE: small N tests the level SEMANTICS only. Nothing here is "
              "a deployment parameter or a security claim.\n", flush=True)
    print(f"channel degrees: {CHANNEL_DEGREES}", flush=True)

    per_partition = run(args.logN, args.depth, args.scale_bits)
    levels = sorted({v["output_level"] for v in per_partition.values()})

    if len(levels) == 1:
        verdict = (f"R2 CONFIRMED: both partitions leave the output at level "
                   f"{levels[0]}. Placement did not change the level.")
        code = 0
    else:
        verdict = (f"R2 REFUTED: partitions gave DIFFERENT output levels "
                   f"{levels}. Condition 2, R3 and R4 all require "
                   f"re-examination. This is the result, not a bug - do not "
                   f"tune.")
        code = 1

    name = "exp1_realignment" if args.logN == 12 else \
        f"exp1_realignment_N{args.logN}"
    out = write_result(name, {
        "experiment": "EXP 1 realignment / partition invariance (R2)",
        "logN": args.logN, "ring_dim": 1 << args.logN,
        "depth_budget": args.depth, "scale_bits": args.scale_bits,
        "channel_degrees": CHANNEL_DEGREES,
        "partitions": per_partition,
        "distinct_output_levels": levels,
        "falsification_condition": FALSIFICATION,
        "level_counting": "OpenFHE GetLevel() counts levels consumed",
        "small_n_caveat": ("logN < 14 tests level semantics only; not a "
                           "security, precision or performance claim"),
    }, verdict)

    banner([f"VERDICT: {verdict}"])
    print(f"wrote {out}", flush=True)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
