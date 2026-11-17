"""
End-to-end encrypted inference on MNIST, step 2 of 2: run the plan of
train_and_plan.py under CKKS in OpenFHE and decrypt (extension).

Needs only numpy, pandas and openfhe, so it runs in the project's container:

    docker run --rm -v "<project folder>:/work" -w /work/code thesis-openfhe \
        python -m experiments.end_to_end_mnist.encrypted_run

Packing: one ciphertext per value of the network, the images of the batch in
the slots (all 10,000 test images in one run). 128-bit security, 40-bit
scaling, no bootstrapping. The linear layers are sums of ciphertexts times
plaintext numbers. Each activation is one Chebyshev evaluation per ciphertext,
with that ciphertext's own degree and coefficients.

The map of each activation input to [-1, 1] and back is folded into the
weights of the layers on either side, which the server holds in plaintext, so
it costs no level.

For every configuration it records: accuracy after decryption, the accuracy
of the same polynomial network in ordinary arithmetic, the largest difference
between the two sets of logits, the time of each stage, and the levels used
by the ciphertexts after each stage.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import src.run_stamp as _stamp  # noqa: E402
from src.run_stamp import banner, write_table  # noqa: E402


def levels_of_degree(d: int) -> int:
    """levels OpenFHE's Chebyshev evaluator uses (thesis Table 6.1, measured)"""
    for top, lv in ((1, 1), (2, 2), (5, 3), (13, 4), (27, 5), (59, 6), (63, 7)):
        if d <= top:
            return lv
    raise ValueError(d)


def used(ct) -> int:
    return int(ct.GetLevel()) + int(ct.GetNoiseScaleDeg()) - 1


class Run:
    def __init__(self, depth: int, slots: int):
        from openfhe import (CCParamsCKKSRNS, GenCryptoContext, PKESchemeFeature,
                             ScalingTechnique, SecurityLevel)
        p = CCParamsCKKSRNS()
        p.SetMultiplicativeDepth(depth)
        p.SetScalingModSize(40)
        p.SetFirstModSize(60)
        p.SetBatchSize(slots)
        p.SetScalingTechnique(ScalingTechnique.FIXEDAUTO)
        p.SetSecurityLevel(SecurityLevel.HEStd_128_classic)
        self.cc = GenCryptoContext(p)
        for f in (PKESchemeFeature.PKE, PKESchemeFeature.KEYSWITCH,
                  PKESchemeFeature.LEVELEDSHE, PKESchemeFeature.ADVANCEDSHE):
            self.cc.Enable(f)
        self.keys = self.cc.KeyGen()
        self.cc.EvalMultKeyGen(self.keys.secretKey)
        self.slots, self.depth = slots, depth
        self.ring_dim = self.cc.GetRingDimension()

    def encrypt(self, v):
        x = np.zeros(self.slots)
        x[:len(v)] = v
        return self.cc.Encrypt(self.keys.publicKey, self.cc.MakeCKKSPackedPlaintext(x.tolist()))

    def decrypt(self, ct, n):
        pt = self.cc.Decrypt(ct, self.keys.secretKey)
        pt.SetLength(self.slots)
        return np.array(pt.GetRealPackedValue())[:n]

    def linear(self, cts, W, b):
        """rows of W times the list of ciphertexts, plus b; None entries are zero"""
        cc, out = self.cc, []
        for i in range(W.shape[0]):
            acc = None
            for j in np.flatnonzero(W[i]):
                if cts[j] is None:
                    continue
                term = cc.EvalMult(cts[j], float(W[i, j]))
                acc = term if acc is None else cc.EvalAdd(acc, term)
            out.append(cc.EvalAdd(acc, float(b[i])))
        return out

    def activate(self, cts, spec):
        cc, out = self.cc, []
        for ct, s in zip(cts, spec):
            c = s["coef"]
            if s["kind"] == -1:
                out.append(None)
            elif s["kind"] == 1:
                out.append(cc.EvalAdd(ct, float(c[0])))
            elif s["degree"] == 1:
                out.append(cc.EvalAdd(cc.EvalMult(ct, float(c[1])), float(c[0])))
            else:
                # openfhe evaluates c0/2 + sum c_k T_k
                out.append(cc.EvalChebyshevSeries(ct, [2 * c[0]] + list(c[1:]), -1.0, 1.0))
        return out


def spread(cts):
    lv = [used(c) for c in cts if c is not None]
    return min(lv), max(lv)


def main() -> None:
    _stamp.use_results("end_to_end_mnist")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--images", type=int, default=None, help="first N test images (trial run)")
    ap.add_argument("--configs", nargs="+", default=None)
    ap.add_argument("--tag", default="", help="added to the output file name (repeat runs)")
    args = ap.parse_args()
    banner(["END TO END ON MNIST, step 2: encrypted run"])
    d = np.load(_stamp.RESULTS_DIR / "plan.npz")
    plan = json.loads((_stamp.RESULTS_DIR / "plan.json").read_text(encoding="utf8"))
    X, y = d["X"], d["y"]
    n = args.images or len(X)
    X, y = X[:n], y[:n]
    A1, c1, A2, c2, A3, c3 = (d[k] for k in ("A1", "c1", "A2", "c2", "A3", "c3"))
    rows = []
    for name, cfg in plan.items():
        if args.configs and not any(name.startswith(c) for c in args.configs):
            continue
        m1, h1, m2, h2 = (np.array(cfg[k]) for k in ("m1", "h1", "m2", "h2"))
        s1, s2 = cfg["layer1"], cfg["layer2"]
        d1, d2 = max(s["degree"] for s in s1), max(s["degree"] for s in s2)
        depth = 3 + levels_of_degree(d1) + levels_of_degree(d2) + 1
        slots = 1 << int(np.ceil(np.log2(max(n, 2))))
        t0 = time.perf_counter()
        run = Run(depth, slots)
        t_setup = time.perf_counter() - t0
        t0 = time.perf_counter()
        cts = [run.encrypt(X[:, j]) for j in range(X.shape[1])]
        t_enc = time.perf_counter() - t0
        times, lv = {}, {}

        def stage(label, fn):
            nonlocal cts
            t = time.perf_counter()
            cts = fn()
            times[label] = time.perf_counter() - t
            lv[label] = spread(cts)

        stage("linear1", lambda: run.linear(cts, A1 / h1[:, None], (c1 - m1) / h1))
        stage("act1", lambda: run.activate(cts, s1))
        stage("linear2", lambda: run.linear(cts, A2 * h1[None, :] / h2[:, None], (c2 - m2) / h2))
        stage("act2", lambda: run.activate(cts, s2))
        stage("linear3", lambda: run.linear(cts, A3 * h2[None, :], c3))
        t0 = time.perf_counter()
        refused = ""
        try:
            logits = np.stack([run.decrypt(c, n) for c in cts], 1)
        except RuntimeError as err:
            # OpenFHE refuses to decode when the values are far larger than the
            # scale allows. With all images in one ciphertext, one image whose
            # polynomial blew up is enough.
            refused = str(err).strip().splitlines()[-1][-110:]
            logits = np.full((n, len(cts)), np.nan)
        t_dec = time.perf_counter() - t0
        ref = d[f"logits::{name}"][:n]
        fin = np.isfinite(ref).all(1)
        acc_enc = 100.0 * (np.isfinite(logits).all(1) & (logits.argmax(1) == y)).mean()
        acc_ref = 100.0 * (fin & (ref.argmax(1) == y)).mean()
        diff = np.abs(logits[fin] - ref[fin])
        server = sum(times.values())
        act = times["act1"] + times["act2"]
        rows.append({
            "config": name, "images": n, "decryption_refused": refused, "ring_dimension": run.ring_dim, "slots": slots,
            "chain_levels": depth, "max_degree_layer1": d1, "max_degree_layer2": d2,
            "encrypted_top1": acc_enc, "plain_polynomial_top1": acc_ref,
            "same_prediction_pct": float("nan") if refused else
            100.0 * (logits.argmax(1) == ref.argmax(1))[fin].mean(),
            "max_logit_difference": float(np.nanmax(diff)) if fin.any() and not refused else float("nan"),
            "median_logit_difference": float(np.nanmedian(diff)) if fin.any() and not refused else float("nan"),
            "p99_logit_difference": float(np.nanpercentile(diff, 99)) if fin.any() and not refused else float("nan"),
            "largest_plain_logit": float(np.abs(ref[fin]).max()) if fin.any() else float("nan"),
            "images_non_finite_in_plain": int((~fin).sum()),
            "seconds_setup_keys": t_setup, "seconds_encrypt": t_enc,
            **{f"seconds_{k}": v for k, v in times.items()},
            "seconds_server_total": server, "seconds_activations": act,
            "activation_share_pct": 100 * act / server, "seconds_decrypt": t_dec,
            **{f"levels_after_{k}_min": v[0] for k, v in lv.items()},
            **{f"levels_after_{k}_max": v[1] for k, v in lv.items()},
        })
        r = rows[-1]
        if refused:
            print(f"{name}: DECRYPTION REFUSED: {refused}", flush=True)
        print(f"{name}: encrypted {acc_enc:.2f}%  plain polynomial {acc_ref:.2f}%  "
              f"max logit diff {r['max_logit_difference']:.2e}  server {server:.0f}s "
              f"(activations {act:.0f}s)  levels after act1 {lv['act1']}, after linear2 "
              f"{lv['linear2']}, final {lv['linear3']}  N={run.ring_dim}", flush=True)
        del run, cts
    out = pd.DataFrame(rows)
    tag = args.tag + ("" if args.images is None else f"_first{args.images}")
    verdict = "; ".join(
        f"{r.config}: encrypted {r.encrypted_top1:.2f}% (plain polynomial "
        f"{r.plain_polynomial_top1:.2f}%), server {r.seconds_server_total:.0f}s of which "
        f"activations {r.seconds_activations:.0f}s, final level "
        f"{int(r.levels_after_linear3_max)} of {int(r.chain_levels)}" for r in out.itertuples())
    write_table("encrypted_run" + tag, out, {
        "experiment": "end to end on MNIST under CKKS (OpenFHE), 128-bit, no bootstrapping"},
        verdict)
    banner([verdict])


if __name__ == "__main__":
    main()
