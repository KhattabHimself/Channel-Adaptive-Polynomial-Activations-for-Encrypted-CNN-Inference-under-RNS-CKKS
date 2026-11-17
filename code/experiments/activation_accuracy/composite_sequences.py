"""
PURPOSE
    O-a: does a composite minimax approximant with PER-COMPONENT degrees
    (Lee et al.'s general form) land BETWEEN the 4-level steps of the
    single-component construction, while holding top-1 loss within 1 point?

    R5 [reports/APPROXIMANT_GRANULARITY.md] rests on the claim that the
    accuracy-viable approximant moves in steps of 4 levels (delta_B(7)) against
    a measured per-channel demand spread of ~1 plateau class, making
    per-channel assignment inexpressible. That claim was measured against a
    composite that iterates ONE degree-7 component. Lee et al.'s general form
    permits a different degree per component, and the depth of a sequence is
    sum_i delta(d_i) — so mixed sequences can in principle hit any integer
    depth. This measures whether they hold accuracy there.

    R5 HOLDS if the smallest depth step between accuracy-viable configurations
    still exceeds the ~1-level demand spread. R5 NARROWS if it does not; the
    report then states by how much and how many distinct per-channel settings
    become available.

COMPONENT FAMILY
    f_n(x) = sum_{i=0}^{n} [C(2i,i)/4^i] * x * (1-x^2)^i,  degree 2n+1.
    The standard sign-iteration family: f_n(+-1) = +-1, odd, and iterating it
    sharpens towards sign. n=1 gives (3x-x^3)/2, n=3 gives the
    (35x-35x^3+21x^5-5x^7)/16 used in the k-fold sweep.

    Cost-efficient components are the degrees 2^k - 1 (3, 7, 15): delta is
    minimal across its plateau there, so degree 5 costs the same as 7 while
    converging more slowly, and 9/11/13 cost the same as 15. The sweep
    therefore uses {3, 7, 15}, whose delta_B costs are 3, 4, 5 — giving depth
    steps of ONE level, which is the whole point of the question.

METHOD
    Reuses the Phase 3 harness: same ResNet-20 checkpoint, same CIFAR-10 test
    set, same observed-range calibration, same no-clamping rule. Only the
    activation is replaced. The block-forward rebinding is duplicated here
    rather than imported so this branch touches no shared file.

INPUTS
    data/cifar-10-batches-py, pretrained cifar10_resnet20.

OUTPUTS
    results/phase5/oa_composite.csv — per sequence:
        sequence, n_components, effective_degree, depth_A, depth_B,
        top1, delta_points, within_1_point
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.provenance import require_clean, write_csv  # noqa: E402
from src.levels import delta_A, delta_B  # noqa: E402
from experiments.activation_accuracy.per_channel_degree_accuracy import calib_loader, loader, top1  # noqa: E402
from src.channel_degrees import load_model  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO.parent / "results"
COMPONENTS = (3, 7, 15)


def sign_component(degree: int) -> np.ndarray:
    """Coefficients of f_n in u = x^2 such that f(x) = x * q(u).

    f_n(x) = sum_i [C(2i,i)/4^i] * x * (1-x^2)^i, so q(u) = sum_i c_i (1-u)^i.
    """
    n = (degree - 1) // 2
    q = np.zeros(n + 1)
    for i in range(n + 1):
        c = math.comb(2 * i, i) / (4.0 ** i)
        # (1-u)^i expanded in u
        for j in range(i + 1):
            q[j] += c * math.comb(i, j) * ((-1.0) ** j)
    return q


class SeqAct(nn.Module):
    """ReLU, calibration, or a per-component composite surrogate."""

    def __init__(self):
        super().__init__()
        self.mode = "relu"
        self.lo = self.hi = None
        self.radius = None
        self.qs: list[torch.Tensor] = []

    def forward(self, x):
        if self.mode == "calibrate":
            flat = x.transpose(0, 1).reshape(x.shape[1], -1)
            lo, hi = flat.min(1).values, flat.max(1).values
            self.lo = lo if self.lo is None else torch.minimum(self.lo, lo)
            self.hi = hi if self.hi is None else torch.maximum(self.hi, hi)
            return torch.relu(x)
        if self.mode == "relu":
            return torch.relu(x)
        v = x / self.radius
        for q in self.qs:                      # one component per entry
            u = v * v
            acc = torch.full_like(v, float(q[-1]))
            for c in reversed(q[:-1].tolist()):  # Horner in u
                acc = acc * u + c
            v = v * acc
        return x * (1.0 + v) / 2.0

    def configure(self, sequence: tuple[int, ...]):
        pad = 0.05 * (self.hi - self.lo)
        self.radius = torch.maximum((self.hi + pad).abs(),
                                    (self.lo - pad).abs()
                                    ).clamp_min(1e-6).view(1, -1, 1, 1)
        self.qs = [torch.tensor(sign_component(d), dtype=torch.float32)
                   for d in sequence]
        self.mode = "seq"


def install_seq(model: nn.Module) -> list[SeqAct]:
    """Give every activation SITE its own module.

    Each BasicBlock reuses one nn.ReLU object for both of its activations, so
    module identity cannot separate them; the block forward is rebound to call
    two distinct modules. (Duplicated from the Phase 3 harness deliberately —
    this branch must not modify a file the other branch also sees.)
    """
    acts: list[SeqAct] = []
    model.relu = stem = SeqAct()
    acts.append(stem)
    for lname in ("layer1", "layer2", "layer3"):
        for blk in getattr(model, lname):
            blk.act_a, blk.act_b = SeqAct(), SeqAct()
            acts += [blk.act_a, blk.act_b]

            def fwd(self, x):
                identity = x
                out = self.act_a(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                if self.downsample is not None:
                    identity = self.downsample(x)
                out = out + identity
                return self.act_b(out)

            blk.forward = types.MethodType(fwd, blk)
    return acts


def sequences(max_len: int, ordered_probe: bool) -> list[tuple[int, ...]]:
    """Non-decreasing sequences over COMPONENTS, plus order probes.

    Non-decreasing is the natural family (coarse refinement first, sharpening
    last) and keeps the sweep to a size the full test set can absorb. A few
    permutations are added to test whether order matters at all.
    """
    seqs = []
    for n in range(2, max_len + 1):
        seqs += list(itertools.combinations_with_replacement(COMPONENTS, n))
    if ordered_probe:
        for s in [(3, 7, 15), (3, 15, 7), (7, 3, 15), (15, 7, 3),
                  (3, 7, 7, 15), (15, 7, 7, 3), (7, 15, 3, 7)]:
            if s not in seqs:
                seqs.append(s)
    return seqs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-len", type=int, default=5)
    ap.add_argument("--images", type=int, default=None)
    ap.add_argument("--calib-images", type=int, default=2000)
    ap.add_argument("--calib-split", choices=["train", "test"], default="train",
                    help="where the calibration images come from")
    ap.add_argument("--no-order-probe", dest="order_probe",
                    action="store_false")
    ap.add_argument("--sequences", nargs="+", default=None,
                    help="explicit sequences, e.g. 15-15-15 7-7-7-15; used to "
                         "re-confirm a screened frontier on the full test set")
    ap.add_argument("--out", default="oa_composite.csv")
    args = ap.parse_args()
    require_clean()

    model = load_model("resnet20")
    acts = install_seq(model)
    dl = loader(args.images)

    baseline = top1(model, dl)
    print(f"baseline (real ReLU) top-1 = {baseline:.2f}%\n")

    for a in acts:
        a.mode = "calibrate"
    top1(model, calib_loader(args.calib_images, args.calib_split))
    for a in acts:
        a.mode = "relu"

    rows = []
    if args.sequences:
        seqs = [tuple(int(x) for x in s.split("-")) for s in args.sequences]
    else:
        seqs = sequences(args.max_len, args.order_probe)
    print(f"evaluating {len(seqs)} component sequences over {COMPONENTS}")
    for seq in seqs:
        for a in acts:
            a.configure(seq)
        acc = top1(model, dl)
        for a in acts:
            a.mode = "relu"
        dA = sum(delta_A(d) for d in seq)
        dB = sum(delta_B(d) for d in seq)
        eff = int(np.prod(seq))
        rows.append({
            "sequence": "-".join(map(str, seq)),
            "n_components": len(seq), "effective_degree": eff,
            "depth_A": dA, "depth_B": dB, "top1": acc,
            "baseline_top1": baseline, "delta_points": acc - baseline,
            "within_1_point": bool(baseline - acc <= 1.0),
        })
        flag = "OK " if rows[-1]["within_1_point"] else "   "
        print(f"  {flag}{rows[-1]['sequence']:<18} eff={eff:<7} "
              f"depth A/B={dA:>2}/{dB:>2}  top-1={acc:6.2f}  "
              f"delta={acc - baseline:+6.2f}")

    df = pd.DataFrame(rows)
    write_csv(df, RESULTS / "activation_accuracy" / args.out,
              extra={"baseline_top1": baseline,
                     "components": str(COMPONENTS),
                     "n_images": args.images or 10000})

    ok = df[df.within_1_point].sort_values("depth_B")
    print("\n=== accuracy-viable configurations (<= 1 point), by depth ===")
    if len(ok):
        print(ok[["sequence", "effective_degree", "depth_A", "depth_B",
                  "top1", "delta_points"]].to_string(index=False))
        for m in ("depth_A", "depth_B"):
            d = sorted(ok[m].unique())
            steps = [b - a for a, b in zip(d, d[1:])]
            print(f"\n{m}: viable depths {d}")
            print(f"  minimum step between viable depths = "
                  f"{min(steps) if steps else 'n/a (single depth)'}")
            print(f"  distinct per-channel settings available = {len(d)}")
    else:
        print("  NONE")


if __name__ == "__main__":
    main()
