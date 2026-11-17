"""
PURPOSE
    Close pre-registered Condition 3 — "accuracy drops > 1 point at any
    latency-reducing setting" — WITHOUT encryption.

    Polynomial approximation is a plaintext operation. Replacing every ReLU
    with the per-channel Chebyshev polynomial the budget assigns, and measuring
    top-1 on CIFAR-10, answers the accuracy question exactly as the encrypted
    version would: CKKS evaluates the same polynomial, it just does so on
    ciphertexts. Nothing about the accuracy consequence needs a GPU or a
    ciphertext.

METHOD
    1. Calibrate: one pass with real ReLUs, recording per-channel mean and std
       of the input to every activation site. 19 sites in ResNet-20 (stem +
       2 per block). Calibration is used rather than BN statistics because the
       second activation in a block consumes the POST-ADD tensor, whose
       distribution is not bn2's.
    2. Interval: either the OBSERVED RANGE per channel (with a 5% margin),
       or mu +/- sigma*std. Coverage turns out to dominate: a Chebyshev fit
       diverges outside its interval, so a sigma-based Omega that misses the
       tail destroys accuracy at ANY degree. Both are swept to show it.
    3. Degrees: the decided budget, minimal d_c with s_c*eps_rel(d_c;t0_c) <= B,
       with the empirical std_c substituted for |a_c| as the interval scale —
       identical for BN outputs, correct for post-add ones.
    4. Substitute p_{d_c} for ReLU at every site and evaluate top-1 over the
       full 10,000-image test set.

    NO CLAMPING. Inputs outside Omega are extrapolated, exactly as they are
    under CKKS. That divergence is a real deployment failure mode, not an
    artifact, so it is left in and the interval rule is swept to price it.

INPUTS
    data/cifar-10-batches-py, pretrained cifar10_resnet20.

OUTPUTS
    results/phase3/accuracy_sweep.csv — per (sigma, B):
        top1, delta_vs_baseline, degree_min/median/max, mean_degree
"""

from __future__ import annotations

import argparse
import sys
import types
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.provenance import require_clean, write_csv  # noqa: E402
from src.channel_degrees import (  # noqa: E402
    bn_sites, invariants, load_model, required_degree,
)

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO.parent / "results"
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)


@lru_cache(maxsize=None)
def cheb_coeffs(degree: int, t0_q: int) -> tuple:
    """Chebyshev coefficients of g(t) = max(0, t - t0) on [-1,1]."""
    t0 = t0_q / 1000.0
    g = lambda t: np.maximum(0.0, t - t0)  # noqa: E731
    return tuple(np.polynomial.chebyshev.Chebyshev.interpolate(
        g, degree, domain=[-1, 1]).coef)


class PolyAct(nn.Module):
    """ReLU, or its per-channel Chebyshev surrogate."""

    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.mode = "relu"
        self.sum_ = self.sumsq = self.count = None
        self.lo = self.hi = None
        self.coef = self.center = self.half = None

    def forward(self, x):
        if self.mode == "calibrate":
            flat = x.transpose(0, 1).reshape(x.shape[1], -1)
            s, sq, n = flat.sum(1), (flat ** 2).sum(1), flat.shape[1]
            self.sum_ = s if self.sum_ is None else self.sum_ + s
            self.sumsq = sq if self.sumsq is None else self.sumsq + sq
            self.count = n if self.count is None else self.count + n
            lo, hi = flat.min(1).values, flat.max(1).values
            self.lo = lo if self.lo is None else torch.minimum(self.lo, lo)
            self.hi = hi if self.hi is None else torch.maximum(self.hi, hi)
            return torch.relu(x)
        if self.mode == "relu":
            return torch.relu(x)
        if self.mode == "composite":
            # No clamping, same standard as the Chebyshev path: FHE cannot
            # clamp, and the radius carries the same 5% margin.
            v = x / self.radius
            for _ in range(self.k):
                v = self._f7(v)
            return x * (1.0 + v) / 2.0
        # Clenshaw over the Chebyshev basis, per channel.
        c, ctr, h = self.coef, self.center, self.half
        t = (x - ctr) / h
        b1 = torch.zeros_like(x)
        b2 = torch.zeros_like(x)
        for k in range(c.shape[1] - 1, 0, -1):
            b1, b2 = 2 * t * b1 - b2 + c[:, k].view(1, -1, 1, 1), b1
        return h * (t * b1 - b2 + c[:, 0].view(1, -1, 1, 1))

    def stats(self):
        mu = self.sum_ / self.count
        var = (self.sumsq / self.count - mu ** 2).clamp_min(1e-12)
        return mu, var.sqrt(), self.lo, self.hi

    def configure_composite(self, radius, k: int):
        """Lee-et-al-style composite minimax: ReLU(x) = x*(1+sign(x))/2 with
        sign approximated by k-fold composition of one low-degree odd minimax
        polynomial, instead of one high-degree interpolant.

        The point is conditioning, not expressiveness. Composing k copies of a
        degree-7 polynomial reaches effective degree 7^k at depth k*delta(7),
        and each component is evaluated on [-1,1] where it is well behaved —
        which is exactly what a single degree-31 or degree-63 interpolant is
        not.
        """
        self.radius = radius.clamp_min(1e-6).view(1, -1, 1, 1)
        self.k = k
        self.mode = "composite"

    @staticmethod
    def _f7(z):
        """Degree-7 odd minimax step for the sign function on [-1,1]."""
        z2 = z * z
        return z * (35.0 + z2 * (-35.0 + z2 * (21.0 - 5.0 * z2))) / 16.0

    def configure(self, center, half, degrees):
        half = half.clamp_min(1e-6)
        mu = center
        t0 = (-center / half).clamp(-1.5, 1.5)
        K = int(max(degrees))
        coef = torch.zeros(len(degrees), K + 1)
        for c, d in enumerate(degrees):
            cc = cheb_coeffs(int(d), int(round(float(t0[c]) * 1000)))
            coef[c, :len(cc)] = torch.tensor(cc, dtype=torch.float32)
        self.coef, self.center, self.half = (
            coef, mu.view(1, -1, 1, 1), half.view(1, -1, 1, 1))
        self.mode = "poly"


def install(model: nn.Module) -> dict[str, PolyAct]:
    """Replace ReLU use-sites with distinct PolyAct modules.

    Each BasicBlock reuses ONE nn.ReLU object for both of its activations, so
    module identity cannot separate them. The block's forward is rebound to
    call two distinct modules instead.
    """
    acts: dict[str, PolyAct] = {}
    model.relu = acts["stem"] = PolyAct("stem")
    for lname in ("layer1", "layer2", "layer3"):
        for i, blk in enumerate(getattr(model, lname)):
            a = acts[f"{lname}.{i}.bn1"] = PolyAct(f"{lname}.{i}.bn1")
            b = acts[f"{lname}.{i}.bn2"] = PolyAct(f"{lname}.{i}.bn2")
            blk.act_a, blk.act_b = a, b

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


def loader(n: int | None):
    tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
    ds = torchvision.datasets.CIFAR10(root=REPO / "data", train=False,
                                      download=False, transform=tf)
    if n:
        ds = torch.utils.data.Subset(ds, range(n))
    return torch.utils.data.DataLoader(ds, batch_size=250)


def calib_loader(n: int, split: str = "train"):
    """The images the ranges are measured on.

    "train" (the default since 2026-11-16) takes the first n images of the
    CIFAR-10 training set, so calibration and scoring never share an image.
    "test" is the original behaviour: the first n TEST images, which are also
    scored. Kept only so the August files can be reproduced.
    """
    if split == "test":
        return loader(n)
    tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
    ds = torchvision.datasets.CIFAR10(root=REPO / "data", train=True,
                                      download=False, transform=tf)
    ds = torch.utils.data.Subset(ds, range(n))
    return torch.utils.data.DataLoader(ds, batch_size=250)


@torch.no_grad()
def top1(model, dl) -> float:
    ok = tot = 0
    for x, y in dl:
        ok += (model(x).argmax(1) == y).sum().item()
        tot += y.numel()
    return 100.0 * ok / tot


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--budgets", type=float, nargs="+",
                    default=[30.0, 10.0, 3.0, 1.0, 0.3])
    ap.add_argument("--intervals", nargs="+",
                    default=["minmax", "6", "3"],
                    help="how Omega is set: minmax = cover the "
                         "observed range; a number = that many "
                         "std either side of the mean")
    ap.add_argument("--max-degree", type=int, default=31)
    ap.add_argument("--approximant", choices=["chebyshev", "composite"],
                    default="chebyshev")
    ap.add_argument("--compositions", type=int, nargs="+", default=[1, 2, 3, 4],
                    help="k, the number of degree-7 minimax steps composed; "
                         "effective degree 7^k at depth k*delta(7)")
    ap.add_argument("--images", type=int, default=None)
    ap.add_argument("--calib-split", choices=["train", "test"], default="train",
                    help="where the 2,000 calibration images come from")
    args = ap.parse_args()
    require_clean()

    model = load_model("resnet20")
    acts = install(model)
    dl = loader(args.images)

    baseline = top1(model, dl)
    print(f"baseline (real ReLU) top-1 = {baseline:.2f}%")

    # v_norm per activation site, from the BN that precedes it.
    inv = invariants("resnet20", 1.0)
    vnorm = {s: g.sort_values("channel_idx").v_norm.to_numpy()
             for s, g in inv.groupby("site")}

    for a in acts.values():
        a.mode = "calibrate"
    top1(model, calib_loader(2000, args.calib_split))
    calib = {k: a.stats() for k, a in acts.items()}
    for a in acts.values():
        a.mode = "relu"

    site_map = {"stem": "bn1"}
    rows = []

    if args.approximant == "composite":
        for k in args.compositions:
            for key, a in acts.items():
                _mu, _std, lo, hi = calib[key]
                pad = 0.05 * (hi - lo)
                radius = torch.maximum((hi + pad).abs(), (lo - pad).abs())
                a.configure_composite(radius, k)
            acc = top1(model, dl)
            for a in acts.values():
                a.mode = "relu"
            depth = k * 4          # delta_B(7) = 4 levels per component
            rows.append({
                "approximant": "composite", "interval": "minmax",
                "compositions_k": k, "effective_degree": 7 ** k,
                "depth_levels_deltaB": depth,
                "top1": acc, "baseline_top1": baseline,
                "delta_points": acc - baseline,
                "within_1_point": bool(baseline - acc <= 1.0)})
            print(f"  composite k={k}  eff.deg={7**k:<6} depth={depth:<3} "
                  f"top-1={acc:6.2f}%  delta={acc - baseline:+6.2f}")
        df = pd.DataFrame(rows)
        write_csv(df, RESULTS / "activation_accuracy/accuracy_composite.csv",
                  extra={"baseline_top1": baseline,
                         "n_images": args.images or 10000,
                         "approximant": "Lee-style composite minimax sign",
                         "calib_split": args.calib_split})
        ok = df[df.within_1_point]
        print("\n=== Condition 3 under composite minimax ===")
        print(ok.to_string(index=False) if len(ok)
              else "  NONE within 1 point")
        return

    for interval in args.intervals:
        for B in args.budgets:
            degs_all = []
            for key, a in acts.items():
                mu, std, lo, hi = calib[key]
                v = vnorm.get(site_map.get(key, key))
                if v is None or len(v) != len(mu):
                    v = np.ones(len(mu))
                if interval == "minmax":
                    # Cover the observed range, with a margin for test data
                    # unseen at calibration. A Chebyshev fit diverges outside
                    # its interval, so coverage is a correctness requirement,
                    # not a tuning choice.
                    pad = 0.05 * (hi - lo)
                    center, half = (hi + lo) / 2, (hi - lo) / 2 + pad
                else:
                    center, half = mu, float(interval) * std
                half = half.clamp_min(1e-6)
                t0 = (-center / half).clamp(-1.5, 1.5)
                s_c = half.numpy() * v
                degs = [required_degree(float(s_c[c]), float(t0[c]), B,
                                        args.max_degree)
                        for c in range(len(mu))]
                a.configure(center, half, degs)
                degs_all.extend(degs)
            acc = top1(model, dl)
            for a in acts.values():
                a.mode = "relu"
            rows.append({
                "interval": interval, "budget_B": B, "top1": acc,
                "baseline_top1": baseline,
                "delta_points": acc - baseline,
                "degree_min": int(min(degs_all)),
                "degree_median": float(np.median(degs_all)),
                "degree_max": int(max(degs_all)),
                "degree_mean": float(np.mean(degs_all)),
                "within_1_point": bool(baseline - acc <= 1.0),
            })
            print(f"  interval={interval:<7} B={B:<8} d_med="
                  f"{rows[-1]['degree_median']:<5} d_max="
                  f"{rows[-1]['degree_max']:<3} top-1={acc:6.2f}%  "
                  f"delta={acc - baseline:+6.2f}")

    df = pd.DataFrame(rows)
    write_csv(df, RESULTS / "activation_accuracy/accuracy_sweep.csv",
              extra={"baseline_top1": baseline, "n_images": args.images or 10000,
                     "no_clamping": "inputs outside Omega are extrapolated",
                     "calib_split": args.calib_split})
    ok = df[df.within_1_point]
    print("\n=== Condition 3: settings within 1 point of baseline ===")
    print(ok[["interval", "budget_B", "degree_median", "degree_max", "top1",
              "delta_points"]].to_string(index=False) if len(ok)
          else "  NONE — no setting tested holds accuracy within 1 point")


if __name__ == "__main__":
    main()
