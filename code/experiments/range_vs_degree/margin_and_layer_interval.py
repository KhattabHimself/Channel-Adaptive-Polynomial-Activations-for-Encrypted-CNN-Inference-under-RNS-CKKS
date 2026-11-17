"""
Three follow-ups to the interval experiment, on the CIFAR ResNets (extension).

A. Margin. The thesis widens the observed range by 5% at each end and never
   varies that. Here the margin is 5, 10, 25, 50 and 100% of the range at
   degrees 15, 31 and 63, one Chebyshev interpolant per channel.
B. Per-channel against per-layer interval. Same degree, same margin; the
   per-layer interval runs from the smallest to the largest value any channel
   of the layer produced. This is the comparison the thesis names as its first
   item of future work.
C. How much the calibration sample matters: the 5% margin, degrees 15 and 31,
   with five different blocks of 2,000 training images as calibration set.
D. Paired tests on the same 10,000 test images: the unmodified network against
   the composite 7-15-15, and against the best single interpolant of part A.

Splits: calibration = training images 0 to 1,999 (parts A, B, D);
validation = training images 40,000 to 41,999; test = the 10,000 test images.
Validation accuracy is reported next to test accuracy so that a margin can be
chosen without looking at the test set. The validation images are training
images, which the network has seen, so accuracy on them is higher than on the
test set and losses are smaller.

run (from code/):
    python -m experiments.range_vs_degree.margin_and_layer_interval --data-root data
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import src.run_stamp as _stamp  # noqa: E402
from src.network_tools import install_cifar_resnet, load_model  # noqa: E402
from src.run_stamp import banner, write_table  # noqa: E402
from experiments.activation_accuracy.composite_sequences import install_seq  # noqa: E402
from experiments.range_vs_degree.range_vs_degree import NORM, TARGETS  # noqa: E402

MARGINS = (0.05, 0.10, 0.25, 0.50, 1.00)
DEGREES = (15, 31, 63)
VALID = (40000, 42000)


def tensors(dataset: str, root: Path, train: bool, lo=None, hi=None):
    import torchvision
    import torchvision.transforms as T
    mean, std = NORM[dataset]
    cls = {"cifar10": torchvision.datasets.CIFAR10,
           "cifar100": torchvision.datasets.CIFAR100}[dataset]
    ds = cls(root=root, train=train, download=True,
             transform=T.Compose([T.ToTensor(), T.Normalize(mean, std)]))
    idx = range(lo or 0, hi if hi is not None else len(ds))
    return (torch.stack([ds[i][0] for i in idx]), torch.tensor([ds[i][1] for i in idx]))


@torch.no_grad()
def correct(model, xs, ys, batch: int = 500):
    """per-image correctness (bool array) and share of non-finite outputs"""
    ok, bad = [], 0
    for i in range(0, len(xs), batch):
        out = model(xs[i:i + batch])
        bad += (~torch.isfinite(out)).any(1).sum().item()
        ok.append((out.argmax(1) == ys[i:i + batch]))
    return torch.cat(ok).numpy(), bad / len(xs)


def calibrate(model, acts, xs):
    for a in acts.values():
        a.reset_calibration()
        a.mode = "calibrate"
    correct(model, xs, torch.zeros(len(xs), dtype=torch.long))
    lohi = {}
    for k, a in acts.items():
        _mu, _sd, lo, hi = a.stats()
        lohi[k] = (lo.clone(), hi.clone())
        a.mode = "relu"
    return lohi


def configure(acts, lohi, degree: int, margin: float, per_layer: bool = False):
    for k, a in acts.items():
        lo, hi = lohi[k]
        if per_layer:
            lo, hi = lo.min().expand_as(lo), hi.max().expand_as(hi)
        pad = margin * (hi - lo)
        a.configure((hi + lo) / 2, (hi - lo) / 2 + pad, [degree] * len(lo))


def relu(acts):
    for a in acts.values():
        a.mode = "relu"


def mcnemar(a: np.ndarray, b: np.ndarray):
    """exact McNemar test and a paired bootstrap interval for acc(a) - acc(b), in points"""
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    p = stats.binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    rng = np.random.default_rng(0)
    d = a.astype(float) - b.astype(float)
    boot = [d[rng.integers(0, len(d), len(d))].mean() * 100 for _ in range(5000)]
    return only_a, only_b, p, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def main() -> None:
    _stamp.use_results("range_vs_degree")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--targets", nargs="+", default=["cifar10_resnet20"], choices=list(TARGETS))
    ap.add_argument("--parts", default="ABCD")
    ap.add_argument("--tag", default="", help="added to the output file names")
    args = ap.parse_args()
    root = Path(args.data_root)
    banner(["MARGIN, PER-LAYER INTERVAL, CALIBRATION SAMPLE, PAIRED TESTS"])
    torch.manual_seed(0)
    rows_a, rows_b, rows_c, rows_d = [], [], [], []
    for name in args.targets:
        dataset, _ = TARGETS[name]
        model = load_model(name)
        acts = install_cifar_resnet(model)
        tx, ty = tensors(dataset, root, False)
        vx, vy = tensors(dataset, root, True, *VALID)
        cx, _ = tensors(dataset, root, True, 0, 2000)
        base_ok, _ = correct(model, tx, ty)
        base_t, base_v = 100 * base_ok.mean(), 100 * correct(model, vx, vy)[0].mean()
        print(f"{name}: unmodified test {base_t:.2f}%, validation {base_v:.2f}%", flush=True)
        lohi = calibrate(model, acts, cx)
        best = None
        if "A" in args.parts:
            for m in MARGINS:
                for d in DEGREES:
                    t = time.time()
                    configure(acts, lohi, d, m)
                    v_ok, _ = correct(model, vx, vy)
                    t_ok, nf = correct(model, tx, ty)
                    relu(acts)
                    r = {"target": name, "margin": m, "degree": d,
                         "validation_top1": 100 * v_ok.mean(),
                         "validation_points_lost": base_v - 100 * v_ok.mean(),
                         "test_top1": 100 * t_ok.mean(), "test_points_lost": base_t - 100 * t_ok.mean(),
                         "test_nonfinite_frac": nf, "unmodified_test_top1": base_t,
                         "unmodified_validation_top1": base_v}
                    rows_a.append(r)
                    if best is None or r["validation_top1"] > best[0]["validation_top1"]:
                        best = (r, t_ok)
                    print(f"  A margin {m:<4} degree {d:<2} validation {r['validation_top1']:6.2f}%  "
                          f"test {r['test_top1']:6.2f}% (lost {r['test_points_lost']:5.2f})  "
                          f"({time.time() - t:.0f}s)", flush=True)
        if "B" in args.parts:
            m_best = best[0]["margin"] if best else 0.25
            for m in sorted({0.05, m_best}):
                for d in (15, 31):
                    for per_layer in (False, True):
                        configure(acts, lohi, d, m, per_layer)
                        t_ok, nf = correct(model, tx, ty)
                        relu(acts)
                        rows_b.append({"target": name, "margin": m, "degree": d,
                                       "interval": "per layer" if per_layer else "per channel",
                                       "test_top1": 100 * t_ok.mean(),
                                       "test_points_lost": base_t - 100 * t_ok.mean(),
                                       "test_nonfinite_frac": nf})
                        print(f"  B margin {m:<4} degree {d:<2} {rows_b[-1]['interval']:<11} "
                              f"test {rows_b[-1]['test_top1']:6.2f}%", flush=True)
        if "C" in args.parts:
            for block in range(5):
                bx, _ = tensors(dataset, root, True, 2000 * block, 2000 * (block + 1))
                lh = calibrate(model, acts, bx)
                for d in (15, 31):
                    configure(acts, lh, d, 0.05)
                    t_ok, nf = correct(model, tx, ty)
                    relu(acts)
                    rows_c.append({"target": name, "calibration_block": block, "degree": d,
                                   "margin": 0.05, "test_top1": 100 * t_ok.mean(),
                                   "test_nonfinite_frac": nf})
                    print(f"  C block {block} degree {d:<2} test {rows_c[-1]['test_top1']:6.2f}%",
                          flush=True)
        if "D" in args.parts and dataset == "cifar10" and name == "cifar10_resnet20":
            m2 = load_model(name)
            seq = install_seq(m2)
            for a in seq:
                a.mode = "calibrate"
            correct(m2, cx, torch.zeros(len(cx), dtype=torch.long))
            for sq in ((7, 15, 15), (7, 7, 15), (15, 15, 15)):
                for a in seq:
                    a.configure(sq)
                s_ok, _ = correct(m2, tx, ty)
                oa, ob, p, lo, hi = mcnemar(base_ok, s_ok)
                rows_d.append({"target": name, "compared": "composite " + "-".join(map(str, sq)),
                               "top1": 100 * s_ok.mean(), "unmodified_top1": base_t,
                               "points_lost": base_t - 100 * s_ok.mean(),
                               "only_unmodified_right": oa, "only_polynomial_right": ob,
                               "mcnemar_p": p, "loss_ci95_lo": lo, "loss_ci95_hi": hi})
                print(f"  D {rows_d[-1]['compared']:<22} lost {rows_d[-1]['points_lost']:.2f} "
                      f"[{lo:.2f}, {hi:.2f}]  McNemar p = {p:.2g}", flush=True)
            if best:
                oa, ob, p, lo, hi = mcnemar(base_ok, best[1])
                b = best[0]
                rows_d.append({"target": name,
                               "compared": f"single interpolant, degree {b['degree']}, margin {b['margin']}"
                                           " (best on validation)",
                               "top1": b["test_top1"], "unmodified_top1": base_t,
                               "points_lost": b["test_points_lost"],
                               "only_unmodified_right": oa, "only_polynomial_right": ob,
                               "mcnemar_p": p, "loss_ci95_lo": lo, "loss_ci95_hi": hi})
                print(f"  D {rows_d[-1]['compared']} lost {b['test_points_lost']:.2f} "
                      f"[{lo:.2f}, {hi:.2f}]  McNemar p = {p:.2g}", flush=True)
    meta = {"experiment": "margin, per-layer interval, calibration sample, paired tests (extension)"}
    if rows_a:
        a = pd.DataFrame(rows_a)
        v = "; ".join(
            f"{t}: best on validation margin {g.loc[g.validation_top1.idxmax()].margin:g} degree "
            f"{int(g.loc[g.validation_top1.idxmax()].degree)}, test loss "
            f"{g.loc[g.validation_top1.idxmax()].test_points_lost:.2f} points "
            f"(at the thesis margin of 0.05 the best test loss is "
            f"{g[g.margin == 0.05].test_points_lost.min():.2f})" for t, g in a.groupby("target"))
        write_table("ext_margin_sweep" + args.tag, a, meta, "Margin sweep. " + v)
        banner([v])
    if rows_b:
        b = pd.DataFrame(rows_b)
        p = b.pivot_table(index=["target", "margin", "degree"], columns="interval", values="test_top1")
        gain = p["per channel"] - p["per layer"]
        write_table("ext_per_layer_interval" + args.tag, b, meta,
                    "Per-channel minus per-layer interval, test points: " + "; ".join(
                        f"{t} margin {m:g} degree {d}: {g:+.2f}" for (t, m, d), g in gain.items()))
    if rows_c:
        c = pd.DataFrame(rows_c)
        s = c.groupby(["target", "degree"]).test_top1.agg(["mean", "std", "min", "max"])
        write_table("ext_calibration_blocks" + args.tag, c, meta,
                    "Five calibration blocks, 5% margin: " + "; ".join(
                        f"{t} degree {d}: {r['mean']:.2f} +/- {r['std']:.2f} "
                        f"({r['min']:.2f} to {r['max']:.2f})" for (t, d), r in s.iterrows()))
    if rows_d:
        d = pd.DataFrame(rows_d)
        write_table("ext_paired_tests" + args.tag, d, meta, "Paired tests against the unmodified network: "
                    + "; ".join(f"{r.compared}: lost {r.points_lost:.2f} "
                                f"[{r.loss_ci95_lo:.2f}, {r.loss_ci95_hi:.2f}], p = {r.mcnemar_p:.2g}"
                                for r in d.itertuples()))


if __name__ == "__main__":
    main()
